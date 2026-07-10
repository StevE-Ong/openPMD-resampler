"""
Particle merging algorithm of P. T. Luu, T. Tueckmantel and A. Pukhov,
"Voronoi particle merging algorithm for PIC codes",
Comput. Phys. Commun. 202 (2016) 165-174, doi:10.1016/j.cpc.2016.01.009.

Particles are grouped into initial spatial cells that are subdivided
recursively along the direction of largest spread, first in position and then
in momentum space, until every cluster is narrower than the given thresholds.
Each surviving cluster is replaced by a single macroparticle carrying the
cluster's total weight, weighted mean position and weighted mean momentum,
which conserves charge and momentum exactly; the energy error is bounded by
the momentum spread threshold.

The implementation follows the particleMerging plugin of the PIConGPU code
(https://github.com/ComputationalRadiationPhysics/picongpu/tree/0.5.0/include/picongpu/plugins/particleMerging),
including its modification of the original algorithm: the position
subdivision is always completed before the momentum spread is examined.
It differs from PIConGPU in two points: the initial Voronoi cells are the
spatial bins given by the user instead of blocks of PIC grid cells (there is
no grid in post-processing), and the minimum mean energy criterion is only
applied in the momentum stage, where the mean momentum is available.

The numerics run on PyTorch tensors in the float32 precision of the input
data, on the GPU when one is available: the "cuda" device covers both NVIDIA
(CUDA) and AMD (ROCm/HIP) builds of PyTorch, and the same code runs on the
CPU otherwise. Since initial cells never interact, the particles are
processed in chunks of whole initial cells sized to the free GPU memory, so
datasets much larger than the GPU still merge; only the chunk at hand lives
on the device. Within a chunk, particles are kept sorted so that each
Voronoi cell is a contiguous segment: cluster statistics are index_add_
scatter sums keyed by the per-particle segment index, and a split is an O(n)
stable partition of the segment, so no sorting happens after the initial
spatial binning. The two splitting stages run as separate passes, each
touching only the coordinates it subdivides, and the splitting itself only
touches the particles of the splitting clusters. On the GPU the
floating-point scatter sums use atomics, so results are not bit-reproducible
between runs; conservation statistics are accumulated in float64 on the CPU
for logging.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .log import logger
from .units import constants

try:
    import torch
except ImportError:  # torch is only needed for Voronoi merging
    torch = None

POSITION_COLUMNS = ["position_x_m", "position_y_m", "position_z_m"]
MOMENTUM_COLUMNS = ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]

# Rough transient GPU bytes per particle at the peak of a splitting pass,
# used to size the chunks of initial cells.
BYTES_PER_PARTICLE = 160


def _resolve_device(device: Optional[str]) -> "torch.device":
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda":
        available = torch.cuda.device_count()
        if available == 0:
            raise ValueError(
                f"device '{device}' requested but PyTorch sees no GPU;"
                " omit the device to fall back to the CPU."
            )
        if resolved.index is not None and resolved.index >= available:
            raise ValueError(
                f"device '{device}' requested but only {available} GPU(s) are"
                f" present (cuda:0..cuda:{available - 1})."
            )
    return resolved


def _cluster_stats(
    values: "torch.Tensor",
    weights: "torch.Tensor",
    segment: "torch.Tensor",
    number_of_segments: int,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Weighted mean and variance of each column of values per segment."""
    weighted = weights[:, None] * values
    zeros = lambda *shape: torch.zeros(*shape, dtype=values.dtype, device=values.device)
    weight_sums = zeros(number_of_segments).index_add_(0, segment, weights)
    mean = zeros((number_of_segments, 3)).index_add_(0, segment, weighted)
    mean /= weight_sums[:, None]
    mean_square = zeros((number_of_segments, 3)).index_add_(0, segment, weighted * values)
    mean_square /= weight_sums[:, None]
    variance = torch.clamp(mean_square - mean**2, min=0.0)
    return weight_sums, mean, variance


class _SplitPlan:
    """
    Partition plan for the splitting segments, computed only on their
    particles. Each particle's side is decided as in PIConGPU: the lower
    sub-cell takes the particles strictly below the mean of the component of
    largest spread.
    """

    def __init__(
        self,
        values: "torch.Tensor",
        counts: "torch.Tensor",
        segment: "torch.Tensor",
        splits: "torch.Tensor",
        mean: "torch.Tensor",
        variance: "torch.Tensor",
    ):
        device = values.device
        self.rows = torch.nonzero(splits[segment]).squeeze(1)
        self.counts = counts[splits]
        self.starts = torch.cumsum(self.counts, dim=0) - self.counts
        self.segment = torch.repeat_interleave(
            torch.arange(self.counts.shape[0], device=device), self.counts
        )

        component = torch.argmax(variance[splits], dim=1)[self.segment]
        self.side = values[self.rows, component] >= mean[splits][self.segment, component]
        self.side_int = self.side.long()
        self.higher_counts = torch.zeros(
            self.counts.shape[0], dtype=torch.long, device=device
        ).index_add_(0, self.segment, self.side_int)

    def degenerate(self) -> "torch.Tensor":
        """Splits that leave one sub-cell empty and thus make no progress."""
        return (self.higher_counts == 0) | (self.higher_counts == self.counts)

    def scatter_to_children(
        self, arrays: Tuple["torch.Tensor", ...]
    ) -> Tuple[Tuple["torch.Tensor", ...], "torch.Tensor"]:
        """
        Compact the particles of the splitting segments into two contiguous
        child segments each (lower side first, preserving order), dropping
        everything else. Returns the compacted arrays and the new starts.
        """
        child_counts = torch.stack(
            (self.counts - self.higher_counts, self.higher_counts), dim=1
        ).reshape(-1)
        child_starts = torch.cumsum(child_counts, dim=0) - child_counts

        rank_in_segment = (
            torch.arange(self.rows.shape[0], device=self.rows.device)
            - self.starts[self.segment]
        )
        higher_before = torch.cumsum(self.side_int, dim=0) - self.side_int
        higher_before = higher_before - higher_before[self.starts][self.segment]
        offset_in_child = torch.where(self.side, higher_before, rank_in_segment - higher_before)
        destination = child_starts[2 * self.segment + self.side_int] + offset_in_child

        compacted = []
        for array in arrays:
            gathered = array[self.rows]
            new = torch.empty_like(gathered)
            new[destination] = gathered
            compacted.append(new)
        return tuple(compacted), child_starts


class VoronoiMerger:
    def __init__(self, df: pd.DataFrame, mass_mev_c2: float, weight_column: str = "weights"):
        self.df = df
        self.mass = mass_mev_c2  # 0.0 for photons
        self.weight_column = weight_column

    def merge(
        self,
        spatial_bins: Tuple[int, int, int] = (16, 16, 16),
        min_particles_to_merge: int = 8,
        pos_spread_threshold: float = 0.5,
        abs_mom_spread_threshold: float = -1.0,
        rel_mom_spread_threshold: float = -1.0,
        min_mean_energy_kev: float = 511.0,
        device: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Merge macroparticles, conserving total weight and momentum exactly and
        energy approximately.

        spatial_bins: number of initial Voronoi cells along (x, y, z).
        min_particles_to_merge: minimum number of macroparticles in a Voronoi
            cell needed to merge them into a single macroparticle.
        pos_spread_threshold: below this spread (standard deviation) in
            position a cell can be merged, in units of the initial spatial
            cell edge length.
        abs_mom_spread_threshold: below this absolute spread in momentum a
            cell can be merged, in units of m_e*c. Disabled for -1.
        rel_mom_spread_threshold: below this spread in momentum relative to
            the cell's mean momentum a cell can be merged. Disabled for -1.
            Exactly one of the two momentum thresholds must be enabled.
        min_mean_energy_kev: minimum mean kinetic energy in keV of a Voronoi
            cell needed to merge it (0 disables the criterion).
        device: PyTorch device string, e.g. "cuda", "cuda:1" or "cpu". The
            default picks the GPU when one is available (NVIDIA and AMD ROCm
            builds of PyTorch both expose it as "cuda") and the CPU otherwise.
            Datasets larger than the GPU memory are processed in chunks of
            whole initial cells, which does not change the result.
        """
        if torch is None:
            raise ImportError(
                "Voronoi merging requires PyTorch; install the 'pytorch-gpu' (or"
                " 'pytorch' for CPU-only) package in the pixi environment."
            )
        if min_particles_to_merge < 2:
            raise ValueError("min_particles_to_merge must be at least 2 for the merge to reduce particles.")
        if (abs_mom_spread_threshold > 0) == (rel_mom_spread_threshold > 0):
            raise ValueError(
                "Exactly one of abs_mom_spread_threshold and rel_mom_spread_threshold"
                " must be positive; disable the other one with -1."
            )
        if min_mean_energy_kev < 0:
            raise ValueError("min_mean_energy_kev must be non-negative.")

        torch_device = _resolve_device(device)
        logger.info("Voronoi merging on device '%s'.\n", torch_device)

        position_np = self.df[POSITION_COLUMNS].to_numpy(np.float32)
        momentum_np = self.df[MOMENTUM_COLUMNS].to_numpy(np.float32)
        weight_np = self.df[self.weight_column].to_numpy(np.float32)

        # Conservation totals for the log, in float64 on the CPU.
        number_initial = position_np.shape[0]
        weight64 = weight_np.astype(np.float64)
        momentum64 = momentum_np.astype(np.float64)
        total_weight_before = weight64.sum()
        total_momentum_before = weight64 @ momentum64
        total_energy_before = weight64 @ np.sqrt(
            np.einsum("ij,ij->i", momentum64, momentum64) + self.mass**2
        )

        # Express positions in units of the initial cell edge length, so that
        # position spreads are compared like in PIConGPU (cell edge units).
        origin = position_np.min(axis=0)
        extent = position_np.max(axis=0) - origin
        cell_edge = np.where(
            extent > 0, extent / np.asarray(spatial_bins, dtype=np.float32), np.float32(1.0)
        ).astype(np.float32)

        # Initial cell key and one stable sort, on the device; the key is
        # built column by column to keep the footprint at a few bytes per
        # particle. Afterwards every initial cell is a contiguous segment.
        key_dtype = torch.int32 if int(np.prod(spatial_bins)) < 2**31 else torch.int64
        key = torch.zeros(number_initial, dtype=key_dtype, device=torch_device)
        for axis in range(3):
            column = torch.tensor(position_np[:, axis], device=torch_device)
            column.sub_(float(origin[axis])).div_(float(cell_edge[axis]))
            column.clamp_(0.0, float(spatial_bins[axis] - 1))
            key.mul_(spatial_bins[axis]).add_(column.int())
        del column
        order = torch.argsort(key, stable=True)
        sorted_key = key[order]
        del key
        order_np = order.cpu().numpy()
        boundary = torch.empty(number_initial, dtype=torch.bool, device=torch_device)
        boundary[0] = True
        boundary[1:] = sorted_key[1:] != sorted_key[:-1]
        cell_starts = torch.nonzero(boundary).squeeze(1).cpu().numpy()
        del order, sorted_key, boundary

        # Group whole initial cells into chunks the free GPU memory can hold;
        # cells never interact, so chunking does not change the result. A
        # single cell larger than the target becomes its own chunk.
        chunk_target = number_initial
        if torch_device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(torch_device)
            chunk_target = max(int(free_bytes * 0.45) // BYTES_PER_PARTICLE, 1_000_000)
        chunk_of_cell = cell_starts // chunk_target
        chunk_start_cells = np.concatenate(
            [[0], np.flatnonzero(np.diff(chunk_of_cell)) + 1]
        )
        cell_offsets = np.append(cell_starts, number_initial)
        chunk_bounds = np.append(chunk_start_cells, cell_starts.size)
        if chunk_start_cells.size > 1:
            logger.info(
                "Processing %d chunks of initial cells to fit the GPU memory.\n",
                chunk_start_cells.size,
            )

        pos_threshold2 = pos_spread_threshold**2
        abs_mom_threshold2 = float(abs_mom_spread_threshold * constants.electron_mass_mev_c2) ** 2
        min_mean_energy_mev = min_mean_energy_kev / 1000.0

        origin_gpu = torch.tensor(origin, device=torch_device)
        cell_edge_gpu = torch.tensor(cell_edge, device=torch_device)

        kept_list = []
        merged_position_list, merged_momentum_list, merged_weight_list = [], [], []
        for first_cell, last_cell in zip(chunk_bounds[:-1], chunk_bounds[1:]):
            first_row, last_row = cell_offsets[first_cell], cell_offsets[last_cell]
            rows_np = order_np[first_row:last_row]
            scaled_position = torch.tensor(position_np[rows_np], device=torch_device)
            scaled_position.sub_(origin_gpu).div_(cell_edge_gpu)
            working_weight = torch.tensor(weight_np[rows_np], device=torch_device)
            row = torch.tensor(rows_np, dtype=torch.long, device=torch_device)
            starts = torch.tensor(
                cell_starts[first_cell:last_cell] - first_row,
                dtype=torch.long,
                device=torch_device,
            )
            kept_np, merged = self._merge_cells(
                scaled_position,
                working_weight,
                row,
                starts,
                position_np,
                momentum_np,
                weight_np,
                origin_gpu,
                cell_edge_gpu,
                min_particles_to_merge,
                pos_threshold2,
                abs_mom_threshold2,
                rel_mom_spread_threshold,
                min_mean_energy_mev,
            )
            kept_list.append(kept_np)
            merged_position_list.append(merged[0])
            merged_momentum_list.append(merged[1])
            merged_weight_list.append(merged[2])

        kept = np.concatenate(kept_list)
        new_position = np.concatenate([position_np[kept]] + merged_position_list)
        new_momentum = np.concatenate([momentum_np[kept]] + merged_momentum_list)
        new_weight = np.concatenate([weight_np[kept]] + merged_weight_list)

        self.log_merge_statistics(
            number_initial,
            new_weight,
            new_momentum,
            total_weight_before,
            total_momentum_before,
            total_energy_before,
        )

        return self.build_dataframe(new_position, new_momentum, new_weight)

    def _merge_cells(
        self,
        scaled_position: "torch.Tensor",
        working_weight: "torch.Tensor",
        row: "torch.Tensor",
        starts: "torch.Tensor",
        position_np: np.ndarray,
        momentum_np: np.ndarray,
        weight_np: np.ndarray,
        origin_gpu: "torch.Tensor",
        cell_edge_gpu: "torch.Tensor",
        min_particles_to_merge: int,
        pos_threshold2: float,
        abs_mom_threshold2: float,
        rel_mom_spread_threshold: float,
        min_mean_energy_mev: float,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Run the two splitting stages on one chunk of whole initial cells,
        already sorted so each cell is a contiguous segment. row carries the
        original row of every particle; positions, momenta and weights of
        specific rows are re-gathered from the CPU arrays when needed, so the
        device only ever holds the chunk. Returns the kept original rows and
        the merged (position, momentum, weight) as numpy arrays.
        """
        torch_device = scaled_position.device
        mass = float(self.mass)
        empty_rows = torch.empty(0, dtype=torch.long, device=torch_device)
        kept_rows = [np.empty(0, dtype=np.int64)]

        # ---- Position stage: split until every cluster is spatially narrow.
        # Only positions and weights are carried; row tracks original rows.
        pending_rows = [empty_rows]
        pending_counts = [empty_rows]

        while working_weight.shape[0] > 0:
            counts = torch.diff(starts, append=starts.new_tensor([working_weight.shape[0]]))
            segment = torch.repeat_interleave(
                torch.arange(starts.shape[0], device=torch_device), counts
            )

            _, mean, variance = _cluster_stats(
                scaled_position, working_weight, segment, starts.shape[0]
            )

            abort = counts < min_particles_to_merge
            splits = ~abort & (variance.amax(dim=1) > pos_threshold2)
            plan = _SplitPlan(scaled_position, counts, segment, splits, mean, variance)

            degenerate = plan.degenerate()
            if degenerate.any().item():
                bad = torch.nonzero(splits).squeeze(1)[degenerate]
                abort[bad] = True
                splits[bad] = False
                plan = _SplitPlan(scaled_position, counts, segment, splits, mean, variance)

            # Spatially narrow clusters queue up for the momentum stage.
            narrow = ~abort & ~splits
            kept_rows.append(row[abort[segment]].cpu().numpy())
            pending_rows.append(row[narrow[segment]])
            pending_counts.append(counts[narrow])

            (scaled_position, working_weight, row), starts = plan.scatter_to_children(
                (scaled_position, working_weight, row)
            )
        del scaled_position

        # ---- Momentum stage: split the spatially narrow clusters until their
        # momentum spread is small, then merge them. Momenta and positions are
        # gathered from the CPU arrays only for the rows that need them.
        row = torch.cat(pending_rows)
        counts = torch.cat(pending_counts)
        starts = torch.cumsum(counts, dim=0) - counts
        rows_np = row.cpu().numpy()
        working_momentum = torch.tensor(momentum_np[rows_np], device=torch_device)
        working_weight = torch.tensor(weight_np[rows_np], device=torch_device)

        merged_positions = [np.empty((0, 3), dtype=np.float32)]
        merged_momenta = [np.empty((0, 3), dtype=np.float32)]
        merged_weights = [np.empty(0, dtype=np.float32)]

        while working_weight.shape[0] > 0:
            counts = torch.diff(starts, append=starts.new_tensor([working_weight.shape[0]]))
            segment = torch.repeat_interleave(
                torch.arange(starts.shape[0], device=torch_device), counts
            )

            weight_sums, mean, variance = _cluster_stats(
                working_momentum, working_weight, segment, starts.shape[0]
            )

            # Choose the momentum spread threshold of each cluster.
            mean_norm2 = torch.sum(mean**2, dim=1)
            if rel_mom_spread_threshold > 0:
                mom_threshold2 = rel_mom_spread_threshold**2 * mean_norm2
            else:
                mom_threshold2 = abs_mom_threshold2
            mean_energy = torch.sqrt(mean_norm2 + mass**2) - mass

            abort = counts < min_particles_to_merge
            abort |= mean_energy < min_mean_energy_mev
            splits = ~abort & (variance.amax(dim=1) > mom_threshold2)
            plan = _SplitPlan(working_momentum, counts, segment, splits, mean, variance)

            degenerate = plan.degenerate()
            if degenerate.any().item():
                bad = torch.nonzero(splits).squeeze(1)[degenerate]
                abort[bad] = True
                splits[bad] = False
                plan = _SplitPlan(working_momentum, counts, segment, splits, mean, variance)

            merges = ~abort & ~splits
            kept_rows.append(row[abort[segment]].cpu().numpy())

            if merges.any().item():
                merge_rows_np = row[merges[segment]].cpu().numpy()
                merge_counts = counts[merges]
                merge_segment = torch.repeat_interleave(
                    torch.arange(merge_counts.shape[0], device=torch_device), merge_counts
                )
                merge_scaled = torch.tensor(position_np[merge_rows_np], device=torch_device)
                merge_scaled.sub_(origin_gpu).div_(cell_edge_gpu)
                merge_weight = torch.tensor(weight_np[merge_rows_np], device=torch_device)
                centroid_sums = torch.zeros(
                    (merge_counts.shape[0], 3), dtype=merge_scaled.dtype, device=torch_device
                ).index_add_(0, merge_segment, merge_weight[:, None] * merge_scaled)
                centroids = centroid_sums / weight_sums[merges][:, None]
                merged_positions.append(
                    (centroids * cell_edge_gpu + origin_gpu).cpu().numpy()
                )
                merged_momenta.append(mean[merges].cpu().numpy())
                merged_weights.append(weight_sums[merges].cpu().numpy())

            (working_momentum, working_weight, row), starts = plan.scatter_to_children(
                (working_momentum, working_weight, row)
            )

        return np.concatenate(kept_rows), (
            np.concatenate(merged_positions),
            np.concatenate(merged_momenta),
            np.concatenate(merged_weights),
        )

    def build_dataframe(
        self, position: np.ndarray, momentum: np.ndarray, weight: np.ndarray
    ) -> pd.DataFrame:
        columns = {}
        for i, name in enumerate(POSITION_COLUMNS):
            columns[name] = position[:, i]
        for i, name in enumerate(MOMENTUM_COLUMNS):
            columns[name] = momentum[:, i]
        columns[self.weight_column] = weight
        if "kinetic_energy_mev" in self.df.columns:
            columns["kinetic_energy_mev"] = (
                np.sqrt(np.sum(momentum**2, axis=1) + self.mass**2) - self.mass
            )

        merged_df = pd.DataFrame(columns, dtype=np.float32)
        return merged_df[[name for name in self.df.columns if name in columns]]

    def log_merge_statistics(
        self,
        number_initial: int,
        weight: np.ndarray,
        momentum: np.ndarray,
        total_weight_before: float,
        total_momentum_before: np.ndarray,
        total_energy_before: float,
    ):
        number_final = weight.shape[0]
        weight = weight.astype(np.float64)
        momentum = momentum.astype(np.float64)
        energy = np.sqrt(np.sum(momentum**2, axis=1) + self.mass**2)
        logger.info(
            "Voronoi merging: %d -> %d macroparticles (%.2f%% reduction).\n",
            number_initial,
            number_final,
            100.0 * (1.0 - number_final / number_initial),
        )
        logger.info(
            "Relative conservation errors: weight %.2e, momentum %.2e, energy %.2e.\n",
            abs(weight.sum() - total_weight_before) / total_weight_before,
            np.linalg.norm(weight @ momentum - total_momentum_before)
            / max(np.linalg.norm(total_momentum_before), np.finfo(float).eps),
            abs(weight @ energy - total_energy_before) / total_energy_before,
        )
