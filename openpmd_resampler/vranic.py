"""
Particle merging algorithm of M. Vranic et al.,
"Merging of macro-particles in particle-in-cell simulations",
Comput. Phys. Commun. 191 (2015) 65-73, doi:10.1016/j.cpc.2015.01.020.

Particles are binned into spatial cells, the momentum space of each cell is
subdivided into momentum cells (spherical or cartesian coordinates), and every
packet of particles sharing a momentum cell is replaced by two macroparticles
that exactly conserve total weight, momentum and energy.

The implementation follows the one in the Smilei PIC code
(https://github.com/SmileiPIC/Smilei/tree/master/src/Merging), with the
momentum cell boundaries computed per spatial cell.

The numerics run on PyTorch tensors in the float32 precision of the input
data, on the GPU when one is available: the "cuda" device covers both NVIDIA
(CUDA) and AMD (ROCm/HIP) builds of PyTorch, and the same code runs on the
CPU otherwise. Since spatial cells never interact, the particles are sorted
by spatial cell once and then processed in chunks of whole cells sized to
the free GPU memory, so datasets much larger than the GPU still merge; only
the chunk at hand lives on the device. Within a chunk, per-cell momentum
bounds are scatter min/max reductions keyed by the cell, a stable sort on
the combined (spatial cell, momentum cell) key groups the particles into
packets, and packet sums are index_add_ scatter reductions. On the GPU the
floating-point scatter sums use atomics, so results are not bit-reproducible
between runs (conservation still holds to float32 precision). Conservation
statistics are accumulated in float64 on the CPU for logging.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .log import logger

try:
    import torch
except ImportError:  # torch is only needed for Vranic merging
    torch = None

POSITION_COLUMNS = ["position_x_m", "position_y_m", "position_z_m"]
MOMENTUM_COLUMNS = ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]

# Rough transient GPU bytes per particle at the peak of a chunk, used to
# size the chunks of spatial cells.
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


def _uniform_bin_indices(values: "torch.Tensor", number_of_bins: int) -> "torch.Tensor":
    """Uniform binning of values between their global min and max."""
    vmin, vmax = torch.aminmax(values)
    width = vmax - vmin
    scaled = torch.where(width > 0, (values - vmin) / width * number_of_bins, 0.0)
    return scaled.long().clamp_(0, number_of_bins - 1)


def _grouped_bin_indices(
    values: "torch.Tensor",
    group_index: "torch.Tensor",
    number_of_groups: int,
    number_of_bins: int,
) -> "torch.Tensor":
    """Uniform binning of values between the min and max of their own group."""
    group_min = torch.zeros(
        number_of_groups, dtype=values.dtype, device=values.device
    ).scatter_reduce_(0, group_index, values, reduce="amin", include_self=False)
    group_max = torch.zeros(
        number_of_groups, dtype=values.dtype, device=values.device
    ).scatter_reduce_(0, group_index, values, reduce="amax", include_self=False)
    low = group_min[group_index]
    width = group_max[group_index] - low
    indices = torch.where(width > 0, (values - low) / width * number_of_bins, 0.0)
    return indices.long().clamp_(0, number_of_bins - 1)


class VranicMerger:
    def __init__(self, df: pd.DataFrame, mass_mev_c2: float, weight_column: str = "weights"):
        self.df = df
        self.mass = mass_mev_c2  # 0.0 for photons
        self.weight_column = weight_column

    def merge(
        self,
        spatial_bins: Tuple[int, int, int] = (16, 16, 16),
        momentum_bins: Tuple[int, int, int] = (16, 16, 16),
        momentum_coordinates: str = "spherical",
        min_packet_size: int = 4,
        max_packet_size: int = 4,
        log_scale: bool = False,
        device: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Merge macroparticles, conserving total weight, momentum and energy.

        spatial_bins: number of position bins along (x, y, z).
        momentum_bins: number of momentum bins, (p, theta, phi) for spherical
            coordinates or (px, py, pz) for cartesian ones.
        momentum_coordinates: "spherical" (recommended in the paper) or "cartesian".
        min_packet_size, max_packet_size: bounds on the number of particles
            merged at once into two macroparticles.
        log_scale: bin the momentum norm logarithmically (spherical only),
            useful for broad energy spectra.
        device: PyTorch device string, e.g. "cuda", "cuda:1" or "cpu". The
            default picks the GPU when one is available (NVIDIA and AMD ROCm
            builds of PyTorch both expose it as "cuda") and the CPU otherwise.
            Datasets larger than the GPU memory are processed in chunks of
            whole spatial cells, which does not change the result.
        """
        if torch is None:
            raise ImportError(
                "Vranic merging requires PyTorch; install the 'pytorch-gpu' (or"
                " 'pytorch' for CPU-only) package in the pixi environment."
            )
        if momentum_coordinates not in ("spherical", "cartesian"):
            raise ValueError("momentum_coordinates must be 'spherical' or 'cartesian'.")
        if log_scale and momentum_coordinates != "spherical":
            raise ValueError("log_scale is only supported with spherical momentum coordinates.")
        if min_packet_size < 3:
            raise ValueError("min_packet_size must be at least 3 for the merge to reduce particles.")
        if max_packet_size < min_packet_size:
            raise ValueError("max_packet_size must be >= min_packet_size.")

        torch_device = _resolve_device(device)
        logger.info("Vranic merging on device '%s'.\n", torch_device)

        position_np = self.df[POSITION_COLUMNS].to_numpy(np.float32)
        momentum_np = self.df[MOMENTUM_COLUMNS].to_numpy(np.float32)
        weight_np = self.df[self.weight_column].to_numpy(np.float32)

        # Conservation totals for the log, in float64 on the CPU.
        number_initial = position_np.shape[0]
        weight64 = weight_np.astype(np.float64)
        momentum64 = momentum_np.astype(np.float64)
        momentum_norm2 = np.einsum("ij,ij->i", momentum64, momentum64)
        total_weight_before = weight64.sum()
        total_momentum_before = weight64 @ momentum64
        total_energy_before = weight64 @ np.sqrt(momentum_norm2 + self.mass**2)

        # The log-scale floor is the smallest positive momentum norm of the
        # whole dataset, as before chunking.
        log_floor = None
        if log_scale:
            positive = momentum_norm2[momentum_norm2 > 0]
            log_floor = float(np.sqrt(positive.min())) if positive.size > 0 else 1.0
        del momentum64, momentum_norm2

        # Spatial cell key and one stable sort, on the device; the key is
        # built column by column to keep the footprint at a few bytes per
        # particle. Afterwards every spatial cell is a contiguous segment.
        key_dtype = torch.int32 if int(np.prod(spatial_bins)) < 2**31 else torch.int64
        key = torch.zeros(number_initial, dtype=key_dtype, device=torch_device)
        for axis in range(3):
            column = torch.tensor(position_np[:, axis], device=torch_device)
            key.mul_(spatial_bins[axis]).add_(
                _uniform_bin_indices(column, spatial_bins[axis]).to(key_dtype)
            )
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

        # Group whole spatial cells into chunks the free GPU memory can hold;
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
                "Processing %d chunks of spatial cells to fit the GPU memory.\n",
                chunk_start_cells.size,
            )

        kept_list = []
        merged_position_list, merged_momentum_list, merged_weight_list = [], [], []
        for first_cell, last_cell in zip(chunk_bounds[:-1], chunk_bounds[1:]):
            first_row, last_row = cell_offsets[first_cell], cell_offsets[last_cell]
            rows_np = order_np[first_row:last_row]
            cell_counts_np = np.diff(
                cell_offsets[first_cell : last_cell + 1]
            )
            kept_np, merged = self._merge_cells(
                rows_np,
                cell_counts_np,
                position_np,
                momentum_np,
                weight_np,
                momentum_bins,
                momentum_coordinates,
                min_packet_size,
                max_packet_size,
                log_floor,
                torch_device,
            )
            kept_list.append(kept_np)
            merged_position_list.append(merged[0])
            merged_momentum_list.append(merged[1])
            merged_weight_list.append(merged[2])

        kept = np.concatenate(kept_list)
        merged_position = np.concatenate([position_np[kept]] + merged_position_list)
        merged_momentum = np.concatenate([momentum_np[kept]] + merged_momentum_list)
        merged_weight = np.concatenate([weight_np[kept]] + merged_weight_list)

        self.log_merge_statistics(
            number_initial,
            merged_weight,
            merged_momentum,
            total_weight_before,
            total_momentum_before,
            total_energy_before,
        )

        return self.build_dataframe(merged_position, merged_momentum, merged_weight)

    def _merge_cells(
        self,
        rows_np: np.ndarray,
        cell_counts_np: np.ndarray,
        position_np: np.ndarray,
        momentum_np: np.ndarray,
        weight_np: np.ndarray,
        momentum_bins: Tuple[int, int, int],
        momentum_coordinates: str,
        min_packet_size: int,
        max_packet_size: int,
        log_floor: Optional[float],
        torch_device: "torch.device",
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Merge one chunk of whole spatial cells, already sorted so each cell
        is a contiguous segment described by cell_counts_np. rows_np carries
        the original row of every particle; only the chunk lives on the
        device. Returns the kept original rows and the merged
        (position, momentum, weight) as numpy arrays.
        """
        number_chunk = rows_np.size
        momentum = torch.tensor(momentum_np[rows_np], device=torch_device)
        weight = torch.tensor(weight_np[rows_np], device=torch_device)
        counts = torch.tensor(cell_counts_np, dtype=torch.long, device=torch_device)
        spatial_cell = torch.repeat_interleave(
            torch.arange(counts.shape[0], device=torch_device), counts
        )

        if momentum_coordinates == "spherical":
            momentum_norm = torch.linalg.norm(momentum, dim=1)
            radial = momentum_norm
            if log_floor is not None:
                radial = torch.log10(torch.clamp(momentum_norm, min=log_floor))
            theta = torch.atan2(momentum[:, 1], momentum[:, 0])
            tiny = torch.finfo(momentum.dtype).tiny
            # For norm 0 the ratio is 0/tiny = 0; the clamp guards against
            # |pz|/|p| exceeding 1 by rounding in float32.
            phi = torch.arcsin(
                (momentum[:, 2] / momentum_norm.clamp_min(tiny)).clamp_(-1.0, 1.0)
            )
            coordinates = (radial, theta, phi)
        else:
            coordinates = (momentum[:, 0], momentum[:, 1], momentum[:, 2])

        # Momentum cell boundaries are local to each spatial cell, as in Smilei.
        number_of_cells = counts.shape[0]
        momentum_index = (
            _grouped_bin_indices(
                coordinates[0], spatial_cell, number_of_cells, momentum_bins[0]
            ) * momentum_bins[1]
            + _grouped_bin_indices(
                coordinates[1], spatial_cell, number_of_cells, momentum_bins[1]
            )
        ) * momentum_bins[2] + _grouped_bin_indices(
            coordinates[2], spatial_cell, number_of_cells, momentum_bins[2]
        )
        cell_key = spatial_cell * int(np.prod(momentum_bins)) + momentum_index
        del coordinates, momentum_index, spatial_cell

        suborder = torch.argsort(cell_key, stable=True)
        sorted_key = cell_key[suborder]
        row = torch.tensor(rows_np, device=torch_device)[suborder]
        momentum = momentum[suborder]
        weight = weight[suborder]
        mass = float(self.mass)
        energy = torch.sqrt(torch.sum(momentum**2, dim=1) + mass**2)

        # Split each momentum cell into packets of at most max_packet_size particles.
        new_cell = torch.empty(number_chunk, dtype=torch.bool, device=torch_device)
        new_cell[0] = True
        new_cell[1:] = sorted_key[1:] != sorted_key[:-1]
        del cell_key, sorted_key, suborder
        cell_index = torch.cumsum(new_cell, dim=0) - 1
        cell_starts = torch.nonzero(new_cell).squeeze(1)
        rank_in_cell = (
            torch.arange(number_chunk, device=torch_device) - cell_starts[cell_index]
        )
        packet_boundary = new_cell | (rank_in_cell % max_packet_size == 0)
        packet_index = torch.cumsum(packet_boundary, dim=0) - 1
        packet_starts = torch.nonzero(packet_boundary).squeeze(1)
        packet_sizes = torch.diff(packet_starts, append=packet_starts.new_tensor([number_chunk]))

        # Per-packet conserved quantities, as scatter sums over the packets.
        number_of_packets = packet_starts.shape[0]
        zeros = lambda *shape: torch.zeros(*shape, dtype=weight.dtype, device=torch_device)
        total_weight = zeros(number_of_packets).index_add_(0, packet_index, weight)
        total_energy = zeros(number_of_packets).index_add_(0, packet_index, weight * energy)
        total_momentum = zeros((number_of_packets, 3)).index_add_(
            0, packet_index, weight[:, None] * momentum
        )
        momentum_norm = torch.linalg.norm(total_momentum, dim=1)

        # Momentum magnitude of the two new particles, from energy conservation:
        # each carries weight wt/2 and energy et/wt. The factored difference of
        # squares loses less precision in float32 than average_energy**2 - mass**2.
        tiny = torch.finfo(weight.dtype).tiny
        average_energy = total_energy / total_weight.clamp_min(tiny)
        new_momentum_norm = torch.sqrt(
            ((average_energy - mass) * (average_energy + mass)).clamp_min(0.0)
        )

        # A packet can only be merged if it is large enough and if the two new
        # momenta can be oriented (impossible when the total momentum vanishes
        # while the particles still carry energy, e.g. two opposite beams).
        mergeable = packet_sizes >= min_packet_size
        mergeable &= ~((momentum_norm == 0.0) & (new_momentum_norm > 0.0))

        # Momentum conservation: both new momenta lie in a plane containing the
        # total momentum, at angles +/- omega from it, with
        # cos(omega) = |pt| / (wt * p_new).
        cos_omega = torch.where(
            new_momentum_norm > 0,
            momentum_norm / (total_weight * new_momentum_norm).clamp_min(tiny),
            1.0,
        ).clamp_(0.0, 1.0)
        sin_omega = torch.sqrt(1.0 - cos_omega**2)

        e1 = torch.where(
            momentum_norm[:, None] > 0,
            total_momentum / momentum_norm.clamp_min(tiny)[:, None],
            total_momentum.new_tensor([1.0, 0.0, 0.0]),
        )
        # The plane is spanned by e1 and the coordinate axis least aligned with it.
        axes = torch.eye(3, dtype=e1.dtype, device=torch_device)
        least_aligned_axis = axes[torch.argmin(torch.abs(e1), dim=1)]
        e3 = torch.linalg.cross(e1, least_aligned_axis)
        e3 = e3 / torch.linalg.norm(e3, dim=1, keepdim=True)
        e2 = torch.linalg.cross(e3, e1)

        parallel = (new_momentum_norm * cos_omega)[:, None] * e1
        transverse = (new_momentum_norm * sin_omega)[:, None] * e2
        new_momentum_a = (parallel + transverse)[mergeable].cpu().numpy()
        new_momentum_b = (parallel - transverse)[mergeable].cpu().numpy()
        new_weight = (total_weight[mergeable] / 2.0).cpu().numpy()

        # The two new particles inherit the positions of the packet's first two
        # members, which lie inside the same spatial cell.
        merged_starts = packet_starts[mergeable]
        rows_first = row[merged_starts].cpu().numpy()
        rows_second = row[merged_starts + 1].cpu().numpy()
        kept_np = row[~mergeable[packet_index]].cpu().numpy()

        merged_position = np.concatenate([position_np[rows_first], position_np[rows_second]])
        merged_momentum = np.concatenate([new_momentum_a, new_momentum_b])
        merged_weight = np.concatenate([new_weight, new_weight])
        return kept_np, (merged_position, merged_momentum, merged_weight)

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
            "Vranic merging: %d -> %d macroparticles (%.2f%% reduction).\n",
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
