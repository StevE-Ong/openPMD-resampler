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

Particles are kept sorted so that each Voronoi cell is a contiguous segment:
cluster statistics are segment sums (np.add.reduceat) over the float32
particle data and a split is an O(n) stable partition of the segment, so no
sorting happens after the initial spatial binning. The two splitting stages
run as separate passes, each touching only the coordinates it subdivides, and
the splitting itself only touches the particles of the splitting clusters.
All arithmetic is done in the single precision of the input data.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from .log import logger
from .units import constants

POSITION_COLUMNS = ["position_x_m", "position_y_m", "position_z_m"]
MOMENTUM_COLUMNS = ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]


def _cluster_stats(
    values: np.ndarray, weights: np.ndarray, starts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted mean and variance of each column of values per segment."""
    weighted = weights[:, np.newaxis] * values
    weight_sums = np.add.reduceat(weights, starts)
    mean = np.add.reduceat(weighted, starts) / weight_sums[:, np.newaxis]
    mean_square = np.add.reduceat(weighted * values, starts) / weight_sums[:, np.newaxis]
    variance = np.maximum(mean_square - mean**2, 0.0)
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
        values: np.ndarray,
        counts: np.ndarray,
        segment: np.ndarray,
        splits: np.ndarray,
        mean: np.ndarray,
        variance: np.ndarray,
    ):
        self.rows = np.flatnonzero(splits[segment])
        self.counts = counts[splits]
        self.starts = np.cumsum(self.counts) - self.counts
        self.segment = np.repeat(np.arange(self.counts.size, dtype=np.int32), self.counts)

        component = variance[splits].argmax(axis=1)[self.segment]
        self.side = values[self.rows, component] >= mean[splits][self.segment, component]
        self.side_int = self.side.astype(np.int32)
        if self.counts.size > 0:
            self.higher_counts = np.add.reduceat(self.side_int, self.starts)
        else:
            self.higher_counts = np.zeros(0, dtype=np.int32)

    def degenerate(self) -> np.ndarray:
        """Splits that leave one sub-cell empty and thus make no progress."""
        return (self.higher_counts == 0) | (self.higher_counts == self.counts)

    def scatter_to_children(
        self, arrays: Tuple[np.ndarray, ...]
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """
        Compact the particles of the splitting segments into two contiguous
        child segments each (lower side first, preserving order), dropping
        everything else. Returns the compacted arrays and the new starts.
        """
        child_counts = np.column_stack((self.counts - self.higher_counts, self.higher_counts)).ravel()
        child_starts = np.cumsum(child_counts) - child_counts

        rank_in_segment = np.arange(self.rows.size) - self.starts[self.segment]
        higher_before = np.cumsum(self.side_int) - self.side_int
        higher_before -= higher_before[self.starts][self.segment]
        offset_in_child = np.where(self.side, higher_before, rank_in_segment - higher_before)
        destination = child_starts[2 * self.segment + self.side_int] + offset_in_child

        compacted = []
        for array in arrays:
            gathered = array[self.rows]
            new = np.empty_like(gathered)
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
        """
        if min_particles_to_merge < 2:
            raise ValueError("min_particles_to_merge must be at least 2 for the merge to reduce particles.")
        if (abs_mom_spread_threshold > 0) == (rel_mom_spread_threshold > 0):
            raise ValueError(
                "Exactly one of abs_mom_spread_threshold and rel_mom_spread_threshold"
                " must be positive; disable the other one with -1."
            )
        if min_mean_energy_kev < 0:
            raise ValueError("min_mean_energy_kev must be non-negative.")

        position = np.ascontiguousarray(self.df[POSITION_COLUMNS].to_numpy())
        momentum = np.ascontiguousarray(self.df[MOMENTUM_COLUMNS].to_numpy())
        weight = self.df[self.weight_column].to_numpy()

        number_initial = position.shape[0]
        total_weight_before = weight.sum()
        total_momentum_before = weight @ momentum
        energy = np.sqrt(np.einsum("ij,ij->i", momentum, momentum) + self.mass**2)
        total_energy_before = weight @ energy
        del energy

        # Express positions in units of the initial cell edge length, so that
        # position spreads are compared like in PIConGPU (cell edge units).
        origin = position.min(axis=0)
        extent = position.max(axis=0) - origin
        bins = np.asarray(spatial_bins, dtype=np.int32)
        cell_edge = np.where(extent > 0, extent / bins, 1.0).astype(position.dtype)
        all_scaled_positions = (position - origin) / cell_edge

        # Sort the particles by initial cell, once; afterwards every Voronoi
        # cell stays a contiguous segment described by its start offset. The
        # cast to the narrowest integer type makes the radix sort cheaper.
        initial_cell = np.ravel_multi_index(
            tuple(
                np.clip(all_scaled_positions[:, i].astype(np.int32), 0, bins[i] - 1)
                for i in range(3)
            ),
            bins,
        ).astype(np.min_scalar_type(int(bins.prod()) - 1))
        order = np.argsort(initial_cell, kind="stable")
        sorted_cell = initial_cell[order]
        new_segment = np.empty(number_initial, dtype=bool)
        new_segment[0] = True
        new_segment[1:] = sorted_cell[1:] != sorted_cell[:-1]
        starts = np.flatnonzero(new_segment)

        pos_threshold2 = pos_spread_threshold**2
        abs_mom_threshold2 = float(abs_mom_spread_threshold * constants.electron_mass_mev_c2) ** 2
        min_mean_energy_mev = min_mean_energy_kev / 1000.0

        row_dtype = np.int32 if number_initial < 2**31 else np.int32
        kept_rows = [np.empty(0, dtype=row_dtype)]

        # ---- Position stage: split until every cluster is spatially narrow.
        # Only positions and weights are carried; row tracks original rows.
        scaled_position = all_scaled_positions[order]
        working_weight = weight[order]
        row = order.astype(row_dtype)
        pending_rows = [np.empty(0, dtype=row_dtype)]
        pending_counts = [np.empty(0, dtype=np.int32)]

        while working_weight.size > 0:
            counts = np.diff(starts, append=working_weight.size)
            segment = np.repeat(np.arange(starts.size, dtype=np.int32), counts)

            _, mean, variance = _cluster_stats(scaled_position, working_weight, starts)

            abort = counts < min_particles_to_merge
            splits = ~abort & (variance.max(axis=1) > pos_threshold2)
            plan = _SplitPlan(scaled_position, counts, segment, splits, mean, variance)

            degenerate = plan.degenerate()
            if degenerate.any():
                bad = np.flatnonzero(splits)[degenerate]
                abort[bad] = True
                splits[bad] = False
                plan = _SplitPlan(scaled_position, counts, segment, splits, mean, variance)

            # Spatially narrow clusters queue up for the momentum stage.
            narrow = ~abort & ~splits
            kept_rows.append(row[abort[segment]])
            pending_rows.append(row[narrow[segment]])
            pending_counts.append(counts[narrow])

            (scaled_position, working_weight, row), starts = plan.scatter_to_children(
                (scaled_position, working_weight, row)
            )

        # ---- Momentum stage: split the spatially narrow clusters until their
        # momentum spread is small, then merge them. Positions are looked up
        # from the original rows only when clusters actually merge.
        row = np.concatenate(pending_rows)
        counts = np.concatenate(pending_counts)
        starts = np.cumsum(counts) - counts
        working_momentum = momentum[row]
        working_weight = weight[row]

        merged_positions = [np.empty((0, 3), dtype=position.dtype)]
        merged_momenta = [np.empty((0, 3), dtype=momentum.dtype)]
        merged_weights = [np.empty(0, dtype=weight.dtype)]

        while working_weight.size > 0:
            counts = np.diff(starts, append=working_weight.size)
            segment = np.repeat(np.arange(starts.size, dtype=np.int32), counts)

            weight_sums, mean, variance = _cluster_stats(working_momentum, working_weight, starts)

            # Choose the momentum spread threshold of each cluster.
            if rel_mom_spread_threshold > 0:
                mom_threshold2 = rel_mom_spread_threshold**2 * np.sum(mean**2, axis=1)
            else:
                mom_threshold2 = abs_mom_threshold2
            mean_energy = np.sqrt(np.sum(mean**2, axis=1) + self.mass**2) - self.mass

            abort = counts < min_particles_to_merge
            abort |= mean_energy < min_mean_energy_mev
            splits = ~abort & (variance.max(axis=1) > mom_threshold2)
            plan = _SplitPlan(working_momentum, counts, segment, splits, mean, variance)

            degenerate = plan.degenerate()
            if degenerate.any():
                bad = np.flatnonzero(splits)[degenerate]
                abort[bad] = True
                splits[bad] = False
                plan = _SplitPlan(working_momentum, counts, segment, splits, mean, variance)

            merges = ~abort & ~splits
            kept_rows.append(row[abort[segment]])

            if merges.any():
                merge_rows = row[merges[segment]]
                merge_counts = counts[merges]
                merge_starts = np.cumsum(merge_counts) - merge_counts
                centroid_sums = np.add.reduceat(
                    weight[merge_rows, np.newaxis] * all_scaled_positions[merge_rows],
                    merge_starts,
                )
                merged_positions.append(centroid_sums / weight_sums[merges, np.newaxis])
                merged_momenta.append(mean[merges])
                merged_weights.append(weight_sums[merges])

            (working_momentum, working_weight, row), starts = plan.scatter_to_children(
                (working_momentum, working_weight, row)
            )

        kept = np.concatenate(kept_rows)
        new_position = np.concatenate(
            [position[kept], np.concatenate(merged_positions) * cell_edge + origin]
        )
        new_momentum = np.concatenate([momentum[kept], np.concatenate(merged_momenta)])
        new_weight = np.concatenate([weight[kept], np.concatenate(merged_weights)])

        self.log_merge_statistics(
            number_initial,
            new_weight,
            new_momentum,
            total_weight_before,
            total_momentum_before,
            total_energy_before,
        )

        return self.build_dataframe(new_position, new_momentum, new_weight)

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
