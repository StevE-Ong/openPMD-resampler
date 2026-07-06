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
"""
from typing import Tuple

import numpy as np
import pandas as pd

from .log import logger

POSITION_COLUMNS = ["position_x_m", "position_y_m", "position_z_m"]
MOMENTUM_COLUMNS = ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]


def _uniform_bin_indices(values: np.ndarray, number_of_bins: int) -> np.ndarray:
    """Uniform binning of values between their global min and max."""
    vmin, vmax = values.min(), values.max()
    if vmax <= vmin:
        return np.zeros(values.shape[0], dtype=np.int64)
    indices = ((values - vmin) / (vmax - vmin) * number_of_bins).astype(np.int64)
    return np.clip(indices, 0, number_of_bins - 1)


def _grouped_bin_indices(
    values: np.ndarray, group_index: np.ndarray, group_starts: np.ndarray, number_of_bins: int
) -> np.ndarray:
    """
    Uniform binning of values between the min and max of their own group.
    All arrays must be sorted by group already.
    """
    group_min = np.minimum.reduceat(values, group_starts)[group_index]
    group_max = np.maximum.reduceat(values, group_starts)[group_index]
    width = group_max - group_min
    with np.errstate(divide="ignore", invalid="ignore"):
        indices = np.where(
            width > 0, (values - group_min) / width * number_of_bins, 0.0
        ).astype(np.int64)
    return np.clip(indices, 0, number_of_bins - 1)


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
        """
        if momentum_coordinates not in ("spherical", "cartesian"):
            raise ValueError("momentum_coordinates must be 'spherical' or 'cartesian'.")
        if min_packet_size < 3:
            raise ValueError("min_packet_size must be at least 3 for the merge to reduce particles.")
        if max_packet_size < min_packet_size:
            raise ValueError("max_packet_size must be >= min_packet_size.")

        position = self.df[POSITION_COLUMNS].to_numpy(np.float64)
        momentum = self.df[MOMENTUM_COLUMNS].to_numpy(np.float64)
        weight = self.df[self.weight_column].to_numpy(np.float64)
        energy = np.sqrt(np.sum(momentum**2, axis=1) + self.mass**2)

        number_initial = position.shape[0]
        total_weight_before = weight.sum()
        total_momentum_before = weight @ momentum
        total_energy_before = weight @ energy

        order = self.sort_into_cells(
            position, momentum, spatial_bins, momentum_bins, momentum_coordinates, log_scale
        )
        position, momentum = position[order], momentum[order]
        weight, energy = weight[order], energy[order]

        # Split each momentum cell into packets of at most max_packet_size particles.
        rank_in_cell = np.arange(number_initial) - self.cell_starts[self.cell_index]
        packet_boundary = self.new_cell | (rank_in_cell % max_packet_size == 0)
        packet_index = np.cumsum(packet_boundary) - 1
        packet_starts = np.flatnonzero(packet_boundary)
        packet_sizes = np.bincount(packet_index)

        # Per-packet conserved quantities.
        number_of_packets = packet_sizes.shape[0]
        total_weight = np.bincount(packet_index, weights=weight)
        total_energy = np.bincount(packet_index, weights=weight * energy)
        total_momentum = np.column_stack(
            [np.bincount(packet_index, weights=weight * momentum[:, i]) for i in range(3)]
        )
        momentum_norm = np.linalg.norm(total_momentum, axis=1)

        # Momentum magnitude of the two new particles, from energy conservation:
        # each carries weight wt/2 and energy et/wt.
        with np.errstate(divide="ignore", invalid="ignore"):
            average_energy = np.where(total_weight > 0, total_energy / total_weight, 0.0)
        new_momentum_norm = np.sqrt(np.maximum(average_energy**2 - self.mass**2, 0.0))

        # A packet can only be merged if it is large enough and if the two new
        # momenta can be oriented (impossible when the total momentum vanishes
        # while the particles still carry energy, e.g. two opposite beams).
        mergeable = packet_sizes >= min_packet_size
        mergeable &= ~((momentum_norm == 0.0) & (new_momentum_norm > 0.0))

        # Momentum conservation: both new momenta lie in a plane containing the
        # total momentum, at angles +/- omega from it, with
        # cos(omega) = |pt| / (wt * p_new).
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_omega = np.where(
                new_momentum_norm > 0,
                momentum_norm / (total_weight * new_momentum_norm),
                1.0,
            )
        cos_omega = np.clip(cos_omega, 0.0, 1.0)
        sin_omega = np.sqrt(1.0 - cos_omega**2)

        e1 = np.divide(
            total_momentum,
            momentum_norm[:, np.newaxis],
            out=np.tile(np.array([1.0, 0.0, 0.0]), (number_of_packets, 1)),
            where=momentum_norm[:, np.newaxis] > 0,
        )
        # The plane is spanned by e1 and the coordinate axis least aligned with it.
        least_aligned_axis = np.eye(3)[np.argmin(np.abs(e1), axis=1)]
        e3 = np.cross(e1, least_aligned_axis)
        e3 /= np.linalg.norm(e3, axis=1, keepdims=True)
        e2 = np.cross(e3, e1)

        parallel = (new_momentum_norm * cos_omega)[:, np.newaxis] * e1
        transverse = (new_momentum_norm * sin_omega)[:, np.newaxis] * e2
        new_momentum_a = (parallel + transverse)[mergeable]
        new_momentum_b = (parallel - transverse)[mergeable]
        new_weight = total_weight[mergeable] / 2.0

        # The two new particles inherit the positions of the packet's first two
        # members, which lie inside the same spatial cell.
        merged_starts = packet_starts[mergeable]
        new_position_a = position[merged_starts]
        new_position_b = position[merged_starts + 1]

        kept = ~mergeable[packet_index]
        merged_position = np.concatenate([position[kept], new_position_a, new_position_b])
        merged_momentum = np.concatenate([momentum[kept], new_momentum_a, new_momentum_b])
        merged_weight = np.concatenate([weight[kept], new_weight, new_weight])

        self.log_merge_statistics(
            number_initial,
            merged_weight,
            merged_momentum,
            total_weight_before,
            total_momentum_before,
            total_energy_before,
        )

        return self.build_dataframe(merged_position, merged_momentum, merged_weight)

    def sort_into_cells(
        self,
        position: np.ndarray,
        momentum: np.ndarray,
        spatial_bins: Tuple[int, int, int],
        momentum_bins: Tuple[int, int, int],
        momentum_coordinates: str,
        log_scale: bool,
    ) -> np.ndarray:
        """
        Sort particles into (spatial cell, momentum cell) groups and return the
        sorting order. Stores the group index, starts and boundaries on self.
        """
        spatial_index = np.ravel_multi_index(
            [_uniform_bin_indices(position[:, i], spatial_bins[i]) for i in range(3)],
            spatial_bins,
        )
        spatial_order = np.argsort(spatial_index, kind="stable")
        sorted_spatial = spatial_index[spatial_order]
        new_spatial = np.empty(sorted_spatial.shape[0], dtype=bool)
        new_spatial[0] = True
        new_spatial[1:] = sorted_spatial[1:] != sorted_spatial[:-1]
        spatial_group = np.cumsum(new_spatial) - 1
        spatial_starts = np.flatnonzero(new_spatial)

        if momentum_coordinates == "spherical":
            sorted_momentum = momentum[spatial_order]
            momentum_norm = np.linalg.norm(sorted_momentum, axis=1)
            radial = momentum_norm
            if log_scale:
                positive = momentum_norm[momentum_norm > 0]
                floor = positive.min() if positive.size > 0 else 1.0
                radial = np.log10(np.maximum(momentum_norm, floor))
            theta = np.arctan2(sorted_momentum[:, 1], sorted_momentum[:, 0])
            with np.errstate(divide="ignore", invalid="ignore"):
                phi = np.where(
                    momentum_norm > 0, np.arcsin(sorted_momentum[:, 2] / momentum_norm), 0.0
                )
            coordinates = (radial, theta, phi)
        else:
            if log_scale:
                raise ValueError("log_scale is only supported with spherical momentum coordinates.")
            coordinates = tuple(momentum[spatial_order, i] for i in range(3))

        # Momentum cell boundaries are local to each spatial cell, as in Smilei.
        momentum_index = np.ravel_multi_index(
            [
                _grouped_bin_indices(coordinates[i], spatial_group, spatial_starts, momentum_bins[i])
                for i in range(3)
            ],
            momentum_bins,
        )
        cell_key = spatial_group * np.prod(momentum_bins) + momentum_index

        suborder = np.argsort(cell_key, kind="stable")
        order = spatial_order[suborder]
        sorted_key = cell_key[suborder]

        self.new_cell = np.empty(sorted_key.shape[0], dtype=bool)
        self.new_cell[0] = True
        self.new_cell[1:] = sorted_key[1:] != sorted_key[:-1]
        self.cell_index = np.cumsum(self.new_cell) - 1
        self.cell_starts = np.flatnonzero(self.new_cell)

        return order

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
