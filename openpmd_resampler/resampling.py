"""
This module contains various particle resampling strategies for particle in cell data.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from .log import logger
from .units import constants
from .utils import dataset_info
from .reader import DataFrameUpdater
from .vranic import VranicMerger


class ParticleResampler:
    def __init__(
        self,
        df: pd.DataFrame,
        weight_column: str = "weights",
        particle_species_mass: float = 1.0,
    ):
        self._df = df.copy()
        self.weight_column = weight_column
        self.particle_species_mass = particle_species_mass
        self.updater = DataFrameUpdater(self, particle_species_mass)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    def set_weights_to(self, new_weight: int = 1) -> pd.DataFrame:
        if new_weight == 1 and self.df[self.weight_column].nunique() != 1:
            raise ValueError("Not all weights are equal. Setting them to 1 might not be a good idea.")
        else:
            logger.info("Multiplicative factor for obtaining original charge: %.2f\n", self.df[self.weight_column].iloc[0])
            self.df[self.weight_column] = new_weight

        return self

    def random_weights(self) -> pd.DataFrame:
        min_weight = self.df[self.weight_column].min()
        max_weight = self.df[self.weight_column].max()
        random_generator = np.random.default_rng(seed=77125)
        self.df[self.weight_column] = random_generator.uniform(
            min_weight, max_weight, size=self.df.shape[0]
        )

        return self

    def simple_thinning(self, number_of_remaining_macroparticles: int) -> pd.DataFrame:
        number_of_remaining_macroparticles = int(number_of_remaining_macroparticles)

        random_generator = np.random.default_rng(seed=77125)
        number_of_initial_macroparticles = self.df.shape[0]

        # Generate random indices for deletion
        delete_indices = random_generator.choice(
            number_of_initial_macroparticles,
            size=number_of_initial_macroparticles - number_of_remaining_macroparticles,
            replace=False,
        )

        # Delete particles and weights at the selected indices
        self.df.drop(delete_indices, inplace=True)

        # Calculate new weight coefficient and update weights
        weight_factor = (
            number_of_initial_macroparticles / number_of_remaining_macroparticles
        )
        self.df[self.weight_column] *= weight_factor

        return self

    def drop_rows_using_mask(self, deletion_mask):
        self.df.loc[deletion_mask] = None
        self.df.dropna(inplace=True)

    def global_leveling_thinning(self, k: float = 2.0) -> pd.DataFrame:
        """
        If the initial number of macroparticles is N, the number after
        thinning will be roughly N/k.
        """
        average_weight = self.df[self.weight_column].mean()
        threshold_weight = k * average_weight

        # Generate random numbers for each particle
        random_generator = np.random.default_rng(seed=77125)
        random_numbers = random_generator.uniform(0.0, 1.0, size=self.df.shape[0])

        # Create a mask for particles to be deleted
        deletion_mask = (self.df[self.weight_column] < threshold_weight) & (
            random_numbers > (self.df[self.weight_column] / threshold_weight)
        )

        # Delete particles
        self.drop_rows_using_mask(deletion_mask)

        # Update weights for the remaining particles
        self.df[self.weight_column] = self.df[self.weight_column].where(
            self.df[self.weight_column] >= threshold_weight, threshold_weight
        )

        logger.info("Dataset after thinning.\n")
        dataset_info(self.df)

        return self

    def vranic_merging(
        self,
        spatial_bins: Tuple[int, int, int] = (16, 16, 16),
        momentum_bins: Tuple[int, int, int] = (16, 16, 16),
        momentum_coordinates: str = "spherical",
        min_packet_size: int = 4,
        max_packet_size: int = 4,
        log_scale: bool = False,
    ) -> pd.DataFrame:
        """
        Merge macroparticles using the algorithm of Vranic et al.,
        Comput. Phys. Commun. 191 (2015) 65-73.

        Particles are binned into spatial cells and momentum cells, and each
        packet of at least min_packet_size particles sharing a cell is replaced
        by two macroparticles conserving total weight, momentum and energy.
        Coarser binning merges more aggressively.

        The particle mass is taken from the particle_species_mass given to the
        constructor (relative to the electron mass, 0.0 for photons).
        """
        merger = VranicMerger(
            self.df,
            mass_mev_c2=self.particle_species_mass * constants.electron_mass_mev_c2,
            weight_column=self.weight_column,
        )
        self.df = merger.merge(
            spatial_bins=spatial_bins,
            momentum_bins=momentum_bins,
            momentum_coordinates=momentum_coordinates,
            min_packet_size=min_packet_size,
            max_packet_size=max_packet_size,
            log_scale=log_scale,
        )

        logger.info("Dataset after Vranic merging.\n")
        dataset_info(self.df)

        return self

    def repeat_and_perturb(self, percentage: float = 0.001) -> pd.DataFrame:
        """
        Repeat each row based on the 'weights' column, set all 'weights' to 1,
        and add a small random value to the position and momentum columns.
        """
        random_generator = np.random.default_rng(seed=77125)

        # Drop energy column
        kinetic_energy_mev_dropped = False
        if "kinetic_energy_mev" in self.df.columns:
            self.df.drop(columns=["kinetic_energy_mev"], inplace=True)
            kinetic_energy_mev_dropped = True

        # Repeat rows based on the 'weights' column
        self.df = self.df.loc[
            self.df.index.repeat(self.df[self.weight_column].astype(int))
        ]
        self.set_weights_to(1)

        # Get all columns except 'weights'
        cols = [col for col in self.df.columns if col != self.weight_column]

        # Compute mean and std for standardization
        mean = self.df.mean()
        std = self.df.std()

        # Standardize and add noise in-place
        for col in cols:
            self.df[col] = (self.df[col] - mean[col]) / std[col]
            epsilon = self.df[col].abs() * percentage
            self.df[col] += random_generator.normal(0, epsilon)
            self.df[col] = self.df[col] * std[col] + mean[col]

        self.df.reset_index(drop=True, inplace=True)

        # Recompute energy column
        if kinetic_energy_mev_dropped:
            self.updater.add_energy_column()

        return self

    def finalize(self) -> pd.DataFrame:
        logger.info("Final dataset to be exported.\n")
        dataset_info(self.df)
        return self.df
