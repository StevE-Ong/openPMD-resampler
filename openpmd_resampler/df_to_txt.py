"""
Module: df_to_txt
This module provides a class for writing pandas DataFrame to a text file with custom headers.
"""

import os

import pandas as pd

from .log import logger
from .units import units
from .utils import format_file_size

MOMENTUM_COLUMNS = ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]


class DataFrameToFile:
    """
    A class used to write a pandas DataFrame to a text file with a custom header.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame to be written to a text file.
        """
        self.df = df
        self.units = units
        self.include_weights = True
        self.include_energy = True
        self.momentum_divisor = None

    def exclude_weights(self):
        self.include_weights = False
        return self

    def exclude_energy(self):
        self.include_energy = False
        return self

    def momentum_in_mc(self, mass_mev_c2: float):
        """
        Write momenta as normalized momentum u = p / (m c) instead of MeV/c,
        where m is the particle species mass (openpmd_viewer's 'ux' convention).
        """
        if mass_mev_c2 <= 0:
            raise ValueError(
                "momentum_in_mc requires a positive species mass;"
                " u = p / (m c) is undefined for massless particles."
            )
        self.momentum_divisor = mass_mev_c2
        return self

    def write_to_file(self, file_path, fortran_binary=False):
        columns_to_write = self.df.columns.tolist()
        if not self.include_weights:
            columns_to_write.remove("weights")
        if not self.include_energy:
            columns_to_write.remove("kinetic_energy_mev")

        logger.info("Writing dataframe to file. This may take a while...\n")
        if fortran_binary:
            self._write_fortran_unformatted(file_path, columns_to_write)
        else:
            self._write_csv(file_path, columns_to_write)
        logger.info("Wrote %s\n", file_path)

        file_size = os.path.getsize(file_path)
        logger.info("Final file size: %s\n", format_file_size(file_size))

    def _column_data(self, column):
        data = self.df[column]
        if self.momentum_divisor is not None and column in MOMENTUM_COLUMNS:
            # float32 / python float stays float32
            data = data / self.momentum_divisor
        return data

    def _column_unit(self, column):
        if self.momentum_divisor is not None and column in MOMENTUM_COLUMNS:
            return "m*c"
        return self.units[column]

    def _write_csv(self, file_path, columns_to_write):
        with open(file_path, "w", encoding="utf-8") as f:
            header = ", ".join(
                f"{column} ({self._column_unit(column)})" for column in columns_to_write
            )
            f.write(header + "\n")
        df = pd.DataFrame({column: self._column_data(column) for column in columns_to_write})
        df.to_csv(
            file_path,
            index=False,
            header=False,
            sep=",",
            float_format="%.7e",
            mode="a",
        )

    def _write_fortran_unformatted(self, file_path, columns_to_write):
        import numpy as np
        from scipy.io import FortranFile

        with FortranFile(file_path, "w") as f:
            f.write_record(np.array([len(self.df)], dtype=np.int32))
            for col in columns_to_write:
                f.write_record(self._column_data(col).to_numpy().astype(np.float32))
