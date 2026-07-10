"""Conservation tests for the resampling algorithms.

Each algorithm's primary value is conserving specific quantities:

- global leveling thinning: total weight (charge), statistically;
- Vranic merging: total weight, momentum and energy, exactly per packet;
- Voronoi merging: total weight and momentum exactly, energy up to the
  momentum spread threshold.

All totals are accumulated in float64 to keep the float32 storage of the
DataFrame from polluting the comparison.
"""
import numpy as np
import pandas as pd
import pytest

from openpmd_resampler.resampling import ParticleResampler
from openpmd_resampler.units import constants

ELECTRON_MASS = constants.electron_mass_mev_c2


def devices():
    result = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            result.append("cuda")
    except ImportError:
        pass
    return result


def synthetic_beam(n=200_000, seed=1234):
    """A relativistic electron bunch with non-uniform weights.

    Mean momentum is nonzero along every axis so that each total-momentum
    component is a meaningful conservation check (no cancellation to ~0).
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "position_x_m": rng.normal(0.0, 2e-6, n),
            "position_y_m": rng.normal(0.0, 2e-6, n),
            "position_z_m": rng.normal(0.0, 5e-6, n),
            "momentum_x_mev_c": rng.normal(3.0, 1.0, n),
            "momentum_y_mev_c": rng.normal(-2.0, 1.0, n),
            "momentum_z_mev_c": rng.normal(50.0, 5.0, n),
            "weights": rng.uniform(0.5, 2.0, n),
        },
        dtype=np.float32,
    )
    p_squared = (
        df["momentum_x_mev_c"].astype(np.float64) ** 2
        + df["momentum_y_mev_c"].astype(np.float64) ** 2
        + df["momentum_z_mev_c"].astype(np.float64) ** 2
    )
    df["kinetic_energy_mev"] = (
        np.sqrt(p_squared + ELECTRON_MASS**2) - ELECTRON_MASS
    ).astype(np.float32)
    return df


def totals(df):
    w = df["weights"].to_numpy(dtype=np.float64)
    return {
        "weight": w.sum(),
        "momentum": np.array(
            [
                (w * df[f"momentum_{c}_mev_c"].to_numpy(dtype=np.float64)).sum()
                for c in "xyz"
            ]
        ),
        "energy": (w * df["kinetic_energy_mev"].to_numpy(dtype=np.float64)).sum(),
    }


@pytest.fixture(scope="module")
def beam():
    return synthetic_beam()


@pytest.mark.parametrize("device", devices())
def test_vranic_conserves_weight_momentum_and_energy(beam, device):
    before = totals(beam)
    df_merged = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .vranic_merging(
            spatial_bins=(4, 4, 4), momentum_bins=(8, 8, 8), device=device
        )
        .finalize()
    )
    after = totals(df_merged)

    assert len(df_merged) < len(beam)
    assert after["weight"] == pytest.approx(before["weight"], rel=1e-5)
    # Compare momentum against the total momentum scale, not per component,
    # so a component close to zero cannot inflate the relative error.
    momentum_scale = np.linalg.norm(before["momentum"])
    assert np.abs(after["momentum"] - before["momentum"]).max() < 1e-5 * momentum_scale
    assert after["energy"] == pytest.approx(before["energy"], rel=1e-5)
    assert list(df_merged.columns) == list(beam.columns)
    assert (df_merged.dtypes == np.float32).all()


@pytest.mark.parametrize("device", devices())
def test_voronoi_conserves_weight_and_momentum(beam, device):
    rel_mom_spread_threshold = 0.1
    before = totals(beam)
    df_merged = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .voronoi_merging(
            spatial_bins=(4, 4, 4),
            min_particles_to_merge=8,
            pos_spread_threshold=0.5,
            rel_mom_spread_threshold=rel_mom_spread_threshold,
            min_mean_energy_kev=0.0,
            device=device,
        )
        .finalize()
    )
    after = totals(df_merged)

    assert len(df_merged) < len(beam)
    assert after["weight"] == pytest.approx(before["weight"], rel=1e-5)
    momentum_scale = np.linalg.norm(before["momentum"])
    assert np.abs(after["momentum"] - before["momentum"]).max() < 1e-5 * momentum_scale
    # Energy is only conserved up to the momentum spread threshold: merging a
    # cluster to its mean momentum loses the spread's contribution, of order
    # (relative spread)^2 per cluster.
    assert after["energy"] == pytest.approx(
        before["energy"], rel=rel_mom_spread_threshold**2
    )
    assert list(df_merged.columns) == list(beam.columns)
    assert (df_merged.dtypes == np.float32).all()


def test_global_leveling_thinning_conserves_weight_statistically(beam):
    k = 10.0
    before = totals(beam)
    df_thin = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .global_leveling_thinning(k=k)
        .finalize()
    )
    after = totals(df_thin)

    # Survivors are kept with probability w / (k * <w>) and re-leveled to
    # k * <w>, so charge is conserved in expectation and the count drops by
    # roughly k; both hold only statistically.
    assert len(df_thin) == pytest.approx(len(beam) / k, rel=0.1)
    assert after["weight"] == pytest.approx(before["weight"], rel=0.02)
    # The thinned mean energy should stay representative of the original beam.
    mean_energy_before = before["energy"] / before["weight"]
    mean_energy_after = after["energy"] / after["weight"]
    assert mean_energy_after == pytest.approx(mean_energy_before, rel=0.02)


@pytest.mark.parametrize("device", devices())
def test_vranic_merging_is_reproducible_on_cpu(beam, device):
    if device != "cpu":
        pytest.skip("GPU float atomics are not bit-reproducible by design")
    kwargs = dict(spatial_bins=(4, 4, 4), momentum_bins=(8, 8, 8), device="cpu")
    first = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .vranic_merging(**kwargs)
        .finalize()
    )
    second = (
        ParticleResampler(beam, particle_species_mass=1.0)
        .vranic_merging(**kwargs)
        .finalize()
    )
    pd.testing.assert_frame_equal(first, second)
