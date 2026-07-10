# Review resolution & real-data validation

This addresses the three review groups from the [review comment](https://github.com/berceanu/openPMD-resampler/pull/2#issuecomment-4924747082), and validates all three algorithms on a production-size PIConGPU dataset, including a CPU vs GPU timing comparison for the two merging algorithms.

## Dependencies & Setup

- **Dependency upgrades split out**: `pixi.toml` is back on the upstream pins (python 3.11, openpmd-api 0.15, pandas 2.0, datashader 0.15, matplotlib 3.7); this PR now only adds `scipy` (Fortran output) and `pytorch` (merging). The upgrades (python 3.12, openpmd-api 0.16, pandas 3, datashader 0.18, matplotlib 3.10) live on a separate `dependency-upgrades` branch cut from upstream `main`, verified against a synthetic run under pandas 3, and will be submitted as their own PR. The full test suite passes under the reverted pins (python 3.11 / pandas 2.0.3), confirming the merging code does not need the upgrades.
- **cupy removed**: it was unused; dropping it (and the `cuda-version` pin) removes the CUDA packages from CPU-only installs. One addition surfaced by re-locking: `pyarrow` is now an explicit dependency, because the dask version pulled in by datashader imports it at runtime without declaring it.
- **CUDA no longer forced**: the global `cuda = "12"` system requirement is gone. The default environment gets the CPU build of PyTorch; GPU support is opt-in through a new `cuda` pixi environment that is the only place with the CUDA system requirement:

  ```console
  $ pixi run start ...            # CPU-only, no CUDA downloaded
  $ pixi run -e cuda start ...    # GPU build of PyTorch
  ```

## Tests & Logic

- **Conservation tests added** (`tests/test_conservation.py`; `pixi run test` for CPU, `pixi run -e test-cuda test` to also exercise the GPU code paths — all green on both, 6 passed / 1 skipped-by-design with a GPU): on a synthetic 200k-particle relativistic bunch with non-uniform weights,
  - Vranic merging conserves total weight, all three momentum components and total kinetic energy to < 10⁻⁵ (relative, float64 accumulation), on CPU and — when available — GPU;
  - Voronoi merging conserves weight and momentum to < 10⁻⁵, and energy within the square of the relative momentum-spread threshold;
  - global leveling thinning reproduces the requested reduction factor and conserves total charge and mean energy statistically (< 2%);
  - CPU Vranic merging is bit-reproducible run-to-run (the GPU is exempt by design: float atomics).
- **`start.py` / `usage.py` unified**: the thinning branch of `start.py` no longer calls `.set_weights_to(1)`. All three algorithms now keep the physical weights in the output, so the file contract is identical regardless of algorithm and the total charge is preserved in the written file.
- **`_write_csv` memory fix**: the writer no longer duplicates the full DataFrame; it streams 5M-row chunks (applying the optional `momentum_in_mc` conversion per chunk), so peak memory is now ~the DataFrame itself plus one chunk. Output is byte-identical to before.

## Docs & Minor Fixes

- README gained a *"Which algorithm should I use?"* guide (thinning = direct reduction factor + statistical charge conservation; Vranic = exact weight/momentum/energy conservation; Voronoi = adaptive, follows local phase-space structure).
- README now states explicitly that the default `--min_mean_energy` of 511 keV silently leaves low-energy clusters unmerged, and that `0` disables the criterion.
- The sample CSV block now shows the real header (`position_x_m (m)`, …, `weights (1)`): positions are exported in meters, momenta in MeV/c.
- Removed the unused `meters_to_microns` constant and the `CLAUDE.local.md` line from `.gitignore`.

## Validation on a production dataset

**Data**: PIConGPU 0.9.0-dev output `gas_205000.bp5` (4.2 GB, iteration 205 000 at 27.08 ps), species `e_forwardPinholeHighGamma`: **110 730 968 macroparticles** (2.41 × 10¹⁰ real electrons, 3858.8 pC, weighted mean energy 211.6 MeV).

**Machine**: 2 × Intel Xeon Gold 6242 (32 physical cores; PyTorch used 32 threads) with 376 GB RAM; NVIDIA Quadro P4000 (8 GB). Reading the file takes ~30 s and is excluded from the timings below. The GPU warm-up (CUDA context) is also excluded.

| Algorithm | Parameters | Device | Time | Macroparticles out | Reduction |
| --- | --- | --- | --- | --- | --- |
| thinning | `k = 100` | CPU | **9.2 s** | 1 106 771 | 100.0× |
| vranic | `--spatial_bins 64 64 64 --momentum_bins 64 64 64` | CPU (32 threads) | 86.3 s | 94 531 696 | 14.6 % |
| vranic | same | GPU | **58.1 s** | 94 531 160 | 14.6 % |
| voronoi | `--spatial_bins 128 128 1024 --min_particles_to_merge 10 --rel_mom_spread_threshold 0.1 --pos_spread_threshold 0.1` | CPU (32 threads) | 115.0 s | 92 915 880 | 16.1 % |
| voronoi | same | GPU | **78.9 s** | 91 506 528 | 17.4 % |

**GPU speedup: 1.5× (Vranic), 1.5× (Voronoi)** over 32 Xeon cores — on a 2016-era 8 GB Quadro P4000 that has to stage the 110M-particle dataset through GPU memory in 44 (Vranic) / 43 (Voronoi) chunks. On a data-center GPU that holds the dataset in fewer chunks, the gap widens accordingly. The CPU and GPU Voronoi outputs differ slightly in count (92.9M vs 91.5M) because float atomics change individual split decisions; the conserved quantities are unaffected (table below).

Conservation on the real dataset (relative errors, float64 accumulation, computed by the built-in conservation logging):

| Run | Weight | Momentum | Energy |
| --- | --- | --- | --- |
| vranic CPU | 8.3 × 10⁻¹¹ | 8.2 × 10⁻⁹ | 2.5 × 10⁻⁹ |
| vranic GPU | 8.5 × 10⁻¹¹ | 8.4 × 10⁻⁹ | 2.1 × 10⁻⁹ |
| voronoi CPU | 6.1 × 10⁻⁹ | 1.4 × 10⁻¹⁰ | 2.5 × 10⁻⁶ |
| voronoi GPU | 3.8 × 10⁻⁹ | 4.3 × 10⁻¹¹ | 2.7 × 10⁻⁶ |

Thinning conserved the total charge to 0.05 % (3858.78 → 3856.91 pC), as expected for a statistical method. The Voronoi energy error (~3 × 10⁻⁶) is bounded by the momentum-spread threshold, exactly as documented.

With these fine bin settings the merging algorithms remove 15–17 % of the macroparticles while leaving the phase space visually indistinguishable — coarser bins / larger thresholds trade fidelity for stronger reduction.

### Comparative phase-space plots (original vs resampled)

| Thinning (k=100) | Vranic (GPU) | Voronoi (GPU) |
| --- | --- | --- |
| ![thinning](plots/comparative_phase_space_thinning.png) | ![vranic](plots/comparative_phase_space_vranic.png) | ![voronoi](plots/comparative_phase_space_voronoi.png) |

*(figures: `plots/comparative_phase_space_{thinning,vranic,voronoi}.png` — drag & drop into the PR comment to upload)*
