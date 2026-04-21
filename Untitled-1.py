"""
Test case runner — imports navigation classes from the main script.
"""

import csv
import numpy as np
from itertools import combinations
from astropy.coordinates import ICRS
from astropy import units as u
from astropy.time import Time
from astropy.io import fits

# ── Import everything from your existing nav script ──────────────────────────
from navigation import (
    loc, sec_d, Light_AU_D,
    RealStar, DeltaScutiStar, Spacecraft, NAV,
    get_unit_vector,
)
import modeling_compiler as MC


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_star():
    freq  = np.random.uniform(5, 15)
    phase = np.random.uniform(0, 2 * np.pi)
    uvec  = np.random.randn(3)
    uvec /= np.linalg.norm(uvec)
    return DeltaScutiStar(freq, 0.01, phase, 1.0, uvec)


def run_case(label, stars, r_true, t_offset_true, t_obs_jd,
             obs_duration=0.5, n_samples=500, noise_sigma=1e-3, n_runs=10):

    star_names_str = " | ".join(s.star_name for s in stars)
    simulator = Spacecraft(r_true, t_offset_true, stars, t_obs=t_obs_jd)
    t_grid    = np.linspace(0, obs_duration, n_samples)
    results   = []

    for run in range(n_runs):
        observations = []
        for star in stars:
            if star.star_name == "Delta Scuti Star (Synthetic)":
                obs = simulator.observe_star_synthetic(
                    star, t_grid, noise_sigma=noise_sigma,
                    scale_factor=np.random.uniform(0.98, 1.02))
            else:
                obs = simulator.observe_star_real(
                    star, t_grid, noise_sigma=noise_sigma,
                    scale_factor=np.random.uniform(0.98, 1.02))
            observations.append(obs)

        solution = NAV(stars).navigate(observations, max_candidates=3)

        base = {
            'test_case':   label,
            'run':         run + 1,
            'stars':       star_names_str,
            'n_stars':     len(stars),
            'n_real':      sum(1 for s in stars if s.star_name != "Delta Scuti Star (Synthetic)"),
            'n_synthetic': sum(1 for s in stars if s.star_name == "Delta Scuti Star (Synthetic)"),
        }

        if solution and solution['success']:
            pos_err  = float(np.linalg.norm(solution['position'] - r_true))
            time_err = float(abs(solution['clock_offset'] - t_offset_true))
            print(f"  [{label}] run {run+1}/{n_runs}  pos={pos_err:.5f} AU  time={time_err:.3f} s")
            results.append({**base, 'position_error_AU': pos_err,
                            'time_error_s': time_err,
                            'residual': float(solution['residual']), 'success': True})
        else:
            print(f"  [{label}] run {run+1}/{n_runs}  FAILED")
            results.append({**base, 'position_error_AU': None,
                            'time_error_s': None, 'residual': None, 'success': False})

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Calibration frames
    bias, dark, flat = [], [], []
    bias.append(fits.getdata("calibration_frames/Bias_1.0ms_Bin1_ISO100_20251205-065105_32.0F_0001.fit").astype(float))
    dark.append(fits.getdata("calibration_frames/NGC0891 darks_00015.fits").astype(float))
    flat.append(fits.getdata("calibration_frames/Flat_300.0ms_Bin1_ISO100_20251205-064251_32.0F_0001.fit").astype(float))
    for _ in range(3):
        bias.append(fits.getdata("calibration_frames/other_calibratoin/Bias_1.0ms_Bin1_ISO100_20251205-065130_32.0F_0010.fit").astype(float))
        dark.append(fits.getdata("calibration_frames/other_calibratoin/Dark_30.0s_Bin1_ISO100_20251205-065700_35.6F_0010.fit").astype(float))
        flat.append(fits.getdata("calibration_frames/other_calibratoin/Flat_300.0ms_Bin1_ISO100_20251205-064753_32.0F_0010.fit").astype(float))

    data_map_paths = [
        "data_maps/real_data_map_Alderamin (Alpha Cephi) 2025-11-15.csv",
        "data_maps/real_data_map_97 Psc.csv",
        "data_maps/real_data_map_Tau Cygni 2025-11-15.csv",
    ]
    star_names        = ["Alderamin", "TIC 381320713", "Tau Cygni"]
    centroid_override = [None, 1, None]
    t_obs             = Time('2025-11-16T02:43:07.685', scale='tdb')
    t_adjustment      = Time('2025-12-05T02:01:46.505157', scale='tdb')

    # Compile light curves
    compiler = MC.ModelingCompiler(bias, dark, flat, data_map_paths, star_names, centroid_override)
    compiler.compile_light_curves()

    # Build real star objects
    real_stars = []
    for i, name in enumerate(star_names):
        star = RealStar(get_unit_vector(name), compiler.COMP_LIST[i],
                        compiler.compiled_dates[i], star_name=name)
        real_stars.append(star)

    real_stars[1].anchored_model_alignment(t_obs, t_adjustment)

    # True state
    r_true        = loc.get_gcrs(t_obs).transform_to(ICRS()).cartesian.xyz.to(u.AU).value
    t_offset_true = 5.0
    t_obs_jd      = t_obs.jd

    # Test configurations: (n_synthetic, n_real)
    configurations = [(9, 1), (3, 1), (2, 2), (1, 1)]
    N_RUNS = 10

    all_results = []
    np.random.seed(42)

    for n_synth, n_real in configurations:
        for real_combo in combinations(range(len(real_stars)), n_real):
            chosen_real = [real_stars[i] for i in real_combo]
            real_names  = " + ".join(s.star_name for s in chosen_real)
            label       = f"{n_synth} synth + {n_real} real [{real_names}]"

            print("=" * 60)
            print(f"TEST CASE: {label}")
            print("=" * 60)

            stars = chosen_real + [make_synthetic_star() for _ in range(n_synth)]
            all_results.extend(run_case(label, stars, r_true, t_offset_true, t_obs_jd, n_runs=N_RUNS))

    # Write CSV
    fieldnames = ['test_case', 'run', 'stars', 'n_stars', 'n_real', 'n_synthetic',
                  'position_error_AU', 'time_error_s', 'residual', 'success']
    with open("nav_test_results.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n✓ Results saved to nav_test_results.csv")


if __name__ == '__main__':
    main()