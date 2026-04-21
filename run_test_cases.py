"""
Test case runner for spacecraft navigation simulation.
Runs 4 configurations varying synthetic vs real star counts,
saves results to CSV.
"""

import numpy as np
import csv
import re
from itertools import product
from astropy.coordinates import ICRS, SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, EarthLocation
from astropy.io import fits
import modeling_compiler as MC

# ── Physical constants ──────────────────────────────────────────────────────
C_L         = 299792458.0
AU_TO_M     = 1.495978707e11
sec_d       = 86400.0
Light_AU_D  = C_L * sec_d / AU_TO_M

loc = EarthLocation(lat=40*u.deg, lon=-88*u.deg, height=200*u.m)


# ── Star classes (copied from main nav script) ──────────────────────────────

class StarBase:
    def __init__(self, unit_vector, time_array_real=None):
        self.uhat = unit_vector / np.linalg.norm(unit_vector)
        self.time_array_real = time_array_real


class DeltaScutiStar(StarBase):
    def __init__(self, frequency, amplitude, phase, offset, unit_vector, time_array_real=None):
        super().__init__(unit_vector, time_array_real)
        self.freq      = frequency
        self.amp       = amplitude
        self.phase     = phase
        self.offset    = offset
        self.star_name = "Delta Scuti Star (Synthetic)"

    def model(self, t, t0=0.0):
        result  = self.amp * np.sin(2 * np.pi * self.freq * (t - t0) + self.phase) + self.offset
        result -= np.mean(result)
        result /= np.max(np.abs(result))
        return (0, result)


class RealStar(StarBase):
    def __init__(self, unit_vector, comparision_object, time_array_real, star_name="Unknown"):
        super().__init__(unit_vector, time_array_real)
        self.std_real  = np.std(comparision_object.model_real)
        self.mean_real = np.mean(comparision_object.model_real)
        self.std_ref   = np.std(comparision_object.model_ref_model)
        self.mean_ref  = np.mean(comparision_object.model_ref_model)
        self.star_name = star_name
        self.comparision_object        = comparision_object
        self.model_string_real         = comparision_object.model_string_real
        self.model_anchored_real_time  = comparision_object.model_anchored_real_time / np.max(np.abs(comparision_object.model_anchored_real_time))
        self.model_real                = comparision_object.model_real / np.max(np.abs(comparision_object.model_real))
        self.model_ref_model_string    = comparision_object.model_ref_model_string
        self.geometric_delay           = 0

    def model(self, t, t0=0.0):
        t_eval     = t - t0
        result     = eval(self.model_string_real, {"np": np, "t": t_eval})
        result     = (result - self.mean_real) / self.std_real
        result_ref = eval(self.model_ref_model_string, {"np": np, "t": t_eval})
        result_ref = (result_ref - self.mean_ref) / self.std_ref
        return (result, result_ref)

    def anchored_model_alignment(self, t_target, t_native):
        r_real   = loc.get_gcrs(t_target).transform_to(ICRS()).cartesian.xyz.to(u.AU).value
        r_native = loc.get_gcrs(t_native).transform_to(ICRS()).cartesian.xyz.to(u.AU).value
        delta_t  = np.dot(self.uhat, r_real - r_native) / Light_AU_D
        self.geometric_delay = delta_t
        print(f"  [{self.star_name}] geometric delay aligned.")


class Observation:
    def __init__(self, time_array, flux_array, star_name="Unknown", true_delta_t=None):
        self.time        = np.array(time_array)
        self.flux        = np.array(flux_array)
        self.star_name   = star_name
        self.true_delta_t = true_delta_t


class Spacecraft:
    def __init__(self, position, clock_offset_seconds, stars, t_obs=None):
        self.r       = np.array(position)
        self.t_offset = clock_offset_seconds / sec_d
        self.stars   = stars
        if t_obs is not None:
            self.t_obs   = Time(t_obs, format='jd', scale='tdb')
            self.r_earth = get_body_barycentric(body="earth", time=self.t_obs).xyz.to(u.au).value

    def observe_star_synthetic(self, star, t_grid, noise_sigma=0.0, scale_factor=1.0):
        geom_delay   = np.dot(star.uhat, self.r) / Light_AU_D
        dt_true      = self.t_offset + geom_delay
        flux_measured = scale_factor * star.model(t_grid + dt_true)[1]
        if noise_sigma > 0:
            flux_measured += np.random.normal(0, noise_sigma, len(flux_measured))
        return Observation(t_grid.copy(), flux_measured, star.star_name, true_delta_t=dt_true)

    def observe_star_real(self, star, t_grid, noise_sigma=0.0, scale_factor=1.0):
        dt_true      = self.t_offset + star.geometric_delay
        flux_measured = star.model(t_grid)[0]
        return Observation(t_grid.copy(), flux_measured, star.star_name, true_delta_t=dt_true)


class NAV:
    def __init__(self, stars):
        self.stars     = stars
        self.num_stars = len(stars)

    def dt_estim(self, star, observations, search_range=0.01, n_grid=1001):
        t_prime      = observations.time
        measured_flux = observations.flux
        dt_grid      = np.linspace(-search_range, search_range, n_grid)
        J_values     = np.zeros(n_grid)
        for i, dt in enumerate(dt_grid):
            model_flux = star.model(t_prime + dt)[1]
            C_opt      = np.dot(measured_flux, model_flux) / np.dot(model_flux, model_flux)
            residual   = measured_flux - C_opt * model_flux
            J_values[i] = np.mean(residual**2)
        candidates = [dt_grid[i] for i in range(1, len(J_values)-1)
                      if J_values[i] < J_values[i-1] and J_values[i] < J_values[i+1]]
        if not candidates:
            candidates = [dt_grid[np.argmin(J_values)]]
        return candidates[:5]

    def solver(self, delta_t_values, sigma_dt=1.0):
        N = len(delta_t_values)
        A = np.zeros((N, 4))
        A[:, 0] = Light_AU_D
        for i in range(N):
            A[i, 1:4] = self.stars[i].uhat
        d      = Light_AU_D * np.array(delta_t_values)
        sigma_d = Light_AU_D * (sigma_dt / sec_d)
        W      = np.eye(N) / sigma_d**2
        try:
            s            = np.linalg.solve(A.T @ W @ A, A.T @ W @ d)
            t_offset_sec = s[0] / Light_AU_D * sec_d
            return {'clock_offset': t_offset_sec, 'position': s[1:4],
                    'residual': np.linalg.norm(A @ s - d), 'success': True}
        except np.linalg.LinAlgError:
            return {'success': False}

    def navigate(self, observations_list, max_candidates=3):
        candidates_per_star = []
        for i, obs in enumerate(observations_list):
            cands = self.dt_estim(self.stars[i], obs)
            candidates_per_star.append(cands[:max_candidates])
        best_solution, best_residual = None, np.inf
        for count, combo in enumerate(product(*candidates_per_star)):
            if count >= 500:
                break
            sol = self.solver(list(combo))
            if sol['success'] and sol['residual'] < best_residual:
                best_residual = sol['residual']
                best_solution = sol
        return best_solution


def get_unit_vector(starname):
    cord = SkyCoord.from_name(starname).transform_to(ICRS()).cartesian.xyz
    return (cord / np.linalg.norm(cord))


def make_synthetic_star(seed_offset=0):
    np.random.seed(None)
    freq  = np.random.uniform(5, 15)
    amp   = 0.01
    phase = np.random.uniform(0, 2 * np.pi)
    uvec  = np.random.randn(3)
    uvec /= np.linalg.norm(uvec)
    return DeltaScutiStar(freq, amp, phase, 1.0, uvec)


def run_case(case_label, stars, r_true, t_offset_true, t_obs,
             obs_duration=0.5, n_samples=500, noise_sigma=1e-3, n_runs=10):
    """Run n_runs Monte Carlo trials for a given star list. Returns list of result dicts."""
    star_names_str = " | ".join(s.star_name for s in stars)
    results = []

    simulator = Spacecraft(r_true, t_offset_true, stars, t_obs=t_obs)
    t_grid    = np.linspace(0, obs_duration, n_samples)

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

        solver   = NAV(stars)
        solution = solver.navigate(observations, max_candidates=3)

        if solution and solution['success']:
            pos_err  = float(np.linalg.norm(solution['position'] - r_true))
            time_err = float(abs(solution['clock_offset'] - t_offset_true))
            results.append({
                'test_case':        case_label,
                'run':              run + 1,
                'stars':            star_names_str,
                'n_stars':          len(stars),
                'n_real':           sum(1 for s in stars if s.star_name != "Delta Scuti Star (Synthetic)"),
                'n_synthetic':      sum(1 for s in stars if s.star_name == "Delta Scuti Star (Synthetic)"),
                'position_error_AU': pos_err,
                'time_error_s':     time_err,
                'residual':         float(solution['residual']),
                'success':          True,
            })
            print(f"  [{case_label}] run {run+1}/{n_runs}  pos={pos_err:.5f} AU  time={time_err:.3f} s")
        else:
            results.append({
                'test_case': case_label, 'run': run+1, 'stars': star_names_str,
                'n_stars': len(stars),
                'n_real': sum(1 for s in stars if s.star_name != "Delta Scuti Star (Synthetic)"),
                'n_synthetic': sum(1 for s in stars if s.star_name == "Delta Scuti Star (Synthetic)"),
                'position_error_AU': None, 'time_error_s': None,
                'residual': None, 'success': False,
            })
            print(f"  [{case_label}] run {run+1}/{n_runs}  FAILED")

    return results


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # ── Calibration & data paths ────────────────────────────────────────────
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
    star_names = ["Alderamin", "TIC 381320713", "Tau Cygni"]
    centroid_override = [None, 1, None]

    t_obs        = Time('2025-11-16T02:43:07.685', scale='tdb')
    t_adjustment = Time('2025-12-05T02:01:46.505157', scale='tdb')

    # ── Compile light curves once ────────────────────────────────────────────
    print("Compiling light curves …")
    compiler = MC.ModelingCompiler(bias, dark, flat, data_map_paths, star_names, centroid_override)
    compiler.compile_light_curves()

    uni  = get_unit_vector(star_names[0])
    uni2 = get_unit_vector(star_names[1])
    uni3 = get_unit_vector(star_names[2])

    star0 = RealStar(uni,  compiler.COMP_LIST[0], compiler.compiled_dates[0], star_name=star_names[0])
    star1 = RealStar(uni2, compiler.COMP_LIST[1], compiler.compiled_dates[1], star_name=star_names[1])
    star2 = RealStar(uni3, compiler.COMP_LIST[2], compiler.compiled_dates[2], star_name=star_names[2])

    star1.anchored_model_alignment(t_obs, t_adjustment)

    real_stars = [star0, star1, star2]

    # ── True spacecraft state ────────────────────────────────────────────────
    r_true       = loc.get_gcrs(t_obs).transform_to(ICRS()).cartesian.xyz.to(u.AU).value
    t_offset_true = 5.0   # seconds
    t_obs_jd     = t_obs.jd

    print(f"\nTrue position : {r_true} AU")
    print(f"True clock offset: {t_offset_true} s\n")

    # ── Define test cases ────────────────────────────────────────────────────
    # For each (n_synth, n_real) combo, run every combination of real stars.
    # e.g. "9 synth + 1 real" → 3 cases: (9s+star0), (9s+star1), (9s+star2)
    #      "2 synth + 2 real" → 3 cases: (2s+star0+star1), (2s+star0+star2), (2s+star1+star2)
    from itertools import combinations as combns

    N_RUNS = 10

    # (n_synthetic, n_real)
    configurations = [
        (9, 1),
        (3, 1),
        (2, 2),
        (1, 1),
    ]

    all_results = []
    np.random.seed(42)

    for n_synth, n_real in configurations:
        for real_combo in combns(range(len(real_stars)), n_real):
            chosen_real = [real_stars[i] for i in real_combo]
            real_names  = " + ".join(s.star_name for s in chosen_real)
            label = f"{n_synth} synthetic + {n_real} real [{real_names}]"

            print("=" * 60)
            print(f"TEST CASE: {label}")
            print("=" * 60)

            stars = chosen_real + [make_synthetic_star() for _ in range(n_synth)]

            results = run_case(
                case_label=label,
                stars=stars,
                r_true=r_true,
                t_offset_true=t_offset_true,
                t_obs=t_obs_jd,
                n_runs=N_RUNS,
            )
            all_results.extend(results)

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = "Legacy_Files/nav_test_results.csv"
    fieldnames = [
        'test_case', 'run', 'stars', 'n_stars', 'n_real', 'n_synthetic',
        'position_error_AU', 'time_error_s', 'residual', 'success'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✓ Results saved to {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    from collections import defaultdict
    by_case = defaultdict(list)
    for r in all_results:
        if r['success']:
            by_case[r['test_case']].append(r)

    seen = []
    [seen.append(r['test_case']) for r in all_results if r['test_case'] not in seen]
    for case in seen:
        rows = by_case[case]
        if rows:
            pos_errs  = [r['position_error_AU'] for r in rows]
            time_errs = [r['time_error_s']      for r in rows]
            print(f"\n{case}")
            print(f"  Success: {len(rows)}/{N_RUNS}")
            print(f"  Pos  error — mean: {np.mean(pos_errs):.5f} AU  std: {np.std(pos_errs):.5f} AU")
            print(f"  Time error — mean: {np.mean(time_errs):.3f} s  std: {np.std(time_errs):.3f} s")
        else:
            print(f"\n{case}  — all runs failed")


if __name__ == '__main__':
    main()
