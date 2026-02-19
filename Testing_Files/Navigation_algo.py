import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from itertools import product

# Physical constants
C_L = 299792458.0  # m/s
AU_TO_M = 1.495978707e11  # meters per AU
sec_d = 86400.0
Light_AU_D = C_L * sec_d / AU_TO_M # Speed of Light in AU/day


class DeltaScutiStar:
    # Class to simulate a delta scuti
    def __init__(self,  frequency, amplitude, phase, offset, unit_vector):
        self.freq = frequency
        self.amp = amplitude
        self.phase = phase
        self.offset = offset
        self.uhat = unit_vector / np.linalg.norm(unit_vector)
    
    def model(self, t, t0=0.0):
        return self.amp * np.sin( 2 * np.pi * self.freq * (t - t0) + self.phase ) + self.offset


class Spacecraft:
    # Spacecraft observation 
    def __init__(self, position, c_off, stars):
        self.r = np.array(position) # source of truth
        self.t_offset = c_off / sec_d  # Built in error of hardware
        self.stars = stars # the stars we would have
    
    def observe_star(self, star, t_grid, noise_sigma=0.0, scale_factor=1.0):

        geom_delay = np.dot(star.uhat, self.r) / Light_AU_D  # added phase shift from the position (days) 
  
        dt_true = self.t_offset + geom_delay # appending the phase shift + our clock offset
     
        T = t_grid + dt_true # added the delay to the time grid to get the true time
        
        # Generate measurements using the star's model
        flux_shifted = star.model(T) # computing the same flux at the shifted time 
        flux_measured = scale_factor * flux_shifted   # applying the scale factor to make everything match 
        
        # Add noise
        if noise_sigma > 0: 
            flux_measured += np.random.normal(0, noise_sigma, len(flux_measured))
        
        return {
            'time': t_grid.copy(),
            'flux': flux_measured,
            'true_delta_t': dt_true
        }


class NAV:
    # Navigation solver class
    def __init__(self, stars):
        self.stars = stars
        self.num_stars = len(stars)
    
    def dt_estim(self, star, observations, search_range=0.05, n_grid=1001):
        t_prime = observations['time'] # spacecraft clock times
        measured_flux = observations['flux']    # measured fluxes

        dt_grid = np.linspace(-search_range, search_range, n_grid)
        J_values = np.zeros(n_grid)
        
        for i, dt in enumerate(dt_grid): # generating our J values
            T = t_prime + dt
            model_flux = star.model(T)
            
            # Derived from taking the derivating and solving for C of the J function
            C_opt = np.dot(measured_flux, model_flux) /  np.dot(model_flux, model_flux)
            residual = measured_flux - C_opt * model_flux
            J_values[i] = np.mean(residual**2)
        
        # Find local minima
        candidates = []
        for i in range(1, len(J_values) - 1):
            if J_values[i] < J_values[i-1] and J_values[i] < J_values[i+1]:
                candidates.append(dt_grid[i])
        
        #if no minima found, use global minimum
        if len(candidates) == 0:
            candidates = [dt_grid[np.argmin(J_values)]]
        
        return  candidates[:5]  # Clip to 5 candidates
    
    def solver(self, delta_t_values, sigma_dt=1.0):

        N = len(delta_t_values)

        A = np.zeros((N, 4))
        A[:, 0] = Light_AU_D  # speed of light column
        for i in range(N):
            A[i, 1:4] = self.stars[i].uhat # add your u hats

        d = Light_AU_D * np.array(delta_t_values)
        
        # covariance
        sigma_d = Light_AU_D * (sigma_dt / sec_d) 
        W = np.eye(N) / sigma_d**2

        #print(A)
        #print(d)    
        




        # least squares solution
        try:
            AtWA = A.T @ W @ A
            AtWd = A.T @ W @ d
            s = np.linalg.solve(AtWA, AtWd)
            
            # Extract results
            c_t_offset = s[0]
            t_offset_sec = c_t_offset / Light_AU_D * sec_d
            r_est = s[1:4]
            
            #residual
            residual = np.linalg.norm(A @ s - d)
            
            return {
                'clock_offset': t_offset_sec,
                'position': r_est,
                'residual': residual,
                'success': True
            }
        
        except np.linalg.LinAlgError:
            return {'success': False}
    
    def navigate(self, observations_list, max_candidates=3):
        #observation_list is a 2 d list of dicts, one per star
        # just finds best solution among combinations of candidates
        candidates_per_star = []
        for i, obs in enumerate(observations_list):
            candidates = self.dt_estim(self.stars[i], obs)
            candidates_per_star.append(candidates[:max_candidates])
            print(f"Star {i}: {len(candidates_per_star[i])} candidates")
        


        best_solution = None
        best_residual = np.inf
        

        max_combinations = 500
        count = 0
        
        for combo in product(*candidates_per_star): # looping through all combinations
            if count >= max_combinations:
                break
            
            solution = self.solver(list(combo))
            
            if solution['success'] and solution['residual'] < best_residual:
                best_residual = solution['residual']
                best_solution = solution
            
            count += 1
        
        print(f"Evaluated {count} combinations")
        return best_solution


def run_single_simulation(r_true, t_offset_true, stars, obs_duration, n_samples, 
                         noise_sigma, seed=None):
    """
    Run a single simulation instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create spacecraft
    simulator = Spacecraft(r_true, t_offset_true, stars)
    
    # Generate time grid
    t_grid = np.linspace(0, obs_duration, n_samples)
    
    # Observe all stars
    observations = []
    for star in stars:
        obs = simulator.observe_star(
            star, t_grid,
            noise_sigma=noise_sigma,
            scale_factor=np.random.uniform(0.98, 1.02)
        )
        observations.append(obs)
    
    # Navigate
    solver = NAV(stars)
    solution = solver.navigate(observations, max_candidates=3)
    
    return solution, observations


def monte_carlo_simulation(n_runs=50, verbose=True):

    print("="*70)
    print("MONTE CARLO SIMULATION")
    print("="*70)

    num_stars = 10
    
    r_true = np.array([np.random.rand(), np.random.rand(), np.random.rand()]) * 1.5  # AU
    t_offset_true = 1  # seconds

    obs_duration = 5  # days
    n_samples = 500
    noise_sigma = 1e-3 #from paper 
    
    # Storage for results
    position_errors = []
    time_errors = []
    residuals = []
    success_count = 0

    np.random.seed(42)
    stars = []
    for i in range(num_stars):
        freq = np.random.uniform(5, 15)  # cycles per day
        amp = 0.01  # 1% variation
        phase = np.random.uniform(0, 2*np.pi)
        baseline = 1.0
        uvec = np.random.randn(3)
        uvec = uvec / np.linalg.norm(uvec)
        stars.append(DeltaScutiStar(freq, amp, phase, baseline, uvec))
    
    print(f"\nTRUE STATE")
    print(f"  Position: {r_true} AU")
    print(f"  Clock offset: {t_offset_true} s")
    print(f"\nOBSERVATION PARAMETERS")
    print(f"  Number of stars: {num_stars}")
    print(f"  Observation duration: {obs_duration} days")
    print(f"  Number of samples: {n_samples}")
    print(f"  Noise sigma: {noise_sigma}")
    print(f"\nRunning {n_runs} Monte Carlo samples...")
    print("-"*70)
    
    # Run Monte Carlo
    for run in range(n_runs):
        # Run simulation with different noise seed
        solution, observations = run_single_simulation(
            r_true, t_offset_true, stars, 
            obs_duration, n_samples, noise_sigma,
            seed=None  # Random seed each time
        )
        
        if solution and solution['success']:
            pos_error = np.linalg.norm(solution['position'] - r_true)
            time_error = abs(solution['clock_offset'] - t_offset_true)
            
            position_errors.append(pos_error)
            time_errors.append(time_error)
            residuals.append(solution['residual'])
            success_count += 1
            
            if verbose and (run + 1) % 10 == 0:
                print(f"Run {run+1}/{n_runs}: "
                      f"Pos Error = {pos_error:.6f} AU, "
                      f"Time Error = {time_error:.3f} s")
        else:
            if verbose:
                print(f"Run {run+1}/{n_runs}: FAILED")
    
    # Convert to arrays
    position_errors = np.array(position_errors)
    time_errors = np.array(time_errors)
    residuals = np.array(residuals)
    
    # Compute statistics
    print("\n" + "="*70)
    print("MONTE CARLO RESULTS")
    print("="*70)
    print(f"\nSuccess rate: {success_count}/{n_runs} ({100*success_count/n_runs:.1f}%)")
    
    if success_count > 0:
        print(f"\nPOSITION ERROR (AU):")
        print(f"  Mean:   {np.mean(position_errors):.6f}")
        print(f"  Median: {np.median(position_errors):.6f}")
        print(f"  Std:    {np.std(position_errors):.6f}")
        print(f"  Min:    {np.min(position_errors):.6f}")
        print(f"  Max:    {np.max(position_errors):.6f}")
        print(f"  1σ:     {np.std(position_errors):.6f}")
        print(f"  3σ:     {3*np.std(position_errors):.6f}")
        
        print(f"\nTIME ERROR (seconds):")
        print(f"  Mean:   {np.mean(time_errors):.3f}")
        print(f"  Median: {np.median(time_errors):.3f}")
        print(f"  Std:    {np.std(time_errors):.3f}")
        print(f"  Min:    {np.min(time_errors):.3f}")
        print(f"  Max:    {np.max(time_errors):.3f}")
        print(f"  1σ:     {np.std(time_errors):.3f}")
        print(f"  3σ:     {3*np.std(time_errors):.3f}")
        
        print(f"\nCOMPARISON TO PAPER (Table 4.2, MapCam nominal case):")
        print(f"  Paper - Position: mean=3.08e-02 AU, std=2.57e-02 AU")
        print(f"  Yours - Position: mean={np.mean(position_errors):.2e} AU, std={np.std(position_errors):.2e} AU")
        print(f"  Paper - Time:     mean=5.40 s, std=2.67 s")
        print(f"  Yours - Time:     mean={np.mean(time_errors):.2f} s, std={np.std(time_errors):.2f} s")
    
    return {
        'position_errors': position_errors,
        'time_errors': time_errors,
        'residuals': residuals,
        'success_rate': success_count / n_runs,
        'r_true': r_true,
        't_offset_true': t_offset_true
    }





def main():
    
    
    np.random.seed(42)
    num_stars = 15
    stars = []
    
    for i in range(num_stars):
        freq = np.random.uniform(5, 15)  # cycles per day
        amp = 0.01  # 1% variation
        phase = np.random.uniform(0, 2*np.pi)
        baseline = 1.0
        
        # Random
        uvec = np.random.randn(3)
        uvec = uvec / np.linalg.norm(uvec)
        
        stars.append(DeltaScutiStar(freq, amp, phase, baseline, uvec))
    
    #initial vals
    r_true = np.array([0.53930622, 0.91569188, 0.2846358])  # AU
    t_offset_true = 5  # seconds
    
    print("TRUE STATE")
    print(f"  Position: {r_true} AU")
    print(f"  Clock offset: {t_offset_true} s")
    print()
    
 



    simulator = Spacecraft(r_true, t_offset_true, stars)
    
    obs_duration = 5 
    n_samples = 500
    t_grid = np.linspace(0, obs_duration, n_samples)
    
    observations = []
    for star in stars:
        obs = simulator.observe_star(
            star, t_grid,
            noise_sigma=1e-3,
            scale_factor=np.random.uniform(0.98, 1.02)
        )
        observations.append(obs)
        print(f"Star: true Δt = {obs['true_delta_t']*sec_d:.3f} s")
    
    print()
    
  
    solver = NAV(stars)
    solution = solver.navigate(observations, max_candidates=3)
    
    # Monte Carlo Testing
    monte_carlo_simulation()

    if solution:
        print("\nESTIMATED STATE")
        print(f"Position: {solution['position']} AU")
        print(f"Clock offset: {solution['clock_offset']:.3f} s")
        print(f"Residual: {solution['residual']:.6e}")
        print()
        print("ERRORS")
        print(f"Position error: {np.linalg.norm(solution['position'] - r_true):.6f} AU")
        print(f"Time error: {abs(solution['clock_offset'] - t_offset_true):.6f} s")
    else:
        print("FALIURE")
    

    """
    plt.figure(figsize=(10, 4))
    star_idx = 0
    obs = observations[star_idx]
    
    plt.plot(obs['time'] * 24, obs['flux'], 'b.', markersize=2, label='Measured')
    
    t_ref = np.linspace(0, obs_duration, 1000)
    plt.plot(t_ref * 24, stars[star_idx].model(t_ref), 'r-', 
             linewidth=1, label='Reference model')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Flux')
    plt.title(f'Star {star_idx} Light Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    """




if __name__ == '__main__':
    main()