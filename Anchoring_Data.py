import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body_barycentric
import matplotlib.pyplot as plt
import lightkurve as lk
import time 
import pandas as pd



class anchoringData:
    """
    Class to anchor stellar light curves observed from different locations,
    """
    def __init__(self, NameStar):
        self.NameStar = NameStar
        self.c  = 299792458  # (m/s) speed of light
        self.time_array, self.flux_array = self.pull_lightcurve_data(NameStar)
        self.r_earth = get_body_barycentric(body = "earth", time=  self.time_array).xyz.to(u.m)  # shape: (3, N)
        print(self.r_earth.unit)
        self.star_coords = SkyCoord.from_name(NameStar)
        self.time_delay = None #computer in get_anchored_lightcurve
        self.t_Earth, self.flux_Earth = self.get_anchored_lightcurve()

         

    @staticmethod
    def pull_lightcurve_data(nameStar):
        #relative to solar-system barycenter
        x = lk.search_targetpixelfile(nameStar)
        x= x[0].download().to_lightcurve()
        return x.time, x.flux

    def compute_light_travel_delay(self, times=None):
        star_direction = self.star_coords.cartesian.xyz
        print(star_direction)  # in meters
        star_direction /= np.linalg.norm(star_direction) # unit vector
        vec_delayed = np.dot(star_direction, self.r_earth.value)  # in meters
        vec_time_delayed = vec_delayed /self.c  # in seconds
        vec_time_delayed_days = vec_time_delayed / 86400.0  # convert to days
        return vec_time_delayed_days

    def get_anchored_lightcurve(self):
        self.time_delay = self.compute_light_travel_delay()
        t_Earth = self.time_array.value - self.time_delay
        return t_Earth, self.flux_array
    

    def plot_comparison(self):
        #time_earth, flux_earth = self.get_anchored_lightcurve()
        #time_earth = time_earth.value
        #flux_earth = flux_earth.value
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=False)
        
        # Original SSB frame
        axes[0].plot(self.time_array.value, self.flux_array, 'b-', alpha=0.7, linewidth=0.5)
        axes[0].set_ylabel('Flux')
        axes[0].set_title(f'{self.NameStar} - SSB Frame')
        axes[0].grid(True, alpha=0.3)
        
        # Earth frame
        axes[1].plot(self.t_Earth, self.flux_Earth, 'r-', alpha=0.7, linewidth=0.5)
        axes[1].set_ylabel('Flux')
        axes[1].set_title(f'{self.NameStar} - Earth Center Frame')
        axes[1].grid(True, alpha=0.3)
        
        # Time delay plot
        #delay = self.compute_light_travel_delay()
        axes[2].plot(self.time_array.value, self.time_delay * 86400, 'g-', alpha=0.7)  # convert to seconds
        axes[2].set_ylabel('Time Delay (seconds)')
        axes[2].set_xlabel('Time (BJD)')
        axes[2].set_title('Roemer Delay (SSB to Earth)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_anchored_lightcurve(self, filename):
        pd.DataFrame({
            'Time_Earth': self.t_Earth,
            'Flux_Earth': self.flux_Earth.value
        }).to_csv(filename, index=False)
        
       
if __name__ == "__main__":
    star_name = "Alderamin"
    anchoring_instance = anchoringData(star_name)
    #t_earth, flux_earth = anchoring_instance.get_anchored_lightcurve()
    
    # Plot comparison
    anchoring_instance.plot_comparison()
    anchoring_instance.save_anchored_lightcurve(f"Shifted_LC/{star_name}_anchored_lightcurve_{int(time.time() * 10000)}.csv")


    plt.show()



