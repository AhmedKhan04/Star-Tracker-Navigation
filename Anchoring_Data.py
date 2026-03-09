import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body_barycentric
import matplotlib.pyplot as plt
import lightkurve as lk
import time 
import unpopular
import pandas as pd

import modeling as sm


class anchoringData:
    """
    Class to anchor stellar light curves observed from different locations,
    """
    def __init__(self, NameStar):
        self.NameStar = NameStar
        self.c  = 299792458  # (m/s) speed of light
        # if NameStar is "V376 Perseus": 
        #     self.time_array, self.flux_array = self.pull_from_file("")
        # else: 
        self.time_array, self.flux_array = self.pull_lightcurve_data(NameStar)
        #print(self.time_array.unit)
        #self.r_earth = get_body_barycentric(body = "earth", time=  self.time_array).xyz.to(u.m)  # shape: (3, N)
        #print(self.r_earth.unit)
        self.star_coords = SkyCoord.from_name(NameStar)
        self.time_delay = None #computer in get_anchored_lightcurve
        #self.t_Earth, self.flux_Earth = self.get_anchored_lightcurve()
        # if (self.flux_Earth.unit is None):
        #     self.flux_Earth = self.flux_Earth * u.luminosity  # assign unit if none
        # if (self.t_Earth.unit is not u.day):
        #     print("Assigning unit to time array")
        #     self.t_Earth = self.t_Earth * u.day  # assign unit if none
        
        self.t_Earth, self.flux_Earth = self.time_array, self.flux_array  # overrite time to be able to not anchor model for now...
        modeling_instance = sm.StarModeling(tuples_values=list(zip(self.t_Earth, self.flux_Earth)))
        self.model_ref_model, _, self.model_ref_model_string = modeling_instance.getCompositeSine2_deep(self.NameStar)
        #plt.figure() 
        #plt.plot(self.time_array, self.flux_array, label="Earth Frame Light Curve")
        #plt.plot(self.time_array, self.model_ref_model, label="Model Reference Model")
        #plt.show() 



         

    @staticmethod
    def pull_lightcurve_data(nameStar):
        #relative to solar-system barycenter
        index = 0 
        x = lk.search_targetpixelfile(nameStar)
        if nameStar == "V376 Perseus":
            index = 1
            x= x[index].download().to_lightcurve()
        elif nameStar == "Delta Scuti": 
            index = 1 
            x = lk.search_lightcurve(nameStar)[-1].download()
        else: 

            x= x[index].download().to_lightcurve()
            #x.plot()
        # lets do some detrending here to get a cleaner light curve for the anchoring.
        # time, flux = [], []
        # result = lk.search_tesscut(nameStar)[-2:] 
        # search_result = result.table['sequence_number']
        # search_result = search_result # limit to first 3 sectors for testing
            
        # print(search_result)
        # tpf_collection = result.download_all(cutout_size=50)
        
        # for l in tpf_collection:
        #     s = unpopular.Source(l.path, remove_bad=True)
        #     s.set_aperture(rowlims=[24, 26], collims=[24, 26])  # adjust these limits based on the star's position in the TPF
        #     s.add_cpm_model(exclusion_size=5, n=64, predictor_method="similar_brightness")
        #     s.set_regs([(1e-5)/2])
        #     s.holdout_fit_predict(k=100);

        #     #aperture_normalized_flux = s.get_aperture_lc(data_type="normalized_flux")
        #     #aperture_cpm_prediction = s.get_aperture_lc(data_type="cpm_prediction", weighting=None)

        #     apt_detrended_flux = s.get_aperture_lc(data_type="cpm_subtracted_flux")
        #     flux.extend(apt_detrended_flux)
        #     time.extend(s.time)

        #print(x)
        return x.time.value, x.flux.value

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
    
    def pull_from_file(self, file_path):
        df = pd.read_csv(file_path)
        time_array = df['Time'].to_numpy()
        flux_array = df['Flux'].to_numpy()
        return time_array, flux_array


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
    #anchoring_instance.save_anchored_lightcurve(f"Shifted_LC/{star_name}_anchored_lightcurve_{int(time.time() * 10000)}.csv")


    plt.show()



