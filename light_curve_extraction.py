import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
from photutils import aperture
import astropy as ap 
from astropy.time import Time
from PIL import Image
import glob
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.table import Table
from photutils.detection import DAOStarFinder, IRAFStarFinder
import astroalign as aa
import os 
from photutils.profiles import  CurveOfGrowth, RadialProfile
from photutils.centroids import centroid_2dg, centroid_com, centroid_1dg 
from photutils.background import Background2D, MedianBackground
import pandas as pd 
from datetime import datetime
import scipy.stats as stats
import copy 
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astroalign as aa


import modeling as sm 



####

# Global variables
initial = True 

photom_list = []
light_curve_extraction = False
plotting = False 
date_array = []


####

class LightCurveExtractor:

    def __init__(self, bias, dark, flat , data_map, star_name, Centroid_override = None):
        #self.file_list = file_list
        self.bias = bias
        self.dark = dark
        self.flat = flat
        self.data_map = data_map
        self.star_name = star_name

        self.photom_list = list([])
        self.date_array = list([])


        self.saving_constant = 0 
        self.Centroid_override = Centroid_override
        self.file_list = pd.read_csv(self.data_map)["FITS File Path"].values
        self.reference_frame = None 
        self.allignment_needed = 0


    def load_calibrated(self, data):
        #data = fits.getdata(file_path).astype(float)
        flat_norm = self.flat / np.median(self.flat)
        try:
            return (data - self.bias - self.dark) / flat_norm
        except Exception as e:
            print(f"Error in load_calibrated: {e}")
            return (data - self.dark)


    def get_max(self, img): 
        maxVal = np.amax(img)
        min = np.min(img)
        mask = np.ones_like(img) 
        mid_point_array = np.array([(mask.shape[0]//2, mask.shape[1]//2)])
        return mid_point_array

    def get_midpoint(self, img):
        ny, nx = img.shape
        x = nx // 2
        y = ny // 2
        return (x, y)


    def drop_outliers(self, data_input, time_input, threshold=2):
        #use Z score to drop outliers
        data = copy.deepcopy(data_input)
        time = copy.deepcopy(time_input)
        z_scores = np.abs(stats.zscore(data))
        #threshold = 2
        outlier_indices = np.where(z_scores > threshold)[0]
        filtered_data = np.delete(data, outlier_indices)
        filtered_time = np.delete(time, outlier_indices)
        return filtered_data, filtered_time

    def extract_light_curve(self, normalize=False):

        cords = None 
        w = None 
        ra = None
        dec = None 

        for filename in self.file_list:
            file_path = filename 
            
        
            print(file_path)
            ############
            # OVERRIDE FOR TESTING

            hdul = fits.open(file_path, memmap=True, do_not_scale_image_data=True)
   

            # Manual scaling (fast)
            bscale = hdul[0].header.get('BSCALE', 1)
            bzero  = hdul[0].header.get('BZERO', 0)
   

            date_obs = hdul[0].header.get('DATE-AVG')

            if date_obs is None:
                date_obs = hdul[0].header.get('DATE-OBS')

            date_obs.split('/')[0].strip()
            
            date_obs_clean = date_obs.split("'/")[0].strip()  
            #print(date_obs)
            if '.' in date_obs_clean:
                date_part, frac = date_obs_clean.split('.')
                frac = frac[:6]  # keep only first 6 digits
                date_obs_clean = f"{date_part}.{frac}"

            t_utc = Time(date_obs_clean, format='isot', scale='utc')
            t_tbd = t_utc.tdb
            btjd = t_tbd.jd - 2457000.0
            self.date_array.append(btjd)
            calibrated_second = hdul[0].data.astype(np.float32) * bscale + bzero # - self.dark  

            if (self.reference_frame is None):
                self.reference_frame = calibrated_second
            #calibrated_second = self.load_calibrated(calibrated_second) ##################################################################

            try: 
                w = WCS(hdul[0].header)
                ra, dec = None , None
                # ra  = hdul[0].header.get('OBJCTRA')
                # dec = hdul[0].header.get('OBJCTDEC')
                if ra is None or dec is None:
                    # fallback to CRVAL if OBJCT keywords missing
                    ra  = hdul[0].header['CRVAL1']
                    dec = hdul[0].header['CRVAL2']

            except Exception as e:
                print("WCS information not found in header:", e)
                print("Using last known coordinates or image center.")


            hdul.close()

            #mean_s, median_s, std_s = sigma_clipped_stats(calibrated_second, sigma=3.0)

            if plotting: 
                plt.figure()
                mean_s, median_s, std_s = sigma_clipped_stats(calibrated_second, sigma=3.0) 
                plt.imshow(calibrated_second, origin='lower', cmap='gray',
                            vmin=median_s - 1*std_s,
                            vmax=median_s + 5*std_s)                
                # axes.axis('off')
                # axes.set_title("Specific Image")

                plt.tight_layout()
                plt.show()
                plt.figure()
            

            #---------------------------

            try: 
                coord = SkyCoord(ra*u.deg, dec*u.deg)
                # ovverride  
                px, py = w.world_to_pixel(coord)
                cords = np.array([(py, px)]) 
                #cords = np.array([(4172.811152197043, 3099.9668610156855)]) # override for testing
                
                print(cords)
            except Exception as e:
                print("WCS conversion failed:", e)
                print("Using last known coordinates or image center.")

            if self.Centroid_override is not None:
                cords = self.Centroid_override # override for testing
                # if(self.allignment_needed//10 == 0):
                #     transform, _ = aa.find_transform(calibrated_second[::5], self.reference_frame[::5])

                #     calibrated_second,_ = aa.apply_transform(
                #         transform,
                #         calibrated_second,
                #         self.reference_frame
                #     )
                #     print("Applied image alignment using astroalign.")
                #     #calibrated_second, footprint = aa.register(calibrated_second, self.reference_frame) # allignement 
                #     max_idx = np.unravel_index(np.argmax(calibrated_second), calibrated_second.shape)
                #     row, col = max_idx
                #     print(row, col)
                #     cords = np.array([(col, row)])  # update cords after alignment
                max_idx = np.unravel_index(np.argmax(calibrated_second), calibrated_second.shape)
                row, col = max_idx
                cords = np.array([(row,col)])
                print("Using Centroid Override:", cords)
                
                self.allignment_needed += 1
            cords_transformed = cords #transform(cords)

            nx, ny = calibrated_second.shape
            print(nx,ny)
            t_p = cords_transformed
            x, y = t_p[0]
            x, y = int(x), int(y)
            half_box = 50
            x1, x2 = max(0, x - half_box), min(nx, x + half_box)
            y1, y2 = max(0, y - half_box), min(ny, y + half_box)

            mask = np.zeros_like(calibrated_second)
            mask[y1:y2, x1:x2] = 1
            print(y1, y2, x1, x2 )

            past_cord = cords_transformed
            
            if(cords_transformed is None):
                #print("No stars found, using previous coordinates") 
                cords_transformed = past_cord
            else:
                background = np.min(calibrated_second)
                
                data_background_subtracted = calibrated_second # - background
                PRE_MASK = data_background_subtracted
                data_background_subtracted = data_background_subtracted[x1:x2, y1:y2]
                #data_background_subtracted = -np.min(data_background_subtracted) + data_background_subtracted
                _, bkg_median, _ = sigma_clipped_stats(data_background_subtracted, sigma=3.0)
                data_background_subtracted = data_background_subtracted - bkg_median #######################################################
               
                smoothed = data_background_subtracted# sp.ndimage.gaussian_filter(data_background_subtracted, sigma=2)
                if plotting and self.Centroid_override is not None: 
                    import matplotlib.patches as patches
        
                    y_plot, x_plot = self.Centroid_override[0]  # coords as seen in imshow ##3154.135590131793, 1994.7199021208542 
                    half_box = 50

                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(calibrated_second, origin='lower', cmap='gray',
                            vmin=median_s - 1*std_s, vmax=median_s + 5*std_s)

                    rect = patches.Rectangle(
                        (x_plot - half_box, y_plot - half_box),  # bottom-left corner
                        half_box * 2, half_box * 2,              # width, height
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.scatter(x_plot, y_plot, color='red', s=20, marker='+')
                    plt.title("Raw image with search box")
                    plt.show()

                x4, y4 = centroid_1dg(smoothed) #centroid_com(smoothed) # centroid_2dg(smoothed)
                #plt.scatter(x4, y4, color='red', s=10)

                #plt.show()
                if plotting:
                    plt.figure()
                    plt.imshow(data_background_subtracted, cmap='gray', origin='lower', vmin=median_s - 1*std_s, vmax=median_s + 5*std_s)    
                    plt.show()
                    plt.figure()
                #plt.imshow(calibrated_second * mask, cmap='blues', origin='lower')    
                #plt.show()
                cords_transformed = np.array([(x4 + y1, y4 + x1)])
                radii = np.arange(1,15)
                cog = CurveOfGrowth(data_background_subtracted, (x4,y4), radii, mask=None)
                #cog = RadialProfile(data_background_subtracted, (x4,y4), radii, mask=None)
                growth_rate = np.diff(cog.profile)
                growth_rate = np.diff(growth_rate) # second derivative
                try:  
                    optimal_index  = np.where(growth_rate < 0)[0][0] 
                    optimal_radius = radii[optimal_index -1 ] # +1 because of the double diff 
                except IndexError as e:
                    print(e)
                    try:
                        optimal_index = optimal_index # use last used index
                        optimal_radius = radii[optimal_index]
                    except Exception as e:
                        print(e)
                        optimal_index = len(growth_rate)//2
                        optimal_radius = radii[optimal_index] # default value if all else fails
                #print("Optimal aperture radius:", optimal_radius)
        
                #threshold = 0.1* np.max(growth_rate)  # e.g., 1% of max growth
                #optimal_index = np.where(growth_rate < threshold)[0][0]
                #optimal_radius = radii[(optimal_index)]
                #print("Optimal aperture radius:", optimal_radius)
                if(plotting == True):
                    plt.figure()
                    plt.imshow(data_background_subtracted, cmap='Grays', origin='lower')    
                    plt.colorbar(label="Flux (ADU)")
                    plt.xlabel("X [pixels]")
                    plt.ylabel("Y [pixels]")

                # overlay aperture circles
                indexed_apertures = []
                
                for r in radii:
                    aperture = CircularAperture((x4, y4), r=r)
                    indexed_apertures.append(aperture)
                    if(plotting == True):
                        aperture.plot(lw=1, alpha=0.5)
                
                #aperture = CircularAperture((x4, y4), r=optimal_radius)
                aperture = indexed_apertures[optimal_index]
                #aperture.plot(lw=2, color='red', label='Optimal Aperture Radius')

                ap_radius = optimal_radius
                ann_inner = optimal_radius + 3
                ann_width = 3

                #aper_t = CircularAperture(cords_transformed[0], r=ap_radius)
                aper_t = aperture
                ann_t = CircularAnnulus((x4,y4), r_in=ann_inner, r_out=ann_inner+ann_width)
                if(plotting == True):
                    ann_t.plot(color='blue')

                    plt.legend()
                    # optionally, mark the star center
                    plt.scatter(x4, y4, color='red', s=10)

                    plt.show()



                    plt.figure()
                    plt.plot(radii, cog.profile, 'bo-')
                    plt.axvline(optimal_radius, color='r', linestyle='--', label='Optimal Radius')
                    plt.legend()
                    plt.xlabel('Aperture Radius (pixels)')
                    plt.ylabel('Cumulative Flux')
                    plt.title('Curve of Growth')
                    plt.grid()
                    plt.show()

            
            apertures = [aper_t, ann_t]
            
            photom_table = aperture_photometry(data_background_subtracted, apertures) # photom_table will contain aperture_sum and annulus_sum for each entry
            #print(photom_table)

            area =  np.pi * (ann_inner + ann_width)**2 - np.pi * (ann_inner)**2


            bkg_mean = photom_table['aperture_sum_1'] / area
            bkg_sum = bkg_mean * area
            final_sum = photom_table['aperture_sum_0'] - bkg_sum
            

            self.saving_constant += 1
            self.photom_list.append(final_sum.value[0])


        self.photom_list, self.date_array = self.drop_outliers(self.photom_list, self.date_array, threshold=2)
        
        if normalize:
            self.photom_list = self.normalize_lightcurve()
        
        if plotting:
            self.plot_lightcurve()

        return self.photom_list, self.date_array
    


    def save_lightcurve(self, output_path=None):
        if not (len(self.photom_list) == 0 or len(self.date_array) == 0 ):
            print("No light curve data to save. Please run extract_light_curve() first.")
            return

        df = pd.DataFrame({
            'Time': self.date_array, 'Flux': self.photom_list
        })

        if output_path is not None:
            output_path = fr"Outputs/{self.star_name}.csv"
            df.to_csv(output_path)
            #print(f"Light curve saved to {output_path}") 
        else: 
            print("No output path provided")
            return 
    
    def plot_lightcurve(self):
        if self.photom_list is None or self.date_array is None:
            print("No light curve data to plot. Please run extract_light_curve() first.")
            return

        plt.figure()
        plt.scatter(self.date_array, self.photom_list)
        plt.xlabel("Time (days since Kepler epoch)")
        plt.ylabel("Flux (arbitrary units)")
        plt.title(f"Light Curve of {self.star_name}")
        plt.show()

    def tuple_format(self):
        if self.photom_list is None or self.date_array is None:
            print("No light curve data to format. Please run extract_light_curve() first.")
            return

        return list(zip(self.date_array, self.photom_list))

    def normalize_lightcurve(self):
        if self.photom_list is None:
            print("No light curve data to normalize. Please run extract_light_curve() first.")
            return

        photom_array = np.array(self.photom_list)
        n_photom_array = 2 * (photom_array - photom_array.min()) / (photom_array.max() - photom_array.min()) - 1
        return n_photom_array

   


        



#plt.scatter(date_array, photom_list)
#plt.show()
#tuples_of_photom = list(zip(date_array, photom_list))
#print(tuples_of_photom[0:5])

"""
model = sm.StarModeling(tuples_of_photom)
model.plotsidebyside_deep('Alderamin')

df = pd.DataFrame({
    'Time': date_array, 'Flux': photom_list

})

df.to_csv(fr"Outputs/Alderamin (Alpha Cephi).csv")

plt.show() 

"""

  
