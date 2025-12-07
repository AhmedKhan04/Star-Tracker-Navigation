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
from photutils.centroids import centroid_2dg
from photutils.background import Background2D, MedianBackground
import pandas as pd 
from datetime import datetime
import scipy.stats as stats
import copy 


import modeling as sm 

#import cv2



####

# Global variables
initial = True 

photom_list = []
light_curve_extraction = False
plotting = False
date_array = []


####

class LightCurveExtractor:

    def __init__(self, bias, dark, flat , data_map, star_name):
        #self.file_list = file_list
        self.bias = bias
        self.dark = dark
        self.flat = flat
        self.data_map = data_map
        self.star_name = star_name

        self.photom_list = list([])
        self.date_array = list([])

        self.saving_constant = 0 


    def load_calibrated(self, data):
        #data = fits.getdata(file_path).astype(float)
        flat_norm = self.flat / np.median(self.flat)
        return (data - self.dark)
        #return (data - self.bias - self.dark) / flat_norm


    def get_max(self, img): 
        maxVal = np.amax(img)
        #print('--------------')
        #print(maxVal)
        print(img.shape)
        min = np.min(img)
        mask = np.ones_like(img) 
        print(mask.shape)
        mid_point_array = np.array([(mask.shape[0]//2, mask.shape[1]//2)])
        print(mid_point_array)
        
        return mid_point_array
        
        #print()
        #mask[0:250, 0:250] = min # removing time_stamp from image 
        #print(np.argmax(img))

        #maxLoc = np.unravel_index(np.argmax(img), img.shape)
        #maxLoc = [(maxLoc[1], maxLoc[0])]
        #print(maxLoc)
        #print('--------------')
        #return maxLoc

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

# i am going to calibrate my picture here since we do not have a proper class structure yet....

#sci_data = fits.getdata(r"images\WASP-12b_example_uncalibrated_images\uncalibrated\WASP-12b_00040.fits").astype(float)


#bias_data = fits.getdata(r"images\WASP-12b_example_raw_biases\bias_00100.fits").astype(float)


#dark_data = fits.getdata(r"C:\Users\ahmed\Downloads\NGC0891 darks_00015.fits").astype(float)
#flat_data = fits.getdata(r"images\WASP-12b_example_raw_flats\flat_r_00002.fits").astype(float)

#dark_corrected = sci_data - bias_data - dark_data
#flat_norm = flat_data / np.median(flat_data)
#calibrated = dark_corrected / flat_norm

#folder_path = r"images\WASP-12b_example_uncalibrated_images\uncalibrated"

    def extract_light_curve(self, normalize=False):
        for filename in pd.read_csv(self.data_map)["FITS File Path"]:
            #print(filename)
            file_path = filename #os.path.join(folder_path, filename)
            
        
            print(file_path)
            # pulling in second image...eventually loop this. 
            #sci_data_second = fits.getdata(fr"{file_path}").astype(float)
            #dark_corrected_second = sci_data_second - bias_data - dark_data
            
            ############
            # OVERRIDE FOR TESTING

            hdul = fits.open(file_path)
            #print("\n=== FITS Header ===")
            #print(repr(hdul[0].header)) 
            #sci_data_second = hdul[0].data.astype(float)

            date_obs = hdul[0].header.get('DATE-AVG')
            date_obs.split('/')[0].strip()
            
            date_obs_clean = date_obs.split("'/")[0].strip()  
            #print(date_obs)
            if '.' in date_obs_clean:
                date_part, frac = date_obs_clean.split('.')
                frac = frac[:6]  # keep only first 6 digits
                date_obs_clean = f"{date_part}.{frac}"

            # (example: ''2025-11-03T00:41:16.2910676' / System Clock:Est. Frame Mid Point')
            #dt = datetime.fromisoformat(date_obs)
            #print(dt)
            # Convert to fractional hours since start of day (UTC)
            #utc_float =  dt.hour + dt.minute / 60 + dt.second / 3600 + dt.microsecond / (3600 * 1e6)
            #rint(f"UTC time (fractional hours): {utc_float}")  

            
            #print(date_obs_clean)
            #print('--------------')
            t_utc = Time(date_obs_clean, format='isot', scale='utc')
            print(t_utc)
            # Define Kepler mission epoch (BJD 2454833.0)
            kepler_epoch = Time(2454833.0, format='jd', scale='utc')

            # Compute days since epoch
            days_since = (t_utc - kepler_epoch).to('day').value
            print(f"Days since Kepler epoch: {days_since:.6f} days")
            self.date_array.append(days_since)


    

            calibrated_second = hdul[0].data.astype(float) # - self.dark  
            #print(calibrated_second) 
            calibrated_second = self.load_calibrated(calibrated_second)
            #calibrated_second[:250, :250] = 0


            hdul.close()
            ##############
            #calibrated_second = load_calibrated(file_path, bias_data, dark_data, flat_norm)


            #from scipy.ndimage import rotate, zoom
            #calibrated_second = rotate(calibrated_second, angle=30.0, reshape=False)
            #calibrated_second = zoom(calibrated_second, 1.5, order=2)

            #mean, median, std = sigma_clipped_stats(calibrated, sigma=3.0)

            mean_s, median_s, std_s = sigma_clipped_stats(calibrated_second, sigma=3.0)

            #img_aligned, footprint = aa.register(calibrated_second, calibrated, detection_sigma=3.0)

            r"""
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(calibrated_second, cmap='grey', interpolation='none', origin='lower', vmin=median_s-2*std_s, vmax=median_s+5*std_s)
            axes[0, 0].axis('off')
            axes[0, 0].set_title("Specific Image")


            axes[0, 1].imshow(calibrated, cmap='grey', interpolation='none', origin='lower', vmin=median-2*std, vmax=median+5*std)
            axes[0, 1].axis('off')
            axes[0, 1].set_title("Target Image")

            axes[1, 0].imshow(img_aligned, cmap='grey', interpolation='none', origin='lower', vmin=median_s-2*std_s, vmax=median_s+5*std_s)
            axes[1, 0].axis('off')
            axes[1, 0].set_title("Specific Image aligned with Target")

            axes[1, 1].imshow(footprint, cmap='grey', interpolation='none', origin='lower')
            axes[1, 1].axis('off')
            axes[1, 1].set_title("Footprint of the transformation")

            axes[1, 0].axis('off')

            plt.tight_layout()
            plt.show()
            plt.figure()
            """

            #---------------------------

            

            FWHM = 3.0


            #daofind = DAOStarFinder(fwhm=FWHM, threshold=5.*std)
            #IRAFfind = IRAFStarFinder(fwhm=FWHM, threshold=5.*std)
            #sources = daofind(calibrated - median)
            #sources_2 = IRAFfind(calibrated - median)

            #cords  = list(zip(sources["xcentroid"], sources["ycentroid"]))

            #cords = np.array([(3499.825554241658, 39.37473823979557)])  # coordinates from image A
            #cords = np.array([(500, 950)])  # coordinates from image A
            #cords = np.array([(980, 560)])  # coordinates from image A
            
            
            #cords = np.array([(560, 980)])
            cords = self.get_max(calibrated_second)
            #cords = np.array([(560, 980)])
            
            print(f"initial centering point: {cords}")
            #cords = get_max(calibrated_second) # coordinates from image A
            #cords = np.array([(2401.93, 2086.15)])  # coordinates from image A
            #transform, (src_list, ref_list) = aa.find_transform(calibrated_second, calibrated)

            #print(transform)  # see what transform was found

            cords_transformed = cords #transform(cords)

            #print("Original coords:", cords)
            #print("Transformed coords:", cords_transformed)


            # override for now for inspection
            #ap_radius  = 1.5 * FWHM
            ##ann_inner = 3 * FWHM
            #ann_width = 2 * FWHM 

            #aper = CircularAperture(cords, r=ap_radius)
            #ann = CircularAnnulus(cords, r_in=ann_inner, r_out=ann_inner+ann_width)

            ny, nx = calibrated_second.shape
            t_p = cords_transformed
            x, y = t_p[0]
            x, y = int(x), int(y)
            print(f"cords: {x, y}")
            half_box = 50
            #print(x)
            #print(y)
            x1, x2 = max(0, x - half_box), min(nx, x + half_box)
            y1, y2 = max(0, y - half_box), min(ny, y + half_box)

            mask = np.zeros_like(calibrated_second)
            mask[y1:y2, x1:x2] = 1

            #masked_data = calibrated_second *  mask

            #plt.imshow(masked_data, cmap='gray', origin='lower',
                    #vmin=median_s - 2*std_s, vmax=median_s + 5*std_s)
            #plt.show()
            past_cord = cords_transformed
            
            #cords_transformed = get_max(masked_data) # DAOStarFinder(fwhm=FWHM, threshold=10*std)(masked_data - median_s)
            
            if(cords_transformed is None):
                print("No stars found, using previous coordinates") 
                cords_transformed = past_cord
            else:
                background = np.min(calibrated_second)
                
                data_background_subtracted = calibrated_second - background
                PRE_MASK = data_background_subtracted
                data_background_subtracted = data_background_subtracted[x1:x2, y1:y2]
                #plt.imshow(calibrated_second, cmap='Blues', origin='lower', vmin=median_s - 2*std_s, vmax=median_s + 5*std_s)
                data_background_subtracted = -np.min(data_background_subtracted) + data_background_subtracted
                
                #plt.imshow(data_background_subtracted, cmap='Grays', origin='lower') #, vmin=median_s - 2*std_s, vmax=median_s + 5*std_s)

                smoothed = sp.ndimage.gaussian_filter(data_background_subtracted, sigma=2)
            
                x4, y4 = centroid_2dg(smoothed)
                #plt.scatter(x4, y4, color='red', s=10)

                #plt.show()

                #plt.imshow(data_background_subtracted, cmap='Grays', origin='lower')    
                #plt.show()
                #print(f"Initial coords: {(x4, y4)}")
                #print(f"Box limits: x({x1}, {x2}) y({y1}, {y2})")
                cords_transformed = np.array([(x4 + y1, y4 + x1)])
                #print(f"Centroided coords: {cords_transformed}")
                #plt.figure()
                
                #plt.imshow(data_background_subtracted, cmap='Blues', origin='lower')
                #plt.plot(x4, y4, 'mx', label='2D Gaussian')
                
                #plt.legend()
                #plt.show()
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
                    optimal_index = optimal_index # use last used index
                    optimal_radius = radii[optimal_index]
                print("Optimal aperture radius:", optimal_radius)
        
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

                #first_deriv  = np.gradient(growth_rate, radii)
                #second_deriv = np.gradient(first_deriv, radii)

                # Find first radius where concavity becomes negative
                #neg_idx = np.where(second_deriv < 0)[0]
                #if len(neg_idx) > 0:
                #    r_turn = radii[neg_idx[0]]
                #else:
                #    r_turn = radii[-1]
                #print(f"Radius where curve starts flattening (2nd derivative < 0): {r_turn:.2f} px")

                

            """    print('going for centroiding')
                #bkg_estimator = MedianBackground()
                #bkg = Background2D(masked_data, box_size=50, filter_size=3, bkg_estimator=bkg_estimator)
                background_subtracted = masked_data 
                plt.imshow(background_subtracted[y1:y2, x1:x2], cmap='gray', origin='lower', 
                    vmin=median_s - 2*std_s, vmax=median_s + 5*std_s)
                plt.show()
                x4, y4 = centroid_2dg(background_subtracted[y1:y2, x1:x2])
                
                cords_transformed = np.array([(y4+y1, x4+x1)])
                print(f"Centroided coords: {cords_transformed}")
                plt.imshow(background_subtracted, cmap='gray', origin='lower', 
                    vmin=median_s - 2*std_s, vmax=median_s + 5*std_s)
                plt.plot(x4, y4, 'mx', label='2D Gaussian Centroid')
                plt.legend()
                plt.show()"""

                #print(degrees)
                #print(direction_matrix) 
            
            


            """
            if(cords_transformed is None):
                print("No stars found, using previous coordinates") 
                cords_transformed = past_cord
            else:
                cords_transformed = get_max(masked_data) 
                increment = 5 # degrees to break up
                radius = 1 #radius between centers 
                degrees = np.arange(360/increment) * increment
                radians = np.radians(degrees)
                direction_matrix = np.array([np.cos(radians), np.sin(radians)]).T
                direction_matrix *= radius
                #plt.figure()
                
                #plt.imshow(calibrated_second, cmap='grey', origin='lower', vmin=median_s-2*std_s, vmax=median_s+5*std_s)
                
                aper_t = CircularAperture(cords_transformed[0], r=ap_radius)
                aper_t.plot(color='green')
                #plt.scatter(cords_transformed[0][0], cords_transformed[0][1])
                phot_table = aperture_photometry(calibrated_second, aper_t)
                
                best_mean = phot_table['aperture_sum'][0] / (np.pi * ap_radius**2)
                best_aperture = aper_t
                best_vec = cords_transformed[0]
                best_achieved = False
                best_previous = cords_transformed[0]
                while(best_achieved == False):
                    for vec in direction_matrix: 
                        #print('NEXT CORD')
                        test_coord = (cords_transformed + np.flip(vec))[0]

                        #print(test_coord)
                        aper_t = CircularAperture(test_coord, r=ap_radius)
                        #aper_t.plot(color='green')
                        #plt.scatter(test_coord[0], test_coord[1])
                        phot_table = aperture_photometry(calibrated_second, aper_t)
                        mean_flux = phot_table['aperture_sum'][0] / (np.pi * ap_radius**2)
                        if mean_flux > best_mean:
                            best_mean = mean_flux
                            best_aperture = aper_t
                            best_vec = vec
                    if(np.array_equal(best_previous, best_vec)):
                        best_achieved = True
                    else: 
                        best_previous = best_vec
                        cords_transformed = cords_transformed + np.flip(best_vec)
                        #print(f"New best mean flux: {best_mean}")
                        #print(f"New best vector offset: {best_vec}")
                        #print(f"New best transformed coords: {(cords_transformed + np.flip(best_vec))[0]}")
                
                if best_aperture is not None:
                    best_aperture.plot(color='red', lw=2)  # highlight the best one
                    #plt.scatter((cords_transformed + np.flip(best_vec))[0][0], (cords_transformed + np.flip(best_vec))[0][1], color='red', s=60, marker='x')
                    print(f"Best aperture mean flux: {best_mean}")
                    print(f"Best vector offset: {best_vec}")
                    print(f"Best transformed coords: {(cords_transformed + np.flip(best_vec))[0]}") 
                
                #print(degrees)
                #print(direction_matrix) 
            """
            
            #print(f"refinied cords {cords_transformed}")
            #plt.scatter((cords_transformed)[0][0], (cords_transformed)[0][1]) # add in our initial centroid. 
            
            #plt.show()
            #plt.close()
            #cords_transformed = DAOStarFinder(fwhm=FWHM, threshold=5.*std, xycoords=cords)
            #cords_transformed = np.array([(3499.825554241658, 39.37473823979557)]) 
            #ap_radius = optimal_radius
            #ann_inner = optimal_radius + 5
            #ann_width = 5

            #aper_t = CircularAperture(cords_transformed[0], r=ap_radius)
            #ann_t = CircularAnnulus(cords_transformed[0], r_in=ann_inner, r_out=ann_inner+ann_width)


            
            

            #apertures = [aper, ann]
            #photom_table = aperture_photometry(calibrated, apertures) # photom_table will contain aperture_sum and annulus_sum for each entry

            #print(photom_table)


            """
            fig, ax = plt.subplots(figsize = (7,7))
            plt.imshow(calibrated, cmap='grey', origin='lower', vmin=median-2*std, vmax=median+5*std)
            aper.plot(color='green')
            ann.plot(color='cyan')
            plt.figure()
            

            fig, ax = plt.subplots(figsize = (7,7))
            plt.imshow(calibrated_second, cmap='grey', origin='lower', vmin=median_s-2*std_s, vmax=median_s+5*std_s)
            aper_t.plot(color='red')
            ann_t.plot(color='cyan')
            #print(aper_t)

            xmin_pixel = -50 + cords_transformed[0][0]
            xmax_pixel = 50 + cords_transformed[0][0]
            ymin_pixel = -50 + cords_transformed[0][1]
            ymax_pixel = 50 + cords_transformed[0][1]
            
            if(xmin_pixel < 0):
                xmin_pixel = 0
            if(ymin_pixel < 0):
                ymin_pixel = 0
            
            if(xmax_pixel > nx):
                xmax_pixel = nx
            if(ymax_pixel > ny):
                ymax_pixel = ny 
            
            ax.set_xlim(xmin_pixel, xmax_pixel)
            ax.set_ylim(ymin_pixel, ymax_pixel)
            #print(xmin_pixel, xmax_pixel, ymin_pixel, ymax_pixel) 
            #plt.scatter(sources['xcentroid'], sources['ycentroid'], s=40, facecolors='none', edgecolors='r', label = "DOA")
            #plt.scatter(sources_2['xcentroid'], sources_2['ycentroid'], s=40, facecolors='none', edgecolors='b', label = "IRAF")
            #plt.legend(loc = "upper left")
            
            plt.title("Calibrated Image with photometry")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            #plt.savefig(fr"images\aperture_masks\{filename}.png")
            plt.show()
            #plt.close()


            # --------------------------
            #Extract the light curve here...
            """
            apertures = [aper_t, ann_t]
            
            photom_table = aperture_photometry(data_background_subtracted, apertures) # photom_table will contain aperture_sum and annulus_sum for each entry
            #print(photom_table)

            area =  np.pi * (ann_inner + ann_width)**2 - np.pi * (ann_inner)**2


            bkg_mean = photom_table['aperture_sum_1'] / area
            bkg_sum = bkg_mean * area
            final_sum = photom_table['aperture_sum_0'] - bkg_sum
            
            print(final_sum)

            self.saving_constant += 1
            self.photom_list.append(final_sum.value[0])

            

            """self.photom_list = photom_list
            self.date_array = date_array"""


            """
            if (saving_constant % 10 == 0 and saving_constant != 0 and light_curve_extraction == True):
                plt.figure()
                plt.plot(photom_list)
                plt.xlabel("Image number")
                plt.ylabel("Flux (arbitrary units)")
                plt.title("Light Curve of Target Star")
                #plt.savefig(r"images\light_curve.png")
                plt.show()
            """

        self.photom_list, self.date_array = self.drop_outliers(self.photom_list, self.date_array, threshold=2)
        
        if normalize:
            self.photom_list = self.normalize_lightcurve()

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
            print(f"Light curve saved to {output_path}") 
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

  
