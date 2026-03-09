# this is the photometry pipeline 

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
from photutils import aperture
import astropy as ap 
from PIL import Image
import glob
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.table import Table

# for when we have jpg data
def load_jpg(path):
    img = Image.open(path).convert("L")  # grayscale
    return np.array(img).astype(float)


#sci_data = load_jpg(r"images\WASP-12b_example_uncalibrated_images\uncalibrated\WASP-12b_00040.fits").astype(float)
#bias_data = load_jpg(r"images\WASP-12b_example_raw_biases\bias_00100.fits").astype(float)
#dark_data = load_jpg(r"images\WASP-12b_example_raw_darks\dark_00150.fits").astype(float)
#flat_data = load_jpg(r"images\WASP-12b_example_raw_flats\flat_r_00002.fits").astype(float)



sci_data = fits.getdata(r"images\WASP-12b_example_uncalibrated_images\uncalibrated\WASP-12b_00040.fits").astype(float)


bias_data = fits.getdata(r"images\WASP-12b_example_raw_biases\bias_00100.fits").astype(float)
dark_data = fits.getdata(r"images\WASP-12b_example_raw_darks\dark_00150.fits").astype(float)
flat_data = fits.getdata(r"images\WASP-12b_example_raw_flats\flat_r_00002.fits").astype(float)




dark_corrected = sci_data - bias_data - dark_data
flat_norm = flat_data / np.median(flat_data)
calibrated = dark_corrected / flat_norm

mean_pre, median_pre, std_pre = sigma_clipped_stats(sci_data, sigma=3.0)
mean, median, std = sigma_clipped_stats(calibrated, sigma=3.0)

plt.figure(figsize=(7,7))
plt.imshow(calibrated, cmap='grey', origin='lower',
           vmin=median-2*std, vmax=median+5*std)
#plt.scatter(sources['xcentroid'], sources['ycentroid'],
#            s=40, facecolors='none', edgecolors='r')
plt.title("Calibrated Image with Detected Stars")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")


plt.figure(figsize=(7,7))
plt.imshow(sci_data, cmap='gray', origin='lower',
           vmin=median_pre-2*std_pre, vmax=median_pre+5*std_pre)
#plt.scatter(sources['xcentroid'], sources['ycentroid'],
#            s=40, facecolors='none', edgecolors='r')
plt.title("Uncalibrated Image with Detected Stars")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.show()