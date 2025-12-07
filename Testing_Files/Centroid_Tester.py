import numpy as np
from photutils.datasets import make_4gaussians_image
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)
import matplotlib.pyplot as plt
from astropy.io import fits


data = fits.getdata(r"C:\Users\ahmed\Downloads\HIP 91726 (delta Sct)_00015.fits").astype(float)


bg_region = data[0:30, 0:125]
bg_median = np.median(bg_region)
bg_std = np.std(bg_region)
print(f"Background median: {bg_median:.2f}, std: {bg_std:.2f}")

bg_median = np.min(data)

data_bg_subtracted = data - bg_median

data = data_bg_subtracted
data = -np.min(data) + data
data = data[480:560, 930:970]

plt.imshow(data, cmap='Blues', origin='lower')#, vmin=median_s-2*std_s, vmax=median_s+5*std_s)

#aper = CircularAperture(cords, r=ap_radius)
#ann = CircularAnnulus(cords, r_in=ann_inner, r_out=ann_inner+ann_width)
"""
cords_transformed = np.array([(3499.825554241658, 39.37473823979557)])

ny, nx = data.shape
t_p = cords_transformed
x, y = t_p[0]
x, y = int(x), int(y)
half_box = 50
#print(x)
#print(y)
x1, x2 = max(0, x - half_box), min(nx, x + half_box)
y1, y2 = max(0, y - half_box), min(ny, y + half_box)

mask = np.zeros_like(data)
mask[y1:y2, x1:x2] = 1

masked_data = data *  mask


"""
#data -= np.median(data[0:30, 0:125])
plt.figure()
plt.imshow(data, cmap='Blues', origin='lower')#, vmin=median_s-2*std_s, vmax=median_s+5*std_s)

#x1, y1 = centroid_com(data)
#print(np.array((x1, y1)))
#x2, y2 = centroid_quadratic(data)
#print(np.array((x2, y2)))
#x3, y3 = centroid_1dg(data)
#print(np.array((x3, y3)))
x4, y4 = centroid_2dg(data)
print(np.array((x4, y4)))
#plt.plot(x1, y1, 'rx', label='COM')
#plt.plot(x2, y2, 'gx', label='Quadratic')
#plt.plot(x3, y3, 'yx', label='1D Gaussian')
plt.plot(x4, y4, 'mx', label='2D Gaussian')
plt.legend()
plt.figure()
plt.hist(data.flatten(), bins=1000)
plt.show()

