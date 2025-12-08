from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#fits_file = pd.read_csv("real_data_map.csv")["FITS File Path"].iloc[0]  # Update this path to your FITS file
fits_file = r"real_data/IM Tauri 2025-11-15/19_02_30/IM Tauri_00001.fits"
    
hdul = fits.open(fits_file)



print("\n=== FITS Header ===")
print(repr(hdul[0].header)) 

data = hdul[0].data
if data is None and len(hdul) > 1:
    data = hdul[1].data

if data is not None:
    data = np.nan_to_num(data)
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
    plt.colorbar(label='Pixel value')
    plt.title(f"FITS Image: {fits_file}")
    plt.show()
else:
    print("No image data found in the FITS file.")

hdul.close()
