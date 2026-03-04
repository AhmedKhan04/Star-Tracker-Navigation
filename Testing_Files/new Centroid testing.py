# new testing 


from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

def test1():
    hdul = fits.open("real_data\97 Psc/Light_97 Piscium_30.0s_Bin1_ISO100_20251204-220742_26.6F_0114.fit")

    #hdul = fits.open("real_data/Alderamin (Alpha Cephi) 2025-11-15/21_15_17/Alderamin (Alpha Cephi)_00001.fits")
    data = hdul[0].data.astype(float)
    header = hdul[0].header
    w = WCS(header)

    # Check what RA/Dec the image center actually points to
    ny, nx = data.shape
    center_coord = w.pixel_to_world(nx//2, ny//2)
    print(f"Image center points to: {center_coord}")

    # Check where your target falls in pixel space
    ra  = header['CRVAL1']
    dec = header['CRVAL2']
    coord = SkyCoord(ra*u.deg, dec*u.deg)
    px, py = w.world_to_pixel(coord)
    print(f"Target pixel coords: ({px:.1f}, {py:.1f})")
    print(f"Image size: {nx} x {ny}")
    print(f"Target in frame: {0 < px < nx and 0 < py < ny}")

    # Check the actual pixel value at that location
    print(f"Pixel value at target: {data[int(py), int(px)]:.2f}")
    print(f"Image median: {np.median(data):.2f}, std: {np.std(data):.2f}")
    for key in ['RA', 'DEC', 'OBJCTRA', 'OBJCTDEC', 'OBJECT']:
        print(key, header.get(key, 'NOT FOUND' ))

def test2():
   

    hdul = fits.open("real_data\97 Psc/Light_97 Piscium_30.0s_Bin1_ISO100_20251204-220742_26.6F_0114.fit")
    data = hdul[0].data.astype(float)

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    print(f"median: {median}, std: {std}")

    daofind = DAOStarFinder(fwhm=5.0, threshold=5.*std)
    sources = daofind(data - median)

    if sources is not None:
        sources.sort('peak')
        sources.reverse()
        print(sources['xcentroid', 'ycentroid', 'peak'][:20])
    else:
        print("No sources found")

def test3():
    hdul = fits.open("real_data\97 Psc/Light_97 Piscium_30.0s_Bin1_ISO100_20251204-220742_26.6F_0114.fit")
    raw = hdul[0].data.astype(float)

    mean, median, std = sigma_clipped_stats(raw, sigma=3.0)
    print(f"Raw - median: {median}, std: {std}, max: {raw.max()}")

    plt.figure(figsize=(12,8))
    plt.imshow(raw, origin='lower', cmap='gray',
            vmin=median - 1*std,
            vmax=median + 5*std)
    plt.title("RAW uncalibrated")
    plt.show()

def test4():
    #hdul = fits.open("real_data\97 Psc/Light_97 Piscium_30.0s_Bin1_ISO100_20251204-195143_28.4F_0004.fit")
    hdul = fits.open("real_data/Alderamin (Alpha Cephi) 2025-11-15/21_15_17/Alderamin (Alpha Cephi)_00001.fits")
    raw = hdul[0].data.astype(float)
    mean, median, std = sigma_clipped_stats(raw, sigma=3.0)
    daofind = DAOStarFinder(fwhm=5.0, threshold=3.*std)
    sources = daofind(raw - median)
    sources.sort('peak')
    sources.reverse()
    sources = sources[sources['peak'] < 64767.0] # filter out saturated sources
    print(sources['xcentroid', 'ycentroid', 'peak'][:20])

    # Plot with labels
    plt.figure(figsize=(12,8))
    plt.imshow(raw, origin='lower', cmap='gray',
            vmin=median - 1*std, vmax=median + 5*std)

    for row in sources[:20]:
        plt.annotate(f"({row['xcentroid']:.0f},{row['ycentroid']:.0f})", 
                    xy=(row['xcentroid'], row['ycentroid']),
                    color='red', fontsize=6)
        plt.scatter(row['xcentroid'], row['ycentroid'], 
                    s=20, facecolors='none', edgecolors='red')

    plt.title("Top 20 brightest sources")
    plt.show()

def test5():
    from astropy.io import fits

    hdul = fits.open("real_data\\97 Psc\\Light_97 Piscium_30.0s_Bin1_ISO100_20251204-195143_28.4F_0004.fit")
    for key, value in hdul[0].header.items():
        print(f"{key}: {value}")
    hdul.close()

test4()

