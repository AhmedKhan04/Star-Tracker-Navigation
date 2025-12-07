# data visualization 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import lightkurve as lk

file_path  = r'Outputs\IM Tauri.csv'

df_prior  = pd.read_csv(file_path)
#time = df['Time']
#flux = df['Flux']

z_scores = np.abs(stats.zscore(df_prior['Flux']))
threshold = 2
outlier_indices = np.where(z_scores > threshold)[0]
df = df_prior.drop(outlier_indices)
print(f"The number of points dropped was: {len(df_prior) - len(df)}")
time = df['Time']
flux = df['Flux']
plt.plot(time, flux)
plt.figure
lc_mod = lk.LightCurve(time=time, flux=flux)
lc = lk.search_targetpixelfile('IM Tauri')
print(lc)
lc = lc.download().to_lightcurve().to_periodogram()
lc.plot()
pg_obs = lc_mod.to_periodogram(frequency = lc.frequency)
pg_obs.plot()
plt.show()

print()