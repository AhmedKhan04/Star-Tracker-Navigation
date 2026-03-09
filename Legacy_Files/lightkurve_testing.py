import lightkurve as lk 
import matplotlib.pyplot as plt

# plot lightkurve 

search = lk.search_targetpixelfile("IM Tauri")
print(search)
lc = search[0].download().to_lightcurve().to_periodogram()
lc.plot()
plt.show()
