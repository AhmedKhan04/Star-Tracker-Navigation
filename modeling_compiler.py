import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
from astropy.io import fits

import modeling as sm 
import light_curve_extraction as lce
import pandas as pd 


#compile all 

class ModelingCompiler:

    def __init__(self, bias, dark, flat_norm, date_maps = None, star_names = None):
        
        self.data_maps = date_maps 
        self.star_names = star_names

        # global settings 

        #self.file_list = file_list
        self.bias = bias
        self.dark = dark
        self.flat = flat

        # storage for results
        self.photom_results = [] 
        self.compiled_dates = []
        self.tuple_results = []
        self.models = [] 


    def compile_from_files(self, file_list_output):
        # load data from output files
        self.data_maps = []
        for i, file_path in enumerate(file_list_output):
            df = pd.read_csv(file_path)
            self.compiled_dates.append(df['Time'].to_numpy())
            self.photom_results.append(df['Flux'].to_numpy())
            self.tuple_results.append( list(zip(df['Time'], df['Flux'])) )

            model, _, model_string = sm.StarModeling(tuples_values=self.tuple_results[i]).getCompositeSine2_deep()
            self.models.append((model, model_string))



    def compile_light_curves(self):
        for i, star_name in enumerate(self.star_names):
            extractor = lce.LightCurveExtractor(self.bias, self.dark, self.flat, self.data_maps[i], star_name)
            photom_list, date_array = extractor.extract_light_curve(normalize = True)
            self.photom_results.append(photom_list)
            self.compiled_dates.append(date_array)
            self.tuple_results.append(extractor.tuple_format())
            #extractor.save_lightcurve(star_name, photom_list, date_array)

            model, _, model_string = sm.StarModeling(tuples_values=self.tuple_results[i]).getCompositeSine2_deep(self.star_names[i])
            self.models.append((date_array, model, model_string))


if __name__ == "__main__":

    # Define our inputs

    #paths

    bias_path = "calibration_frames/Bias_1.0ms_Bin1_ISO100_20251205-065105_32.0F_0001.fit"
    dark_path = "calibration_frames/NGC0891 darks_00015.fits"
    flat_path = "calibration_frames/Flat_300.0ms_Bin1_ISO100_20251205-064251_32.0F_0001.fit"
    data_map_paths = [
        #"data_maps/real_data_map_Alderamin (Alpha Cephi) 2025-11-15.csv",
        "data_maps/real_data_map_IM Tauri 2025-11-15.csv"
    ]

    # star names

    star_names = [
        #"Alderamin",
        "IM Tauri"
    ]

    # Load calibration frames
    bias = fits.getdata(bias_path).astype(float)
    dark = fits.getdata(dark_path).astype(float)
    flat = fits.getdata(flat_path).astype(float)
    
    #create ModelingCompiler instance
    compiler = ModelingCompiler(bias, dark, flat, data_map_paths, star_names)
    compiler.compile_light_curves()
    # Alternatively, compile from existing light curve files
    #file_list_output = ["light_curves/Alderamin_light_curve.csv", "light_curves/IM Tauri_light_curve.csv"]
    #compiler.compile_from_files(file_list_output)

    # Access results
    for i, star_name in enumerate(star_names):
        print(f"Results for {star_name}:")
        print(f"Dates: {compiler.compiled_dates[i][0:5]} ...")
        print(f"Photometry: {compiler.photom_results[i][0:5]} ...")
        model, model_string = compiler.models[i][1], compiler.models[i][2]
        print(f"Model String: {model_string}")

        plt.figure()
        plt.plot(compiler.compiled_dates[i], compiler.photom_results[i], label='Observed Light Curve')
        plt.plot(compiler.compiled_dates[i], model, label='Fitted Model', linestyle='--')
        plt.title(f"Light Curve and Model for {star_name}")
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.legend()

    plt.show()









    











