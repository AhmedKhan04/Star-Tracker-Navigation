import os 
import pandas as pd 

folder_path = "real_data/"
star_names = os.listdir(folder_path)

for star in star_names:
    final_map = []
    #print(star)
    star_data_path = f"{folder_path}{star}"
    star_data_files = os.listdir(star_data_path)
    for file in star_data_files:
        data_point_path = f"{star_data_path}/{file}"
        #print(data_point_path)
        for file in os.listdir(data_point_path): # only want fits files to be pulled
            #print(file)
            if file.endswith(".fits"):
                final_string = f"{data_point_path}/{file}"
                print(f"{data_point_path}/{file}")
                final_map.append(final_string)
    pd.DataFrame(final_map, columns=["FITS File Path"]).to_csv(f"real_data_map_{star}.csv", index=False)


# Save the final map to a CSV file

print("FITS file paths have been saved to 'real_data_map.csv'.")



print("-----------------------------")



