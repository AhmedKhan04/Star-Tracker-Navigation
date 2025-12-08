import os 
import pandas as pd 

folder_path = r"real_data\97 Psc"
star_names = os.listdir(folder_path)
final_map = []
star_name = "97 Psc"
print(star_names)
for file in star_names:
   
    if file.endswith(".fit"):
        print("found!")
        final_string = f"{folder_path}/{file}"
        print(f"{folder_path}/{file}")
        final_map.append(final_string)
    
pd.DataFrame(final_map, columns=["FITS File Path"]).to_csv(f"data_maps/real_data_map_{star_name}.csv", index=False)


# Save the final map to a CSV file

print(f"FITS file paths have been saved to 'date_maps/real_data_map_{star_name}.csv'.")


print("-----------------------------")



