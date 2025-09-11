import pandas as pd

final_location = "Funtane"

pinova = pd.read_excel(f"../data/Pinova/{final_location}.xlsx")

# Convert to DataFrame
df = pd.DataFrame(pinova)

# Convert 'time' column to datetime
df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M")

# Define the split timestamp
split_time = pd.to_datetime("01/01/2021 0:00", format="%d/%m/%Y %H:%M")

# Split data
df_19_20 = df[df["time"] < split_time]
df_2021 = df[df["time"] >= split_time]

# Save to Excel files
df_19_20.to_excel(f"../data/Pinova/{final_location}_19_20_Pinova.xlsx", index=False)
df_2021.to_excel(f"../data/Pinova/{final_location}_2021_Pinova.xlsx", index=False)

print("Files saved ")
