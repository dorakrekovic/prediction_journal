import pandas as pd

# Load the Excel file
df = pd.read_excel("../data/Suhopolje_19_20.xlsx")  # Replace with your file name

# Check column names
print(df.columns)

# Assuming the columns are 'timestamp' and 'temp_kelvin'
df['temp'] = df['t2m'] - 273.15

# Save to a new Excel file
df.to_excel("Suhopolje_19_20.xlsx", index=False)

