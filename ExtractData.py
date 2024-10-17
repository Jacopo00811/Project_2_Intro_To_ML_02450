import numpy as np
import pandas as pd
import re


def map_engine_size(value):
    if pd.isna(value):
        return np.nan  
    value = str(value)
    if value.lower() in ["electric", "electric motor", "hybrid", "-", "electric (tri-motor)", "electric (93 kwh)", "electric (100 kwh)"]:
        return 0
    if value.lower() == "hybrid (4.0)":
        return 4.0
    match = re.match(
        r'(\d*\.?\d+)(?:\s*\+\s*(?:Electric|Electric Motor|Hybrid|electric|electric motor|hybrid))?(?:\s*\(.*\))?$',
        value,
        re.IGNORECASE)
    if match:
        return float(match.group(1))  # Return the numeric part
    return np.nan

def map_number(value):
    if pd.isna(value):
        return np.nan  
    value = str(value).strip()
    if value.lower() == "-":
        return 0
    
    # Match the number (integer or decimal) anywhere in the string
    match = re.search(r'(\d*\.?\d+)', value)
    if match:
        return float(match.group(1))  # Return the numeric part as a float
    return np.nan


# Load and read the csv file
path = "Project_2\\Project_2_Intro_To_ML_02450\\Data\\car_data_with_country.csv"
df = pd.read_csv(path)

# Extract raw_data and rows and columns
raw_data = df.values  
rows, columns=df.shape
cols = range(0, columns) # Counts the number of columns and returns it in a range

# Save the attribute names
attributes = np.asarray(df.columns[cols])

# Save the Car Make and Country columns
CarMake = raw_data[:,0]
Country = raw_data[:,-1]

# Make them unique
CarMakers = np.unique(CarMake)
Countries = np.unique(Country)

# Create a dictionary for the Car Makers and Countries
CarMakersDict = dict(zip(CarMakers,range(len(CarMakers))))
CountriesDict = {
    'Germany': 0,
    'Italy': 1,
    'Japan': 2,
    'UK': 3,
    'USA': 4,
    'France': 5,
    'Sweden': 6,
    'Croatia': 7,
    'UAE': 8,
    'South Korea': 9,
    'Others': 10
}

# Replace the Car Make and Country columns with the dictionary values to 
for i in range(rows):
    df.loc[i, "Car Make"]= CarMakersDict[df.loc[i, "Car Make"]]
for i in range(rows):
    df.loc[i, "Country"]= CountriesDict[df.loc[i, "Country"]]

df.drop(df.columns[1], axis=1, inplace=True) # Drop the Model column
columns -= 1  # Update the number of columns
df['Price (in USD)'] = df['Price (in USD)'].str.replace(',', '')  # Remove commas

# Filter the Engine Size, 0-60 MPH Time, Horsepower and Torque columns
df['Engine Size (L)'] = df['Engine Size (L)'].apply(map_engine_size)
df['0-60 MPH Time (seconds)'] = df['0-60 MPH Time (seconds)'].apply(map_number)
df['Horsepower'] = df['Horsepower'].apply(map_number)
df['Torque (lb-ft)'] = df['Torque (lb-ft)'].apply(map_number)

# Change the data type of the columns to numeric
change_type_cols = df.columns[df.dtypes.eq('object')]
df[change_type_cols] = df[change_type_cols].apply(pd.to_numeric, errors='coerce')

# Substitute the incorrected values with the mean of the column
min_value_index = df['Horsepower'].idxmin() 
df.at[min_value_index, 'Horsepower'] = 1500
max_values_index = df['Horsepower'].idxmax()
df.at[max_values_index, 'Horsepower'] = 1500
max_values_index = df['Horsepower'].idxmax()
df.at[max_values_index, 'Horsepower'] = 1500
max_values_index = df['Horsepower'].idxmax()
df.at[max_values_index, 'Horsepower'] = 1500

# Drop the rows with missing values
df.fillna(df.mean(), inplace=True)

# Save the clean data and check shape
X = df.values




# Redefine the updated variables
# CarMake = X[:,0]
# Country = X[:,-1]
# Countries = np.unique(Country)
# CarMakersDict = dict(zip(CarMakers,range(len(CarMakers))))
# CountriesDict = {
#     'Germany': 0,
#     'Italy': 1,
#     'Japan': 2,
#     'UK': 3,
#     'USA': 4,
#     'France': 5,
#     'Sweden': 6,
#     'Croatia': 7,
#     'UAE': 8,
#     'South Korea': 9,
#     'Others': 10
# }
# print(CarMakersDict)

print(f"The original data matrix has shape: {X.shape}")
