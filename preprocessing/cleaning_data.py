
# This file will contain all the code that will be used to preprocess 
# the data you will receive to predict a new price. (fill the NaN values, handle text data, etc...).

# def preprocess() will take a new house's data as input and return those data preprocessed as output.

# If your data doesn't contain the required information, you should return an error to the user.

import pandas as pd
import numpy as np
import urllib.parse
import re
from sklearn.impute import KNNImputer
pd.set_option('display.max_columns', None)

def load_and_preprocess_data(file_path, property_type):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.drop(df[df["Type"]=="house group"].index)
    df = df.drop(df[df["Type"]=="apartment group"].index)
    df = df[df["Type"]==property_type]

    return df
def select_and_rename_columns(df):
    df = df[['id','Price','Zip','Type','Subtype','location',
       'Surroundings type',
       'Living area',
       'Bedrooms','Kitchen type','Bathrooms',
       'Building condition',
       'Construction year',
       'Number of frontages',
       'Covered parking spaces', 'Outdoor parking spaces',
       'Swimming pool',
       'Furnished',
       'How many fireplaces?','Surface of the plot',
       'Terrace','Terrace surface',
       'Garden','Garden surface',
       'Primary energy consumption','Energy class','Heating type'
    ]]

    df = df.rename(columns={
        'location' :'Locality',
        'Transaction Type' : 'Type_of_sale',
        'Type' :'Type_of_property',
        'Subtype' : 'Subtype_of_property',
        'Number of frontages': 'Number_of_facades',
        'Bedrooms':'Number_of_rooms',
        'Kitchen type' : 'Fully_equipped_kitchen',
        'How many fireplaces?' : 'Open_fire',
        'Surface of the plot' :'Surface_of_the_land',
    })

    return df

def convert_and_clean(df, common_cols):
    def clean_and_convert(column):
        column = column.apply(lambda x: re.sub('\D+', '', str(x)))
        column = column.replace('', np.nan)
        return column

    for col in common_cols:
        df[col] = clean_and_convert(df[col])

    return df

def handle_garden_terrace(df):
    for feature in ['Garden', 'Terrace']:
        conditions = [
            df[feature]== "Yes",
            (df[feature].isna()) & (df[feature + " surface"].isna()),
            df[feature + " surface"].notna()
        ]
        values = [1, 0, 1]
        df[feature] = np.select(conditions, values)

        df.loc[(df[feature] == 0 ) & (df[feature + " surface"].isna()), feature + ' surface'] = 0

    return df

def nan_replacement(df, cols):
    for col in cols:
        df[col] = df[col].replace("Yes", 1).replace("No", 0).replace('', np.nan).fillna(0)
    return df

def handle_categorical_columns(df, kitchen_mapping, building_cond_mapping):
    df['Kitchen values'] = df['Fully_equipped_kitchen'].map(kitchen_mapping).fillna(df['Fully_equipped_kitchen'])
    df['Building Cond. values'] = df['Building condition'].map(building_cond_mapping).fillna(df['Building condition'])

    df = df.drop(columns=['Fully_equipped_kitchen', 'Building condition'])

    return df

def handle_parking(df):
    conditions = [
        (df["Covered parking spaces"].notna()) & (df["Outdoor parking spaces"].notna()),
        (df["Covered parking spaces"].isna()) & (df["Outdoor parking spaces"].isna()),
        (df["Covered parking spaces"].isna()) & (df["Outdoor parking spaces"].notna()),
        (df["Covered parking spaces"].notna()) & (df["Outdoor parking spaces"].isna())
    ]
    values = [(df["Covered parking spaces"]+df["Outdoor parking spaces"]), 0, df["Outdoor parking spaces"],df["Covered parking spaces"]]
    df['Parking'] = np.select(conditions, values)

    df = df.drop(columns=["Covered parking spaces","Outdoor parking spaces"])

    return df

def get_province(zip_code):
    province_map = {
        (1000, 1299): 'Brussels Capital Region',
        (1300, 1499): 'Walloon Brabant',
        (1500, 1999): 'Flemish Brabant',
        (2000, 2999): 'Antwerp',
        (3000, 3499): 'Flemish Brabant',
        (3500, 3999): 'Limburg',
        (4000, 4999): 'Liège',
        (5000, 5999): 'Namur',
        (6000, 6599): 'Hainaut',
        (6600, 6999): 'Luxembourg',
        (7000, 7999): 'Hainaut',
        (8000, 8999): 'West Flanders',
        (9000, 9999): 'East Flanders'
    }

    for zip_range, province in province_map.items():
        if zip_range[0] <= zip_code <= zip_range[1]:
            return province

    return 'Unknown'


def remove_outliers(df, columns, n_std):
    for col in columns:
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean+(n_std*sd))]
    return df

def one_convert_to_nan(column):
    column = column.replace(1.0, np.nan)
    return column

def knn_imputer(df, exclude_cols):
    other_cols = [col for col in df.columns if col not in exclude_cols]
    impute_knn = KNNImputer(n_neighbors=5)
    df[other_cols] = impute_knn.fit_transform(df[other_cols]).astype(float)
    return df

def preprocess(house_df: pd.DataFrame):
    kitchen_mapping = {'Not installed': 0, 'Installed': 1, 'Semi equipped': 2, 'Hyper equipped': 3, 'USA uninstalled': 0,
                       'USA installed': 1, 'USA semi equipped': 2, 'USA hyper equipped': 3}
    building_cond_mapping = {'To restore': 0, 'To be done up': 2, 'Just renovated': 3, 'To renovate': 1, 'Good': 3, 'As new': 4}
    exclude_cols = ["Price","Type_of_property","Subtype_of_property","Locality","Surroundings type","Energy class","Heating type","Province"]
    # Apartment code
    
    apt_df = select_and_rename_columns(apt_df)
    apt_df = apt_df.drop(columns=['Surface_of_the_land'])
    common_cols = ['Living area', 'Terrace surface', 'Garden surface', 'Primary energy consumption']
    apt_df = convert_and_clean(apt_df, common_cols)
    apt_df = handle_garden_terrace(apt_df)
    apt_df = handle_categorical_columns(apt_df, kitchen_mapping, building_cond_mapping)
    apt_df = handle_parking(apt_df)
    apt_df = nan_replacement(apt_df, ['Furnished', 'Swimming pool', 'Open_fire'])
    apt_df = knn_imputer(apt_df, exclude_cols)
    apt_df = apt_df.drop(apt_df[apt_df["Living area"].isna()].index)

    apt_df['Locality'] = apt_df['Locality'].apply(urllib.parse.unquote)
    apt_df['Province'] = apt_df['Zip'].apply(get_province)

    apt_df = apt_df.astype({"Price": "float", "Number_of_rooms": "float", "Living area": "float",
                            "Terrace surface": "float", "Garden surface": "float",
                            "Number_of_facades": "float", "Primary energy consumption": "float"})

    aptdf = apt_df.copy()
    aptdf = remove_outliers(aptdf, ['Price'], 4)
    apt_df = remove_outliers(aptdf, ['Living area'], 3)
    # apt_df.to_csv("final_apartment.csv")


    # House code
    common_cols = ['Living area', 'Surface_of_the_land', 'Terrace surface', 'Garden surface', 'Primary energy consumption']
    house_df = load_and_preprocess_data("raw_data.csv", "house")
    house_df = select_and_rename_columns(house_df)
    house_df = convert_and_clean(house_df, common_cols)
    print(house_df.columns)
    
    house_df = handle_garden_terrace(house_df)
    house_df = handle_categorical_columns(house_df, kitchen_mapping, building_cond_mapping)

    house_df = nan_replacement(house_df, ['Furnished', 'Swimming pool', 'Open_fire'])
    house_df = handle_parking(house_df)
    house_df = knn_imputer(house_df, exclude_cols)
    house_df = house_df.drop(house_df[house_df["Living area"].isna()].index)
    house_df = house_df.drop(house_df[house_df["Surface_of_the_land"].isna()].index)

    house_df['Locality'] = house_df['Locality'].apply(urllib.parse.unquote)
    house_df['Province'] = house_df['Zip'].apply(get_province)

    house_df = house_df.astype({"Price": "float", "Number_of_rooms": "float", "Living area": "float",
                                "Surface_of_the_land": "float", "Terrace surface": "float", "Garden surface": "float",
                                "Number_of_facades": "float", "Primary energy consumption": "float"})

    housedf = house_df.copy()
    housedf = remove_outliers(housedf, ['Price'], 4)
    house_df = remove_outliers(housedf, ['Living area', 'Surface_of_the_land'], 3)
    house_df['Surface_of_the_land'] = one_convert_to_nan(house_df['Surface_of_the_land'])

    return house_df

if __name__ == "__main__":
    df = pd.read_csv("raw_data.csv")
    cleaned_data = preprocess(df)