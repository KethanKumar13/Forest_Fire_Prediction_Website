import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

data = pd.read_csv("Forest_Fire_Dataset.csv")
City = data['City']
data = data.drop(columns=['City'])
Month = data['Month']
data = data.drop(columns=['Month'])
Wildlife = data['Wildlife']
data = data.drop(columns=['Wildlife'])
new_data = pd.get_dummies(data, columns=["Oxygen level","Temperature", "Humidity", "Wind speed", "Drought", "Topography","Intensity","Possibility of fire"])

new_data["Oxygen level"] = new_data["Oxygen level_High"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Oxygen level_High", "Oxygen level_Low"], inplace=True)


# Combine "Marital status_Yes" and "Marital status_No" into a single binary column
new_data["Temperature"] = new_data["Temperature_High"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Temperature_High", "Temperature_Low"], inplace=True)

new_data["Humidity"] = new_data["Humidity_High"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Humidity_High", "Humidity_Low"], inplace=True)

new_data["Wind speed"] = new_data["Wind speed_High"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Wind speed_High", "Wind speed_Low"], inplace=True)

new_data["Drought"] = new_data["Drought_High"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Drought_High", "Drought_Low"], inplace=True)
new_data["Topography"] = new_data["Topography_High"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Topography_High", "Topography_Low"], inplace=True)

new_data["Intensity"] = new_data["Intensity_High"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Intensity_High", "Intensity_Low"], inplace=True)

new_data["Possibility of fire"] = new_data["Possibility of fire_High"]
new_data.drop(columns=["Possibility of fire_High", "Possibility of fire_Low"], inplace=True)

data_final = new_data.copy()
data_final.insert(0, 'Month', Month)
data_final.insert(1, 'Wildlife', Wildlife)
data_final.insert(2, 'City', City)

data_final = np.array(data_final)
# Data Slicing
X = data_final[0:, 4:-1]
y = data_final[0:, -1]
X = X.astype('float')
y = y.astype('float')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier

# Create and train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)



pickle.dump(rf_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))






