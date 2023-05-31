import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle


def file_copy():
    pass


# Select the relevant features for the regression model
# 'Snow on Grnd (cm)'
features = ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 'Total Rain (mm)', 'Total Snow (cm)',
            'Total Precip (mm)', 'Snow on Grnd (cm)', 'Spd of Max Gust (km/h)']

all_y = ['Berri1', 'Boyer', 'Brébeuf', 'CSC (Côte Sainte-Catherine)', 'Maisonneuve_1', 'Maisonneuve_2',
         'Maisonneuve_3', 'Notre-Dame', 'Parc', 'Parc U-Zelt Test', 'PierDup', 'Pont_Jacques_Cartier',
         'Rachel / Hôtel de Ville', 'Rachel / Papineau', 'René-Lévesque', 'Saint-Antoine',
         'Saint-Laurent U-Zelt Test', 'Saint-Urbain', 'Totem_Laurier', 'University', 'Viger']

y_columns = ['Berri1', 'Boyer', 'Brébeuf', 'CSC (Côte Sainte-Catherine)', 'Maisonneuve_2',
             'Maisonneuve_3', 'Notre-Dame', 'Parc', 'PierDup',
             'Rachel / Hôtel de Ville', 'Rachel / Papineau', 'René-Lévesque', 'Saint-Antoine',
             'Saint-Urbain', 'Totem_Laurier', 'University', 'Viger']


def load_one_column_data(column):
    merged_data = load_merged_data()
    X = merged_data[features]
    y = merged_data[column]
    return X, y


def load_data(load_lag_data=False, show_info=False):
    merged_data = load_merged_data()
    if load_lag_data:
        # Add previous three days' values of 'y' as features
        for i in range(3):
            for column in y_columns:
                merged_data[f'{column}_lag{i + 1}'] = merged_data[column].shift(i + 1)

        # Drop any rows with missing values
        merged_data.dropna(subset=features + [f'{column}_lag{i + 1}' for column in y_columns for i in range(3)],
                           inplace=True)

        X = merged_data[features + [f'{column}_lag{i + 1}' for column in y_columns for i in range(3)]]
    else:
        merged_data.dropna(subset=features, inplace=True)
        X = merged_data[features]
    y = merged_data[y_columns]

    if show_info:
        show_feature_target_correlation(X, y)
    return X, y


def show_feature_target_correlation(X, y):
    # Create a grid of scatter plots
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    # Iterate over each x feature and create a scatter plot
    for i, feature in enumerate(X.columns):
        axes[i].scatter(X[feature], y.iloc[:, 0])
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('')
    # Adjust spacing and display the plot
    fig.tight_layout()
    plt.show()


def load_merged_data(show_missing_values=False):
    # Load the data sets
    bike_lane_data = pd.read_csv('datasets/MontrealBikeLane.csv')
    weather_data = pd.read_csv('datasets/WeatherInfo.csv')
    # Convert the date columns to datetime format
    bike_lane_data['Date'] = pd.to_datetime(bike_lane_data['Date'], format="%d/%m/%Y")
    weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
    # Merge the data sets on the 'Date' column
    merged_data = pd.merge(bike_lane_data, weather_data, left_on='Date', right_on='Date/Time')

    if show_missing_values:
        plot_missing_values_in_features(merged_data)
        plot_missing_values_in_targets(merged_data)

    # Replace '<31' with 31 in 'Spd of Max Gust (km/h)' column
    merged_data['Spd of Max Gust (km/h)'] = merged_data['Spd of Max Gust (km/h)'].replace('<31', 31)
    # Convert columns to numeric type
    merged_data[features] = merged_data[features].astype(float)
    # Fill missing values in 'Snow on Grnd (cm)' column with 0
    merged_data['Snow on Grnd (cm)'] = merged_data['Snow on Grnd (cm)'].fillna(0)
    # Fill missing values with the mean of previous and next rows
    merged_data[features] = merged_data[features].fillna(merged_data[features].shift())
    return merged_data


def plot_missing_values_in_features(merged_data):
    # Calculate the proportion of missing values in the features columns
    missing_values_prop = merged_data[features].isnull().mean()
    print(missing_values_prop)
    # Set a custom color palette
    colors = sns.color_palette("mako", len(missing_values_prop))
    # Create a horizontal bar plot with seaborn
    plt.figure(figsize=(10 * 0.65, 6 * 0.65))
    sns.barplot(x=missing_values_prop.values, y=missing_values_prop.index, palette=colors)
    # Customize the plot
    plt.xlabel('Proportion of Missing Values')
    plt.ylabel('Features')
    plt.title('Proportion of Missing Values in Features Columns')
    plt.grid(axis='x')  # Add a grid to the x-axis
    plt.tight_layout()
    file_name = './missing_data_features.eps'
    plt.savefig(file_name, format='eps')
    file_copy(file_name)
    plt.show()


def plot_missing_values_in_targets(merged_data):
    # Calculate the proportion of missing values in the target columns
    missing_values_prop = merged_data[all_y].isnull().mean()
    print(missing_values_prop)
    # Set a custom color palette
    colors = sns.color_palette("mako", len(missing_values_prop))
    # Create a horizontal bar plot with seaborn
    plt.figure(figsize=(10 * 0.65, 6 * 0.65))
    sns.barplot(x=missing_values_prop.values, y=missing_values_prop.index, palette=colors)
    # Customize the plot
    plt.xlabel('Proportion of Missing Values')
    plt.ylabel('Target Variables')
    plt.title('Proportion of Missing Values in Target Variables')
    plt.grid(axis='x')
    plt.tight_layout()
    file_name = './missing_data_targets.eps'
    plt.savefig(file_name, format='eps')
    file_copy(file_name)
    plt.show()


def baseline_comparison():
    X, y = load_data(load_lag_data=True)
    X = np.array(X)
    y = np.array(y)
    # Shuffle the data
    X, y = shuffle(X, y)
    y = y[:, 0]
    # Define a list of models to try
    models = [
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        SVR()
    ]
    # Iterate over the models and evaluate their performance
    for model in models:
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        mean_r2 = scores.mean()
        print(f"Model: {model.__class__.__name__}")
        print("Mean R-squared:", mean_r2, scores)


if __name__ == '__main__':
    baseline_comparison()
    # load_merged_data(show_missing_values=True)
    # load_data(show_info=True)
