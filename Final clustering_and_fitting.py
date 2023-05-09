# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Reads a file and returns the original dataframe, 
# the dataframe with countries as columns, 
# and the dataframe with year as columns.
    
def read_df(filename: str):
  # Read the file into a pandas dataframe
  df = pd.read_csv(filename)
    
  # Transpose the dataframe
  df_trans = df.transpose()
    
  # Populate the header of the transposed dataframe with the header information 
  # silice the dataframe to get the year as columns
  df_trans.columns = df_trans.iloc[1]

  # As year is now columns so we don't need it as rows
  df_trans_year = df_trans[0:].drop('year')
    
  # silice the dataframe to get the country as columns
  df_trans.columns = df_trans.iloc[0]
    
  # As country is now columns so we don't need it as rows
  df_trans_country = df_trans[0:].drop('country')
    
  return df, df_trans_country, df_trans_year

# load data from World Bank website or a similar source
df, df_country, df_year = read_df('worldbank.csv')

# removes null values from a given feature

def remove_null_values(feature):

  # drop null values from the feature
  return np.array(feature.dropna())

# Passing Features to remove_null_values function 

nitrous_oxide = remove_null_values(df[['nitrous_oxide']])
greenhouse_gas_emissions = remove_null_values(df[['greenhouse_gas_emissions']])
agricultural_land = remove_null_values(df[['agricultural_land']])
min_length = min(len(nitrous_oxide), len(greenhouse_gas_emissions), len(agricultural_land))
 
# after removing the null values we will create datafram 
clean_data = pd.DataFrame({ 
                                'country': [df['country'].iloc[x] for x in range(min_length)],
                                'year': [df['year'].iloc[x] for x in range(min_length)],
                                'nitrous_oxide': [nitrous_oxide[x][0] for x in range(min_length)],
                                'greenhouse_gas_emissions': [greenhouse_gas_emissions[x][0] for x in range(min_length)],
                                 'agricultural_land': [agricultural_land[x][0] for x in range(min_length)]
                                 })

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
transform_data = scaler.fit_transform(clean_data[['nitrous_oxide', 'greenhouse_gas_emissions', 'agricultural_land']])

# Use KMeans to find clusters in the clean_data
kmeans = KMeans(n_clusters=3)
kmeans.fit(transform_data)

# Add the cluster assignments as a new column to the clean_data
clean_data['cluster'] = kmeans.labels_

# create a plot showing the clusters and cluster centers using pyplot
for i in range(3):
    cluster_data = clean_data[clean_data['cluster'] == i]
    plt.scatter(cluster_data['nitrous_oxide'], cluster_data['greenhouse_gas_emissions'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Nitrous Oxide')
plt.ylabel('Greenhouse Gas Emissions')
plt.title('Clusters')
plt.legend()
plt.show()

# create a plot showing the clusters and cluster centers using pyplot
for i in range(3):
    cluster_data = clean_data[clean_data['cluster'] == i]
    plt.scatter(cluster_data['agricultural_land'], cluster_data['greenhouse_gas_emissions'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Agricultural Land')
plt.ylabel('Greenhouse Gas Emissions')
plt.title('Clusters')
plt.legend()
plt.show()

jp = clean_data[clean_data['country'] == 'Japan']

# create a plot showing the clusters and cluster centers using pyplot
for i in range(3):
    cluster_data = jp[jp['cluster'] == i]
    plt.scatter(jp['agricultural_land'], jp['greenhouse_gas_emissions'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Agricultural Land')
plt.ylabel('Greenhouse Gas Emissions')
plt.title('Japan Clusters')
plt.legend()
plt.show()

us = clean_data[clean_data['country'] == 'United States']

# create a plot showing the clusters and cluster centers using pyplot
for i in range(3):
    cluster_data = us[us['cluster'] == i]
    plt.scatter(cluster_data['nitrous_oxide'], cluster_data['greenhouse_gas_emissions'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Nitrous Oxide')
plt.ylabel('Greenhouse Gas Emissions')
plt.title('United States Cluster')
plt.legend()
plt.show()

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

# Define the exponential function
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Filter Data for cluster 0

c0 = clean_data[(clean_data['cluster'] == 0)]

# Define x values 
x = c0['agricultural_land']

# Define y values
y = c0['nitrous_oxide']

popt, pcov = curve_fit(exp_func, x, y)

popt

pcov

# Use err_ranges function to estimate lower and upper limits of the confidence range
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_ranges(x, exp_func, popt,sigma)

# Use pyplot to create a plot showing the best fitting function and the confidence range
plt.plot(x, y, 'o', label='data')
plt.plot(x, exp_func(x, *popt), '-', label='fit')
plt.fill_between(x, lower, upper, color='pink', label='confidence interval')
plt.legend()
plt.xlabel('Agricultural Land')
plt.ylabel('Nitrous Oxide')
plt.show()

# Define the range of future x-values 
future_x = np.arange(70, 80)

# Predict the future y-values
future_y = exp_func(future_x, *popt)

# Plot the predictions along with the original data
plt.plot(x, y, 'o', label='data')
plt.plot(x, exp_func(x, *popt), '-', label='fit')
plt.plot(future_x, future_y, 'o', label='future predictions')
plt.xlabel('Agricultural Land')
plt.ylabel('Nitrous Oxide')
plt.legend()
plt.show()