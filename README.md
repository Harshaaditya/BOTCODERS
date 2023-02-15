# BOTCODERS
#ALGORITHM:
1. Load the CSV file into a pandas DataFrame using the 'read_csv function
2. Calculate the total number of rows in the DataFrame using the ' shape* attribute.
3. Calculate the number of unique values in each column using the "unique method.
4. Calculate the mean, median, mode, and standard deviation of numerical columns using the describe method.
5. Group the data by a categorical column and calculate the mean, median, mode, and standard deviation of numerical columns for each group using the groupby method.
6. Plot the data using various visualization tools such as histograms, bar charts, scatter plots, and box plots to gain insights into the distribution and patterns of the data.

SOURCE CODE:


#NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline 

# Seaborn for easier visualization
import seaborn as sns

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# StandardScaler from Scikit-Learn
from sklearn.preprocessing import StandardScaler

# PCA from Scikit-Learn
from sklearn.decomposition import PCA

# Scikit-Learn's KMeans algorithm
from sklearn.cluster import KMeans

# Adjusted Rand index
from sklearn.metrics import adjusted_rand_score

 

#DEMOGRAPHY
import pandas as pd
df = pd.read_csv('data.csv')

df.head() # to see the first few rows of the DataFrame
df.info() # to see information about the DataFrame, such as data types and null values

demography = df.groupby('demographic_variable').size()

demography.plot(kind='bar')

import matplotlib.pyplot as plt

demography.plot(kind='bar')
plt.title('Demographic Variable Counts')
plt.xlabel('Demographic Variable')
plt.ylabel('Count')
plt.show() 
#PURCHASE BEHAVIOUR:
# Print the first few rows of the data
print(df.head())

# Calculate the total number of purchases
num_purchases = df.shape[0]
print('Total number of purchases:', num_purchases)

# Calculate the total revenue
total_revenue = df['Quantity'].sum()
print('Total revenue:', total_revenue)

# Calculate the average purchase price
avg_purchase_price = total_revenue / num_purchases
print('Average purchase price:', avg_purchase_price)

# Calculate the number of unique customers
num_customers = df['CustomerID'].nunique()
print('Number of unique customers:', num_customers)

# Calculate the most popular item
popular_item = df['Description'].mode()[0]
print('Most popular item:', popular_item)

# Calculate the total revenue by item
revenue_by_item = df.groupby('Description')['Quantity'].sum()
print('Revenue by item:\n', revenue_by_item)

# Calculate the total revenue by customer
revenue_by_customer = df.groupby('CustomerID')['Quantity'].sum()
print('Revenue by customer:\n', revenue_by_customer)


