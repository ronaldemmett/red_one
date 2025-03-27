# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 01:08:30 2025

@author: Romuald
"""

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show first 5 rows
dataframe.head(5)

#Creating a new data frame

# Load library
import pandas as pd

# Create a dictionary
dictionary = {"Name": ['Jacky Jackson', 'Steven Stevenson'],
  "Age": [38, 25],
  "Driver": [True, False]
}

# Create DataFrame
dataframe = pd.DataFrame(dictionary)

# Show DataFrame
dataframe

# Add a column for eye color
dataframe["Eyes"] = ["Brown", "Blue"]

#3.2 Getting information about the data

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Show two rows
dataframe.head(2)

# Show dimensions
dataframe.shape

#We can then get descriptive statistics for any numeric columns

# Show statistics
dataframe.describe()

# Show info
dataframe.info()

dataframe.head(2)

dataframe.tail(2)

#3.3 Slicing DataFrames

#you need to select individual data or slices of a Dataframe

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Select first row
dataframe.iloc[0]

dataframe.iloc[1]

#Select the first four rows

# Select four rows
dataframe.iloc[1:4]

# Select three rows
dataframe.iloc[:4]

# Set index
dataframe = dataframe.set_index(dataframe['Name'])

#We can use the new indexed set of variable to select a specific row

#Selectings rows based on conditionals 

#Problem : you want to select data frame rows based on some condition

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Show top two rows where column 'sex' is 'female'
gen2 = dataframe[dataframe['Sex'] == 'female'].head(2)

# Filter rows
dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]

#Sorting values 

#Problem
#You need to sort a dataframe by the values in a column.

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Sort the dataframe by Age
data_sort_age = dataframe.sort_values(by=["Age"], ascending=False)

#3.6 Replacing values 

#You need to replace values in a dataframe

# Replacement of any instance of "female" into "woman" in the sex column

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Replace values, show two rows
dataframe['Sex'].replace("female", "Woman").head(2)

#We can also replace multiple values at the same time

# Replace "female" and "male with "Woman" and "Man"
dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5)

#3.7 Renaming columns

#Problem: you want to rename a column in a pandas dataframe

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Rename column, show two rows
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)

# Rename columns, show two rows
dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)

#3.8 Finding the minimum, maximum and sum

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Calculate statistics
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())

# Show counts
dataframe.count()

#3.9 Finding unique values

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Select unique values
dataframe['Sex'].unique()
#array(['female', 'male'], dtype=object)

# Show counts
dataframe['Sex'].value_counts()

# Show counts
dataframe['PClass'].value_counts()

# Show number of unique values
dataframe['PClass'].nunique()

#3.10 Handling missing values 

#you want to select missing values in a Dataframe

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Delete column
dataframe.drop('Age', axis=1).head(2)

drop_age = dataframe.drop('Age', axis=1).head(2)


#3.12 deleting a row 
#you want to delete one or more rows 

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Delete rows, show first three rows of output
drop_male = dataframe[dataframe['Sex'] != 'male'].head(3)

# Delete row, show first two rows of output
#the conditions specify the elements that are being kept
data_without_allison = dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2)

# Delete row, by specifying the row index show first two rows of output
ddata_wout_fist_row = dataframe[dataframe.index != 0]

#3.13 Dropping duplicate rows 

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Drop duplicates, show first two rows of output
data_unique = dataframe.drop_duplicates()

# Show number of rows
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))

#Drop duplicates from a subset of elements 

# Drop duplicates
drop_duplicates_subset = dataframe.drop_duplicates(subset=['Sex'])

#3.14 Grouping rows by values 

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Group rows by the values of the column 'Sex', calculate mean # of each group
dataframe.groupby('Sex').mean(numeric_only=True)

# Group rows, count rows
dataframe.groupby('Survived')['Name'].count()


# Group rows, calculate mean
dataframe.groupby(['Sex','Survived'])['Age'].mean()

#3.15 Grouping Rows by Time

# Load libraries
import pandas as pd
import numpy as np

# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Create DataFrame
dataframe = pd.DataFrame(index=time_index)

# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
dataframe.resample('W').sum()


help(np.random.randint)

#3.16 Aggregating operations and statistics

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Get the minimum of every column
dataframe.agg("min")

# Application of specific function to specific sets of column : Mean Age, min and max SexCode
dataframe.agg({"Age":["mean"], "SexCode":["min", "max"]})

# Number of people who survived and didn't survive in each class
dataframe.groupby(
    ["PClass","Survived"]).agg({"Survived":["count"]}
  ).reset_index()


#3.17 Looping over a column : iterate over every element in a column and apply some action

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Print first two names uppercased
for name in dataframe['Name'][0:2]:
    print(name.upper())

#you can use it as a Python object even if it is a dataset 

#3.18 Applying a Function over all elements in a column 

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Create function
def uppercase(x):
    return x.upper()

# Apply function, show two rows
dataframe['Name'].apply(uppercase)[0:2]

#It is a way to work like in Stata to go with  it 

#3.19 Applying a Function to Groups 

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Group rows, apply function to groups

# Group rows, apply function to groups
dataframe.groupby('Sex').apply(lambda x: x.count())

#3.20 Concatening data frames 

# Load library
import pandas as pd

# Create DataFrame
data_a = {'id': ['1', '2', '3'],
          'first': ['Alex', 'Amy', 'Allen'],
          'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# Create DataFrame
data_b = {'id': ['4', '5', '6'],
          'first': ['Billy', 'Brian', 'Bran'],
          'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

# Concatenate DataFrames by rows
pd.concat([dataframe_a, dataframe_b], axis=0)


# Concatenate DataFrames by columns
pd.concat([dataframe_a, dataframe_b], axis=1)

#3.21 Merging dataframes

# Load library
import pandas as pd

# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                 'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
                                                              'name'])

# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
                                                      'total_sales'])

# Merge DataFrames
data_merge = pd.merge(dataframe_employees, dataframe_sales, on='employee_id')

# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')

# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')


pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')

# Merge DataFrames
pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id')











