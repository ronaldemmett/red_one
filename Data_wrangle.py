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



















