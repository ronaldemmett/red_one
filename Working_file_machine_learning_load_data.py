# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:24:25 2025

@author: Romuald
"""

#Loading a sample dataset

#Chapter 2 : Loading data

# Load scikit-learn's datasets
from sklearn import datasets

# Load digits dataset
digits = datasets.load_digits()

# Create features matrix
features = digits.data

# Create target vector
target = digits.target

 # View first observation
features[0]

iris = datasets.load_iris()

#Create feature matrix for the iris data 
feature_iris=iris.data

feature_iris[0]

#Creating a simulated dataset

 # Load library
from sklearn.datasets import make_regression

# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples = 100, n_features = 3,
n_informative = 3,
n_targets = 1,
noise = 0.0,
coef = True,
random_state = 1)

print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

print(features)

# Load library
from sklearn.datasets import make_blobs

# Generate feature matrix and target vector
features, target = make_blobs(n_samples = 100,
n_features = 2,
centers = 3,
cluster_std = 0.5,
shuffle = True,
random_state = 1)

 # View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])


# Load library
import matplotlib.pyplot as plt

# View scatterplot
plt.scatter(features[:,0], features[:,1], c=target)
plt.show()



#2.3 Loading a CSV File
# Load library
import pandas as pd



# Create URL
url = 'https://tinyurl.com/simulated_data'

csv_file='E:\Work\Machine_learning_Python\Current_working_machine_learning\country_database_def.csv'

# Load dataset
dataframe = pd.read_csv(csv_file, encoding='latin-1')

dataframe.head(2)


#Loading excel file

# Load library
import pandas as pd

# Create URL
url = 'E:\Work\Machine_learning_Python\Current_working_machine_learning\country_database.xlsx'

dataframe_excel=pd.read_excel(url, sheet_name=0, header=0)



# View the first two rows
dataframe_excel.head(2)


# Load libraries
import pandas as pd
from sqlalchemy import create_engine
# Create a connection to the database
database_connection = create_engine('sqlite:///sample.db')
# Load data
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)
# View first two rows
dataframe.head(2)


# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.csv'

# Load dataset
dataframe = pd.read_csv(url)

# View first two rows
dataframe.head(2)


#2.5 Loading a JSON File

# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.json'

# Load data
dataframe = pd.read_json(url, orient='columns')

# View the first two rows
dataframe.head(2)

#2.9 Querying a SQLite Database

# Load libraries
import pandas as pd
from sqlalchemy import create_engine

#Creation of table

import sqlite3
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


#There is no table named data, we want to create it 

import sqlite3
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS data (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value INTEGER
                )''')
conn.commit()
conn.close()

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///your_database.db')
Base = declarative_base()

class Data(Base):
    __tablename__ = 'data'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Integer)

Base.metadata.create_all(engine)

# Create a connection to the database
database_connection = create_engine('sqlite:///your_database.db')

# Load data
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)

# View first two rows
dataframe.head(2)

#2.11 Loading Data from a Google Sheet




