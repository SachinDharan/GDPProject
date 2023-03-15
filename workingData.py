import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

import requests




with open('workingData.txt', 'r') as f:
    data_str = f.read()

data = json.loads(data_str)

df = pd.DataFrame(data)

df.to_csv('workingData.csv')

