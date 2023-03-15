import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import requests

# api_key = '7A0JXiyThxo6l9471M7n97tW5LXwXZcx1qePcPYE'



# url = 'https://api.eia.gov/v2/coal/aggregate-production/data/?frequency=annual&data[0]=average-employees&data[1]=labor-hours&data[2]=number-of-mines&data[3]=production&data[4]=productivity&start=2010&end=2020&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=7A0JXiyThxo6l9471M7n97tW5LXwXZcx1qePcPYE'
# response = requests.get(url)
# js = response.json()
# data = pd.json_normalize(js)


with open('dataACC.txt', 'r') as f:
    data_str = f.read()

data = json.loads(data_str)

df = pd.DataFrame(data)
df.to_csv('dataACC.csv')