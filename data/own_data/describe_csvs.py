import pandas as pd 

data = pd.read_csv('data/own_data/19_03_25-29_03_25.csv')

print(
    '\ncsv description and info\n',
    data.describe(),
    data.info(),
    '\n\ncsv head\n',
    data.head(),
    '\n\nunique classes\n',
    data.iloc[:, 0].unique(),
)