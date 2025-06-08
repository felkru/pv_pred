import pandas as pd
import matplotlib.pyplot as plt

start = 35063-267191

# df = pd.read_csv('data/transfer_learning_data/PV Plants Datasets_84071570.csv')
df = pd.read_csv('data/transfer_learning_data/PV Plants Datasets_62032213.csv')
df.plot(x="Date", y="Specific Energy (kWh/kWp)", figsize=(40,7))
plt.savefig('station_1_actual.png')
plt.show()