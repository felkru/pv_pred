import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/transfer_learning_data/PV Plants Datasets_62030198.csv')
df[10000:22000].plot(x="Date", y="Specific Energy (kWh/kWp)")
plt.show()