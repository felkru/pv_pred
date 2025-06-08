import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/transfer_learning_data/PV Plants Datasets_84071570.csv')
df.plot(x="Date", y="Specific Energy (kWh/kWp)")
plt.show()