import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/psm/train.csv')
data = data.drop(columns='timestamp_(min)')
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15, 15))
# plot heat map
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
