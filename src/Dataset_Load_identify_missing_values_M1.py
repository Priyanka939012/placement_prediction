import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('C:/Users/HP/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv'
)
print("first five rows:")
print(df.head(5))

print("print first 8  ")
subset=df.iloc[:,0:6]
print(subset)

missing_counts=subset.isnull().sum()
print ("......missing values per column....")
print(missing_counts)
print("-" * 40)

duplicate_rows=df[df.duplicated()]
print(f"Total duplicated rows: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)

plt.figure(figsize=(10,6))

sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap="viridis")
plt.title("Missing values Heatmap")
plt.show()



