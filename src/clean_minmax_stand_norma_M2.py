import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , MinMaxScaler , Normalizer
df=pd.read_csv('C:/Users/HP/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv'
)
print("Original Dataset")
print(df.head())
print("dataset Shape:" , df.shape)
print("\nData Types:")
print("------------------")
print(df.types)
print(df.isnull().sum())
print("\nDuplicate Records")
df=df.drop_duplicates()
numerical_columns=df.select_dtypes(include=['int64','float64']).columns
for column in numerical_columns:
    df[column]=df[column].fillna(df[column].mean())

categorical_columns=df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column]=df[column].fillna(df[column].mode()[0])

for column in categorical_columns:
    df[column]=df[column].str.strip()

numeric_columns=df.select_dtypes(include=['int 64','float64']).columns
print("\nNumeric Columns")
print(list(numeric_columns))

#standardization(z-score)
#mean =0, standard deviation=1

standard_scaler=StandardScaler()
standardized=standard_scaler.fit_transform(df[numeric_columns])
