import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================================
# FILE PATHS
# ============================================

input_file = (
    'C:/Users/HP/PycharmProjects/placement_prediction/'
    'dataset/placement_predict_50K_Raw.csv'
)

output_file = (
    'C:/Users/HP/PycharmProjects/placement_prediction/'
    'dataset/final_preprocess_M2.csv'
)

# ============================================
# READ DATASET
# ============================================

df = pd.read_csv(input_file)

# Create a copy so the original dataset remains unchanged
processed_df = df.copy()

print("Original Dataset Shape:", processed_df.shape)

# ============================================
# REMOVE DUPLICATE RECORDS
# ============================================

processed_df = processed_df.drop_duplicates().reset_index(drop=True)

# ============================================
# IDENTIFY COLUMNS
# ============================================

# Numeric columns
numeric_cols = processed_df.select_dtypes(
    include=[np.number]
).columns.tolist()

# Categorical/text columns
categorical_cols = processed_df.select_dtypes(
    include=['object', 'string']
).columns.tolist()

print("\nNumeric Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# ============================================
# HANDLE MISSING VALUES
# ============================================

# ---------- Numeric Columns ----------
for col in numeric_cols:

    median_value = processed_df[col].median()

    processed_df[col] = processed_df[col].fillna(
        median_value
    )

# ---------- Categorical Columns ----------
for col in categorical_cols:

    mode_values = processed_df[col].mode()

    if not mode_values.empty:
        mode_value = mode_values.iloc[0]
        processed_df[col] = processed_df[col].fillna(
            mode_value
        )
    else:
        # If the column contains no valid value
        processed_df[col] = processed_df[col].fillna(
            "unknown"
        )

# ============================================
# CLEAN TEXT DATA
# ============================================

for col in categorical_cols:

    # Convert to string
    processed_df[col] = processed_df[col].astype("string")

    # Remove leading and trailing spaces
    processed_df[col] = processed_df[col].str.strip()

    # Convert text to lowercase
    processed_df[col] = processed_df[col].str.lower()

    # Replace empty strings with "unknown"
    processed_df[col] = processed_df[col].replace(
        "",
        "unknown"
    )

# ============================================
# LABEL ENCODING
# ============================================

for col in categorical_cols:

    encoder = LabelEncoder()

    processed_df[col] = encoder.fit_transform(
        processed_df[col].astype(str)
    )

# ============================================
# FEATURE SCALING
# ============================================

# Scale only numeric columns
if len(numeric_cols) > 0:

    scaler = StandardScaler()

    processed_df[numeric_cols] = scaler.fit_transform(
        processed_df[numeric_cols]
    )

# ============================================
# FINAL CHECK FOR MISSING VALUES
# ============================================

missing_values = processed_df.isnull().sum().sum()

print("\nTotal Missing Values After Preprocessing:",
      missing_values)

# ============================================
# SAVE PREPROCESSED DATASET
# ============================================

processed_df.to_csv(
    output_file,
    index=False
)

# ============================================
# FINAL INFORMATION
# ============================================

print("\n============================================")
print("Preprocessing Completed Successfully!")
print("============================================")

print("Original Dataset Shape :",
      df.shape)

print("Processed Dataset Shape:",
      processed_df.shape)

print("Duplicate Rows Removed :",
      df.shape[0] - processed_df.shape[0])

print("Total Missing Values   :",
      missing_values)

print("Saved File             :",
      output_file)