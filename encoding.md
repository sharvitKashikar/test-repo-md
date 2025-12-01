# Encoding Categorical Data in Python  
This document explains Label Encoding and One-Hot Encoding using the exact code provided.  
It is intended as a standalone `encoding.md` file for GitHub.

---

## 📌 Introduction

Most machine learning models require **numerical** input.  
However, real datasets often contain **categorical** values such as colors, cities, product types, etc.

To use these values in ML models, we convert them into numbers using **encoding techniques**.  
This file focuses on:

- **One-Hot Encoding using pandas**
- **One-Hot Encoding using scikit-learn**

The entire explanation is based on the following code:

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Original Data
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']
})
print("Original Data:")
print(data)

# One-Hot Encoding with pandas get_dummies
encoded_pandas = pd.get_dummies(data, columns=["Color"])
print("\nOne-Hot Encoding with pandas.get_dummies:")
print(encoded_pandas)

# One-Hot Encoding with sklearn
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(data[['Color']])

encoded_sklearn = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(['Color'])
)
print("\nOne-Hot Encoding with sklearn OneHotEncoder:")
print(encoded_sklearn)
