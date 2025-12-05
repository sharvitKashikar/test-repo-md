# One-Hot Encoding and Decoding with Pandas and Scikit-learn

This document describes how to perform One-Hot Encoding on categorical data using both `pandas.get_dummies` and `sklearn.preprocessing.OneHotEncoder`, and crucially, how to decode the data back to its original format.

One-Hot Encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.

## Python Implementation

To run this example, ensure you have `pandas` and `scikit-learn` installed:

```bash
pip install pandas scikit-learn
```

### Code Example: `label_encoding_one_hot_encoding.py`

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

# We recover the original category by selecting the column with value = 1
decoded_pandas = encoded_pandas.idxmax(axis=1).str.replace("Color_", "")
decoded_pandas = decoded_pandas.to_frame(name="Decoded_Color")
print("\nDecoded back using pandas:")
print(decoded_pandas)


# One-Hot Encoding with sklearn
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(data[['Color']])

encoded_sklearn = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(['Color'])
)
print("\nOne-Hot Encoding with sklearn OneHotEncoder:")
print(encoded_sklearn)
# ---- Decoding sklearn OneHotEncoder ----
decoded_sklearn = encoder.inverse_transform(encoded_array)
decoded_sklearn = pd.DataFrame(decoded_sklearn, columns=["Decoded_Color"])
print("\nDecoded back using sklearn inverse_transform:")
print(decoded_sklearn)
```

### Explanation of One-Hot Encoding

- **Original Data**: We start with a simple DataFrame containing a 'Color' column with categorical values.
- **Pandas `get_dummies`**: This function converts categorical variables into dummy/indicator variables. Each unique category becomes a new column, with 1 indicating the presence of that category and 0 otherwise.
- **Scikit-learn `OneHotEncoder`**: A more explicit way to perform one-hot encoding, especially useful in machine learning pipelines. `sparse_output=False` is used to get a dense NumPy array.

### Decoding One-Hot Encoded Data

- **Decoding with Pandas**: To inverse the `get_dummies` operation, we use `idxmax(axis=1)` to find the column with the value `1` for each row, which corresponds to the original category. We then clean up the column prefix (`'Color_'`).
- **Decoding with Scikit-learn**: The `OneHotEncoder` object has a convenient `inverse_transform` method that directly converts the encoded numerical array back into the original categorical values, making the decoding process straightforward and robust.

This example provides a complete workflow for handling one-hot encoded data, from encoding to decoding, using common Python libraries.