A project to implement one hot encoding in Python on categorical data 
to use the project use the command git clone

## One-Hot Encoding and Decoding Example

This project provides an example of One-Hot Encoding categorical data (specifically 'Color') using both `pandas.get_dummies` and `sklearn.preprocessing.OneHotEncoder`. It also demonstrates how to decode the one-hot encoded data back to its original categorical format.

### Encoding with pandas.get_dummies

```python
import pandas as pd

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
```

### Decoding with pandas

To decode data that was one-hot encoded using `pandas.get_dummies`, you can use `idxmax` to find the column with value `1` and then clean up the column names.

```python
# We recover the original category by selecting the column with value = 1
decoded_pandas = encoded_pandas.idxmax(axis=1).str.replace("Color_", "")
decoded_pandas = decoded_pandas.to_frame(name="Decoded_Color")
print("\nDecoded back using pandas:")
print(decoded_pandas)
```

### Encoding with sklearn.preprocessing.OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

# Original Data (re-shown for clarity)
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']
})

# One-Hot Encoding with sklearn
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(data[['Color']])

encoded_sklearn = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(['Color'])
)
print("\nOne-Hot Encoding with sklearn OneHotEncoder:")
print(encoded_sklearn)
```

### Decoding with sklearn.preprocessing.OneHotEncoder

Scikit-learn's `OneHotEncoder` provides an `inverse_transform` method to easily convert one-hot encoded data back to its original categorical format.

```python
# ---- Decoding sklearn OneHotEncoder ----
decoded_sklearn = encoder.inverse_transform(encoded_array)
decoded_sklearn = pd.DataFrame(decoded_sklearn, columns=["Decoded_Color"])
print("\nDecoded back using sklearn inverse_transform:")
print(decoded_sklearn)
```

To run the example code, execute the `label_encoding_one_hot_encoding.py` script:

```bash
python label_encoding_one_hot_encoding.py
```