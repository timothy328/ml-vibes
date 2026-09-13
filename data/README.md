# California housing data schema

The data loader in `california_housing.py` calls
`sklearn.datasets.fetch_california_housing` and keeps the dataset in memory.
It does not write the downloaded data or transformed features to a CSV file.

The input matrix has 20,640 rows and eight numeric feature columns:

| Position | Name | Description |
| ---: | --- | --- |
| 0 | `MedInc` | Median income in the block group |
| 1 | `HouseAge` | Median house age in the block group |
| 2 | `AveRooms` | Average number of rooms per household |
| 3 | `AveBedrms` | Average number of bedrooms per household |
| 4 | `Population` | Block group population |
| 5 | `AveOccup` | Average number of household members |
| 6 | `Latitude` | Block group latitude |
| 7 | `Longitude` | Block group longitude |

The original continuous target is median house value, supplied by
scikit-learn as `target`. The model uses a derived binary target named
`y_binary`:

- `0`: original house value is at or below the dataset median
- `1`: original house value is above the dataset median

The loader returns `(X, y_binary, feature_names)`, where `X` has shape
`(n_samples, 8)` and `y_binary` has shape `(n_samples,)` with integer values
only in `{0, 1}`.
