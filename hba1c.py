from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

results = []
target = 'fad8_hba1c'

file_path = f'D:/fadsdiet-foundational/data/fadsdiet_preprocessed_{target}.csv'
df = pd.read_csv(file_path)

X = df.drop(columns=[target])
y = df[target]

print(f'Shape of X: {X.shape}')

print(f'Target {target} in X:', target in X.columns)  # Should be False

# Drop all columns starting with 'fad8' in X
columns_to_drop = X.filter(regex='^fad8').columns
X = X.drop(columns=columns_to_drop)

pd.DataFrame(X.columns).to_csv('D:/fadsdiet-foundational/data/features.csv', index=False, header=False)

# Normalize the target variable
target_scaler = StandardScaler()
y = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Normalize the features
feature_scaler = StandardScaler()
X = feature_scaler.fit_transform(X)

# Drop outliers in the target variable using IQR method
Q1 = pd.Series(y).quantile(0.25)
Q3 = pd.Series(y).quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask = (y >= lower_bound) & (y <= upper_bound)
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

model = TabPFNRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Inverse transform the predictions to the original scale
y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Calculate regression metrics
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"Mean Absolute Error (MAE) for {target}: {mae}")
print(f"Mean Squared Error (MSE) for {target}: {mse}")
print(f"R-squared (R²) for {target}: {r2}")

# Append results to the list
results.append({
    'Target': target,
    'MAE': mae,
    'MSE': mse,
    'R²': r2
})

# Print the first 10 true and predicted values
print(pd.DataFrame({'True': y_test_original, 'Predicted': y_pred_original}).head(15))

# Create a DataFrame from the results and display it
results_df = pd.DataFrame(results)
print("\nComparison of Errors and R-squared for each target:")
print(results_df)