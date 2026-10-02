import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Simple synthetic dataset: House Size (sq ft) vs Price ($)
data = {
    'Size': [1500, 1800, 2400, 3000, 3500, 4000, 1200, 2000, 2800, 3200],
    'Price': [300000, 360000, 480000, 600000, 700000, 800000, 240000, 400000, 560000, 640000]
}

df = pd.DataFrame(data)

# Split features and target
X = df[['Size']]
y = df['Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Example prediction
sample_size = np.array([[2500]])
predicted_price = model.predict(sample_size)
print(f"Predicted price for 2500 sq ft: ${predicted_price[0]:.2f}")