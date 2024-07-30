# Importing necessary libraries
from sklearn.linear_model import LinearRegression
import numpy as np

# Creating sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Features
y = np.array([2, 4, 6, 8, 10])               # Target variable

# Creating an instance of the linear regression model
model = LinearRegression()

# Fitting the model with the data
model.fit(X, y)

# Making predictions
predictions = model.predict(np.array([6]).reshape(-1, 1))

print(predictions)
