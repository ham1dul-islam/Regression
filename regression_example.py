# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Linear Regression Example
# This example demonstrates how to perform linear regression using scikit-learn
# 1. Generate some sample data (replace with your own data if available)
def main():
    np.random.seed(0)  # For reproducibility
    X = 2 * np.random.rand(100, 1)  # 100 data points, 1 feature (independent variable)
    y = 4 + 3 * X + np.random.randn(100, 1)  # Target variable (dependent variable) with some noise

    # 2. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% train, 20% test

    # 3. Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 5. Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # 6. Visualize the results
    plt.scatter(X_test, y_test, color='blue', label='Actual data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted line')
    plt.xlabel('X (Independent Variable)')
    plt.ylabel('y (Dependent Variable)')
    plt.title('Linear Regression Example')
    plt.legend()
    plt.show()

    # 7. Print the model's coefficients (slope and intercept)
    print(f"Intercept: {model.intercept_}")
    print(f"Slope: {model.coef_}")

    #Example of predicting a new value.
    new_x = np.array([[1.5]]) #Must be a 2D array
    new_y_pred = model.predict(new_x)
    print(f"Predicted y for x = 1.5: {new_y_pred}")
    
if __name__ == "__main__":
    main()