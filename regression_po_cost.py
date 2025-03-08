import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime as dt


def main():
    # Sample Data with Dates and Total PO Cost
# Sample Data with Fluctuating Total PO Cost
# Sample Data with Fluctuating Total PO Cost
    data = {'Date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01']),
            'TotalPOCost': [5000, 5600, 4000, 4500, 3000, 3500, 4000, 4500, 3322, 3600]}
    df = pd.DataFrame(data)

    # Extract features directly from the Date column
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek #Monday=0, Sunday=6

    X = df[['Month', 'DayOfWeek']] #Time has been removed.
    y = df['TotalPOCost']

    degree = 2
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)

    # ... (rest of the code for evaluation and plotting)

    # Example of predicting new values (including month and day of week).
    future_dates = pd.to_datetime(['2023-11-01', '2023-12-01', '2024-01-01'])
    future_months = [11, 12, 1]
    future_days = [3, 5, 0] #Wed, Fri, Mon.
    future_X = np.column_stack((future_months, future_days))
    future_predictions = model.predict(future_X)

    # Displaying Predictions
    print("Future PO Cost Predictions:")
    for i in range(len(future_dates)):
        print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${future_predictions[i]:.2f}")

    #Visualizing future predictions.

    plt.scatter(df['Date'], df['TotalPOCost'], label='Actual Data')
    plt.plot(df['Date'], y_pred, color='red', label='Predicted Line')
    plt.scatter(future_dates, future_predictions, color = 'green', label = "Future Predictions") #add future predictions to the graph.
    plt.xlabel('Date')
    plt.ylabel('Total PO Cost')
    plt.title('Total PO Cost Prediction')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()