import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample Data with Fluctuating Total PO Cost
data = {'Date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01']),
        'TotalPOCost': [5000, 5600, 4900, 4300, 7100, 7900, 6900, 6433, 8900, 9300]}
df = pd.DataFrame(data)

# Extract features directly from the Date column
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Convert to PyTorch tensors
X = torch.tensor(df[['Month', 'DayOfWeek']].values, dtype=torch.float32)
y = torch.tensor(df['TotalPOCost'].values, dtype=torch.float32).view(-1, 1)

# Define the model
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 10) #Added a hidden layer to increase flexibility.
        self.relu = nn.ReLU() #Relu activation.
        self.linear2 = nn.Linear(10, output_size)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = RegressionModel(input_size=2, output_size=1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Make predictions
with torch.no_grad():
    predicted = model(X).numpy()

# Plotting
plt.scatter(df['Date'], df['TotalPOCost'], label='Actual Data')
plt.plot(df['Date'], predicted, color='red', label='Predicted Line')
plt.xlabel('Date')
plt.ylabel('Total PO Cost')
plt.title('Total PO Cost Prediction (PyTorch)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Future Predictions
future_dates = pd.to_datetime(['2023-11-01', '2023-12-01', '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01'])
future_months = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
future_days = [3, 5, 0, 3, 3, 1, 3, 5, 0, 2, 4, 6] #Example days.
future_X = torch.tensor(np.column_stack((future_months, future_days)), dtype=torch.float32)

with torch.no_grad():
    future_predictions = model(future_X).numpy()

print("Future PO Cost Predictions:")
for i in range(len(future_dates)):
    print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${future_predictions[i][0]:.2f}")

plt.scatter(future_dates, future_predictions, color='green', label='Future Predictions')
plt.show()