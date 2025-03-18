import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime, timedelta

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
        self.linear1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
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

# Future Predictions (Day by Day)
today = datetime.now()
future_dates = [today + timedelta(days=i) for i in range(1, 366)]  # Predict for the next year (365 days)
future_months = [date.month for date in future_dates]
future_days = [date.weekday() for date in future_dates]
future_X = torch.tensor(np.column_stack((future_months, future_days)), dtype=torch.float32)

with torch.no_grad():
    future_predictions = model(future_X).numpy()

# Create DataFrame for results
future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Cost': future_predictions.flatten()})

print(future_df)