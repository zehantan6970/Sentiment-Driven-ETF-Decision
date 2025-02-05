import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Prepare the data
data = {
    'Date': [
        '2025-01-24', '2025-01-23', '2025-01-22', '2025-01-21', '2025-01-20',
        '2025-01-17', '2025-01-16', '2025-01-15', '2025-01-14', '2025-01-13',
        '2025-01-10', '2025-01-09', '2025-01-08', '2025-01-07', '2025-01-06',
        '2025-01-03', '2025-01-02', '2024-12-31', '2024-12-30', '2024-12-27',
        '2024-12-26', '2024-12-25', '2024-12-24', '2024-12-23', '2024-12-20',
        '2024-12-19', '2024-12-18', '2024-12-17', '2024-12-16', '2024-12-13',
        '2024-12-12', '2024-12-11', '2024-12-10', '2024-12-09', '2024-12-06',
        '2024-12-05', '2024-12-04', '2024-12-03', '2024-12-02', '2024-11-29',
        '2024-11-28', '2024-11-27', '2024-11-26', '2024-11-25', '2024-11-22',
        '2024-11-21', '2024-11-20', '2024-11-19', '2024-11-18', '2024-11-15',
        '2024-11-14', '2024-11-13', '2024-11-12', '2024-11-11', '2024-11-08',
        '2024-11-07', '2024-11-06', '2024-11-05', '2024-11-04', '2024-11-01',
        '2024-10-31', '2024-10-30', '2024-10-29', '2024-10-28', '2024-10-25',
        '2024-10-24', '2024-10-23', '2024-10-22', '2024-10-21', '2024-10-18',
        '2024-10-17', '2024-10-16', '2024-10-15', '2024-10-14', '2024-10-11',
        '2024-10-10', '2024-10-09', '2024-10-08', '2024-09-30'
    ],
    'Close': [
        1.783, 1.772, 1.739, 1.766, 1.763, 1.763, 1.765, 1.756, 1.756, 1.715,
        1.725, 1.743, 1.751, 1.752, 1.738, 1.732, 1.759, 1.825, 1.868, 1.849,
        1.844, 1.847, 1.846, 1.814, 1.816, 1.811, 1.823, 1.814, 1.817, 1.824,
        1.870, 1.855, 1.863, 1.846, 1.850, 1.827, 1.818, 1.829, 1.815, 1.806,
        1.785, 1.793, 1.769, 1.756, 1.765, 1.829, 1.823, 1.827, 1.819, 1.814,
        1.848, 1.861, 1.863, 1.888, 1.907, 1.954, 1.864, 1.875, 1.822, 1.795,
        1.789, 1.770, 1.790, 1.794, 1.803, 1.803, 1.809, 1.801, 1.799, 1.829,
        1.770, 1.790, 1.764, 1.809, 1.773, 1.772, 1.843, 2.048, 1.862
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocess the data (scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Prepare data for LSTM (convert to time-series format)
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time-step for LSTM
time_step = 10  # Use 10 previous days' data to predict the next day
X, y = create_dataset(scaled_data, time_step)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape data for LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the LSTM model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # Take the output of the last time step
        return out

# Create the model
model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(torch.FloatTensor(X_train))
    loss = criterion(y_pred, torch.FloatTensor(y_train).view(-1, 1))
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Predict on the test set
model.eval()
predicted_price = model(torch.FloatTensor(X_test)).detach().numpy()

# Inverse scale the predictions and actual values
predicted_price = scaler.inverse_transform(predicted_price)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.plot(y_test_actual, label='True Price')
plt.plot(predicted_price, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('LSTM - Stock Price Prediction')
plt.legend()
plt.show()

# Predict the stock price for 2025-01-27 using the last 10 days of data
last_10_days = scaled_data[-time_step:].reshape(1, time_step, 1)
predicted_27th = model(torch.FloatTensor(last_10_days)).detach().numpy()
predicted_27th = scaler.inverse_transform(predicted_27th)
print(f"Predicted stock price for 2025-01-27: {predicted_27th[0][0]}")
