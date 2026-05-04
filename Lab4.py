import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from sklearn.metrics import mean_squared_error

# 🔹 Create dummy time series data
data = np.sin(np.linspace(0, 100, 500))

# 🔹 Prepare dataset
X, y = [], []
for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

# 🔹 Function to build model
def build_model(model_type):
    model = Sequential()

    if model_type == "RNN":
        model.add(SimpleRNN(32, input_shape=(10, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(32, input_shape=(10, 1)))
    elif model_type == "GRU":
        model.add(GRU(32, input_shape=(10, 1)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

# 🔹 Train and evaluate models
results = {}

for model_type in ["RNN", "LSTM", "GRU"]:
    print(f"\nTraining {model_type}...")

    model = build_model(model_type)

    start = time.time()
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    end = time.time()

    preds = model.predict(X, verbose=0)
    mse = mean_squared_error(y, preds)

    results[model_type] = {
        "MSE": mse,
        "Training Time": end - start
    }

# 🔹 Print Results
print("\n📊 Results Comparison:")
for k, v in results.items():
    print(f"{k}: MSE = {v['MSE']:.4f}, Time = {v['Training Time']:.2f} sec")