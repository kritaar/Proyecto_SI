import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

# Descargar datos de Amazon
df = yf.download('AMZN', start='2014-01-01', end='2024-05-23')

# Usar todas las columnas para RNN
df_rnn = df.copy()

# Usar solo el precio de cierre para KNN y SVM
df = df[['Close']].copy()
df.rename(columns={'Close': 'close'}, inplace=True)

# Escalar datos
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Crear datos de entrenamiento y prueba
def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 10
X, y = create_dataset(df_scaled, time_step)

# Dividir en conjunto de entrenamiento y prueba
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# KNN
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train[:, -1].reshape(-1, 1), y_train)  # Usar solo la última columna para KNN

# SVM
svm = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svm.fit(X_train[:, -1].reshape(-1, 1), y_train)  # Usar solo la última columna para SVM

# RNN con arquitectura ajustada
rnn = Sequential()
rnn.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))  # Más unidades
rnn.add(Dropout(0.2))  # Capa de Dropout para prevenir el sobreajuste
rnn.add(LSTM(100, return_sequences=True))  # Segunda capa LSTM
rnn.add(Dropout(0.2))  # Capa de Dropout
rnn.add(LSTM(50))  # Tercera capa LSTM con menos unidades
rnn.add(Dropout(0.2))  # Capa de Dropout
rnn.add(Dense(1))  # Capa de salida
rnn.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo RNN
rnn.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, batch_size=32, epochs=20)  

# Evaluar el modelo ajustado
y_pred_rnn = rnn.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
rnn_mse = mean_squared_error(y_test, y_pred_rnn)
rnn_r2 = r2_score(y_test, y_pred_rnn)

print(f"RNN Mean Squared Error: {rnn_mse}")

# Crear la interfaz gráfica
def predict_price():
    days = int(entry.get())
    start_date = df_rnn.index[-time_step]
    last_days = df_scaled[-time_step:]
    future_prices_knn = []
    future_prices_svm = []
    future_prices_rnn = []

    for _ in range(days):
        knn_pred = knn.predict(last_days[-1].reshape(1, -1))
        svm_pred = svm.predict(last_days[-1].reshape(1, -1))
        rnn_pred = rnn.predict(last_days.reshape(1, time_step, 1))
        
        knn_pred = knn_pred.reshape(-1, 1)
        svm_pred = svm_pred.reshape(-1, 1)
        rnn_pred = rnn_pred.flatten().reshape(-1, 1)
        
        last_days = np.concatenate((last_days[1:], rnn_pred), axis=0)
        future_prices_knn.append(knn_pred[0])
        future_prices_svm.append(svm_pred[0])
        future_prices_rnn.append(rnn_pred[0])

    future_prices_knn = scaler.inverse_transform(np.array(future_prices_knn).reshape(-1, 1))
    future_prices_svm = scaler.inverse_transform(np.array(future_prices_svm).reshape(-1, 1))
    future_prices_rnn = scaler.inverse_transform(np.array(future_prices_rnn).reshape(-1, 1))

    knn_r2_train = knn.score(X_train[:, -1].reshape(-1, 1), y_train)
    knn_r2_test = knn.score(X_test[:, -1].reshape(-1, 1), y_test)
    svm_r2_train = svm.score(X_train[:, -1].reshape(-1, 1), y_train)
    svm_r2_test = svm.score(X_test[:, -1].reshape(-1, 1), y_test)
    rnn_r2_train = rnn.evaluate(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, verbose=0)
    rnn_r2_test = rnn.evaluate(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test, verbose=0)

    messagebox.showinfo("Predicción y Precisión",
                        f"Predicción a partir de: {start_date.strftime('%Y-%m-%d')}\n\n"
                        f"Precios estimados para {days} días en el futuro:\n\n"
                        f"KNN: {future_prices_knn[-1][0]:.2f}\n"
                        f"SVM: {future_prices_svm[-1][0]:.2f}\n"
                        f"RNN: {future_prices_rnn[-1][0]:.2f}\n\n"
                        "Resultados de precisión:\n"
                        f"KNN: {knn_r2_train:.4f},  (test): {knn_r2_test:.4f}\n"
                        f"SVM: {svm_r2_train:.4f},  (test): {svm_r2_test:.4f}\n"
                        f"RNN: {rnn_r2}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, days+1), future_prices_knn, label='KNN')
    plt.plot(range(1, days+1), future_prices_svm, label='SVM')
    plt.plot(range(1, days+1), future_prices_rnn, label='RNN')
    plt.xlabel('Días en el futuro')
    plt.ylabel('Precio de las acciones de Amazon')
    plt.title('Predicción de precios futuros')
    plt.legend()
    plt.show()

# Crear ventana
window = tk.Tk()
window.title("Predicción de Precios")
window.geometry("300x100")

label = tk.Label(window, text="Ingrese los días futuros:")
label.pack()
entry = tk.Entry(window)
entry.pack()

button = tk.Button(window, text="Predecir", command=predict_price)
button.pack()

window.mainloop()
