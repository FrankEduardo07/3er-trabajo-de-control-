import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(0)
X = np.linspace(-10, 10, 200).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)

plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='black', label='Datos reales')
plt.scatter(X_test, y_pred_mlp, color='blue', label='Predicción MLP')
plt.scatter(X_test, y_pred_knn, color='red', label='Predicción k-NN')
plt.title('Comparación de Predicciones')
plt.legend()
plt.show()
