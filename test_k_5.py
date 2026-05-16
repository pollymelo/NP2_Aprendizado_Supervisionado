"""
Implementação 100% manual do KNN (NumPy) + pré-processamento em data_utils.

Mesmas features e target dos demais testes; k=5; Manhattan e Euclidiana.
"""

import numpy as np
import pandas as pd

from data_utils import TARGET, preparar_dados
from knn_custom import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

# KNN Regressor Manhattan (distância + treino + inferência)

class KNNRegressorManhattan:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        previsoes = []
        for x in X_test:
            distancias = np.sum(np.abs(self.X_train - x), axis=1)
            indices_vizinhos = np.argsort(distancias)[: self.n_neighbors]
            previsoes.append(np.mean(self.y_train[indices_vizinhos]))
        return np.array(previsoes)

# KNN Regressor Euclidiana (distância + treino + inferência)

class KNNRegressorEuclidiana:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        previsoes = []
        for x in X_test:
            distancias = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            indices_vizinhos = np.argsort(distancias)[: self.n_neighbors]
            previsoes.append(np.mean(self.y_train[indices_vizinhos]))
        return np.array(previsoes)


def executar_treino():
    n_neighbors = 5
    X_train, X_test, y_train, y_test = preparar_dados()
    y_true = list(y_test.values)

    resultados = []
    modelos = [
        ("Manhattan", KNNRegressorManhattan),
        ("Euclidiana", KNNRegressorEuclidiana),
    ]

    for nome_distancia, RegressorClass in modelos:
        knn = RegressorClass(n_neighbors=n_neighbors)

        # Treinamento
        knn.fit(X_train.values, y_train.values)

        # Inferência
        y_pred = knn.predict(X_test.values)
        y_pred_lista = list(y_pred)

        # Métricas de erro (funções manuais de knn_custom)
        mse = mean_squared_error(y_true, y_pred_lista)
        rmse = root_mean_squared_error(y_true, y_pred_lista)
        mae = mean_absolute_error(y_true, y_pred_lista)
        r2 = r2_score(y_true, y_pred_lista)

        print("\n" + "=" * 60)
        print(f"KNN Manual (NumPy) — distância {nome_distancia}")
        print("=" * 60)
        print(f"Target: {TARGET}")
        print(f"K: {n_neighbors}")
        print(f"MSE  : {mse:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"MAE  : {mae:.6f}")
        print(f"R2   : {r2:.6f}")

        print("\nPrimeiras 10 previsões vs valores reais:")
        print(
            pd.DataFrame(
                {"Real": y_test.iloc[:10].values, "Previsto": y_pred[:10]}
            ).round(4)
        )

        resultados.append(
            {
                "Modelo": "KNN Manual",
                "Distância": nome_distancia,
                "K": n_neighbors,
                "Target": TARGET,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            }
        )

    return resultados


if __name__ == "__main__":
    executar_treino()
