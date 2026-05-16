"""
k = 5: KNN Custom (knn_custom.py) e KNN Manual (NumPy).

Mesmas features e target dos demais testes; Manhattan e Euclidiana.
"""

import numpy as np
import pandas as pd

from data_utils import TARGET, preparar_dados
from knn_custom import (
    KNNRegressor,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

METRICAS = {
    "Manhattan": "manhattan",
    "Euclidiana": "euclidean",
}

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


MANUAL_REGRESSORS = {
    "Manhattan": KNNRegressorManhattan,
    "Euclidiana": KNNRegressorEuclidiana,
}


def _calcular_metricas(y_true, y_pred):
    y_pred_lista = list(y_pred)
    return {
        "MSE": mean_squared_error(y_true, y_pred_lista),
        "RMSE": root_mean_squared_error(y_true, y_pred_lista),
        "MAE": mean_absolute_error(y_true, y_pred_lista),
        "R2": r2_score(y_true, y_pred_lista),
    }


def _imprimir_resultado(rotulo_modelo, nome_distancia, n_neighbors, metricas, y_test, y_pred):
    print("\n" + "=" * 60)
    print(f"{rotulo_modelo} — distância {nome_distancia}")
    print("=" * 60)
    print(f"Target: {TARGET}")
    print(f"K: {n_neighbors}")
    print(f"MSE  : {metricas['MSE']:.6f}")
    print(f"RMSE : {metricas['RMSE']:.6f}")
    print(f"MAE  : {metricas['MAE']:.6f}")
    print(f"R2   : {metricas['R2']:.6f}")
    print("\nPrimeiras 10 previsões vs valores reais:")
    print(
        pd.DataFrame(
            {"Real": y_test.iloc[:10].values, "Previsto": np.asarray(y_pred)[:10]}
        ).round(4)
    )


def executar_treino():
    n_neighbors = 5
    X_train, X_test, y_train, y_test = preparar_dados()
    y_true = list(y_test.values)
    resultados = []

    # KNN Custom (knn_custom.py) — k = 5

    for nome_distancia, metric in METRICAS.items():
        knn = KNNRegressor(n_neighbors=n_neighbors, metric=metric)
        knn.fit(X_train.values, y_train.values)
        y_pred = knn.predict(X_test.values)

        metricas = _calcular_metricas(y_true, y_pred)
        _imprimir_resultado("KNN Custom", nome_distancia, n_neighbors, metricas, y_test, y_pred)

        resultados.append(
            {
                "Modelo": "KNN Custom",
                "Distância": nome_distancia,
                "K": n_neighbors,
                "Target": TARGET,
                **metricas,
            }
        )

    # KNN Manual (NumPy) — k = 5
   
    for nome_distancia, RegressorClass in MANUAL_REGRESSORS.items():
        knn = RegressorClass(n_neighbors=n_neighbors)
        knn.fit(X_train.values, y_train.values)
        y_pred = knn.predict(X_test.values)

        metricas = _calcular_metricas(y_true, y_pred)
        _imprimir_resultado("KNN Manual (NumPy)", nome_distancia, n_neighbors, metricas, y_test, y_pred)

        resultados.append(
            {
                "Modelo": "KNN Manual",
                "Distância": nome_distancia,
                "K": n_neighbors,
                "Target": TARGET,
                **metricas,
            }
        )

    return resultados


if __name__ == "__main__":
    executar_treino()
