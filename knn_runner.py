"""Pipeline KNN: distância → treino → inferência → métricas."""

import pandas as pd

from data_utils import TARGET, preparar_dados
from knn_custom import (
    KNNRegressor,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

ROTULO_DISTANCIA = {
    "manhattan": "Manhattan",
    "euclidean": "Euclidiana",
}


def executar_knn(n_neighbors, metric, modelo="KNN Custom"):
    X_train, X_test, y_train, y_test = preparar_dados()
    y_true = list(y_test.values)

    # Cálculo de distância + KNN (treino e inferência em knn_custom)

    knn = KNNRegressor(n_neighbors=n_neighbors, metric=metric)

    # Treinamento do modelo
    knn.fit(X_train.values, y_train.values)

    # Inferência do modelo
    y_pred = knn.predict(X_test.values)

    # Métricas de erro (implementação manual)
  
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    nome_distancia = ROTULO_DISTANCIA[metric]

    print("\n" + "=" * 60)
    print(f"KNN ({modelo}) — distância {nome_distancia}")
    print("=" * 60)
    print(f"Target: {TARGET}")
    print(f"K: {n_neighbors}")
    print("\nResultados:")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R2   : {r2:.6f}")

    print("\nPrimeiras 10 previsões vs valores reais:")
    print(
        pd.DataFrame({"Real": y_test.iloc[:10].values, "Previsto": y_pred[:10]}).round(4)
    )

    return {
        "Modelo": modelo,
        "Distância": nome_distancia,
        "K": n_neighbors,
        "Target": TARGET,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }
