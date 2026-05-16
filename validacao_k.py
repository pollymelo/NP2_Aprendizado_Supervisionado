"""
Seleção de hiperparâmetros por validação cruzada (apenas treino).

Evita overfitting: k é escolhido pelo RMSE médio na validação,
não pelo desempenho no próprio treino (onde k=1 “decora” os dados).
"""

import numpy as np
import pandas as pd

from data_utils import (
    FEATURES,
    TARGET,
    MinMaxScalerManual,
    carregar_dados,
    preparar_dados,
    separar_treino_teste,
)
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

# k >= 3 reduz overfitting severo observado em k=1 e k=2
K_MINIMO_ANTI_OVERFITTING = 3
VALORES_K_PADRAO = [3, 4, 5, 6, 7]

# k muito alto → média de muitos vizinhos ≈ valor global (underfitting)
VALORES_K_UNDERFITTING = [50, 100, 200, 500, 800]
K_DEMO_UNDERFITTING = 800


def classificar_ajuste(rmse_tr, rmse_te, r2_tr, r2_te, gap):
    """
    Classifica overfitting / underfitting / equilíbrio (bias-variance).

    Underfitting: R² baixo em treino E teste, gap pequeno (modelo rígido demais).
    Overfitting: R² alto no treino, pior no teste, gap grande.
    """
    if r2_te < 0.65 and r2_tr < 0.68 and abs(gap) < 0.05:
        return "underfitting"
    if r2_te < 0.72 and r2_tr < 0.76 and gap < 0.025 and rmse_te > 0.275:
        return "underfitting leve"
    if gap > 0.15:
        return "overfitting forte"
    if gap > 0.06:
        return "overfitting leve"
    return "equilibrado"


def obter_treino_bruto(caminho="dataset.csv", test_size=0.2, random_state=42):
    data = carregar_dados(caminho)
    X = data[FEATURES]
    y = data[TARGET]
    return separar_treino_teste(X, y, test_size=test_size, random_state=random_state)


def validacao_cruzada_rmse(
    X_train,
    y_train,
    k,
    metric="manhattan",
    n_folds=5,
    random_state=42,
):
    """RMSE médio em k folds; scaler e KNN ajustados só no fold de treino."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X_train))
    folds = np.array_split(indices, n_folds)
    rmses = []

    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        scaler = MinMaxScalerManual()
        X_tr_n = scaler.fit_transform(X_tr)
        X_val_n = scaler.transform(X_val)

        knn = KNNRegressor(n_neighbors=k, metric=metric)
        knn.fit(X_tr_n.values, y_tr.values)
        pred = knn.predict(X_val_n.values)
        rmses.append(root_mean_squared_error(list(y_val.values), pred))

    return float(np.mean(rmses))


def selecionar_k_e_distancia(
    valores_k=None,
    metricas=None,
    n_folds=5,
    random_state=42,
):
    """
    Escolhe (k, distância) com menor RMSE na validação cruzada do treino.
    Por padrão só considera k >= 3 para evitar overfitting extremo.
    """
    if valores_k is None:
        valores_k = VALORES_K_PADRAO
    if metricas is None:
        metricas = ["manhattan", "euclidean"]

    X_train, _, y_train, _ = obter_treino_bruto(random_state=random_state)
    linhas = []

    for metric in metricas:
        for k in valores_k:
            rmse_cv = validacao_cruzada_rmse(
                X_train, y_train, k, metric, n_folds=n_folds, random_state=random_state
            )
            linhas.append(
                {
                    "K": k,
                    "metric": metric,
                    "Distância": ROTULO_DISTANCIA[metric],
                    "RMSE_CV": rmse_cv,
                }
            )

    tabela_cv = pd.DataFrame(linhas).sort_values("RMSE_CV").reset_index(drop=True)
    melhor = tabela_cv.iloc[0]
    return {
        "K": int(melhor["K"]),
        "metric": melhor["metric"],
        "Distância": melhor["Distância"],
        "RMSE_CV": float(melhor["RMSE_CV"]),
    }, tabela_cv


def _k_efetivo(k, n_treino):
    return max(1, min(int(k), n_treino - 1))


def _metricas_treino_teste(X_train, X_test, y_train, y_test, k, metric):
    k = _k_efetivo(k, len(X_train))
    y_tr = list(y_train.values)
    y_te = list(y_test.values)

    knn = KNNRegressor(n_neighbors=k, metric=metric)
    knn.fit(X_train.values, y_train.values)
    pred_tr = knn.predict(X_train.values)
    pred_te = knn.predict(X_test.values)

    rmse_tr = root_mean_squared_error(y_tr, pred_tr)
    rmse_te = root_mean_squared_error(y_te, pred_te)
    r2_tr = r2_score(y_tr, pred_tr)
    r2_te = r2_score(y_te, pred_te)
    gap = rmse_te - rmse_tr
    status = classificar_ajuste(rmse_tr, rmse_te, r2_tr, r2_te, gap)

    return {
        "K": k,
        "Distância": ROTULO_DISTANCIA[metric],
        "RMSE_treino": rmse_tr,
        "RMSE_teste": rmse_te,
        "R2_treino": r2_tr,
        "R2_teste": r2_te,
        "gap_RMSE": gap,
        "status": status,
    }


def diagnostico_treino_teste(k, metric):
    """Compara RMSE/R² treino vs teste (overfitting / underfitting / equilíbrio)."""
    X_train, X_test, y_train, y_test = preparar_dados()
    return _metricas_treino_teste(X_train, X_test, y_train, y_test, k, metric)


def diagnostico_bias_variance(
    metric="manhattan",
    valores_k=None,
    random_state=42,
):
    """Tabela completa treino vs teste para vários k (inclui faixa de underfitting)."""
    if valores_k is None:
        valores_k = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30] + VALORES_K_UNDERFITTING

    X_train, X_test, y_train, y_test = preparar_dados(random_state=random_state)
    linhas = [
        _metricas_treino_teste(X_train, X_test, y_train, y_test, k, metric)
        for k in valores_k
    ]
    return pd.DataFrame(linhas)


def executar_diagnostico_underfitting(metric="manhattan"):
    """Exibe comparação k baixo (overfitting) vs ótimo vs k alto (underfitting)."""
    diag_k1 = diagnostico_treino_teste(1, metric)
    diag_otimo = diagnostico_treino_teste(4, metric)
    diag_under = diagnostico_treino_teste(K_DEMO_UNDERFITTING, metric)

    tabela = pd.DataFrame([diag_k1, diag_otimo, diag_under])
    tabela_completa = diagnostico_bias_variance(metric=metric)

    print("\n" + "=" * 80)
    print("DIAGNÓSTICO BIAS-VARIANCE (overfitting × equilíbrio × underfitting)")
    print("=" * 80)
    print(tabela.round(6).to_string(index=False))

    print("\n" + "=" * 80)
    print("CURVA COMPLETA — classificação por k (Manhattan)")
    print("=" * 80)
    print(tabela_completa.round(6).to_string(index=False))

    return tabela, tabela_completa


def executar_modelo_ajustado(valores_k=None, n_folds=5, random_state=42):
    """
    Modelo final: k e distância escolhidos por CV no treino (k >= 3).
    Retorna métricas no teste e diagnóstico treino vs teste.
    """
    escolha, tabela_cv = selecionar_k_e_distancia(
        valores_k=valores_k, n_folds=n_folds, random_state=random_state
    )
    k = escolha["K"]
    metric = escolha["metric"]

    X_train, X_test, y_train, y_test = preparar_dados(random_state=random_state)
    y_te = list(y_test.values)

    knn = KNNRegressor(n_neighbors=k, metric=metric)
    knn.fit(X_train.values, y_train.values)
    y_pred = knn.predict(X_test.values)

    mse = mean_squared_error(y_te, y_pred)
    rmse = root_mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)

    diag = diagnostico_treino_teste(k, metric)

    print("\n" + "=" * 80)
    print("SELEÇÃO DE k (validação cruzada 5-fold — só treino, k >= 3)")
    print("=" * 80)
    print(tabela_cv.round(6).to_string(index=False))

    print("\n" + "=" * 80)
    print("MODELO AJUSTADO (anti-overfitting)")
    print("=" * 80)
    print(f"k escolhido por CV : {k}")
    print(f"Distância          : {escolha['Distância']}")
    print(f"RMSE médio na CV   : {escolha['RMSE_CV']:.6f}")
    print(f"\nMétricas no TESTE:")
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  R2   : {r2:.6f}")
    print(f"\nDiagnóstico treino vs teste:")
    print(f"  RMSE treino : {diag['RMSE_treino']:.6f}")
    print(f"  RMSE teste  : {diag['RMSE_teste']:.6f}")
    print(f"  R2 treino   : {diag['R2_treino']:.6f}")
    print(f"  R2 teste    : {diag['R2_teste']:.6f}")
    print(f"  Gap RMSE    : {diag['gap_RMSE']:+.6f}")
    print(f"  Situação    : {diag['status']}")

    diag_under = diagnostico_treino_teste(K_DEMO_UNDERFITTING, metric)
    print(f"\nContraste underfitting (k={diag_under['K']}):")
    print(f"  R2 teste    : {diag_under['R2_teste']:.6f}")
    print(f"  Situação    : {diag_under['status']}")

    resultado = {
        "Modelo": "KNN Custom (CV)",
        "Distância": escolha["Distância"],
        "K": k,
        "Target": TARGET,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "RMSE_CV": escolha["RMSE_CV"],
        "status_overfitting": diag["status"],
    }
    return resultado, tabela_cv, diag
