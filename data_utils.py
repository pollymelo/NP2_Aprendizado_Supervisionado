"""Pré-processamento manual (sem scikit-learn)."""

import numpy as np
import pandas as pd

FEATURES = [
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "physical_activity",
]

TARGET = "academic_performance"


def carregar_dados(caminho="dataset.csv"):
    return pd.read_csv(caminho)


def separar_treino_teste(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    tamanho_teste = int(len(X) * test_size)
    teste_idx = indices[:tamanho_teste]
    treino_idx = indices[tamanho_teste:]
    return (
        X.iloc[treino_idx],
        X.iloc[teste_idx],
        y.iloc[treino_idx],
        y.iloc[teste_idx],
    )


def minmax_fit_transform(X_train):
    minimo = X_train.min()
    maximo = X_train.max()
    denominador = (maximo - minimo).replace(0, 1)
    return (X_train - minimo) / denominador, minimo, maximo


def minmax_transform(X_test, minimo, maximo):
    denominador = (maximo - minimo).replace(0, 1)
    return (X_test - minimo) / denominador


def preparar_dados(caminho="dataset.csv", test_size=0.2, random_state=42):
    data = carregar_dados(caminho)
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = separar_treino_teste(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_norm, minimo, maximo = minmax_fit_transform(X_train)
    X_test_norm = minmax_transform(X_test, minimo, maximo)
    return X_train_norm, X_test_norm, y_train, y_test
