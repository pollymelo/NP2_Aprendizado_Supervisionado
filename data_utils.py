"""Pré-processamento manual (sem scikit-learn).

Normalização MinMax SEM vazamento de dados:
  1. Separar treino e teste primeiro
  2. fit() aprende min/max APENAS com X_train
  3. transform() aplica esses parâmetros em treino e teste
"""

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


class MinMaxScalerManual:
    """MinMax [0, 1] com fit somente no conjunto de treino (evita data leakage)."""

    def __init__(self):
        self.minimo_ = None
        self.maximo_ = None
        self._ajustado = False

    def fit(self, X_train):
        """Aprende min e max exclusivamente a partir dos dados de treino."""
        self.minimo_ = X_train.min()
        self.maximo_ = X_train.max()
        self._ajustado = True
        return self

    def transform(self, X):
        """Aplica min/max aprendidos no fit (treino ou teste)."""
        if not self._ajustado:
            raise RuntimeError("Chame fit() no conjunto de treino antes de transform().")
        denominador = (self.maximo_ - self.minimo_).replace(0, 1)
        return (X - self.minimo_) / denominador

    def fit_transform(self, X_train):
        """fit + transform apenas em X_train."""
        return self.fit(X_train).transform(X_train)


def preparar_dados(caminho="dataset.csv", test_size=0.2, random_state=42):
    """
    Pipeline sem vazamento:
      split → scaler.fit(treino) → transform(treino) e transform(teste)
    """
    data = carregar_dados(caminho)
    X = data[FEATURES]
    y = data[TARGET]

    # 1) Split ANTES de qualquer estatística de normalização
    X_train, X_test, y_train, y_test = separar_treino_teste(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2) Min/max só do treino; teste nunca entra no fit
    scaler = MinMaxScalerManual()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_test_norm, y_train, y_test


# Compatibilidade com chamadas antigas (se existirem)
def minmax_fit_transform(X_train):
    scaler = MinMaxScalerManual()
    return scaler.fit_transform(X_train), scaler.minimo_, scaler.maximo_


def minmax_transform(X_test, minimo, maximo):
    scaler = MinMaxScalerManual()
    scaler.minimo_ = minimo
    scaler.maximo_ = maximo
    scaler._ajustado = True
    return scaler.transform(X_test)
