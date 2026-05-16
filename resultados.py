"""
Compara todos os experimentos KNN para academic_performance.

Requisitos cobertos em knn_custom.py / test_k_5.py:
  distância → treino (fit) → inferência (predict) → métricas de erro
"""

import pandas as pd

from test_k_1 import executar_treino as test_k_1
from test_k_2 import executar_treino as test_k_2
from test_k_3 import executar_treino as test_k_3
from test_k_4 import executar_treino as test_k_4
from test_k_1_eucli import executar_treino as test_k_1_eucli
from test_k_2_eucli import executar_treino as test_k_2_eucli
from test_k_3_eucli import executar_treino as test_k_3_eucli
from test_k_4_eucli import executar_treino as test_k_4_eucli
from test_k_5 import executar_treino as test_k_5


def executar_comparacao():
    resultados = []

    for executar in (
        test_k_1,
        test_k_2,
        test_k_3,
        test_k_4,
        test_k_1_eucli,
        test_k_2_eucli,
        test_k_3_eucli,
        test_k_4_eucli,
    ):
        resultados.append(executar())

    # k=5: KNN Custom + KNN Manual, Manhattan e Euclidiana (4 resultados)
    resultados.extend(test_k_5())

    comparacao = (
        pd.DataFrame(resultados)
        .sort_values(by="RMSE")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print("COMPARAÇÃO DOS MODELOS KNN")
    print("Target: academic_performance | Features: 4 variáveis numéricas")
    print("=" * 80)
    print(comparacao.round(6))

    melhor = comparacao.iloc[0]
    print("\n" + "=" * 80)
    print("MELHOR MODELO (menor RMSE)")
    print("=" * 80)
    print(f"Modelo    : {melhor['Modelo']}")
    print(f"Distância : {melhor['Distância']}")
    print(f"K         : {melhor['K']}")
    print(f"MSE       : {melhor['MSE']:.6f}")
    print(f"RMSE      : {melhor['RMSE']:.6f}")
    print(f"MAE       : {melhor['MAE']:.6f}")
    print(f"R2        : {melhor['R2']:.6f}")

    return comparacao, melhor


if __name__ == "__main__":
    executar_comparacao()
