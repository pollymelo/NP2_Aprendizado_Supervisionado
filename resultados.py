"""
Compara experimentos KNN e exibe modelo ajustado (k por validação cruzada).
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
from validacao_k import (
    K_DEMO_UNDERFITTING,
    diagnostico_treino_teste,
    executar_diagnostico_underfitting,
    executar_modelo_ajustado,
)


def executar_comparacao():
    resultados = []

    print("=" * 80)
    print("EXPERIMENTOS POR k (k=1 tende a overfitting — ver modelo ajustado ao final)")
    print("=" * 80)

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

    resultados.extend(test_k_5())

    comparacao = (
        pd.DataFrame(resultados)
        .sort_values(by="RMSE")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print("COMPARAÇÃO — métricas no TESTE (target: academic_performance)")
    print("=" * 80)
    print(comparacao.round(6))

    modelo_cv, _, diag_cv = executar_modelo_ajustado()
    tabela_bv, tabela_bv_completa = executar_diagnostico_underfitting()

    print("\n" + "=" * 80)
    print("RECOMENDAÇÃO FINAL (equilíbrio bias-variance)")
    print("=" * 80)
    print(f"Evitar k=1 (overfitting) e k>={K_DEMO_UNDERFITTING} (underfitting)")
    print(f"Usar k = {modelo_cv['K']} com distância {modelo_cv['Distância']} (escolhido por CV)")
    print(f"RMSE teste : {modelo_cv['RMSE']:.6f}")
    print(f"R2 teste   : {modelo_cv['R2']:.6f}")
    print(f"Situação   : {diag_cv['status']}")

    return comparacao, modelo_cv, tabela_bv_completa


if __name__ == "__main__":
    executar_comparacao()
