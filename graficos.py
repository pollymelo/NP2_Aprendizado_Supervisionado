"""
Gera gráficos para o relatório NP2:
  - Dispersão Real vs Predito
  - RMSE por k (treino e teste)
  - RMSE na validação cruzada por k
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import TARGET, preparar_dados
from knn_custom import KNNRegressor, root_mean_squared_error, r2_score
from validacao_k import (
    K_DEMO_UNDERFITTING,
    ROTULO_DISTANCIA,
    VALORES_K_PADRAO,
    diagnostico_bias_variance,
    obter_treino_bruto,
    selecionar_k_e_distancia,
    validacao_cruzada_rmse,
)

PASTA_SAIDA = Path("graficos")


def _treinar_prever(k, metric):
    X_train, X_test, y_train, y_test = preparar_dados()
    knn = KNNRegressor(n_neighbors=k, metric=metric)
    knn.fit(X_train.values, y_train.values)
    y_true = list(y_test.values)
    y_pred = knn.predict(X_test.values)
    y_pred_train = knn.predict(X_train.values)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_train": list(y_train.values),
        "y_pred_train": y_pred_train,
        "rmse_teste": root_mean_squared_error(y_true, y_pred),
        "r2_teste": r2_score(y_true, y_pred),
        "rmse_treino": root_mean_squared_error(
            list(y_train.values), y_pred_train
        ),
    }


def scatter_real_vs_predito(y_true, y_pred, titulo, nome_arquivo):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.45, edgecolors="k", linewidths=0.3)
    lim_min = min(min(y_true), min(y_pred))
    lim_max = max(max(y_true), max(y_pred))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="Ideal (y=x)")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predito")
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    caminho = PASTA_SAIDA / nome_arquivo
    plt.savefig(caminho, dpi=150)
    plt.close()
    return caminho


def grafico_rmse_por_k(metric="manhattan", ks=None):
    if ks is None:
        ks = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 800]
    rmses_treino = []
    rmses_teste = []
    for k in ks:
        res = _treinar_prever(k, metric)
        rmses_treino.append(res["rmse_treino"])
        rmses_teste.append(res["rmse_teste"])

    nome = ROTULO_DISTANCIA[metric]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, rmses_treino, "o-", label="RMSE treino")
    plt.plot(ks, rmses_teste, "s-", label="RMSE teste")
    plt.xlabel("k (número de vizinhos)")
    plt.ylabel("RMSE")
    plt.title(f"RMSE por k — distância {nome}")
    plt.axvspan(1, 2.5, alpha=0.08, color="red", label="Overfitting (k baixo)")
    plt.axvspan(3, 30, alpha=0.08, color="green", label="Faixa equilibrada")
    plt.axvspan(200, max(ks) + 50, alpha=0.08, color="blue", label="Underfitting (k alto)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    caminho = PASTA_SAIDA / f"rmse_por_k_{metric}.png"
    plt.savefig(caminho, dpi=150)
    plt.close()
    return caminho, pd.DataFrame(
        {"K": ks, "RMSE_treino": rmses_treino, "RMSE_teste": rmses_teste}
    )


def grafico_rmse_cv():
    X_train, _, y_train, _ = obter_treino_bruto()
    linhas = []
    for metric in ("manhattan", "euclidean"):
        for k in VALORES_K_PADRAO:
            rmse_cv = validacao_cruzada_rmse(X_train, y_train, k, metric)
            linhas.append(
                {
                    "K": k,
                    "metric": metric,
                    "Distância": ROTULO_DISTANCIA[metric],
                    "RMSE_CV": rmse_cv,
                }
            )
    df = pd.DataFrame(linhas)

    plt.figure(figsize=(8, 5))
    for metric in ("manhattan", "euclidean"):
        sub = df[df["metric"] == metric]
        plt.plot(
            sub["K"],
            sub["RMSE_CV"],
            "o-",
            label=ROTULO_DISTANCIA[metric],
        )
    plt.xlabel("k (número de vizinhos)")
    plt.ylabel("RMSE médio (validação cruzada 5-fold)")
    plt.title("Seleção de k — apenas conjunto de treino")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    caminho = PASTA_SAIDA / "rmse_validacao_cruzada.png"
    plt.savefig(caminho, dpi=150)
    plt.close()
    return caminho, df


def grafico_r2_por_k(metric="manhattan"):
    df = diagnostico_bias_variance(metric=metric)
    plt.figure(figsize=(9, 5))
    plt.plot(df["K"], df["R2_treino"], "o-", label="R² treino")
    plt.plot(df["K"], df["R2_teste"], "s-", label="R² teste")
    plt.xlabel("k (número de vizinhos)")
    plt.ylabel("R²")
    plt.title(f"R² por k — {ROTULO_DISTANCIA[metric]} (underfitting quando ambos caem)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    caminho = PASTA_SAIDA / f"r2_por_k_{metric}.png"
    plt.savefig(caminho, dpi=150)
    plt.close()
    return caminho, df


def gerar_todos_graficos():
    PASTA_SAIDA.mkdir(exist_ok=True)
    arquivos = []

    # Dispersão: k=1 (overfitting), k=4 (bom no teste), modelo CV
    k_under = K_DEMO_UNDERFITTING
    casos = [
        (1, "manhattan", "k=1 Manhattan (overfitting)", "scatter_k1_manhattan.png"),
        (4, "manhattan", "k=4 Manhattan (equilibrado)", "scatter_k4_manhattan.png"),
        (
            k_under,
            "manhattan",
            f"k={k_under} Manhattan (underfitting)",
            "scatter_k800_manhattan.png",
        ),
    ]
    escolha, _ = selecionar_k_e_distancia()
    k_cv = int(escolha["K"])
    metric_cv = escolha["metric"]
    casos.append(
        (
            k_cv,
            metric_cv,
            f"k={k_cv} {escolha['Distância']} (escolhido por CV)",
            "scatter_modelo_cv.png",
        )
    )

    for k, metric, titulo, arquivo in casos:
        res = _treinar_prever(k, metric)
        titulo_completo = (
            f"{titulo}\nRMSE={res['rmse_teste']:.4f} | R²={res['r2_teste']:.4f}"
        )
        arquivos.append(
            scatter_real_vs_predito(
                res["y_true"], res["y_pred"], titulo_completo, arquivo
            )
        )

    _, df_man = grafico_rmse_por_k("manhattan")
    arquivos.append(PASTA_SAIDA / "rmse_por_k_manhattan.png")
    _, df_euc = grafico_rmse_por_k("euclidean")
    arquivos.append(PASTA_SAIDA / "rmse_por_k_euclidean.png")
    _, df_cv = grafico_rmse_cv()
    arquivos.append(PASTA_SAIDA / "rmse_validacao_cruzada.png")
    _, df_r2 = grafico_r2_por_k("manhattan")
    arquivos.append(PASTA_SAIDA / "r2_por_k_manhattan.png")

    df_man.to_csv(PASTA_SAIDA / "tabela_rmse_manhattan.csv", index=False)
    df_euc.to_csv(PASTA_SAIDA / "tabela_rmse_euclidiana.csv", index=False)
    df_cv.to_csv(PASTA_SAIDA / "tabela_rmse_cv.csv", index=False)
    df_r2.to_csv(PASTA_SAIDA / "tabela_bias_variance.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Gráficos salvos em: {PASTA_SAIDA.resolve()}")
    print("=" * 60)
    for arq in arquivos:
        print(f"  - {arq.name}")

    return arquivos


if __name__ == "__main__":
    gerar_todos_graficos()
