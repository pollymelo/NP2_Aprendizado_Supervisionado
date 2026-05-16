# NP2 — K-Nearest Neighbors (Regressão)

Implementação manual de KNN para prever `academic_performance`, atendendo aos requisitos:

| Requisito | Onde está |
|-----------|-----------|
| Cálculo de distância (Manhattan / Euclidiana) | `knn_custom.py`, classes em `test_k_5.py` |
| Treinamento do modelo | `KNNRegressor.fit()` / `KNNRegressorManhattan.fit()` |
| Inferência do modelo | `KNNRegressor.predict()` / `*.predict()` |
| Métricas de erro (MSE, RMSE, MAE, R²) | `knn_custom.py` (laços `for`, sem sklearn) |

**Não utiliza** `sklearn.neighbors` nem métricas prontas do scikit-learn.

## Estrutura do projeto

```
knn_custom.py    # KNN + distâncias + métricas (núcleo do trabalho)
data_utils.py    # Carregamento, split e normalização MinMax manual
knn_runner.py    # Pipeline: dados → fit → predict → métricas
test_k_1..4      # KNN Custom, Manhattan, k = 1..4
test_k_*_eucli   # KNN Custom, Euclidiana, k = 1..4
test_k_5.py      # k = 5: KNN Custom + KNN Manual (NumPy), Manhattan e Euclidiana
validacao_k.py   # CV no treino para escolher k (anti-overfitting)
resultados.py    # Compara experimentos + modelo ajustado por CV
graficos.py      # Scatter Real vs Predito + RMSE por k
RELATORIO_NP2.md # Texto do relatório (todas as seções do enunciado)
main.py          # Executa resultados + gráficos + CSV
graficos/        # Saída: PNGs e tabelas (gerado ao rodar main.py)
dataset.csv      # Dados
requirements.txt
```

## Features e target

- **Target:** `academic_performance`
- **Features:** `daily_social_media_hours`, `sleep_hours`, `screen_time_before_sleep`, `physical_activity`

Todos os testes usam o **mesmo** conjunto de variáveis e split (80/20, `random_state=42`).

### Valores de k testados

| k | Arquivos | Modelo |
|---|----------|--------|
| 1 | `test_k_1`, `test_k_1_eucli` | KNN Custom |
| 2 | `test_k_2`, `test_k_2_eucli` | KNN Custom |
| 3 | `test_k_3`, `test_k_3_eucli` | KNN Custom |
| 4 | `test_k_4`, `test_k_4_eucli` | KNN Custom |
| 5 | `test_k_5` | KNN Custom **e** KNN Manual (NumPy) |

`python main.py` executa **12 experimentos** (8 com k=1..4 + 4 com k=5).

## Normalização sem vazamento de dados

A normalização MinMax em `data_utils.py` segue o padrão correto:

1. **Separar** treino e teste (`separar_treino_teste`)
2. **`fit`** do `MinMaxScalerManual` **somente** em `X_train` (min/max do treino)
3. **`transform`** em treino e teste com os parâmetros aprendidos no passo 2

O conjunto de teste **não participa** do cálculo de mínimo e máximo. Isso evita *data leakage* e mantém a avaliação honesta.

## Como executar

```bash
pip install -r requirements.txt
python main.py
```

Isso gera:

- Tabela comparativa no terminal e `graficos/tabela_comparacao_completa.csv`
- Gráficos de dispersão e RMSE em `graficos/`

Só o PDF:

```bash
python graficos.py   # se ainda não tiver as figuras
```

Comandos separados:

```bash
python resultados.py   # só tabelas e diagnóstico
python graficos.py       # só figuras
```

Teste individual: `python test_k_3.py`, `python test_k_5.py`

## Resultados (comparação justa)

Após `python main.py`, os modelos são ordenados por RMSE. Exemplo típico:

| Modelo | Distância | K | RMSE (aprox.) | R² (aprox.) |
|--------|-----------|---|---------------|-------------|
| KNN Custom | Manhattan | 4 | 0.259 | 0.787 |
| KNN Manual | Euclidiana | 5 | 0.261 | 0.784 |
| KNN Manual | Manhattan | 5 | 0.261 | 0.784 |
| KNN Custom | Manhattan | 3 | 0.266 | 0.776 |

### Overfitting e correção

| k | Problema | Correção aplicada |
|---|----------|-------------------|
| **k = 1** | Overfitting forte (R² treino ≈ 1,0, gap grande no teste) | Mantido só como experimento; **não** usar como modelo final |
| **k = 2** | Overfitting moderado | Comparado na tabela; CV prefere k maior |
| **k = 3–7** | Equilibrado | Faixa usada por `validacao_k.py` |
| **k ≥ 800** | Underfitting (R² baixo em treino e teste) | Ver `scatter_k800_manhattan.png` e `tabela_bias_variance.csv` |

O módulo **`validacao_k.py`** escolhe **k e distância** com **validação cruzada 5-fold só no treino** (k mínimo = 3), evitando escolher k=1 por desempenho inflado no treino.

```bash
python resultados.py   # inclui tabela CV + modelo ajustado ao final
```

### Conclusão

1. **Modelo recomendado:** o exibido ao final de `resultados.py` (**KNN Custom com k escolhido por CV**).
2. **k = 4** costuma ser o melhor no teste quando fixado manualmente; a CV confirma k na faixa 3–5.
3. **Manhattan** tende a superar Euclidiana neste dataset.
4. **test_k_5** valida KNN Custom vs KNN Manual (NumPy) no mesmo k=5.

## Entrega da NP2 (checklist)

| Seção do enunciado | Arquivo |
|--------------------|---------|
| Contextualização | `RELATORIO_NP2.md` §1 |
| Desenvolvimento algorítmico | `RELATORIO_NP2.md` §2 + `knn_custom.py` |
| Ajuste de hiperparâmetros (k) | `test_k_*.py`, `validacao_k.py` |
| Análise (tabelas + gráficos) | `resultados.py`, `graficos/` |
| Dificuldades e limitações | `RELATORIO_NP2.md` §5 |
| Conclusão (bias-variance) | `RELATORIO_NP2.md` §6 |

## Dependências

- `pandas`, `numpy`, `matplotlib`
- Sem `scikit-learn` para KNN ou métricas
