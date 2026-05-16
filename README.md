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
test_k_5.py      # KNN Manual (NumPy), k = 5, Manhattan e Euclidiana
resultados.py    # Compara todos os experimentos
main.py          # Executa resultados.py
dataset.csv      # Dados
requirements.txt
```

## Features e target

- **Target:** `academic_performance`
- **Features:** `daily_social_media_hours`, `sleep_hours`, `screen_time_before_sleep`, `physical_activity`

Todos os testes usam o **mesmo** conjunto de variáveis e split (80/20, `random_state=42`).

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

Ou apenas a comparação:

```bash
python resultados.py
```

Teste individual:

```bash
python test_k_3.py
python test_k_5.py
```

## Resultados (comparação justa)

Após `python main.py`, os modelos são ordenados por RMSE. Exemplo típico:

| Modelo | Distância | K | RMSE (aprox.) | R² (aprox.) |
|--------|-----------|---|---------------|-------------|
| KNN Custom | Manhattan | 4 | 0.259 | 0.787 |
| KNN Manual | Euclidiana | 5 | 0.261 | 0.784 |
| KNN Manual | Manhattan | 5 | 0.261 | 0.784 |
| KNN Custom | Manhattan | 3 | 0.266 | 0.776 |

### Conclusão

1. **Melhor configuração na comparação:** KNN Custom com **k = 4** e distância **Manhattan**.
2. **k = 3** continua muito competitivo (próximo na tabela).
3. **Manhattan** tende a superar Euclidiana neste dataset, para a maioria dos valores de k.
4. **test_k_5** valida a mesma lógica com implementação NumPy independente de `knn_custom.KNNRegressor`.

## Dependências

- `pandas` — leitura do CSV
- `numpy` — operações em `test_k_5.py`

Sem `scikit-learn`.
