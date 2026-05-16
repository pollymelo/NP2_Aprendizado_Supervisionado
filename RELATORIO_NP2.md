# NP2 — K-Nearest Neighbors (KNN) para Regressão

**Target:** `academic_performance`  
**Features:** horas de redes sociais, sono, tela antes de dormir, atividade física  

---

## 1. Contextualização

O **K-Nearest Neighbors (KNN)** é um algoritmo de aprendizado supervisionado **não paramétrico**: não estima coeficientes globais, mas usa os **k exemplos de treino mais próximos** de um novo ponto para fazer a previsão. Em **regressão**, a saída costuma ser a **média** dos valores alvo dos vizinhos.

A proximidade é medida por uma **função de distância**. Neste trabalho foram implementadas:

- **Manhattan** (L1): soma das diferenças absolutas entre atributos;
- **Euclidiana** (L2): raiz da soma dos quadrados das diferenças.

O hiperparâmetro principal é **k** (número de vizinhos). Valores pequenos tornam o modelo mais flexível (risco de **overfitting**); valores grandes o tornam mais suave (risco de **underfitting**).

---

## 2. Desenvolvimento algorítmico

A implementação está em `knn_custom.py` e segue o fluxo exigido:

| Etapa | Implementação |
|-------|----------------|
| Distância | `_manhattan_distance`, `_euclidean_distance` (laços `for`) |
| Treinamento | `fit()` — armazena `X_train` e `y_train` |
| Inferência | `predict()` — calcula distâncias, seleciona k menores, retorna média |
| Métricas | `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score` |

O arquivo `test_k_5.py` repete o KNN com **NumPy** (validação independente). O pré-processamento (`data_utils.py`) faz **split 80/20** e **normalização MinMax** com `fit` apenas no treino, evitando vazamento de dados.

**Trecho central da inferência:**

```python
dists.sort(key=lambda x: x[0])
k_neigh = dists[: self.n_neighbors]
pred = sum(v for _, v in k_neigh) / self.n_neighbors
```

---

## 3. Ajuste de hiperparâmetros

Foram testados **k = 1, 2, 3, 4 e 5** com Manhattan e Euclidiana (`test_k_1` … `test_k_5`, `*_eucli.py`).

Para reduzir overfitting, `validacao_k.py` aplica **validação cruzada 5-fold** somente no treino, considerando **k ∈ {3, 4, 5, 6, 7}**, e escolhe o par (k, distância) com menor RMSE médio na CV.

Execute:

```bash
python resultados.py
python graficos.py
```

---

## 4. Análise de resultados

### 4.1 Tabelas

A tabela completa é gerada por `resultados.py` (métricas no **conjunto de teste**). Arquivos auxiliares em `graficos/`:

- `tabela_rmse_manhattan.csv`
- `tabela_rmse_euclidiana.csv`
- `tabela_rmse_cv.csv`

### 4.2 Gráficos (Real vs Predito)

Gerados por `graficos.py` na pasta `graficos/`:

| Arquivo | Descrição |
|---------|-----------|
| `scatter_k1_manhattan.png` | k=1 — overfitting (pontos longe da reta ideal) |
| `scatter_k4_manhattan.png` | k=4 — bom equilíbrio no teste |
| `scatter_modelo_cv.png` | Modelo com k escolhido por validação cruzada |
| `rmse_por_k_manhattan.png` | RMSE treino vs teste por k |
| `rmse_por_k_euclidean.png` | Idem para Euclidiana |
| `rmse_validacao_cruzada.png` | RMSE na CV para seleção de k |

**Interpretação dos scatter plots:** pontos próximos da reta **y = x** indicam boas previsões; dispersão grande indica erro alto.

### 4.3 Resumo típico (Manhattan, teste)

| k | RMSE teste (aprox.) | R² teste (aprox.) | Observação |
|---|---------------------|-------------------|------------|
| 1 | 0,34 | 0,64 | **Overfitting** (R² treino ≈ 1,0) |
| 4 | 0,26 | 0,79 | **Equilibrado** |
| 7 (CV) | 0,26 | 0,79 | **Equilibrado** (escolhido por CV) |
| 800 | 0,35 | 0,61 | **Underfitting** (R² baixo em treino e teste) |

### 4.4 Underfitting — o que é e como apareceu no projeto

**Underfitting** ocorre quando o modelo é **simples demais** para capturar o padrão dos dados: erra tanto no **treino** quanto no **teste**, com desempenho parecido nos dois (gap pequeno).

No KNN, isso aparece com **k muito alto**: a previsão vira a média de centenas de vizinhos, aproximando-se da **média global** de `academic_performance` e ignorando as features.

**Como detectamos no código** (`validacao_k.py`, função `classificar_ajuste`):

- R² **baixo** no treino **e** no teste;
- **Gap** RMSE (teste − treino) **pequeno** (não é “decoração” do treino, é incapacidade geral).

**Gráficos relacionados:**

| Arquivo | Conteúdo |
|---------|----------|
| `scatter_k800_manhattan.png` | Dispersão com k alto — pontos afastados da reta ideal |
| `r2_por_k_manhattan.png` | Queda de R² treino e teste para k grande |
| `rmse_por_k_manhattan.png` | Faixas coloridas: overfitting (k baixo), equilíbrio, underfitting (k alto) |
| `tabela_bias_variance.csv` | Classificação automática por k |

---

## 5. Dificuldades e limitações

**Técnicas**

- Implementação manual do KNN é **lenta** para muitos pontos (complexidade ~O(n) por previsão).
- **k=1** memoriza o treino (R² treino ≈ 1,0) — foi necessário CV e k mínimo ≥ 3 na seleção automática.
- Garantir **normalização sem vazamento** exigiu separar split e `fit` do scaler antes de transformar o teste.

**Conceituais**

- KNN assume que **proximidade nas features** implica desempenho acadêmico similar — pode falhar se a relação for não linear e pouco local.
- Apenas **4 features numéricas**; variáveis categóricas (gênero, plataforma) não foram usadas nos testes principais.

**Dataset**

- ~2500 registros; teste com 20% (~500 amostras).
- Possível **ruído** e limites nas variáveis; o modelo explica ~79% da variância (R²), não 100%.

---

## 6. Conclusão — bias-variance e melhor hiperparâmetro

### Trade-off viés–variância (overfitting × equilíbrio × underfitting)

| Região de k | Viés | Variância | Sintoma | Diagnóstico |
|-------------|------|-----------|---------|-------------|
| **k = 1** | Baixo no treino | Alto | R² treino ≈ 1; teste pior | **Overfitting** |
| **k = 3–7** | Moderado | Moderada | Bom R² no teste; gap pequeno | **Equilibrado** |
| **k ≥ 800** | Alto | Baixo | R² baixo em treino **e** teste | **Underfitting** |

O gráfico em U do erro: RMSE alto nas extremidades (k muito baixo ou muito alto) e menor na faixa intermediária — comportamento clássico do KNN.

### Melhor configuração

- **Evitar:** k = 1 (overfitting) e k muito grande, ex. **≥ 800** (underfitting neste dataset).
- **Usar:** k escolhido por **validação cruzada** (`validacao_k.py`, faixa 3–7) — tipicamente **k = 7**, distância **Euclidiana** ou **k = 4**, **Manhattan** no teste fixo.

**Por quê Manhattan vs Euclidiana?** Depende de k; ambas ficam próximas na faixa equilibrada. L1 é mais robusta a outliers em algumas features.

**Por quê não k=1?** Memoriza o treino. **Por quê não k=800?** Suaviza demais; o modelo deixa de usar a informação local das features.

---

## Referência rápida — estrutura do código

```
knn_custom.py    → KNN + métricas
data_utils.py    → dados + MinMax sem leakage
knn_runner.py    → pipeline dos testes k=1..4
validacao_k.py   → CV + anti-overfitting
test_k_5.py      → KNN NumPy (k=5)
resultados.py    → comparação + diagnóstico
graficos.py      → figuras para o relatório
main.py          → executa tudo
```
