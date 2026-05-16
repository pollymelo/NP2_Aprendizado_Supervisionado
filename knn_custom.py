"""KNN para regressão com distâncias e métricas implementadas manualmente.

Requisitos do trabalho:
  1. Cálculo de distância (Manhattan / Euclidiana)
  2. Treinamento (fit)
  3. Inferência (predict)
  4. Métricas de erro (MSE, RMSE, MAE, R²)
"""

class KNNRegressor:
    """KNN regressor (Manhattan and Euclidean distance)."""

    def __init__(self, n_neighbors=3, metric='manhattan'):
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Treinamento: armazena exemplos de treino."""
        self.X_train = [list(x) for x in X_train]
        self.y_train = list(y_train)

    @staticmethod
    def _manhattan_distance(a, b):
        s = 0.0
        for i in range(len(a)):
            s += abs(a[i] - b[i])
        return s

    @staticmethod
    def _euclidean_distance(a, b):
        s = 0.0
        for i in range(len(a)):
            s += (a[i] - b[i]) ** 2
        return s ** 0.5

    def predict(self, X_test):
        """Inferência: previsão por média dos k vizinhos mais próximos."""
        preds = []
        distance_func = self._euclidean_distance if self.metric == 'euclidean' else self._manhattan_distance
        
        for xt in X_test:
            dists = []
            for i, xr in enumerate(self.X_train):
                d = distance_func(xt, xr)
                dists.append((d, self.y_train[i]))
            dists.sort(key=lambda x: x[0])
            k_neigh = dists[: self.n_neighbors]
            pred = sum(v for _, v in k_neigh) / self.n_neighbors
            preds.append(pred)
        return preds


def mean_squared_error(y_true, y_pred):
    """Cálculo manual do MSE."""
    s = 0.0
    n = len(y_true)
    for i in range(n):
        s += (y_true[i] - y_pred[i]) ** 2
    return s / n


def root_mean_squared_error(y_true, y_pred):
    """Cálculo manual do RMSE."""
    return mean_squared_error(y_true, y_pred) ** 0.5


def mean_absolute_error(y_true, y_pred):
    """Cálculo manual do MAE."""
    s = 0.0
    n = len(y_true)
    for i in range(n):
        s += abs(y_true[i] - y_pred[i])
    return s / n


def r2_score(y_true, y_pred):
    """Cálculo manual do R^2 (coeficiente de determinação)."""
    n = len(y_true)
    mean_y = sum(y_true) / n
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        ss_res += (y_true[i] - y_pred[i]) ** 2
        ss_tot += (y_true[i] - mean_y) ** 2
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)
