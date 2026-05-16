from knn_runner import executar_knn


def executar_treino():
    return executar_knn(n_neighbors=4, metric="euclidean")


if __name__ == "__main__":
    executar_treino()
