# Agrupamiento y recomendador de StyleBot. Responsable: Luis Angel Mendosa Gaytán.
# Usa los datos comunes de datos_stylebot.py. Fuentes al final del notebook 05.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

SEMILLA = 42


def a_vectores(X):
    """(n, 28, 28) -> (n, 784). No vuelve a normalizar: los píxeles ya están entre 0 y 1."""
    return X.reshape(len(X), -1)


def ajustar_pca(X_train, n_componentes=50, semilla=SEMILLA):
    """Ajusta PCA solo con entrenamiento. Validación y prueba únicamente se transforman."""
    pca = PCA(n_components=n_componentes, random_state=semilla)
    pca.fit(a_vectores(X_train))
    return pca


def ajustar_kmeans(Z_train, k=10, semilla=SEMILLA, n_init=10):
    """K-Means sobre las características de entrenamiento."""
    modelo = KMeans(n_clusters=k, n_init=n_init, random_state=semilla)
    modelo.fit(Z_train)
    return modelo


def tabla_contingencia(grupos, y, k=10, n_categorias=10):
    """Filas = grupos de K-Means, columnas = categorías reales."""
    tabla = np.zeros((k, n_categorias), dtype=np.int64)
    np.add.at(tabla, (grupos, y), 1)
    return tabla


def metricas_agrupamiento(grupos, y, k=10):
    """Pureza, ARI y NMI. Las etiquetas solo se usan para evaluar, no para agrupar."""
    tabla = tabla_contingencia(grupos, y, k)
    return {
        "pureza": float(tabla.max(axis=1).sum() / tabla.sum()),
        "ari": float(adjusted_rand_score(y, grupos)),
        "nmi": float(normalized_mutual_info_score(y, grupos)),
    }


def construir_recomendador(Z_catalogo, metrica="cosine"):
    """Índice de vecinos más cercanos sobre el catálogo (entrenamiento)."""
    return NearestNeighbors(metric=metrica, algorithm="brute").fit(Z_catalogo)


def recomendar(indice, Z_consultas, k=5):
    """Devuelve, para cada consulta, las posiciones de las k prendas más parecidas del catálogo."""
    _, posiciones = indice.kneighbors(Z_consultas, n_neighbors=k)
    return posiciones


def recomendar_mismo_grupo(Z_catalogo, grupos_catalogo, Z_consultas, grupos_consultas,
                           k=5, metrica="cosine"):
    """Variante: busca vecinos solo entre las prendas del mismo grupo de K-Means."""
    posiciones = np.empty((len(Z_consultas), k), dtype=np.int64)
    for g in np.unique(grupos_consultas):
        en_grupo = np.flatnonzero(grupos_catalogo == g)
        consultas = np.flatnonzero(grupos_consultas == g)
        indice = construir_recomendador(Z_catalogo[en_grupo], metrica)
        _, locales = indice.kneighbors(Z_consultas[consultas], n_neighbors=k)
        posiciones[consultas] = en_grupo[locales]
    return posiciones


def recomendaciones_aleatorias(n_consultas, n_catalogo, k=5, semilla=SEMILLA):
    """Línea base: k prendas del catálogo elegidas al azar, sin repetir dentro de cada consulta."""
    rng = np.random.default_rng(semilla)
    return np.stack([rng.choice(n_catalogo, size=k, replace=False) for _ in range(n_consultas)])


def aciertos(y_consultas, y_catalogo, posiciones):
    """Matriz booleana: True si la recomendación es de la misma categoría que la consulta."""
    return y_catalogo[posiciones] == y_consultas[:, None]


def precision_en_k(y_consultas, y_catalogo, posiciones):
    """Precision@k promedio: fracción de recomendaciones de la misma categoría."""
    return float(aciertos(y_consultas, y_catalogo, posiciones).mean())
