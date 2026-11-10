# Aportación de agrupamiento y recomendador

**Responsable:** Luis Angel Mendosa Gaytán.
**Notebook:** `notebooks/04_no_supervisado.ipynb`. **Funciones:** `src/recomendador_stylebot.py`.

## Objetivo

Explorar si las prendas de Fashion-MNIST forman grupos naturales sin usar sus etiquetas y construir un recomendador que, a partir de una imagen, devuelva las cinco prendas más parecidas del catálogo.

## Uso de las categorías reales

Las categorías reales se usan para analizar y evaluar los resultados, no para escoger artificialmente los grupos ni las recomendaciones:

- K-Means se ajusta solo con las 50 componentes de PCA de entrenamiento, sin etiquetas.
- El recomendador busca vecinos solo con las características de las imágenes. La categoría de la consulta y la del catálogo nunca intervienen en la búsqueda.
- Las categorías se usan después para nombrar cada grupo por su categoría mayoritaria, calcular pureza, ARI, NMI y Precision@5, y elegir ejemplos de aciertos y fallos.
- La comparación de variantes del recomendador usa Precision@5 en validación para elegir el método completo. No cambia qué prendas recomienda cada variante.
- El número de grupos (k = 10) lo fija el plan del equipo. Se revisaron otros valores de k con inercia y silueta, que no usan etiquetas.

## Datos y características

Se usó la división común del equipo: 48,000 imágenes de entrenamiento, 12,000 de validación y 10,000 de prueba, con semilla 42. El notebook comprueba que las huellas SHA-256 de los índices coinciden con `datos/protocolo_datos.json`.

Cada imagen se aplanó en 784 píxeles y se redujo a 50 componentes con PCA, reutilizando las funciones de `02_caracteristicas.ipynb`. PCA se ajustó solo con entrenamiento y conserva 86.3 % de la varianza. Las etiquetas no se usan para agrupar ni para recomendar, solo para evaluar.

## Entrega del domingo: agrupamiento, observaciones y descripción del recomendador

### Agrupamiento con K-Means

Primero se probó un ejemplo pequeño con 2,000 imágenes y 10 grupos para confirmar que el procedimiento funcionaba (`kmeans_ejemplo_pequeno.png`). Después se aplicó a las 48,000 imágenes de entrenamiento.

Se usaron 10 grupos, uno por categoría. Antes se revisaron la inercia y la silueta para k entre 4 y 14. La inercia baja de forma gradual, sin un codo claro, y la silueta es baja en todos los casos (0.18 a 0.22). Esto indica que los grupos se traslapan.

| Conjunto | Pureza | ARI | NMI |
|---|---:|---:|---:|
| Entrenamiento | 0.552 | 0.347 | 0.511 |
| Validación | 0.558 | 0.347 | 0.512 |

Observaciones:

- Los bolsos y los botines forman los grupos más limpios: los dos grupos de bolso tienen alrededor de 94 % de bolsos y uno de los grupos de botín tiene 90 %.
- El calzado se separa del resto de la ropa, pero tenis, sandalias y botines se reparten entre varios grupos.
- Pantalón y vestido comparten un grupo (60 % pantalón y 36 % vestido) porque ambos son siluetas verticales y alargadas.
- Suéter, abrigo y camisa se mezclan: un grupo tiene 37 % abrigo y 37 % suéter, y otro no supera 28 % en ninguna categoría.
- Un grupo reúne prendas de distintas categorías con tonos oscuros. K-Means sobre píxeles agrupa también por brillo, no solo por forma.
- Los resultados de validación son casi iguales a los de entrenamiento, así que los grupos generalizan a imágenes nuevas.

### Visualización t-SNE

En el mapa t-SNE de 3,000 imágenes, pantalones y bolsos forman islas separadas. El calzado ocupa otra región. Camiseta, camisa, suéter, abrigo y vestido quedan juntos en una zona amplia. Esto coincide con lo que muestra K-Means.

### Descripción del recomendador

El recomendador funciona por vecinos más cercanos:

1. El catálogo son las 48,000 imágenes de entrenamiento, representadas con sus 50 componentes de PCA.
2. La imagen de consulta se aplana a 784 píxeles y se transforma con el mismo PCA, sin volver a ajustarlo.
3. Se calcula la distancia coseno entre la consulta y cada prenda del catálogo. La distancia coseno compara la dirección de los vectores en lugar de su magnitud; en validación dio un resultado ligeramente mejor que la distancia euclidiana.
4. Se devuelven las cinco prendas más cercanas.

Las consultas vienen de validación o prueba, por lo que nunca forman parte del catálogo. También se probó una variante que busca solo dentro del grupo de K-Means de la consulta, para aprovechar el agrupamiento y reducir el número de comparaciones. La figura `recomendador_ejemplos_validacion.png` muestra una consulta por categoría con sus cinco recomendaciones.

## Entrega del lunes: evaluación del recomendador

Se evaluó con **Precision@5**: la fracción de las cinco recomendaciones que pertenece a la misma categoría que la consulta. Recomendar al azar da 0.10 porque las 10 categorías están equilibradas.

Comparación de variantes en validación (12,000 consultas):

| Variante | Precision@5 |
|---|---:|
| Píxeles 784 + coseno | 0.828 |
| PCA 50 + euclidiana | 0.825 |
| **PCA 50 + coseno** (elegida) | **0.832** |
| PCA 50 + coseno, solo dentro del mismo grupo de K-Means | 0.829 |
| Aleatorio (5 semillas) | 0.100 |

Se eligió PCA 50 con distancia coseno porque obtuvo el valor más alto en validación. Las diferencias entre variantes son pequeñas. PCA logra lo mismo que los 784 píxeles con 50 valores por imagen. Limitar la búsqueda al mismo grupo de K-Means no mejora la precisión, pero la hace más rápida.

**Evaluación final en prueba (10,000 consultas, una sola vez):**

| Variante | Precision@5 |
|---|---:|
| PCA 50 + coseno | **0.821** |
| Aleatorio (5 semillas) | 0.100 |

El recomendador es unas 8 veces mejor que el azar. En 63.9 % de las consultas las cinco recomendaciones son de la categoría correcta. En 9.5 %, solo una o ninguna lo es.

Precision@5 por categoría en prueba: pantalón 0.96, bolso 0.95, botín 0.94, tenis 0.91, sandalia 0.87, vestido 0.83, camiseta 0.76, abrigo 0.71, suéter 0.71 y camisa 0.56.

Confusiones más frecuentes: camisa ↔ camiseta (17 % y 16 % de sus recomendaciones), abrigo ↔ suéter (14 % y 13 %), suéter y abrigo → camisa y sandalia → tenis. En los ejemplos de fallos se ve que la prenda recomendada tiene casi la misma silueta. Por ejemplo, un suéter blanco de manga larga recibe abrigos de forma casi idéntica.

## Revisión con aprendizaje clásico

Se revisó `03_modelos.ipynb` junto con `02_caracteristicas.ipynb`:

- Los tres notebooks usan la celda común de carga sin cambios, la misma huella de la división (`idx_train` = `1c0cffec…`) y el mismo orden de etiquetas.
- En los tres, PCA se ajusta solo con entrenamiento, y la prueba se usa solo para la evaluación final después de elegir con validación.
- **Diferencia:** la clasificación usa PCA con 100 componentes (91.2 % de varianza), mientras que características, agrupamiento y recomendador usan 50 (86.3 %). No invalida resultados, pero el informe debe decirlo. Para unificar, basta cambiar el valor de la sección 7 de `04_no_supervisado.ipynb`.

## Limitaciones

- Las imágenes son de 28 x 28 en escala de grises. La similitud depende de la forma y el brillo, no del color, la textura fina ni el estilo.
- Precision@5 solo considera correcta una recomendación de la misma categoría. Una camisa parecida a una camiseta puede ser útil para un usuario aunque cuente como error.
- La silueta y la pureza muestran que las categorías de ropa superior no forman grupos separados en el espacio de píxeles.
- Los valores provienen de una ejecución con semilla 42. K-Means y t-SNE pueden variar un poco con otra versión de scikit-learn.

## Archivos producidos

- `notebooks/04_no_supervisado.ipynb`, `notebooks/05_aprendizaje_no_supervisado.ipynb` y `notebooks/06_sistema_recomendacion.ipynb`
- `src/recomendador_stylebot.py`
- `resultados/tablas/`: `kmeans_*.csv` y `recomendador_*.csv`
- `resultados/graficas/`: `kmeans_*.png`, `tsne_categorias_y_grupos.png` y `recomendador_*.png`

## Para la integración (`E1_StyleBot.ipynb`)

Las secciones 5 y 6 están separadas en dos notebooks que se ejecutan solos, cada uno con la carga común al inicio:

- `notebooks/05_aprendizaje_no_supervisado.ipynb` → **sección 5** de E1: PCA, ejemplo pequeño de K-Means, revisión de k, K-Means con 10 grupos, ejemplos de cada grupo y t-SNE.
- `notebooks/06_sistema_recomendacion.ipynb` → **sección 6** de E1: recomendador, Precision@5, comparación contra el azar y ejemplos de aciertos y fallos. Repite al inicio las funciones, el PCA y K-Means de la sección 5 para funcionar solo; dentro de E1 basta copiar desde "## 6 Sistema de recomendación".

`notebooks/04_no_supervisado.ipynb` reúne ambas partes con la explicación completa.

Ajustes hechos para que no choquen con otras secciones:

- **PCA propio de 50 componentes.** En la sección 4, `03_modelos` reescribe `X_train_pca` con 100 componentes; si mi parte reutilizara ese nombre, el agrupamiento cambiaría. Mis celdas calculan su propio PCA (`Z_train`, `Z_val`, `Z_test`).
- **Nombres separados.** `CARPETA_NO_SUPERVISADO`, `guardar_tabla_ns` y `guardar_figura_ns` evitan sobrescribir `CARPETA_SALIDA` y `guardar_csv` de la carga común.
- Se probó ejecutando E1 completo con una sección 4 simulada que deja `X_train_pca` de 100 componentes. Los resultados fueron idénticos a los de `04_no_supervisado.ipynb` (Precision@5 en prueba = 0.821).
- Solo necesita la carga común de la sección 1 (`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`, `SEMILLA`, `CATEGORIAS`). No agrega bibliotecas fuera de `requirements.txt`.

## Fuentes

- Scikit-learn. KMeans, PCA, TSNE y NearestNeighbors. https://scikit-learn.org/stable/
- Scikit-learn. Evaluación de agrupamiento. https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
- Manning, C., Raghavan, P. y Schütze, H. (2008). *Introduction to Information Retrieval*, cap. 8. https://nlp.stanford.edu/IR-book/
- Van der Maaten, L. y Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9.
- Xiao, H., Rasul, K. y Vollgraf, R. (2017). Fashion-MNIST. https://github.com/zalandoresearch/fashion-mnist

El código se elaboró con asistencia de IA y fue revisado por el responsable, según las reglas del curso.
