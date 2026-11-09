# StyleBot Preparación común de datos

Este material prepara Fashion-MNIST para la primera entrega. El objetivo es que los cinco integrantes utilicen las mismas imágenes, divisiones y categorías.

## Abrir y ejecutar

1. Abre https://colab.research.google.com/ e inicia sesión.
2. Utiliza Archivo y Subir notebook para abrir `notebooks/01_datos.ipynb`.
3. Guarda una copia en tu Drive y escribe tu nombre en el primer bloque de texto.
4. Ejecuta las celdas de arriba hacia abajo. La celda de carga descarga Fashion-MNIST automáticamente.
5. Revisa que no aparezcan errores y descarga el ZIP de resultados de la última celda.
6. Descarga también el notebook ejecutado como `.ipynb` para conservar sus resultados.

## Acuerdo de datos

| Conjunto | Imágenes | Uso |
|---|---:|---|
| Entrenamiento | 48,000 | Aprender y hacer validación cruzada |
| Validación | 12,000 | Seleccionar ajustes antes de medir la prueba |
| Prueba | 10,000 | Evaluación final, conservando la prueba oficial |

- Semilla: 42.
- Separación: de las 60,000 imágenes originales de entrenamiento se reserva 20 % para validación mediante `train_test_split` estratificado.
- Imágenes de 28 por 28 píxeles, un canal en escala de grises.
- Píxeles `float32` entre 0 y 1, normalizados una sola vez.
- Etiquetas enteras de 0 a 9.
- El código común se encuentra en el notebook y también en `src/datos_stylebot.py`.
- Las versiones se registran al ejecutar. Comparar las huellas de índices si los entornos difieren.

| ID | Categoría |
|---:|---|
| 0 | Camiseta |
| 1 | Pantalón |
| 2 | Suéter |
| 3 | Vestido |
| 4 | Abrigo |
| 5 | Sandalia |
| 6 | Camisa |
| 7 | Tenis |
| 8 | Bolso |
| 9 | Botín |

## Archivos para compartir

- `notebooks/01_datos.ipynb`: carga, comprobaciones y ejemplos explicados. Ejecutarlo antes de la entrega académica.
- `src/datos_stylebot.py`: las mismas funciones comunes, útiles para la integración posterior.
- `datos/categorias.csv`: correspondencia entre números y categorías.
- `datos/particiones_semilla42.npz`: índices de las imágenes, generados al ejecutar.
- `datos/protocolo_datos.json`: reglas, huellas y versiones, generado al ejecutar.
- `resultados/tablas/`: conteos y comprobaciones.
- `resultados/graficas/`: ejemplos y distribución.

En Colab, cada compañero puede guardar una copia del notebook con el nombre de su parte y conservar las celdas comunes. El módulo de Python es opcional para quien prefiera importar funciones; en ese caso hay que cargarlo también en la sesión.

## Reglas de trabajo

- Conservar la semilla, las categorías y las divisiones comunes.
- No volver a normalizar entre 255 las imágenes ya preparadas.
- Ajustar escaladores y PCA solo con entrenamiento; dentro de cada partición cuando se hace validación cruzada.
- No usar la prueba para elegir parámetros.
- Subir cambios desde la cuenta de cada integrante y explicar qué cambió.
- Registrar resultados reales, fuentes y aportaciones. Leer y comprender el código asistido, y reconocer esa asistencia según las reglas del curso.

## Fuentes

- TensorFlow, Fashion-MNIST load_data: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
- TensorFlow, Basic classification: https://www.tensorflow.org/tutorials/keras/classification
- Scikit-learn, train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Zalando Research, Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
