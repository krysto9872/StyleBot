# Aportación del líder y responsable de datos

## Objetivo

Preparar una base de datos común y reproducible para que todos los integrantes trabajen con las mismas imágenes, categorías y divisiones durante la primera entrega de StyleBot.

## Dataset

Se utilizó Fashion-MNIST, un conjunto de 70,000 imágenes en escala de grises. Cada imagen tiene una resolución de 28 por 28 píxeles y pertenece a una de diez categorías: camiseta, pantalón, suéter, vestido, abrigo, sandalia, camisa, tenis, bolso y botín.

## Preparación realizada

La división oficial de prueba, formada por 10,000 imágenes, se conservó para la evaluación final. Las 60,000 imágenes restantes se dividieron de manera estratificada en 48,000 ejemplos de entrenamiento y 12,000 de validación. Se utilizó una semilla de 42 para permitir que la separación pueda reproducirse.

Los píxeles, que originalmente tenían valores enteros entre 0 y 255, se transformaron a valores `float32` entre 0 y 1. La normalización facilita que los modelos trabajen con una escala común y se realiza una sola vez.

## Comprobaciones

Se verificaron las cantidades, la forma de las imágenes, el rango de los píxeles, las etiquetas y la ausencia de valores no finitos. Entrenamiento y validación no comparten índices. Las diez categorías quedaron equilibradas: cada una contiene 4,800 imágenes de entrenamiento, 1,200 de validación y 1,000 de prueba.

Estas comprobaciones validan la estructura y los valores del dataset. No constituyen una búsqueda completa de imágenes duplicadas ni una inspección individual de daños visuales.

## Archivos producidos

- `notebooks/01_datos.ipynb`: carga, división, verificaciones y gráficas.
- `datos/categorias.csv`: correspondencia entre identificadores y categorías.
- `datos/particiones_semilla42.npz`: índices de la división reproducible.
- `datos/protocolo_datos.json`: reglas, versiones y huellas de los índices.
- `resultados/tablas/`: cantidades y comprobaciones.
- `resultados/graficas/`: ejemplos y distribución de categorías.

## Fuentes

- TensorFlow. Fashion-MNIST `load_data`. https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
- TensorFlow. Basic classification: Classify images of clothing. https://www.tensorflow.org/tutorials/keras/classification
- Scikit-learn. `train_test_split`. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
