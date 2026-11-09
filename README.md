# StyleBot

StyleBot es un proyecto académico que desarrolla un sistema para identificar prendas de ropa y simular su acomodo en un almacén. La primera entrega utiliza Fashion-MNIST para preparar los datos, comparar métodos clásicos de aprendizaje automático y explorar prendas similares.

## Integrantes y responsabilidades

- Krysto Zuñiga: Líder y preparación de datos.
- Ángel Godinez: Aprendizaje automático clásico.
- Luis Gaytan: Simulación y aprendizaje por refuerzo.
- Jacqueline Soto: Integración y documentación.
- Juan Flores: Redes neuronales.

## Primera entrega

La preparación común de los datos ya incluye:

- 48,000 imágenes de entrenamiento.
- 12,000 imágenes de validación.
- 10,000 imágenes de prueba.
- Imágenes de 28 por 28 píxeles normalizadas entre 0 y 1.
- Diez categorías equilibradas.
- Semilla 42 para reproducir la división.

El notebook principal de esta parte es [`notebooks/01_datos.ipynb`](notebooks/01_datos.ipynb). Las reglas detalladas están en [`README_DATOS.md`](README_DATOS.md).

## Estructura inicial

```text
StyleBot/
├── datos/                 Categorías, índices y protocolo reproducible
├── documentacion/         Aportaciones para el informe
├── notebooks/             Notebooks de cada parte
├── resultados/            Tablas y gráficas generadas
├── src/                   Funciones reutilizables
├── README.md              Descripción general
├── README_DATOS.md        Instrucciones de preparación de datos
└── requirements.txt       Bibliotecas utilizadas
```

## Ejecución

1. Abrir `notebooks/01_datos.ipynb` en Google Colab.
2. Ejecutar las celdas de arriba hacia abajo.
3. Conservar la semilla, categorías y división establecidas.
4. No utilizar el conjunto de prueba para entrenar o seleccionar parámetros.

## Fuentes principales

- [TensorFlow Fashion-MNIST](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data)
- [Tutorial de clasificación de TensorFlow](https://www.tensorflow.org/tutorials/keras/classification)
- [División de datos en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

