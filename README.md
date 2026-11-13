# StyleBot

StyleBot es un proyecto académico que desarrolla un sistema para identificar prendas de ropa y simular su acomodo en un almacén. La primera entrega utiliza Fashion-MNIST para preparar los datos, comparar métodos clásicos de aprendizaje automático y explorar prendas similares.

## Integrantes y responsabilidades

- Krysto Alberto Adan Jauregui Zuñiga: Coordinador del proyecto y responsable de datos
- Jacqueline Soto: Responsable de integración, documentación y demostración
- Juan José Aranda Flores: Responsable de redes neuronales (CNN)
- Luis Ángel Mendosa Gaytan: Responsable de simulación y aprendizaje por refuerzo
- Ángel Uriel Martínez Godinez: Responsable de aprendizaje automático clásico

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
├── documentacion/         Informe final y aportaciones
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

## Notebooks de la primera entrega

- [`01_datos.ipynb`](notebooks/01_datos.ipynb): carga, división y comprobación de Fashion-MNIST.
- [`02_caracteristicas.ipynb`](notebooks/02_caracteristicas.ipynb): extracción de características con PCA, HOG y momentos de Hu.
- [`03_modelos.ipynb`](notebooks/03_modelos.ipynb): modelos clásicos supervisados, validación cruzada y métricas.
- [`04_no_supervisado.ipynb`](notebooks/04_no_supervisado.ipynb): K-Means, t-SNE y sistema de recomendación completo.
- [`05_aprendizaje_no_supervisado.ipynb`](notebooks/05_aprendizaje_no_supervisado.ipynb): sección separada de agrupamiento.
- [`06_sistema_recomendacion.ipynb`](notebooks/06_sistema_recomendacion.ipynb): sección separada del recomendador.
- [`E1_StyleBot.ipynb`](notebooks/E1_StyleBot.ipynb): notebook final integrado de la primera entrega.

El informe de la primera entrega está en [`documentacion/ProyectoFinal_StyleBot.docx`](documentacion/ProyectoFinal_StyleBot.docx).

Antes de entregar, se debe abrir `notebooks/E1_StyleBot.ipynb` en Google Colab, ejecutar todas las celdas de arriba hacia abajo y comprobar que las secciones 1 a 8, las tablas, las gráficas y las conclusiones aparezcan sin errores.

## Segunda entrega

Esta entrega desarrolla y compara redes neuronales convolucionales para clasificar las prendas de Fashion-MNIST.
Se conserva la misma división de la Entrega 1.

- 48,000 imágenes de entrenamiento.
- 12,000 imágenes de validación.
- 10,000 imágenes de prueba.
- Semilla 42.
- Píxeles normalizados entre 0 y 1.
- Diez categorías equilibradas.
- Imágenes adaptadas a la forma `(28, 28, 1)`.
- Aumento de datos aplicado solamente al entrenamiento.
- El conjunto de prueba no se utiliza para seleccionar arquitecturas.

En esta entrega, el notebook principal esta en `notebooks/entrega2/00_datos_cnn.ipynb` y las reglas detalladas están en [`README_DATOS.md`](README_DATOS.md).

## Estructura inicial

```text
StyleBot/
├── notebooks/
│   └── entrega2/
│       ├── 00_datos_cnn.ipynb
│       ├── 01_arquitecturas_cnn.ipynb
│       ├── 02_metricas_comparacion.ipynb
│       ├── 03_inferencia_integracion.ipynb
│       └── E2_StyleBot.ipynb
├── modelos/
│   └── stylebot_cnn_final.keras
├── resultados/
│   └── entrega2/
├── documentacion/
│   └── ProyectoFinal_StyleBot.docx
├── README.md
└── requirements.txt
```

## Ejecución

1. Abrir `notebooks/entrega2/E2_StyleBot.ipynb` en Google Colab.
2. Ejecutar todas las celdas de arriba hacia abajo.
3. Comprobar que las tablas y gráficas aparezcan.
4. Confirmar que el modelo final se pueda guardar y cargar.
5. No utilizar el conjunto de prueba para elegir arquitecturas o hiperparámetros.

## Notebooks de la segunda entrega

- [`00_datos_cnn.ipynb`](notebooks/entrega2/00_datos_cnn.ipynb): carga, preparación, división y aumento de datos para las CNN.
- [`01_arquitecturas_cnn.ipynb`](notebooks/entrega2/01_arquitecturas_cnn.ipynb): desarrollo y entrenamiento de tres arquitecturas CNN de diferente profundidad, con BatchNormalization, Dropout y callbacks.
- [`02_metricas_comparacion.ipynb`](notebooks/entrega2/02_metricas_comparacion.ipynb): evaluación de modelos, curvas de aprendizaje, matrices de confusión y comparación contra los modelos clásicos de la primera entrega.
- [`03_inferencia_integracion.ipynb`](notebooks/entrega2/03_inferencia_integracion.ipynb): inferencia, integración del modelo CNN y pruebas de clasificación.
- [`E2_StyleBot.ipynb`](notebooks/entrega2/E2_StyleBot.ipynb): notebook final integrado de la segunda entrega, incluyendo CNN y transfer learning con MobileNetV2.

## Fuentes principales

- [TensorFlow Fashion-MNIST](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data)
- [Tutorial de clasificación de TensorFlow](https://www.tensorflow.org/tutorials/keras/classification)
- [División de datos en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Inverted residuals and linear bottlenecks](https://doi.org/10.1109/CVPR.2018.00474)
- [EarlyStopping](https://keras.io/api/callbacks/early_stopping/)
- [Dropout](https://jmlr.org/papers/v15/srivastava14a.html)
- [Convolutional neural networks Stanford University](https://cs231n.github.io/convolutional-networks/)
