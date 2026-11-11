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

## Notebooks de la primera entrega

- [`01_datos.ipynb`](notebooks/01_datos.ipynb): carga, división y comprobación de Fashion-MNIST.
- [`02_caracteristicas.ipynb`](notebooks/02_caracteristicas.ipynb): extracción de características con PCA, HOG y momentos de Hu.
- [`03_modeloss.ipynb`](notebooks/03_modeloss.ipynb): modelos clásicos supervisados, validación cruzada y métricas.
- [`04_no_supervisado.ipynb`](notebooks/04_no_supervisado.ipynb): K-Means, t-SNE y sistema de recomendación completo.
- [`05_aprendizaje_no_supervisado.ipynb`](notebooks/05_aprendizaje_no_supervisado.ipynb): sección separada de agrupamiento.
- [`06_sistema_recomendacion.ipynb`](notebooks/06_sistema_recomendacion.ipynb): sección separada del recomendador.
- [`E1_StyleBot.ipynb`](notebooks/E1_StyleBot.ipynb): notebook final integrado de la primera entrega.

Antes de entregar, se debe abrir `notebooks/E1_StyleBot.ipynb` en Google Colab, ejecutar todas las celdas de arriba hacia abajo y comprobar que las secciones 1 a 8, las tablas, las gráficas y las conclusiones aparezcan sin errores.

## Fuentes principales

- [TensorFlow Fashion-MNIST](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data)
- [Tutorial de clasificación de TensorFlow](https://www.tensorflow.org/tutorials/keras/classification)
- [División de datos en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

