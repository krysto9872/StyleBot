# StyleBot

StyleBot es un proyecto académico que desarrolla un sistema para identificar prendas de ropa y simular su acomodo en un almacén. La primera entrega utiliza Fashion-MNIST para preparar los datos, comparar métodos clásicos de aprendizaje automático y explorar prendas similares.

## Integrantes y responsabilidades

- Krysto Alberto Adan Jauregui Zuñiga: Coordinador del proyecto y responsable de datos
- Jacqueline Soto: Responsable de integración, documentación y demostración
- Juan José Aranda Flores: Responsable de redes neuronales (CNN)
- Luis Ángel Mendoza Gaytán: Responsable de simulación y aprendizaje por refuerzo
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

## Tercera entrega

Esta entrega integra la CNN congelada de la Entrega 2 con un almacén simulado de 8 por 8. El robot recibe una prenda, usa la predicción de la CNN como percepción y aprende a llevarla al estante correspondiente mediante Q-Learning y Dyna-Q.

El protocolo común está en [`notebooks/entrega3/00_protocolo_entrega3.ipynb`](notebooks/entrega3/00_protocolo_entrega3.ipynb) y la especificación completa del MDP está en [`documentacion/entrega3/ESPECIFICACION_MDP.md`](documentacion/entrega3/ESPECIFICACION_MDP.md).

Reglas comunes de esta entrega:

- Almacén de 8 por 8 con recepción, pasillos, obstáculos y diez estantes.
- Estado tabular de 1,408 posibilidades: posición, prenda percibida y disponibilidad del estante destino.
- Seis acciones: arriba, abajo, izquierda, derecha, tomar y colocar.
- Recompensas: +10 colocación correcta, -8 incorrecta, -3 estante lleno, -2 acción inválida y -1 paso normal.
- Semilla común 42 para reproducibilidad.
- Comparación de Q-Learning, Dyna-Q y un agente aleatorio con los mismos episodios de evaluación.
- La CNN final se carga desde `modelos/stylebot_cnn_final.keras` y no se vuelve a entrenar.
- La demostración final procesa 50 prendas y reporta el porcentaje de éxito.

### Organización de la tercera entrega

```text
StyleBot/
├── datos/entrega3/          Protocolo y datos de evaluación
├── demo/entrega3/           GIF o video final de hasta un minuto
├── documentacion/entrega3/  MDP, plan y aportaciones para el informe
├── modelos/entrega3/        Tablas Q entrenadas
├── notebooks/entrega3/      Desarrollo e integración de la Entrega 3
├── resultados/entrega3/     Tablas, gráficas, métricas y revisión
└── src/entrega3/            Entorno, agentes y conexión con la CNN
```

### Orden de ejecución de la tercera entrega

1. Ejecutar `notebooks/entrega3/00_protocolo_entrega3.ipynb` para comprobar las reglas comunes.
2. Probar `src/entrega3/entorno_almacen.py` y `src/entrega3/probar_entorno.py`.
3. Entrenar Q-Learning y Dyna-Q en `notebooks/entrega3/01_qlearning_dynaq.ipynb`.
4. Integrar la CNN congelada en `notebooks/entrega3/02_integracion_cnn_rl.ipynb`.
5. Comparar algoritmos y agente aleatorio en `notebooks/entrega3/03_evaluacion_rl.ipynb`.
6. Ejecutar el notebook integrado `notebooks/entrega3/E3_StyleBot.ipynb` de arriba hacia abajo.
7. Verificar el informe, las gráficas, las tablas y la demostración de 50 prendas antes de entregar.

## Cuarta y última entrega

La Entrega 4 integra en un solo flujo reproducible los componentes desarrollados en las tres entregas anteriores: preparación de datos, modelos clásicos, análisis no supervisado, recomendador, CNN y agentes de aprendizaje por refuerzo.

La entrega oficial es el miércoles 2 de diciembre de 2026 antes de las 19:00. El cierre interno del equipo será el martes 1 de diciembre a las 19:00.

### Organización de la Entrega 4

~~~text
StyleBot/
├── app/                         Aplicación Streamlit opcional
├── demo/entrega4/              Evidencia de la demostración final
├── documentacion/entrega4/     Plan, checklist, contrato y coevaluación
├── notebooks/entrega4/         Notebook final y comparación global
├── presentacion/               Diapositivas de la exposición
├── resultados/entrega4/        Tabla global, gráficas y pruebas
└── src/entrega4/               Funciones finales de CNN y simulación
~~~

El plan, la lista de verificación y las reglas comunes se encuentran en:

- [PLAN_TRABAJO_ENTREGA4.md](documentacion/entrega4/PLAN_TRABAJO_ENTREGA4.md)
- [CHECKLIST_FINAL.md](documentacion/entrega4/CHECKLIST_FINAL.md)
- [CONTRATO_INTEGRACION.md](documentacion/entrega4/CONTRATO_INTEGRACION.md)

### Responsables

- Krysto Alberto Adan Jauregui Zúñiga: estructura, integración, notebook final, revisión y entrega.
- Ángel Uriel Martínez Godinez: comparación global de modelos, tabla, gráfica y análisis.
- Juan José Aranda Flores: función reutilizable de inferencia CNN y pruebas.
- Luis Ángel Mendoza Gaytán: función reutilizable de simulación de aprendizaje por refuerzo y pruebas.
- Jacqueline Soto: informe final, PDF, presentación, demostración, coevaluación e interfaz web opcional.

### Orden de ejecución final

1. Abrir [notebooks/entrega4/E4_StyleBot.ipynb](notebooks/entrega4/E4_StyleBot.ipynb) en Google Colab.
2. Ejecutar todas las celdas de arriba hacia abajo.
3. Comprobar la carga de la CNN y las tablas Q guardadas.
4. Ejecutar la clasificación, selección de estante y simulación del agente.
5. Verificar la demostración de 50 prendas y la separación de fallos de CNN y de política.
6. Confirmar que las tablas, gráficas y conclusiones permanezcan visibles.
7. Descargar el notebook ejecutado y reemplazar la plantilla del repositorio.

### Entregables finales obligatorios

- Informe técnico final en PDF.
- Notebook E4_StyleBot.ipynb ejecutado sin errores.
- Repositorio completo y reproducible.
- README y requirements.txt actualizados.
- Modelos guardados o enlaces funcionales.
- Presentación final de diez minutos.
- Formato de coevaluación.
- Demostración funcional de 50 prendas.
- Evidencias de participación mediante commits.
- Correcciones recibidas durante las entregas anteriores.

### Aplicación web opcional

La aplicación con Streamlit o Gradio es una actividad adicional para obtener puntos extra. Permitirá seleccionar o cargar una imagen, mostrar la categoría y confianza de la CNN y presentar el estante, recorrido, pasos, recompensa y resultado del agente.

La aplicación no genera imágenes y no debe retrasar los entregables obligatorios.

## Fuentes principales

- [TensorFlow Fashion-MNIST](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data)
- [Tutorial de clasificación de TensorFlow](https://www.tensorflow.org/tutorials/keras/classification)
- [División de datos en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Inverted residuals and linear bottlenecks](https://doi.org/10.1109/CVPR.2018.00474)
- [EarlyStopping](https://keras.io/api/callbacks/early_stopping/)
- [Dropout](https://jmlr.org/papers/v15/srivastava14a.html)
- [Convolutional neural networks Stanford University](https://cs231n.github.io/convolutional-networks/)
