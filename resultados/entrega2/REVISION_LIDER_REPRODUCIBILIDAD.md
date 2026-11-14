# Revisión del líder: reproducibilidad y fuga de datos

**Responsable:** Krysto Alberto Adan Jauregui Zuñiga  
**Fecha de revisión:** 13 de noviembre de 2026  
**Entrega:** 2 - Redes neuronales convolucionales

## Archivos revisados

- `notebooks/entrega2/00_datos_cnn.ipynb`
- `notebooks/entrega2/00_experimento_aumento.ipynb`
- `notebooks/entrega2/01_arquitecturas_cnn.ipynb`

## Comprobaciones aprobadas

| Control | Resultado |
|---|---|
| Semilla común | 42 en los tres notebooks |
| Entrenamiento | 48,000 imágenes |
| Validación | 12,000 imágenes |
| Prueba | 10,000 imágenes oficiales, sin mezclar con entrenamiento |
| Estratificación | `stratify=y_base` con `random_state=42` |
| Normalización | Píxeles `float32` entre 0 y 1 |
| Forma para CNN | `(28, 28, 1)` |
| Categorías | Diez categorías en el mismo orden |
| Aumento | Solo se aplica a `train_ds` |
| Validación y prueba | `val_ds` y `test_ds` no contienen aumento |
| Selección de modelos | Debe hacerse con validación, no con prueba |

## Revisión de fuga de datos

No se detectó fuga de datos en los archivos disponibles. La división se hace antes del aumento y las transformaciones aleatorias se aplican únicamente después de crear el conjunto de entrenamiento. El conjunto oficial de prueba permanece separado.

Regla para el resto de la Entrega 2: el conjunto de prueba solo se utilizará una vez, el lunes 16, después de seleccionar con validación la mejor CNN propia y el modelo de transfer learning.

## Experimento controlado del líder

Se comparó la misma CNN base, con la misma inicialización, ocho épocas, lote 128 y Adam con tasa de aprendizaje 0.001.

| Condición | Accuracy de validación | F1 macro de validación | Tiempo | Parámetros |
|---|---:|---:|---:|---:|
| Sin aumento | 0.9188 | 0.9183 | 32.9020 s | 402,442 |
| Con aumento | 0.8790 | 0.8793 | 58.2762 s | 402,442 |

En este experimento la condición sin aumento obtuvo mejor F1 macro. Esto no significa que el aumento sea incorrecto; significa que, con esta arquitectura y ocho épocas, las transformaciones utilizadas hicieron el entrenamiento más difícil. El resultado debe documentarse sin cambiarlo ni ocultarlo.

## Estado del notebook de Juan José

El archivo `01_arquitecturas_cnn.ipynb` ya contiene:

- Arquitectura A: 402,442 parámetros.
- Arquitectura B: 421,930 parámetros.
- Arquitectura C: 1,749,034 parámetros.
- `EarlyStopping` con paciencia 5 y restauración de los mejores pesos.
- `ReduceLROnPlateau` con factor 0.5 y paciencia 3.
- La misma división, semilla y canal de entrada que los notebooks del líder.

Pendiente observado al momento de esta revisión: el archivo todavía no contiene el entrenamiento de la Arquitectura A, sus métricas de validación, historial, curvas ni modelo guardado. Es trabajo asignado a Juan José y no se sustituye desde la parte del líder.

## Configuración obligatoria para A, B y C

- Semilla: 42.
- División: 48,000 / 12,000 / 10,000.
- Batch size: 128, salvo que se documente una limitación de memoria.
- Mismo número máximo de épocas para A, B y C.
- Mismos callbacks para A, B y C.
- Misma normalización y mismas categorías.
- Selección de arquitectura solamente con validación.
- Prohibido usar prueba para ajustar parámetros o elegir el modelo.

## Próxima revisión del líder

Cuando Juan José suba los entrenamientos A, B y C, se deben comprobar los historiales, las épocas realmente ejecutadas, los callbacks, los parámetros y que las métricas provengan de validación. Después se registrará la arquitectura seleccionada.
