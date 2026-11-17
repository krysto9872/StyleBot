# Revisión del líder: reproducibilidad, integración y uso de datos

**Responsable:** Krysto Alberto Adan Jauregui Zuñiga  
**Fecha de actualización:** 16 de noviembre de 2026  
**Entrega:** 2 - Redes neuronales convolucionales  
**Corte revisado:** revisión técnica del lunes 16 de noviembre

## Archivos revisados

- `notebooks/entrega2/00_datos_cnn.ipynb`
- `notebooks/entrega2/00_experimento_aumento.ipynb`
- `notebooks/entrega2/01_arquitecturas_cnn.ipynb`
- `notebooks/entrega2/02_metricas_comparacion.ipynb`
- `notebooks/entrega2/03_inferencia_integracion.ipynb`
- `modelos/arquitectura_a.keras`
- `modelos/arquitectura_b.keras`
- `modelos/stylebot_cnn_final.keras`
- Los CSV de entrenamiento, comparación e inferencia publicados en `resultados/entrega2/`.

## Controles de reproducibilidad aprobados

| Control | Resultado verificado |
|---|---|
| Semilla común | 42 |
| Entrenamiento | 48,000 imágenes |
| Validación | 12,000 imágenes |
| Prueba | 10,000 imágenes oficiales |
| Estratificación | `stratify=y_base` y `random_state=42` |
| Normalización | `float32` entre 0 y 1 |
| Entrada de CNN propias | `(28, 28, 1)` |
| Categorías | Diez clases en el orden oficial de Fashion-MNIST |
| Aumento | Únicamente en `train_ds` |
| Validación y prueba | Sin transformaciones aleatorias |
| Selección del modelo | Con validación, no con prueba |

No se detectó mezcla de índices entre entrenamiento y validación. La división se realiza antes del aumento y las transformaciones aleatorias se aplican después de construir el conjunto de entrenamiento. La Arquitectura C fue seleccionada utilizando exclusivamente métricas de validación.

## Experimento controlado del líder

Se comparó la misma CNN base con la misma inicialización, ocho épocas, lote 128 y Adam con tasa de aprendizaje 0.001.

| Condición | Accuracy de validación | F1 macro de validación | Tiempo | Parámetros |
|---|---:|---:|---:|---:|
| Sin aumento | 0.9188 | 0.9183 | 32.9020 s | 402,442 |
| Con aumento | 0.8790 | 0.8793 | 58.2762 s | 402,442 |

En este experimento el aumento no mejoró la validación. El resultado se conserva sin ocultarlo ni modificarlo. Esto no demuestra que el aumento sea incorrecto en general; indica que las transformaciones y ocho épocas no favorecieron a esta arquitectura bajo la configuración probada.

## Resultados de validación verificados

| Modelo | Parámetros | Tiempo de entrenamiento | Accuracy de validación | F1 macro de validación |
|---|---:|---:|---:|---:|
| Arquitectura A sin aumento | 402,442 | 49.07 s | 0.9213 | 0.9215 |
| Arquitectura A con aumento | 402,442 | 199.93 s | 0.9010 | 0.9006 |
| Arquitectura B | 421,930 | 251.71 s | 0.9025 | 0.9013 |
| Arquitectura C | 1,749,034 | 360.68 s | 0.9321 | 0.9319 |
| MobileNetV2 | 2,428,362 | 316.98 s | 0.4172 | 0.4261 |

La mejor CNN propia es la Arquitectura C. El archivo publicado como modelo final provisional es `modelos/stylebot_cnn_final.keras`. La palabra provisional se mantiene hasta realizar la evaluación final programada del conjunto de prueba.

## Revisión de modelos guardados e inferencia

Los archivos A, B y C se encuentran publicados y el notebook de inferencia fue ejecutado completamente, sin errores. La recarga produjo los siguientes resultados en validación:

| Modelo | Tamaño `.keras` | Tiempo individual por imagen | Accuracy recalculada | F1 macro recalculado |
|---|---:|---:|---:|---:|
| Arquitectura A | 4.63 MB | 8.923 ms | 0.9213 | 0.9215 |
| Arquitectura B | 4.88 MB | 16.827 ms | 0.8990 | 0.8984 |
| Arquitectura C | 20.11 MB | 34.069 ms | 0.9321 | 0.9319 |

La Arquitectura A es la alternativa más ligera y rápida. La Arquitectura C obtiene el mejor desempeño y permanece como modelo elegido. La prueba de sesión limpia cargó el modelo final y generó predicciones válidas sin depender de variables creadas previamente.

### Diferencia observada en la Arquitectura B

El CSV de entrenamiento registra 0.9025 de accuracy y 0.9013 de F1 macro para B, mientras que el archivo `arquitectura_b.keras` recargado obtuvo 0.8990 y 0.8984. La diferencia no cambia la selección de C, pero Juan José y Luis Ángel deben confirmar el lunes si el archivo B contiene exactamente los mejores pesos restaurados por `EarlyStopping`.

## Revisión de transfer learning del domingo 15

La rama MobileNetV2 conserva la misma división y semilla. Las imágenes se convierten de `(28, 28, 1)` a RGB, se redimensionan a `(32, 32, 3)` y se escalan al intervalo esperado por MobileNetV2. La base preentrenada permanece congelada durante el entrenamiento de la cabeza clasificadora.

El modelo MobileNetV2 obtuvo 0.4172 de accuracy y 0.4261 de F1 macro en validación, considerablemente por debajo de C. El notebook de inferencia reconstruye MobileNetV2 sin pesos entrenados cuando no encuentra su archivo, por lo que sus tiempos y tamaño sirven como referencia de arquitectura, pero sus predicciones recalculadas no sustituyen las métricas del entrenamiento original.

## Incidencia: uso anticipado del conjunto de prueba

El notebook `02_metricas_comparacion.ipynb` contiene una evaluación del conjunto de prueba realizada antes de la fecha acordada. Esta ejecución incumple el protocolo del equipo. Sus números no deben utilizarse para modificar arquitecturas, hiperparámetros, aumento, callbacks ni la selección del modelo.

La decisión de conservar la Arquitectura C ya estaba sustentada por validación. Para mantener trazabilidad, el lunes 16 se deberá:

1. Congelar los modelos y la configuración antes de cualquier evaluación final.
2. Registrar que existió una ejecución anticipada.
3. Ejecutar la evaluación final sin realizar ajustes posteriores basados en prueba.
4. Separar claramente los resultados de validación de los resultados finales de prueba.

## Estado de integración al cierre del sábado

| Responsable | Evidencia revisada | Estado técnico |
|---|---|---|
| Krysto | Datos, aumento, protocolo, revisión e integración | Actualizado |
| Juan José | A, B, C, MobileNetV2, historiales y modelos | Completo |
| Ángel Uriel | Métricas y matrices; incluye prueba anticipada | Requiere corrección de protocolo |
| Luis Ángel | Notebook ejecutado, inferencia, sesión limpia y resultados | Completo |
| Jacqueline | Informe actualizado; faltan curvas incrustadas y actualizar estados | Requiere corrección documental |

## Decisiones del líder

- Arquitectura seleccionada con validación: C.
- Modelo final provisional: `modelos/stylebot_cnn_final.keras`.
- No se permiten cambios de arquitectura basados en resultados de prueba.
- La diferencia de B se documenta, pero no afecta la selección.
- Las curvas y matrices deben integrarse al informe y al notebook final.
- `E2_StyleBot.ipynb` debe ejecutarse completamente desde cero antes de la entrega.

## Próximas comprobaciones

- Confirmar los mejores pesos guardados de la Arquitectura B.
- Publicar o corregir las evidencias de validación de Ángel sin mezclar prueba.
- Agregar las curvas reales al informe.
- Realizar el lunes la evaluación final documentada sin ajustes posteriores.
- Completar el notebook integrado, las conclusiones, contribuciones y referencias.


## Actualización del líder: lunes 16 de noviembre de 2026

### Evidencias nuevas revisadas

- `modelos/mobilenetv2_transfer.keras` fue publicado y puede cargarse como modelo entrenado.
- `notebooks/entrega2/01_arquitecturas_cnn.ipynb` contiene 13 celdas de código ejecutadas, sin salidas de error, y verifica la recarga de la Arquitectura C y de MobileNetV2.
- Ángel Uriel publicó `resultados/tablas/tabla_comparativa_validacion.csv` y `resultados/graficas/matriz_confusion_mobilenetv2_validacion.png`.
- Luis Ángel ejecutó nuevamente la inferencia con el archivo entrenado de MobileNetV2 y generó la comparación contra la Arquitectura C.

### Corrección de las métricas de MobileNetV2

El valor de 0.4172 de accuracy y 0.4261 de F1 macro corresponde al experimento inicial con la base congelada. Después del entrenamiento controlado y la verificación del archivo final, `modelos/mobilenetv2_transfer.keras` obtuvo los siguientes resultados reproducidos en validación:

| Modelo verificado | Accuracy de validación | F1 macro de validación | Estado |
|---|---:|---:|---|
| MobileNetV2 inicial con base congelada | 0.4172 | 0.4261 | Referencia experimental |
| MobileNetV2 entrenado y guardado | 0.9128 | 0.9122 | Modelo final de transferencia |

Para la comparación final debe utilizarse 0.9128 y 0.9122. La tabla de Ángel y el resumen textual del notebook de Juan todavía muestran el resultado inicial, por lo que deben distinguir las dos etapas y agregar el modelo final entrenado.

### Comparación de inferencia verificada

| Modelo | Tamaño del archivo | Tiempo individual | Tiempo por imagen en lote | Imágenes por segundo | Accuracy de validación | F1 macro |
|---|---:|---:|---:|---:|---:|---:|
| Arquitectura C | 20.11 MB | 26.740 ms | 1.3212 ms | 757 | 0.9321 | 0.9319 |
| MobileNetV2 entrenado | 11.12 MB | 211.154 ms | 9.4831 ms | 105 | 0.9128 | 0.9122 |

La Arquitectura C supera a MobileNetV2 en accuracy y F1 macro y también es considerablemente más rápida en la medición realizada. MobileNetV2 ocupa menos espacio en disco, pero no compensa la pérdida de precisión ni el mayor tiempo de inferencia para el robot.

### Congelamiento de modelos y autorización de prueba final

A partir de esta revisión quedan congelados los siguientes archivos:

- `modelos/stylebot_cnn_final.keras`, Arquitectura C.
- `modelos/mobilenetv2_transfer.keras`, modelo final de transferencia.

Se autoriza la evaluación final con las 10,000 imágenes oficiales de prueba. Esta evaluación deberá realizarse una sola vez con los archivos congelados. No se permite modificar arquitectura, hiperparámetros, aumento, callbacks o pesos después de observar los resultados de prueba. Los resultados anticipados existentes no se utilizaron para seleccionar el modelo.

### Pendientes detectados antes de la integración final

1. **Juan José:** corregir el resumen de `01_arquitecturas_cnn.ipynb` para separar el MobileNetV2 inicial (0.4172/0.4261) del modelo entrenado final (0.9128/0.9122).
2. **Ángel Uriel:** actualizar su tabla comparativa con la fila del MobileNetV2 entrenado final y realizar la evaluación final de prueba sin ajustar posteriormente los modelos.
3. **Luis Ángel:** su actualización del lunes fue subida dentro de `src/notebooks/entrega2/` y `src/resultados/entrega2/`. Debe colocar el notebook y sus resultados en `notebooks/entrega2/` y `resultados/entrega2/`, evitando carpetas duplicadas.
4. **Jacqueline:** integrar las gráficas, corregir la tabla de contribuciones y actualizar el informe con las métricas finales verificadas.

### Decisión definitiva del líder

La Arquitectura C permanece como modelo seleccionado para StyleBot. Su elección se realizó con validación antes de autorizar la prueba final. El modelo de transferencia se conserva como comparación experimental y evidencia del uso de aprendizaje por transferencia.
