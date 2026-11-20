# Plan de trabajo — Entrega 3 de StyleBot

**Fecha establecida en el documento del proyecto:** miércoles 25 de noviembre de 2026 a las 23:59.

**Meta interna del equipo:** miércoles 25 de noviembre de 2026 a las 19:00, para dejar margen antes del cierre oficial.

**Cierre técnico interno:** martes 24 de noviembre a las 18:00.

El miércoles no se aceptan nuevos desarrollos del equipo: se reserva para la ejecución completa, revisión y entrega del líder.

Todos deben trabajar en sus propios archivos y subir avances funcionales al repositorio. Nadie debe modificar el archivo de otro integrante sin avisar al líder.

## Krysto — líder y responsable de datos

| Fecha y hora límite | Entregable en GitHub |
|---|---|
| Jueves 19, 22:30 | Crear carpetas, protocolo común, MDP, reglas, semillas, nombres de archivos y lista de revisión. |
| Sábado 21, 18:00 | Comprobar que entorno, CNN y evaluación usan las mismas categorías, acciones, recompensas y rutas. Registrar diferencias. |
| Lunes 23, 20:00 | Revisar avances de todos, asignar correcciones concretas y comprobar que no se use prueba para elegir parámetros. |
| Martes 24, 18:00 | Cerrar contenido técnico, revisar integración, informe, demo y evidencias. |
| Miércoles 25, antes de 19:00 | Ejecutar notebook final, comprobar enlaces y dejar la entrega lista. No se aceptan desarrollos nuevos. |

## Luis Ángel — simulación y aprendizaje por refuerzo

| Fecha y hora límite | Entregable en GitHub |
|---|---|
| Viernes 20, 18:00 | Entorno Gymnasium 8×8 probado: recepción, pasillos, obstáculos, estantes, seis acciones, recompensas y terminación. |
| Sábado 21, 16:00 | `agentes_rl.py` con Q-Learning y Dyna-Q, además de primera corrida reproducible. |
| Domingo 22, 18:00 | Notebook con dos configuraciones por algoritmo, curvas iniciales y tablas Q guardadas. |
| Martes 24, 14:00 | Correcciones terminadas y archivos finales listos para integración. |

## Juan José — redes neuronales CNN

| Fecha y hora límite | Entregable en GitHub |
|---|---|
| Viernes 20, 18:00 | `percepcion_cnn.py` que cargue `stylebot_cnn_final.keras` sin reentrenar y devuelva categoría y confianza. |
| Sábado 21, 16:00 | Prueba con imágenes y conexión al entorno sin revelar la etiqueta real al agente. |
| Domingo 22, 18:00 | `02_integracion_cnn_rl.ipynb` con resultados reproducibles y conteo de aciertos/errores CNN. |
| Martes 24, 14:00 | Correcciones terminadas y sección lista para notebook final. |

## Ángel Uriel — aprendizaje automático clásico y evaluación

| Fecha y hora límite | Entregable en GitHub |
|---|---|
| Viernes 20, 18:00 | Estructura de `03_evaluacion_rl.ipynb`, métricas y agente aleatorio. |
| Sábado 21, 16:00 | Código para evaluar con las mismas semillas y exportar resultados sin entrenar durante la evaluación. |
| Domingo 22, 18:00 | Comparación Q-Learning, Dyna-Q y aleatorio; gráficas y separación de fallos CNN/política. |
| Martes 24, 14:00 | CSV, gráficas, conclusiones y fuentes corregidas. |

## Jacqueline — integración, documentación y demostración

| Fecha y hora límite | Entregable en GitHub |
|---|---|
| Viernes 20, 18:00 | Crear estructura de la nueva sección del informe y del notebook `E3_StyleBot.ipynb`. |
| Domingo 22, 20:00 | Integrar avances disponibles y documentar el MDP, algoritmos, configuraciones y métricas. |
| Lunes 23, 20:00 | Primer borrador completo de 4 a 6 páginas y demostración preliminar. |
| Martes 24, 16:00 | Notebook final, informe, evidencias y GIF o video de 50 prendas listos para revisión. |
| Martes 24, 18:00 | Aplicar últimas correcciones y cerrar la entrega. |

## Archivos finales que deben existir

- `notebooks/entrega3/E3_StyleBot.ipynb`
- `src/entrega3/entorno_almacen.py`
- `src/entrega3/agentes_rl.py`
- `src/entrega3/percepcion_cnn.py`
- `modelos/entrega3/q_table_qlearning.npz`
- `modelos/entrega3/q_table_dynaq.npz`
- tablas y gráficas en `resultados/entrega3/`
- informe actualizado en `documentacion/`
- demostración de 50 prendas en `demo/entrega3/`
- matriz de cobertura en `documentacion/entrega3/MATRIZ_REQUISITOS_ENTREGA3.md`
