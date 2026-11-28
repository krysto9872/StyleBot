# Análisis global de StyleBot

## Alcance de la comparación

StyleBot combina componentes que resuelven tareas diferentes. Por ello, **no se deben comparar directamente** la varianza explicada, NMI, Precision@5, F1 macro y recompensa de aprendizaje por refuerzo. Cada métrica se interpreta dentro de su propio componente.

## Resultados principales

- **PCA:** con 50 componentes conserva 86.3% de la varianza. Sirve para reducir la representación de las imágenes antes de agrupar o recomendar.
- **K-Means:** con k=10 obtuvo NMI de 0.5121 en validación. Detecta agrupaciones parciales, aunque varias prendas superiores se traslapan.
- **Recomendador:** PCA 50 con distancia coseno logró Precision@5 de 0.8321, el mejor resultado de sus variantes en validación.
- **CNN:** Arquitectura C obtuvo F1 macro de 0.9259 en prueba. Es el modelo elegido para reconocer la categoría de la prenda.
- **Aprendizaje por refuerzo:** Dyna-Q obtuvo 93.6% de éxito de prendas, ligeramente por encima de Q-Learning (92.4%) y muy por encima del agente aleatorio (1.6%). Dyna-Q también usó menos pasos promedio: 78.64 frente a 90.72 de Q-Learning.

## Conclusión

No existe un modelo único que resuelva todo el sistema. PCA reduce dimensiones, K-Means ayuda a explorar similitudes, el recomendador propone prendas similares, la CNN identifica la prenda y Dyna-Q decide cómo acomodarla. La integración permite que la predicción de la CNN se convierta en una decisión de almacenamiento.

## Limitaciones

- K-Means no usa etiquetas y sus grupos no representan necesariamente categorías exactas.
- El recomendador puede confundir prendas con siluetas similares.
- La CNN puede cometer errores de clasificación, que se deben separar de los fallos de la política RL.
- Dyna-Q depende de una representación del entorno para generar experiencias simuladas.

## Fuentes internas

- `documentacion/APORTE_AGRUPAMIENTO_RECOMENDADOR.md`
- `resultados/tablas/kmeans_metricas.csv`
- `resultados/tablas/recomendador_comparacion_validacion.csv`
- `resultados/entrega2/tabla_evaluacion_final_prueba.csv`
- `resultados/entrega3/qlearning_dynaq_resumen.csv`
