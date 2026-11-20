# Especificación común del MDP — Entrega 3

**Proyecto:** StyleBot

**Responsable de la especificación:** Krysto Alberto Adan Jauregui Zúñiga

**Versión:** 1.0 — 19 de noviembre de 2026

## 1. Objetivo

Entrenar y comparar agentes de decisión secuencial que acomoden prendas en un almacén simulado. La CNN final de la Entrega 2 funciona como percepción: predice la categoría de la prenda recogida y permanece congelada. La política de movimiento se aprende con Q-Learning y Dyna-Q.

## 2. Almacén 8×8

La recepción ocupa `(7,3)` y `(7,4)`. Los símbolos son `E` estante, `R` recepción, `#` obstáculo y `.` pasillo.

```text
    0 1 2 3 4 5 6 7
0   E E E E E E E E
1   . . . . . . . .
2   . # # . . # # .
3   . . . . . . . .
4   E . # . . # . E
5   . . . . . . . .
6   . # . . . . # .
7   . . . R R . . .
```

Los estantes de las categorías 0 a 7 están en la fila 0. Bolso, categoría 8, está en `(4,0)` y Botín, categoría 9, está en `(4,7)`. El mapa coincide con `src/entrega3/entorno_almacen.py`.

## 3. Formulación del MDP

### Estado

El estado observable contiene:

1. Posición del robot: 64 celdas posibles.
2. Carga percibida por la CNN: categorías 0 a 9 o valor 10 cuando no lleva prenda.
3. Indicador de que el estante destino está lleno: 0 o 1.

El total es `64 × 11 × 2 = 1,408` estados. La ocupación completa de los diez estantes se conserva dentro del entorno y se devuelve en `info`, pero no se agrega a la tabla Q para mantener manejable el aprendizaje tabular.

La codificación oficial es:

```python
estado = ((fila * 8 + columna) * 11 + carga) * 2 + int(destino_lleno)
```

### Acciones

| Código | Acción |
|---:|---|
| 0 | Arriba |
| 1 | Abajo |
| 2 | Izquierda |
| 3 | Derecha |
| 4 | Tomar una prenda en recepción |
| 5 | Colocar la prenda o devolverla cuando el destino está lleno |

### Recompensas

| Evento | Recompensa |
|---|---:|
| Colocación en el estante de la categoría real | +10 |
| Colocación en un estante incorrecto | -8 |
| Estante lleno o devolución por falta de espacio | -3 |
| Acción inválida o choque | -2 |
| Movimiento o acción normal sin colocación | -1 |

Cada acción recibe una sola recompensa. El valor -2 para una acción inválida evita que el agente confunda un choque con un movimiento válido.

### Transición, terminación y truncamiento

- El movimiento es determinista y los obstáculos no se pueden atravesar.
- `tomar` solo funciona en recepción, sin carga y con prendas pendientes.
- `colocar` deja la prenda en el estante actual si tiene capacidad.
- El episodio termina cuando no quedan prendas y el robot ya no lleva una.
- El episodio se trunca al alcanzar 60 pasos por prenda.

## 4. Integración de la CNN

Se carga `modelos/stylebot_cnn_final.keras` sin reentrenarlo. Al tomar una prenda, el entorno conserva dos valores:

- `carga_real`: etiqueta real, utilizada únicamente para calcular el resultado.
- `carga_percibida`: predicción de la CNN, utilizada por el agente para decidir el destino.

Clasificación de fallos:

- **Fallo de CNN:** el estante elegido coincide con la predicción, pero la predicción no coincide con la categoría real.
- **Fallo de política:** la prenda queda mal colocada y el estante elegido tampoco coincide con la predicción. Esto incluye el caso en que la CNN acertó pero el agente eligió otro estante.

La etiqueta real nunca se entrega al agente como parte del estado.

## 5. Entrenamiento y comparación

Se deben probar por lo menos estas dos configuraciones en ambos algoritmos:

| Configuración | Alpha | Gamma | Epsilon inicial | Epsilon mínimo | Decaimiento |
|---|---:|---:|---:|---:|---:|
| A | 0.10 | 0.95 | 1.00 | 0.05 | 0.995 |
| B | 0.20 | 0.99 | 1.00 | 0.05 | 0.997 |

Para Dyna-Q se comparan además 10 y 30 pasos de planeación. La selección se realiza con episodios de validación, nunca con el conjunto de prueba.

## 6. Protocolo reproducible de evaluación

- Semilla general y de entrenamiento: 42.
- Evaluación: semillas 1000 a 1049, sin exploración ni actualización de la tabla Q.
- Agente aleatorio: usa exactamente las mismas semillas y prendas que los agentes entrenados.
- Demostración: 50 prendas fijas generadas con semilla 2026.
- CNN congelada durante entrenamiento, validación, prueba y demostración.

Métricas obligatorias:

1. Recompensa acumulada por episodio y promedio móvil.
2. Tasa de colocación correcta.
3. Pasos promedio por prenda.
4. Colocaciones incorrectas.
5. Intentos en estantes llenos y acciones inválidas.
6. Tiempo de entrenamiento.
7. Fallos de CNN y fallos de política por separado.
8. Comparación de Q-Learning, Dyna-Q y agente aleatorio.

## 7. Responsables y archivos

| Responsable | Trabajo técnico | Archivos principales |
|---|---|---|
| Krysto Alberto Adan Jauregui Zúñiga | Protocolo, organización, revisión e integración de criterios | `00_protocolo_entrega3.ipynb`, esta especificación y `REVISION_LIDER_ENTREGA3.md` |
| Luis Ángel Mendoza Gaytán | Entorno, Q-Learning, Dyna-Q y tablas Q | `entorno_almacen.py`, `agentes_rl.py`, `01_qlearning_dynaq.ipynb` |
| Juan José Aranda Flores | Carga de la CNN congelada y conexión de la percepción | `percepcion_cnn.py`, `02_integracion_cnn_rl.ipynb` |
| Ángel Uriel Martínez Godinez | Comparación, agente aleatorio, gráficas, métricas y análisis de fallos | `03_evaluacion_rl.ipynb` y `resultados/entrega3/` |
| Jacqueline Soto | Notebook final, informe, demostración y evidencias | `E3_StyleBot.ipynb`, informe y `demo/entrega3/` |

## 8. Criterio de cierre

La entrega está lista solamente cuando `E3_StyleBot.ipynb` se ejecuta completo de arriba hacia abajo, produce las tablas y gráficas esperadas, carga la CNN y las tablas Q, evalúa los tres agentes, procesa las 50 prendas y coincide con el informe y la demostración.
