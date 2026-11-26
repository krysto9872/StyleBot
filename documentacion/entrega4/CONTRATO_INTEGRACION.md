# Contrato común de integración de StyleBot

Este documento define cómo se comunicarán los módulos de CNN, simulación, resultados y aplicación. El objetivo es evitar que cada integrante utilice nombres o formatos diferentes.

## Categorías comunes

```python
CATEGORIAS = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Tenis", "Bolso", "Botín"
]
```

## Rutas comunes

- CNN final: `modelos/stylebot_cnn_final.keras`.
- Tabla Q-Learning: `modelos/entrega3/q_table_qlearning.npz`.
- Tabla Dyna-Q: `modelos/entrega3/q_table_dynaq.npz`.
- Predicciones de 50 prendas: `datos/entrega3/predicciones_50_prendas.csv`.
- Resultados de agentes: `resultados/entrega3/`.

## Función de inferencia CNN

Archivo esperado: `src/entrega4/inferencia_cnn.py`.

```python
def predecir_prenda(imagen) -> dict:
    """Devuelve categoría, índice, confianza e imagen procesada."""
```

Campos mínimos de salida:

```python
{
    "indice_categoria": 2,
    "categoria": "Suéter",
    "confianza": 0.95,
    "imagen_procesada": imagen_28x28
}
```

## Función de simulación

Archivo esperado: `src/entrega4/simulacion_final.py`.

```python
def simular_acomodo(categoria, agente="dyna_q", semilla=2026) -> dict:
    """Carga la tabla Q y devuelve el resultado de una simulación."""
```

Campos mínimos de salida:

```python
{
    "agente": "dyna_q",
    "categoria": "Suéter",
    "estante": [0, 2],
    "trayectoria": [[7, 3], [6, 3]],
    "acciones": ["tomar", "arriba"],
    "pasos": 18,
    "recompensa": -7,
    "exito": True,
    "tipo_fallo": None
}
```

## Comparación global

Archivo esperado: `resultados/entrega4/tabla_comparacion_global.csv`.

Columnas mínimas:

```text
bloque,modelo,metrica_principal,valor,tiempo,tamano,observacion
```

No se deben comparar directamente métricas con significados distintos. La tabla debe explicar si el valor corresponde a accuracy, F1, Precision@5, tasa de éxito o recompensa.

## Reglas de integración

1. Las funciones no deben entrenar modelos al importarse.
2. Las tablas Q no pueden modificarse durante la evaluación.
3. Las rutas se resolverán desde la raíz del repositorio.
4. Los errores deben producir mensajes claros y no cerrar el notebook o la aplicación.
5. Los módulos deben poder probarse de forma independiente antes de conectarse al notebook final.

