# Verificación de la simulación en E4 y en la aplicación

**Responsable:** Luis Ángel Mendoza Gaytán (simulación y aprendizaje por refuerzo)
**Fecha:** 28 de noviembre de 2026
**Commit revisado:** c5391ce
**Módulo:** `src/entrega4/simulacion_final.py`

## 1. Importación desde E4_StyleBot.ipynb

Se ejecutaron, sin modificarlas, las celdas de preparación, del protocolo, de las rutas y de la sección 6 del notebook. La celda 13 importó `simular_acomodo` desde `src.entrega4.simulacion_final` y mostró «Simulación validada»: devolvió todos los campos del contrato.

Después se llamó la misma función con ambos agentes, con las 10 categorías y con la semilla 2026:

| Categoría | Estante | Q-Learning: pasos / recompensa | Dyna-Q: pasos / recompensa |
|---|---|---|---|
| Camiseta/top | (0, 0) | 14 / −3 | 12 / −1 |
| Pantalón | (0, 1) | 13 / −2 | 11 / 0 |
| Suéter | (0, 2) | 12 / −1 | 10 / +1 |
| Vestido | (0, 3) | 11 / 0 | 9 / +2 |
| Abrigo | (0, 4) | 10 / +1 | 10 / +1 |
| Sandalia | (0, 5) | 11 / 0 | 11 / 0 |
| Camisa | (0, 6) | 12 / −1 | 12 / −1 |
| Tenis | (0, 7) | 13 / −2 | 13 / −2 |
| Bolso | (4, 0) | 10 / +1 | 8 / +3 |
| Botín | (4, 7) | 9 / +2 | 9 / +2 |

- Resultado: 20 de 20 simulaciones con éxito y con los campos del contrato completos (agente, categoría, estante, trayectoria, acciones, pasos, recompensa, éxito y tipo de fallo).
- `ejecutar_stylebot()` (celda 15) también funcionó con `agente="dyna_q"` y con `agente="q_learning"`; en esta prueba la CNN se sustituyó por una predicción fija de «Suéter».
- Las categorías de `predecir_prenda()` de Juan José coinciden exactamente con las del módulo de simulación (`protocolo_entrega3.json`).

## 2. Las tablas Q no se modifican

| Tabla | SHA-256 antes y después de todas las pruebas |
|---|---|
| `q_table_qlearning.npz` | `9a84a356c4fc3ef89f46dbefbf6dd4de83abc22ad3bc000d07095062ff7748ac` |
| `q_table_dynaq.npz` | `b58f699629cc90e2c7bebd8dc58823bdedbce2dc2a2036a27324dca87425e7a0` |

- Las huellas fueron idénticas antes y después.
- La tabla en memoria queda marcada como de solo lectura: si algún código intenta modificarla, Python lanza `ValueError: assignment destination is read-only`.
- No hay entrenamiento al importar ni al simular.

## 3. Conexión con app/app.py

Se ejecutaron las funciones `preparar_servicios()`, `ejecutar()` y `dibujar_recorrido()` de `app/app.py`, sin modificarlas. Como en la prueba no había TensorFlow ni Streamlit, la CNN se reemplazó por las predicciones ya publicadas de las 50 prendas (`predicciones_50_prendas.csv`).

| Agente | Semilla | Prendas | Correctas | Fallos de CNN | Otros fallos | Pasos promedio |
|---|---|---|---|---|---|---|
| Dyna-Q | 2026 | 50 | 49 | 1 | 0 | 10.5 |
| Dyna-Q | 0 | 50 | 49 | 1 | 0 | 10.5 |
| Dyna-Q | 2147483647 | 50 | 49 | 1 | 0 | 10.5 |
| Q-Learning | 2026 | 50 | 49 | 1 | 0 | 11.5 |
| Q-Learning | 0 | 50 | 49 | 1 | 0 | 11.5 |
| Q-Learning | 2147483647 | 50 | 49 | 1 | 0 | 11.5 |

- **Sin excepciones:** las 300 ejecuciones terminaron sin errores. En todas, la cantidad de acciones, posiciones y eventos coincide con los pasos, y el agente y la semilla del resultado coinciden con los elegidos.
- **El error es de la CNN:** la prenda fallida es el suéter que la CNN clasificó como camiseta. La aplicación lo muestra como `tipo_fallo = cnn`, no como un error del agente.
- **Exportación y mapa:** la descarga en JSON se generó sin errores. El mapa del recorrido se dibujó con `posicion_inicial` y `trayectoria`.
- **Imagen subida:** cuando no se conoce la categoría real, el agente llega al estante de la categoría predicha. La aplicación muestra el aviso de que eso no confirma la clasificación.

**Nota para Jacqueline:** las líneas de `preparar_servicios()` que reasignan `rl.RAIZ`, `rl.RUTA_PROTOCOLO` y `rl.RUTAS_TABLAS` no causan errores, pero tampoco son necesarias. El módulo ya calcula la raíz del repositorio con `Path(__file__).resolve().parents[2]` y funciona desde cualquier carpeta. Se pueden conservar o eliminar.

## 4. Pendiente

Para completar la tabla de pruebas de `app/README_APP.md` hay que abrir la aplicación real con Streamlit y TensorFlow en una computadora con el repositorio completo y tomar las capturas. La simulación no requiere cambios para esas pruebas.
