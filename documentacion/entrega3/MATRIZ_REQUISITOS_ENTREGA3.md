# Matriz de requisitos de la Entrega 3

Esta tabla convierte cada requisito del documento del proyecto en un archivo verificable y un responsable. El estado inicial indica que la estructura común está preparada; los resultados técnicos todavía deben ser producidos por sus responsables.

| Requisito del proyecto | Responsable principal | Evidencia esperada | Estado inicial |
|---|---|---|---|
| Entorno Gymnasium de almacén 8×8 con recepción, pasillos y estantes por categoría | Luis Ángel Mendoza Gaytán | `src/entrega3/entorno_almacen.py` y pruebas | Entorno inicial disponible; falta integrarlo con los agentes finales |
| MDP con posición, carga, disponibilidad de estantes y seis acciones | Krysto y Luis | `ESPECIFICACION_MDP.md`, protocolo y entorno | Especificado y validado; la disponibilidad completa permanece en el entorno y el estado tabular conserva la del destino relevante |
| Recompensas +10, -8, -3 y -1; cualquier ajuste debe justificarse | Krysto y Luis | Especificación, entorno y explicación del -2 por acción inválida | Especificado y validado |
| CNN predice la categoría al tomar una prenda | Juan José Aranda Flores | `src/entrega3/percepcion_cnn.py` y `02_integracion_cnn_rl.ipynb` | Pendiente |
| Q-Learning tabular | Luis Ángel Mendoza Gaytán | `agentes_rl.py`, notebook y tabla Q | Pendiente |
| Dyna-Q y comparación de velocidad de aprendizaje | Luis Ángel Mendoza Gaytán | Notebook, tiempos, episodios y tabla Q | Pendiente |
| Ajustar alpha, gamma, epsilon y decaimiento con dos configuraciones como mínimo | Luis y Ángel Uriel | `configuraciones_hiperparametros.csv` y notebook | Dos configuraciones definidas; falta ejecutarlas |
| Curva de recompensa acumulada por episodio | Ángel Uriel Martínez Godinez | `curva_recompensa.png` y datos CSV | Pendiente |
| Comparación contra agente aleatorio | Ángel Uriel Martínez Godinez | `comparacion_agentes.csv` y gráfica | Pendiente |
| Separar fallos causados por CNN y por política | Juan José y Ángel Uriel | `analisis_fallos.csv`, ejemplos y explicación | Definición común lista; falta medir |
| Notebook o scripts con entorno, agente y entrenamiento | Luis y Jacqueline | Notebooks 01 a 03 y `E3_StyleBot.ipynb` | Parcial |
| Sección de informe de 4 a 6 páginas con MDP, resultados y análisis | Jacqueline Soto | Documento del proyecto actualizado | Pendiente |
| Video de máximo un minuto o GIF con 50 prendas y porcentaje de aciertos | Jacqueline Soto | `demo/entrega3/stylebot_50_prendas.gif` o video equivalente | Pendiente |
| Repositorio actualizado y reproducible | Todo el equipo; revisión de Krysto | README, requirements, commits y ejecución final | Estructura común lista; faltan entregables individuales |

## Condiciones para marcar un requisito como completo

1. El archivo existe en la ruta acordada y pertenece al responsable indicado.
2. El código se ejecuta desde una sesión limpia siguiendo el README.
3. Los resultados visibles coinciden con los CSV, gráficas e informe.
4. No se usa el conjunto de prueba para seleccionar hiperparámetros.
5. La CNN final permanece congelada durante toda la Entrega 3.
6. El notebook final se ejecuta completo de arriba hacia abajo sin errores.
