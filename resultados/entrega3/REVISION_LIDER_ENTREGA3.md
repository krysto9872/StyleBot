# Revisión del líder — Entrega 3

**Responsable:** Krysto Alberto Adan Jauregui Zúñiga

**Corte de la revisión:** 20 de noviembre de 2026, hasta el commit `e335814`.

**Próximo cierre del equipo:** sábado 21 de noviembre, según el plan de trabajo.

## Trabajo realizado por el líder

- [x] Carpetas y protocolo común de la Entrega 3 publicados.
- [x] Semilla 42, categorías, acciones, recompensas y estado tabular de 1,408 posibilidades definidos.
- [x] MDP, matriz de requisitos, reglas y nombres de entregables documentados.
- [x] Entorno 8×8 de Luis ejecutado y comprobado.
- [x] Validación automática agregada en `src/entrega3/validar_integracion.py`.
- [x] Compatibilidad actual comprobada: mapa, recepción, estantes, acciones, recompensas, estados y rutas coinciden con el protocolo.
- [ ] Volver a ejecutar la validación cuando Juan, Luis y Ángel publiquen sus siguientes archivos.
- [ ] Cerrar la revisión de resultados, informe, demostración y notebook final.

## Estado comprobado por integrante

| Integrante | Estado al corte | Evidencia encontrada | Corrección o siguiente entrega |
|---|---|---|---|
| Krysto | Completo en sus tareas vencidas | Protocolo, plan, MDP, matriz, notebook común y validación del líder. | Repetir la validación tras integrar los siguientes avances. |
| Luis Ángel | Completo para el avance del viernes | `entorno_almacen.py` y `probar_entorno.py`; entorno válido y compatible. | Entregar el sábado `agentes_rl.py` y la primera corrida reproducible. |
| Juan José | Pendiente para el avance del viernes | No existe un archivo suyo dentro de `src/entrega3/` al corte. | Subir `percepcion_cnn.py`, cargar `modelos/stylebot_cnn_final.keras` sin reentrenar y devolver categoría y confianza. |
| Ángel Uriel | Avance parcial | Subió `03_evaluacion_rl.ipynb`, con agente aleatorio, métricas y exportaciones, pero está sin ejecutar. | Usar 50 episodios con semillas 1000–1049, ejecutar el notebook y conservar los nombres comunes de resultados. Las comparaciones con Q-Learning y Dyna-Q esperan los archivos de Luis. |
| Jacqueline | Avance parcial | Amplió el informe con la estructura de la Entrega 3. | Crear `notebooks/entrega3/E3_StyleBot.ipynb`; corregir “Mendosa Gaytan” por “Mendoza Gaytán” en el informe. |

## Archivos que todavía faltan

- `src/entrega3/agentes_rl.py`
- `src/entrega3/percepcion_cnn.py`
- `notebooks/entrega3/01_qlearning_dynaq.ipynb`
- `notebooks/entrega3/02_integracion_cnn_rl.ipynb`
- `notebooks/entrega3/E3_StyleBot.ipynb`
- `modelos/entrega3/q_table_qlearning.npz`
- `modelos/entrega3/q_table_dynaq.npz`
- CSV y gráficas finales en `resultados/entrega3/`
- GIF o video de 50 prendas en `demo/entrega3/`

## Verificación final pendiente

- [ ] Q-Learning y Dyna-Q entrenados con dos configuraciones como mínimo.
- [ ] Integración con `modelos/stylebot_cnn_final.keras` comprobada.
- [ ] Comparación contra el agente aleatorio terminada.
- [ ] Errores de CNN y de política separados.
- [ ] Demostración final de 50 prendas generada.
- [ ] `E3_StyleBot.ipynb` ejecutado completo sin errores.
- [ ] Sección de 4 a 6 páginas terminada en el informe.
- [ ] PDF, GIF o video y repositorio revisados antes de la entrega.

No se asigna una sanción nueva mediante esta revisión. Primero se debe comprobar si venció el plazo correspondiente y si existió una prórroga autorizada. La ausencia o el retraso quedan registrados aquí para aplicar después la regla acordada por el equipo.
