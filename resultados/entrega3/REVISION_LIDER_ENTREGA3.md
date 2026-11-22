# Revisión del líder — Entrega 3

**Responsable:** Krysto Alberto Adan Jauregui Zúñiga

**Corte de la revisión:** sábado 21 de noviembre de 2026, hasta el commit `4bf5815`.

**Próximo cierre del equipo:** domingo 22 de noviembre a las 18:00 para los entregables técnicos y a las 20:00 para la integración documental.

## Trabajo realizado por el líder

- [x] Protocolo común, MDP, matriz de requisitos, reglas y rutas publicados.
- [x] Entorno 8×8 ejecutado y compatible con categorías, acciones, recompensas y 1,408 estados.
- [x] `percepcion_cnn.py` revisado sin importar ni modificar sus pesos.
- [x] `02_integracion_cnn_rl.ipynb` comprobado: 4 de 4 celdas ejecutadas y sin errores.
- [x] `03_evaluacion_rl.ipynb` comprobado: 2 de 2 celdas ejecutadas y sin errores.
- [x] Semillas 1000–1049 y 50 episodios verificados en la evaluación.
- [x] Nombres comunes de CSV y gráficas comprobados.
- [x] Validador del líder ampliado para revisar automáticamente los componentes nuevos.
- [ ] Repetir la validación cuando Luis entregue los agentes, el notebook y las tablas Q.
- [ ] Comprobar el notebook final integrado y su demostración antes del cierre.

## Estado comprobado por integrante

| Integrante | Estado al corte | Evidencia verificada | Qué falta |
|---|---|---|---|
| Krysto | Completo para el sábado | Protocolo, organización, revisión y validación técnica actualizada. | Volver a validar después de la entrega de Luis y registrar las correcciones del lunes. |
| Luis Ángel | Incompleto y fuera del plazo del sábado | El entorno del viernes continúa válido, pero su último avance de Entrega 3 sigue siendo del 19 de noviembre. | Faltan `src/entrega3/agentes_rl.py`, `notebooks/entrega3/01_qlearning_dynaq.ipynb`, la primera corrida reproducible y las dos tablas Q. |
| Juan José | Trabajo técnico completo, pero entregado tarde | `percepcion_cnn.py`, CSV de 50 prendas y `02_integracion_cnn_rl.ipynb` ejecutado. Resultado: 49/50 correctas, 1 fallo CNN y 0 fallos de política. | Conservar estos archivos y atender solamente las correcciones de integración que aparezcan. |
| Ángel Uriel | Completo dentro del plazo del sábado | Evaluación ejecutada con 50 episodios, semillas 1000–1049, agente aleatorio, CSV y gráficas comunes. | Integrar Q-Learning y Dyna-Q cuando Luis entregue sus tablas y resultados. |
| Jacqueline | Estructura creada, integración pendiente | `E3_StyleBot.ipynb` existe con 47 celdas de texto y las secciones generales. | El notebook aún tiene 0 celdas de código. Debe integrar resultados disponibles y avanzar el informe para el domingo a las 20:00. También debe corregir “Mendosa Gaytan” por “Mendoza Gaytán”. |

## Retrasos comprobados

- Luis no entregó el avance del sábado programado para las 16:00.
- Juan publicó `percepcion_cnn.py` aproximadamente a las 22:48 y el notebook de integración aproximadamente a las 23:12; el plazo era a las 16:00.
- Jacqueline creó la estructura inicial después del plazo del viernes y continuó ampliándola el sábado. Su siguiente integración vence el domingo a las 20:00.
- Ángel publicó los resultados del agente aleatorio antes del plazo del sábado.

Esta revisión registra los retrasos, pero no cambia calificaciones por sí sola. La sanción debe aplicarse según la regla previamente comunicada por el líder y considerando únicamente prórrogas autorizadas antes del vencimiento.

## Archivos que todavía faltan

- `src/entrega3/agentes_rl.py`
- `notebooks/entrega3/01_qlearning_dynaq.ipynb`
- `modelos/entrega3/q_table_qlearning.npz`
- `modelos/entrega3/q_table_dynaq.npz`
- resultados finales que comparen Q-Learning, Dyna-Q y agente aleatorio
- código integrado y ejecutado en `notebooks/entrega3/E3_StyleBot.ipynb`
- GIF o video de 50 prendas en `demo/entrega3/`

## Verificación final pendiente

- [ ] Q-Learning y Dyna-Q entrenados con dos configuraciones como mínimo.
- [ ] Comparación final contra el agente aleatorio terminada.
- [x] Integración de la CNN congelada con el entorno comprobada.
- [x] Errores de CNN y de política separados en la prueba de Juan.
- [ ] Demostración final de 50 prendas generada.
- [ ] `E3_StyleBot.ipynb` integrado y ejecutado completo sin errores.
- [ ] Sección de 4 a 6 páginas terminada en el informe.
- [ ] PDF, GIF o video y repositorio revisados antes de la entrega.
