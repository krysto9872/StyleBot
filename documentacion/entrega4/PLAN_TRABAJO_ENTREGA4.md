# Plan de trabajo de la cuarta y última entrega de StyleBot

**Entrega oficial:** miércoles 2 de diciembre de 2026 antes de las 19:00.  
**Cierre interno:** martes 1 de diciembre de 2026 a las 19:00.  
**Responsable de coordinación:** Krysto Alberto Adan Jauregui Zúñiga.

## Objetivo

Integrar en un solo flujo reproducible la preparación de datos, los modelos clásicos, el análisis no supervisado, la CNN y los agentes Q-Learning y Dyna-Q. El resultado debe incluir un notebook final ejecutable, el informe técnico en PDF, la presentación, la coevaluación y una demostración funcional.

La aplicación web con Streamlit es opcional y se realizará únicamente después de completar los entregables obligatorios.

## Estructura de la Entrega 4

~~~text
StyleBot/
├── app/                         Aplicación Streamlit opcional
├── demo/entrega4/              Evidencia de la demostración final
├── documentacion/entrega4/     Plan, lista de verificación y coevaluación
├── notebooks/entrega4/         Notebook final y comparación global
├── presentacion/               Diapositivas de la exposición
├── resultados/entrega4/        Tabla global, gráficas y pruebas finales
└── src/entrega4/               Funciones reutilizables de CNN y simulación
~~~

## Responsables y entregables

| Responsable | Entregable principal | Fecha interna |
|---|---|---|
| Krysto | Estructura, integración, E4_StyleBot.ipynb, README, revisión y entrega | 1 dic., 19:00 |
| Ángel Uriel | Comparación global de modelos, tabla, gráfica y análisis | 30 nov., 17:00 |
| Juan José | Función reutilizable de inferencia CNN y pruebas | 30 nov., 17:00 |
| Luis Ángel | Función reutilizable de simulación RL y pruebas | 30 nov., 17:00 |
| Jacqueline | Informe final, PDF, presentación, demostración, coevaluación e interfaz web opcional | 1 dic., 17:00 |

## Calendario común

| Fecha | Cierre esperado |
|---|---|
| Jueves 26 | Estructura, reglas y contratos de integración publicados. |
| Viernes 27 | Primeras funciones de CNN y RL, comparación global y borradores de informe y presentación. |
| Sábado 28 | Primera ejecución integrada y aplicación web local opcional. |
| Domingo 29 | Resultados globales, documentación, referencias, conclusiones y coevaluación. |
| Lunes 30 | Notebook ejecutado en sesión limpia, PDF y presentación terminados. |
| Martes 1 | Correcciones, ensayo final y congelamiento del repositorio a las 19:00. |
| Miércoles 2 | Solamente revisión de enlaces y entrega antes de las 19:00. |

## Reglas comunes

1. No modificar ni borrar los entregables de las Entregas 1, 2 y 3.
2. Cada integrante trabajará en los archivos de su rol y publicará sus propios commits.
3. La CNN y las tablas Q se cargarán desde los modelos guardados; no se volverán a entrenar durante la demostración final.
4. Se conservarán las semillas establecidas en las entregas anteriores.
5. Las funciones integradas devolverán diccionarios con campos documentados en CONTRATO_INTEGRACION.md.
6. El notebook final deberá funcionar desde una sesión limpia de Google Colab.
7. Los números del notebook, informe, README y presentación deben coincidir.
8. La aplicación web es opcional y no puede retrasar el PDF, notebook, presentación o coevaluación.
9. Después del martes 1 a las 19:00 solamente el líder podrá autorizar correcciones.

## Entregables finales obligatorios

- documentacion/ProyectoFinal_StyleBot.pdf.
- notebooks/entrega4/E4_StyleBot.ipynb ejecutado completamente.
- Repositorio reproducible con README y dependencias actualizadas.
- Presentación final de diez minutos con participación de los cinco integrantes.
- Formato de coevaluación del equipo.
- Demostración funcional de 50 prendas.
- Evidencias de participación mediante commits.
- Correcciones recibidas durante las tres entregas anteriores.
