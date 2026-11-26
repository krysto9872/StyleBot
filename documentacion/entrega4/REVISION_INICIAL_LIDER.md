# Revisión inicial del líder — Entrega 4

**Proyecto:** StyleBot  
**Responsable de la revisión:** Krysto Alberto Adan Jauregui Zúñiga  
**Fecha:** jueves 26 de noviembre de 2026  
**Momento de revisión:** antes del límite interno de las 18:00

## Resultado general

La estructura inicial de la Entrega 4 está creada y publicada. El repositorio cuenta con las reglas comunes, el plan, el checklist, el contrato de integración, la coevaluación y la plantilla del notebook final. Los modelos necesarios de las entregas anteriores también están disponibles.

Sin embargo, la integración final todavía no es ejecutable: el notebook E4 es una plantilla sin resultados y aún no se observan en GitHub los entregables técnicos asignados hoy a Ángel Uriel, Juan José, Luis Ángel y Jacqueline. Como el plazo de hoy todavía no había vencido al realizar esta revisión, su estado se registra como **pendiente dentro del plazo**, no como atraso.

## Evidencia verificada en GitHub

| Elemento revisado | Estado | Evidencia u observación |
| --- | --- | --- |
| Plan de trabajo | Completo | documentacion/entrega4/PLAN_TRABAJO_ENTREGA4.md |
| Checklist final | Completo | documentacion/entrega4/CHECKLIST_FINAL.md |
| Contrato de integración | Completo | documentacion/entrega4/CONTRATO_INTEGRACION.md |
| Coevaluación final | Preparada | documentacion/entrega4/COEVALUACION_FINAL.md; faltan las calificaciones de E3 y E4 al cierre. |
| README principal | Actualizado | Incluye objetivo, responsables, estructura y ejecución de la Entrega 4. |
| Modelo CNN final | Disponible | modelos/stylebot_cnn_final.keras |
| Tablas Q | Disponibles | modelos/entrega3/q_table_qlearning.npz y q_table_dynaq.npz |
| Notebook E4 | Plantilla creada | 14 celdas, 4 celdas de código, 0 ejecutadas y 0 salidas. |
| Funciones finales en src/entrega4 | Pendiente | Solamente existe README.md. |
| Resultados de Entrega 4 | Pendiente | Solamente existe README.md. |
| Demostración de Entrega 4 | Pendiente | Solamente existe README.md. |
| Presentación final | Pendiente | Solamente existe README.md. |
| Aplicación web opcional | Pendiente | Solamente existe README_APP.md; no existe app.py. |
| requirements.txt | Requiere revisión posterior | Todavía no incluye Streamlit; se agregará solamente si se desarrolla la aplicación opcional. |
| Informe técnico | Avance disponible | documentacion/ProyectoFinal_StyleBot.docx fue actualizado el 25 de noviembre; falta revisar su contenido final y exportarlo a PDF. |

## Estado por integrante

### Krysto Alberto Adan Jauregui Zúñiga

**Estado de hoy: completo.**

Ya realizó:

- Estructura de carpetas de la Entrega 4.
- Plan de trabajo y reglas comunes.
- Checklist y contrato de integración.
- Plantilla de E4_StyleBot.ipynb.
- Actualización del README.
- Preparación de la coevaluación.
- Esta revisión inicial del líder.

Siguiente actividad: integrar el viernes 27 las funciones y resultados que entregue el equipo en notebooks/entrega4/E4_StyleBot.ipynb.

### Ángel Uriel Martínez Godinez

**Estado: pendiente dentro del plazo de hoy.**

Debe publicar:

- Resumen de los resultados definitivos de modelos clásicos, PCA, K-Means y recomendador.
- Definición de las métricas que sí pueden compararse y de las que no.
- Borrador de la tabla comparativa global.

La revisión no encontró todavía notebooks/entrega4/01_comparacion_global.ipynb ni archivos nuevos en resultados/entrega4.

### Juan José Aranda Flores

**Estado: pendiente dentro del plazo de hoy.**

Debe publicar src/entrega4/inferencia_cnn.py con una función que:

- Reciba una imagen.
- La convierta a 28 por 28 en escala de grises.
- Cargue modelos/stylebot_cnn_final.keras sin volver a entrenarlo.
- Devuelva categoría, confianza y datos procesados.

La revisión encontró el modelo CNN, pero src/entrega4 todavía contiene solamente README.md.

### Luis Ángel Mendoza Gaytán

**Estado: pendiente dentro del plazo de hoy.**

Debe publicar src/entrega4/simulacion_final.py con una función que:

- Reciba la categoría predicha y el agente elegido.
- Cargue la tabla Q correspondiente sin modificarla.
- Devuelva recorrido, acciones, recompensa, pasos y éxito.

Las dos tablas Q están disponibles, pero src/entrega4 todavía contiene solamente README.md.

### Jacqueline Soto

**Estado: pendiente de confirmación dentro del plazo de hoy.**

El Word del proyecto tiene un avance del 25 de noviembre. Debe confirmar mediante un commit que realizó las correcciones iniciales de la Entrega 4:

- Completar la sección 5.4.
- Eliminar afirmaciones antiguas que describen la integración como trabajo futuro.
- Incorporar los resultados integrados ya existentes.
- Ordenar el documento según las diez secciones solicitadas.

Todavía no se requiere el PDF final, la presentación terminada ni la aplicación web. Esos entregables tienen fechas posteriores.

## Dependencias confirmadas

- Juan José y Luis Ángel pueden trabajar desde ahora de manera independiente utilizando el contrato de integración.
- Ángel Uriel puede preparar su comparación con los resultados ya existentes; no necesita esperar la aplicación web.
- Jacqueline puede corregir la estructura del informe sin esperar a los resultados finales; después actualizará las cifras definitivas.
- Krysto necesita las funciones de Juan y Luis para realizar la primera integración completa.
- La aplicación web depende de las funciones finales de CNN y simulación, por lo que no debe iniciarse antes de validar esos módulos.

## Riesgos detectados

| Riesgo | Consecuencia | Acción preventiva |
| --- | --- | --- |
| Funciones de CNN o simulación entregadas tarde | Retrasa el notebook E4 y la aplicación. | Revisar commits al vencer el plazo de hoy y solicitar corrección inmediata. |
| Nombres, parámetros o salidas diferentes al contrato | Falla la integración. | Validar cada función contra CONTRATO_INTEGRACION.md. |
| Reentrenar modelos dentro del notebook final | Ejecución lenta o imposible en la demostración. | Cargar únicamente el modelo CNN y las tablas Q guardadas. |
| Informe con resultados antiguos | Inconsistencias con notebook y presentación. | Jacqueline debe comparar todas las cifras antes del cierre. |
| Dedicar tiempo temprano a Streamlit | Riesgo para entregables obligatorios. | Trabajar la web solamente después de una ejecución integral correcta. |
| Notebook sin salidas visibles | No demuestra funcionamiento. | Ejecutarlo desde una sesión limpia de Colab antes de descargarlo. |

## Acciones inmediatas del líder

1. A las 18:00 revisar los commits de los cuatro responsables.
2. Registrar como atraso solamente lo que siga ausente después del plazo.
3. Verificar que cada archivo respete el contrato de integración.
4. Informar a cada integrante una corrección concreta, sin pedirle repetir trabajo correcto.
5. Integrar el viernes las funciones disponibles en E4_StyleBot.ipynb.
6. Mantener la aplicación web como actividad opcional hasta completar el flujo obligatorio.

## Conclusión

El trabajo del líder programado para iniciar la Entrega 4 está completo. El proyecto cuenta con una base organizada y con los modelos anteriores necesarios. El siguiente punto crítico es recibir y validar los cuatro avances técnicos y documentales de hoy; después podrá comenzar la integración real del notebook final.
