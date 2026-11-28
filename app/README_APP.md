# Aplicación web opcional de StyleBot

La aplicación con Streamlit es una actividad adicional para obtener puntos extra. Clasifica una prenda y muestra la decisión del agente; no genera imágenes.

## Avance disponible

La primera versión de la interfaz está en [app.py](app.py). Fue preparada por Krysto como apoyo a Jacqueline y reutiliza los módulos de CNN y simulación del equipo.

Incluye selección de una de las 50 muestras oficiales o carga de PNG/JPG, categoría y confianza de la CNN, elección de Q-Learning o Dyna-Q, estante objetivo, mapa con recorrido, pasos, recompensa, eventos y descarga del resultado en JSON.

Cada ejecución representa una sola prenda y comienza con el almacén vacío. No sustituye la evaluación por lotes ni la demostración completa del notebook E4.

## Cómo abrirla

1. Descargar o clonar el repositorio completo y descomprimirlo. Descargar únicamente app.py no es suficiente.
2. Abrir una terminal en la carpeta principal StyleBot, donde está requirements.txt.
3. Instalar las dependencias del proyecto:

   ~~~bash
   python -m pip install -r requirements.txt
   ~~~

4. Iniciar la aplicación:

   ~~~bash
   python -m streamlit run app/app.py
   ~~~

5. Abrir la dirección local que muestre la terminal. Mantener la terminal abierta mientras se utiliza la página.

La primera carga de muestras necesita Internet para descargar Fashion-MNIST. La aplicación usa modelos ya entrenados; no vuelve a entrenarlos. Subir el código a GitHub no publica automáticamente una página accesible en Internet.

## Archivos que deben estar presentes

- app/app.py
- src/entrega4/inferencia_cnn.py y src/entrega4/simulacion_final.py, junto con sus módulos importados.
- src/entrega3/entorno_almacen.py
- modelos/stylebot_cnn_final.keras
- modelos/entrega3/q_table_qlearning.npz
- modelos/entrega3/q_table_dynaq.npz
- datos/entrega3/protocolo_entrega3.json
- datos/entrega3/predicciones_50_prendas.csv

## Conexión con el trabajo del equipo

La interfaz llama a predecir_imagen del módulo de Juan José y entrega el índice de categoría a simular_acomodo del módulo de simulación. No hace falta reemplazar el módulo de CNN para usar esta versión.

Se pasan rutas absolutas al cargador de la CNN y se ajustan las rutas del módulo de simulación desde la interfaz para localizar protocolo y tablas en la raíz del repositorio. Esto evita depender de la carpeta desde la que se ejecute el comando.

## Interpretación del resultado

Con una muestra oficial se utiliza su etiqueta real para evaluar el resultado. Con una imagen subida por el usuario no se conoce la categoría real: llegar al estante de la categoría predicha no demuestra que la clasificación sea correcta. Las fotografías distintas de Fashion-MNIST pueden producir predicciones equivocadas incluso con confianza alta.

## Pruebas pendientes antes de entregar

La sintaxis Python de app.py fue comprobada. La ejecución completa de la interfaz con los modelos todavía está pendiente; no se declara terminada ni validada por esta comprobación.

| Prueba | Qué comprobar | Evidencia |
| --- | --- | --- |
| Muestra oficial con Dyna-Q | Aparecen imagen procesada, categoría, estante, mapa y métricas; no hay excepciones. | Captura y JSON descargado. |
| Misma muestra con Q-Learning | Se utiliza el agente elegido y se actualiza el recorrido. | Captura y JSON descargado. |
| Imagen PNG/JPG propia | Se procesa y aparece el aviso sobre categoría real desconocida. | Captura. |
| Archivo inválido o demasiado grande | Se muestra un mensaje comprensible y no un resultado anterior. | Captura del mensaje. |
| Cambio de muestra, agente o semilla | Se solicita ejecutar nuevamente y no se confunden resultados anteriores. | Registro de prueba. |

Jacqueline puede incorporar estas evidencias y las instrucciones al informe. Juan José puede revisar la predicción y Luis Ángel la simulación. El líder revisará el flujo completo y registrará los problemas encontrados antes de la entrega.
