"""Inferencia de StyleBot: guardar, cargar y usar modelos .keras.

Responsable: Luis Angel Mendoza Gaytán (Entrega 2, inferencia e integración).
Uso desde la terminal, en una sesión limpia:
    python inferencia_stylebot.py --modelo modelos/stylebot_cnn_final.keras --n 5

Formato para la Entrega 3 (simulación del robot):
    Entrada: una imagen de 28 x 28 en escala de grises, prenda clara sobre fondo oscuro,
             como arreglo de NumPy (28, 28) o (28, 28, 1), uint8 de 0 a 255 o float de 0 a 1.
    Salida:  diccionario con categoria (0-9), nombre, probabilidad, top (3 opciones) y tiempo_ms.
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

CATEGORIAS = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Tenis", "Bolso", "Botín",
]


def guardar_modelo(modelo, ruta):
    """Guarda el modelo completo en formato .keras y devuelve la ruta."""
    ruta = os.fspath(ruta)
    if not ruta.endswith(".keras"):
        raise ValueError("La ruta debe terminar en .keras")
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    modelo.save(ruta)
    return ruta


def cargar_modelo(ruta):
    """Carga un modelo .keras guardado con guardar_modelo o con modelo.save()."""
    ruta = os.fspath(ruta)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe el modelo: {ruta}")
    return tf.keras.models.load_model(ruta)


def tamano_modelo_mb(ruta):
    """Tamaño del archivo guardado, en megabytes."""
    return os.path.getsize(ruta) / (1024 ** 2)


def preparar_imagenes(imagenes, modelo=None):
    """Convierte imágenes al formato que espera el modelo.

    Acepta una imagen (28, 28) o (28, 28, 1), o un lote (n, 28, 28) o (n, 28, 28, 1),
    con píxeles uint8 de 0 a 255 o float de 0 a 1. Si el modelo espera otra entrada
    (por ejemplo MobileNetV2 con 32 x 32 x 3), la convierte a RGB y la redimensiona.
    """
    original = np.asarray(imagenes)
    x = original.astype("float32")
    if x.shape in [(28, 28), (28, 28, 1)]:
        x = x.reshape(1, 28, 28, 1)
    elif x.ndim == 3 and x.shape[1:] == (28, 28):
        x = x[..., np.newaxis]
    elif not (x.ndim == 4 and x.shape[1:] == (28, 28, 1)):
        raise ValueError(
            f"Forma no válida {original.shape}: se espera (28, 28), (28, 28, 1) o un lote de esas imágenes."
        )
    if np.issubdtype(original.dtype, np.integer) or x.max() > 1.0:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)

    if modelo is not None:
        alto, ancho, canales = modelo.input_shape[1:4]
        if canales == 3:
            x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x))
        if (alto, ancho) != (28, 28):
            x = tf.image.resize(x, [alto, ancho])
        x = np.asarray(x, dtype="float32")
    return x


def predecir_prenda(modelo, imagen, top=3):
    """Recibe UNA imagen de 28 x 28 y devuelve categoría, nombre de la prenda y probabilidad."""
    x = preparar_imagenes(imagen, modelo)
    if len(x) != 1:
        raise ValueError("predecir_prenda recibe una sola imagen; para varias usa predecir_lote.")
    inicio = time.perf_counter()
    probabilidades = np.asarray(modelo(x, training=False))[0]
    tiempo_ms = 1000 * (time.perf_counter() - inicio)
    orden = np.argsort(probabilidades)[::-1][:top]
    return {
        "categoria": int(orden[0]),
        "nombre": CATEGORIAS[int(orden[0])],
        "probabilidad": round(float(probabilidades[orden[0]]), 4),
        "top": [
            {"categoria": int(i), "nombre": CATEGORIAS[int(i)], "probabilidad": round(float(probabilidades[i]), 4)}
            for i in orden
        ],
        "tiempo_ms": round(tiempo_ms, 2),
    }


def predecir_lote(modelo, imagenes, tamano_lote=256):
    """Predice varias imágenes. Devuelve categorías, probabilidad de la categoría y matriz de probabilidades."""
    x = preparar_imagenes(imagenes, modelo)
    probabilidades = modelo.predict(x, batch_size=tamano_lote, verbose=0)
    return probabilidades.argmax(axis=1), probabilidades.max(axis=1), probabilidades


def medir_tiempo_prediccion(modelo, imagenes, repeticiones=100, calentamiento=10, tamano_lote=256):
    """Tiempo promedio de predicción.

    - Individual: una imagen por llamada, como la recibiría el robot. Se descartan las
      primeras llamadas de calentamiento.
    - En lote: todas las imágenes con predict y lotes de tamano_lote, dividido entre el número de imágenes.
    El tiempo individual incluye solo el modelo; la conversión de imágenes se hace antes.
    """
    x = preparar_imagenes(imagenes, modelo)
    for i in range(calentamiento):
        np.asarray(modelo(x[i % len(x)][np.newaxis], training=False))
    tiempos = []
    for i in range(repeticiones):
        imagen = x[i % len(x)][np.newaxis]
        inicio = time.perf_counter()
        np.asarray(modelo(imagen, training=False))
        tiempos.append(time.perf_counter() - inicio)
    modelo.predict(x[:tamano_lote], batch_size=tamano_lote, verbose=0)
    inicio = time.perf_counter()
    modelo.predict(x, batch_size=tamano_lote, verbose=0)
    total = time.perf_counter() - inicio
    ms = np.array(tiempos) * 1000
    return {
        "ms_por_imagen_individual": float(ms.mean()),
        "ms_desviacion_individual": float(ms.std()),
        "ms_por_imagen_en_lote": float(1000 * total / len(x)),
        "imagenes_por_segundo_en_lote": float(len(x) / total),
    }


def _principal():
    """Prueba en sesión limpia: carga el modelo y predice imágenes de validación sin variables previas."""
    parser = argparse.ArgumentParser(description="Prueba de inferencia de StyleBot en una sesión limpia")
    parser.add_argument("--modelo", default="modelos/stylebot_cnn_final.keras")
    parser.add_argument("--n", type=int, default=5, help="Número de imágenes de validación")
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split

    inicio = time.perf_counter()
    modelo = cargar_modelo(args.modelo)
    print(f"Modelo cargado: {args.modelo} ({tamano_modelo_mb(args.modelo):.2f} MB) "
          f"en {time.perf_counter() - inicio:.2f} s")

    (X_base, y_base), _ = tf.keras.datasets.fashion_mnist.load_data()
    _, idx_val = train_test_split(np.arange(len(y_base), dtype=np.int64), test_size=0.20,
                                  random_state=42, stratify=y_base)
    for i in idx_val[: args.n]:
        resultado = predecir_prenda(modelo, X_base[i])  # imagen original uint8 de 0 a 255
        resultado["categoria_real"] = int(y_base[i])
        resultado["nombre_real"] = CATEGORIAS[int(y_base[i])]
        print(json.dumps(resultado, ensure_ascii=False))


if __name__ == "__main__":
    _principal()
