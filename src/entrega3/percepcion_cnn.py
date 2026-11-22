
"""Módulo de percepción CNN para StyleBot (Entrega 3).

Responsable: Juan José Aranda Flores (redes neuronales CNN).

Carga el modelo congelado `modelos/stylebot_cnn_final.keras` sin reentrenarlo
ni modificar sus pesos, recibe imágenes de 28x28 y devuelve la categoría
predicha, la confianza y la coordenada del estante objetivo según
`datos/entrega3/protocolo_entrega3.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

RUTA_PROTOCOLO_DEFECTO = Path("datos/entrega3/protocolo_entrega3.json")
RUTA_MODELO_DEFECTO = Path("modelos/stylebot_cnn_final.keras")

CATEGORIAS_DEFECTO = [
    "Camiseta/top",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Tenis",
    "Bolso",
    "Botín",
]

ESTANTES_DEFECTO = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    3: (0, 3),
    4: (0, 4),
    5: (0, 5),
    6: (0, 6),
    7: (0, 7),
    8: (4, 0),
    9: (4, 7),
}


def cargar_protocolo(ruta_protocolo: str | Path = RUTA_PROTOCOLO_DEFECTO) -> Dict[str, Any]:
    """Lee el archivo JSON del protocolo común de la Entrega 3."""
    ruta = Path(ruta_protocolo)
    if ruta.exists():
        with ruta.open("r", encoding="utf-8") as archivo:
            return json.load(archivo)
    return {
        "version": "entrega3-1.0",
        "semilla": 42,
        "semilla_demo": 2026,
        "categorias": CATEGORIAS_DEFECTO,
        "estantes": {str(k): list(v) for k, v in ESTANTES_DEFECTO.items()},
        "prendas_demo": 50,
        "max_pasos_por_prenda": 60,
        "modelo_cnn": str(RUTA_MODELO_DEFECTO),
    }


def cargar_modelo_cnn(
    ruta_modelo: str | Path | None = None,
    ruta_protocolo: str | Path = RUTA_PROTOCOLO_DEFECTO,
) -> tf.keras.Model:
    """Carga la CNN final entrenada y congela sus pesos (solo inferencia)."""
    if ruta_modelo is None:
        protocolo = cargar_protocolo(ruta_protocolo)
        ruta_modelo = protocolo.get("modelo_cnn", str(RUTA_MODELO_DEFECTO))

    ruta = Path(ruta_modelo)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el modelo CNN congelado en: {ruta}")

    modelo = tf.keras.models.load_model(ruta, compile=False)
    modelo.trainable = False
    return modelo


def preprocesar_imagenes(imagenes: np.ndarray) -> np.ndarray:
    """Normaliza y ajusta imágenes 28x28 a formato (N, 28, 28, 1) en [0, 1].

    Acepta una sola imagen (28, 28) o (28, 28, 1), o un lote (N, 28, 28)
    o (N, 28, 28, 1). Evita dividir entre 255 si ya está normalizada.
    """
    arr = np.asarray(imagenes, dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape != (28, 28):
            raise ValueError(f"Se esperaba imagen de 28x28, se recibió {arr.shape}")
        arr = arr[np.newaxis, ..., np.newaxis]
    elif arr.ndim == 3:
        if arr.shape == (28, 28, 1):
            arr = arr[np.newaxis, ...]
        elif arr.shape[1:] == (28, 28):
            arr = arr[..., np.newaxis]
        else:
            raise ValueError(f"Forma de entrada 3D no válida para CNN 28x28: {arr.shape}")
    elif arr.ndim == 4:
        if arr.shape[1:] != (28, 28, 1):
            raise ValueError(f"Se esperaba lote (N, 28, 28, 1), se recibió {arr.shape}")
    else:
        raise ValueError(f"Dimensión no soportada para imagen: {arr.shape}")

    if arr.size > 0 and float(np.max(arr)) > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def obtener_estante_objetivo(
    categoria_id: int,
    protocolo: Dict[str, Any] | None = None,
) -> Tuple[int, int]:
    """Devuelve la coordenada (fila, columna) del estante para una categoría (0-9)."""
    categoria_id = int(categoria_id)
    if not (0 <= categoria_id < 10):
        raise ValueError(f"La categoría debe estar entre 0 y 9, se recibió {categoria_id}")

    if protocolo is not None and "estantes" in protocolo:
        coord = protocolo["estantes"][str(categoria_id)]
        return (int(coord[0]), int(coord[1]))
    return ESTANTES_DEFECTO[categoria_id]


def predecir_prenda(
    modelo: tf.keras.Model,
    imagen: np.ndarray,
    protocolo: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Clasifica una sola imagen 28x28 y devuelve categoría, confianza y estante."""
    protocolo = protocolo or cargar_protocolo()
    categorias = protocolo.get("categorias", CATEGORIAS_DEFECTO)

    entrada = preprocesar_imagenes(imagen)
    if entrada.shape[0] != 1:
        raise ValueError("predecir_prenda recibe una sola imagen. Para lotes usa predecir_lote.")

    probabilidades = modelo(entrada, training=False).numpy()[0]
    categoria_id = int(np.argmax(probabilidades))
    confianza = float(probabilidades[categoria_id])
    estante = obtener_estante_objetivo(categoria_id, protocolo)

    return {
        "categoria_id": categoria_id,
        "categoria": categorias[categoria_id],
        "confianza": round(confianza, 6),
        "estante_objetivo": estante,
        "probabilidades": [float(p) for p in probabilidades],
    }


def predecir_lote(
    modelo: tf.keras.Model,
    imagenes: np.ndarray,
    protocolo: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predice un lote de imágenes 28x28 devolviendo (predicciones, confianzas, probabilidades)."""
    entrada = preprocesar_imagenes(imagenes)
    probabilidades = modelo.predict(entrada, verbose=0)
    predicciones = np.argmax(probabilidades, axis=1).astype(np.int64)
    confianzas = np.max(probabilidades, axis=1).astype(np.float32)
    return predicciones, confianzas, probabilidades


def seleccionar_50_prendas(
    semilla_demo: int | None = None,
    n_prendas: int | None = None,
    protocolo: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Selecciona 50 imágenes reproducibles del conjunto oficial de prueba de Fashion-MNIST.

    Toma 5 prendas por cada una de las 10 categorías usando `semilla_demo` (2026)
    para asegurar representación equilibrada en todos los estantes del almacén.
    """
    protocolo = protocolo or cargar_protocolo()
    semilla = int(protocolo.get("semilla_demo", 2026) if semilla_demo is None else semilla_demo)
    total = int(protocolo.get("prendas_demo", 50) if n_prendas is None else n_prendas)

    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    rng = np.random.default_rng(semilla)

    if total % 10 == 0:
        por_clase = total // 10
        indices_sel = []
        for clase in range(10):
            idx_clase = np.where(y_test == clase)[0]
            elegidos = rng.choice(idx_clase, size=por_clase, replace=False)
            indices_sel.append(elegidos)
        indices = np.concatenate(indices_sel)
        rng.shuffle(indices)
    else:
        indices = rng.choice(len(y_test), size=total, replace=False)

    imagenes = x_test[indices].astype(np.float32) / 255.0
    etiquetas = y_test[indices].astype(np.int64)
    return indices.astype(np.int64), imagenes, etiquetas


def generar_csv_predicciones_50(
    modelo: tf.keras.Model | None = None,
    ruta_salida: str | Path = "datos/entrega3/predicciones_50_prendas.csv",
    ruta_protocolo: str | Path = RUTA_PROTOCOLO_DEFECTO,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Genera el CSV oficial con las 50 prendas, sus predicciones, confianzas y errores."""
    protocolo = cargar_protocolo(ruta_protocolo)
    categorias = protocolo.get("categorias", CATEGORIAS_DEFECTO)

    if modelo is None:
        modelo = cargar_modelo_cnn(ruta_protocolo=ruta_protocolo)

    indices_test, imagenes, etiquetas_reales = seleccionar_50_prendas(protocolo=protocolo)
    predicciones, confianzas, _ = predecir_lote(modelo, imagenes, protocolo=protocolo)

    filas = []
    for orden in range(len(etiquetas_reales)):
        real_id = int(etiquetas_reales[orden])
        pred_id = int(predicciones[orden])
        conf = float(confianzas[orden])
        est_real = obtener_estante_objetivo(real_id, protocolo)
        est_pred = obtener_estante_objetivo(pred_id, protocolo)
        es_error = pred_id != real_id

        filas.append(
            {
                "id_prenda": orden,
                "indice_test": int(indices_test[orden]),
                "etiqueta_real": real_id,
                "categoria_real": categorias[real_id],
                "prediccion": pred_id,
                "categoria_predicha": categorias[pred_id],
                "confianza": round(conf, 6),
                "es_error": es_error,
                "estante_real": f"({est_real[0]}, {est_real[1]})",
                "estante_objetivo": f"({est_pred[0]}, {est_pred[1]})",
            }
        )

    df = pd.DataFrame(filas)
    ruta_csv = Path(ruta_salida)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_csv, index=False, encoding="utf-8")
    return df, imagenes, etiquetas_reales, predicciones


if __name__ == "__main__":
    modelo_cnn = cargar_modelo_cnn()
    df_50, _, _, _ = generar_csv_predicciones_50(modelo=modelo_cnn)
    errores = int(df_50["es_error"].sum())
    exactitud = float((~df_50["es_error"]).mean())
    print(f"Predicciones generadas: {len(df_50)} prendas | Exactitud: {exactitud:.4f} | Errores CNN: {errores}")
