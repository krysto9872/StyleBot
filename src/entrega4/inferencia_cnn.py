"""Módulo reutilizable de inferencia CNN para StyleBot (Entrega 4).

Responsable: Juan José Aranda Flores (redes neuronales CNN).

Carga el modelo congelado `modelos/stylebot_cnn_final.keras` (Arquitectura C)
sin reentrenarlo. Acepta imágenes desde archivos, objetos PIL, bytes/UploadedFile
de Streamlit o arreglos NumPy (escala de grises, RGB o RGBA y cualquier tamaño),
las convierte a 28x28 en escala de grises normalizadas en [0, 1] y devuelve
la categoría predicha, confianza, estante objetivo y los datos procesados.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

RUTA_MODELO_DEFECTO = Path("modelos/stylebot_cnn_final.keras")
RUTA_PROTOCOLO_DEFECTO = Path("datos/entrega3/protocolo_entrega3.json")

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

NOTA_DOMINIO_FASHION_MNIST = (
    "La CNN (Arquitectura C) fue entrenada exclusivamente con Fashion-MNIST "
    "(imágenes de 28x28 en escala de grises, prenda centrada sobre fondo oscuro). "
    "Las imágenes de otro tamaño o en color se convierten automáticamente a 28x28 "
    "en escala de grises para evitar errores de inferencia; sin embargo, fotografías "
    "externas con fondos complejos o colores reales pueden presentar menor confianza."
)

# Caché en memoria para evitar recargar el modelo repetidamente en notebooks o Streamlit
_CACHE_MODELOS: Dict[str, tf.keras.Model] = {}


def _buscar_archivo_proyecto(ruta: str | Path) -> Path:
    """Resuelve rutas relativas tanto desde la raíz del repo como desde subcarpetas."""
    p = Path(ruta)
    if p.exists():
        return p
    raiz_repo = Path(__file__).resolve().parents[2]
    candidato = raiz_repo / p
    if candidato.exists():
        return candidato
    return p


def cargar_protocolo(ruta_protocolo: str | Path = RUTA_PROTOCOLO_DEFECTO) -> Dict[str, Any]:
    """Carga categorías, estantes y ruta oficial del modelo desde el protocolo JSON."""
    ruta = _buscar_archivo_proyecto(ruta_protocolo)
    if ruta.exists():
        with ruta.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": "entrega4-1.0",
        "semilla": 42,
        "semilla_demo": 2026,
        "categorias": CATEGORIAS_DEFECTO,
        "estantes": {str(k): list(v) for k, v in ESTANTES_DEFECTO.items()},
        "modelo_cnn": str(RUTA_MODELO_DEFECTO),
    }


def cargar_modelo_cnn(
    ruta_modelo: str | Path | None = None,
    ruta_protocolo: str | Path = RUTA_PROTOCOLO_DEFECTO,
    usar_cache: bool = True,
) -> tf.keras.Model:
    """Carga `modelos/stylebot_cnn_final.keras` congelado (sin reentrenar)."""
    if ruta_modelo is None:
        protocolo = cargar_protocolo(ruta_protocolo)
        ruta_modelo = protocolo.get("modelo_cnn", str(RUTA_MODELO_DEFECTO))

    ruta = _buscar_archivo_proyecto(ruta_modelo)
    clave = str(ruta.resolve())

    if usar_cache and clave in _CACHE_MODELOS:
        return _CACHE_MODELOS[clave]

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el modelo CNN congelado en: {ruta}")

    modelo = tf.keras.models.load_model(ruta, compile=False)
    modelo.trainable = False

    if usar_cache:
        _CACHE_MODELOS[clave] = modelo
    return modelo


def _extraer_arreglo_imagen(imagen: Any) -> Tuple[np.ndarray, str, str]:
    """Convierte ruta, archivo Streamlit, bytes, PIL.Image o ndarray a ndarray."""
    if imagen is None:
        raise ValueError("La imagen recibida es None.")

    if isinstance(imagen, (str, Path)):
        ruta = Path(imagen)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe la imagen en la ruta: {ruta}")
        with Image.open(ruta) as img_pil:
            modo = img_pil.mode
            arr = np.asarray(img_pil)
        return arr, f"archivo:{ruta.name}", modo

    if isinstance(imagen, Image.Image):
        return np.asarray(imagen), "PIL.Image", imagen.mode

    if isinstance(imagen, (bytes, bytearray)):
        with Image.open(io.BytesIO(imagen)) as img_pil:
            modo = img_pil.mode
            arr = np.asarray(img_pil)
        return arr, "bytes", modo

    # Compatibilidad directa con st.file_uploader (UploadedFile) y BytesIO
    if hasattr(imagen, "getvalue"):
        contenido = imagen.getvalue()
        with Image.open(io.BytesIO(contenido)) as img_pil:
            modo = img_pil.mode
            arr = np.asarray(img_pil)
        return arr, type(imagen).__name__, modo

    if hasattr(imagen, "read"):
        posicion_inicial = imagen.tell() if hasattr(imagen, "tell") else None
        contenido = imagen.read()
        if posicion_inicial is not None and hasattr(imagen, "seek"):
            imagen.seek(posicion_inicial)
        with Image.open(io.BytesIO(contenido)) as img_pil:
            modo = img_pil.mode
            arr = np.asarray(img_pil)
        return arr, type(imagen).__name__, modo

    arr = np.asarray(imagen)
    if arr.size == 0:
        raise ValueError("El arreglo de imagen está vacío.")
    modo = "ndarray"
    return arr, "numpy.ndarray", modo


def preparar_imagen_cnn(
    imagen: Any,
    invertir_colores: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convierte cualquier imagen soportada a (1, 28, 28, 1) en escala de grises [0, 1].

    Devuelve
    --------
    tensor_entrada : np.ndarray de forma (1, 28, 28, 1) float32 en [0, 1].
    imagen_28x28   : np.ndarray de forma (28, 28) float32 en [0, 1].
    metadatos      : diccionario con los detalles del preprocesamiento aplicado.
    """
    arr, origen, modo_original = _extraer_arreglo_imagen(imagen)
    forma_original = tuple(int(d) for d in arr.shape)
    dtype_original = str(arr.dtype)

    arr = arr.astype(np.float32)

    # Limpiar valores no finitos si existieran
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    min_orig = float(np.min(arr))
    max_orig = float(np.max(arr))

    # Permitir lote unitario (1, H, W) o (1, H, W, C)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Se esperaba una sola imagen, pero se recibió un lote de tamaño {arr.shape[0]}."
            )
        arr = arr[0]

    # Normalizar a [0, 1] sin dividir dos veces si ya viene normalizada
    normalizada = False
    if max_orig > 1.0:
        escala = 65535.0 if max_orig > 255.0 else 255.0
        arr = arr / escala
        normalizada = True
    arr = np.clip(arr, 0.0, 1.0)

    # Convertir a escala de grises 2D
    convertida_a_gris = False
    if arr.ndim == 2:
        gris = arr
        canales_orig = 1
    elif arr.ndim == 3:
        canales_orig = int(arr.shape[-1])
        if canales_orig == 1:
            gris = arr[..., 0]
        elif canales_orig == 2:
            # Escala de grises + canal alfa (LA) sobre fondo oscuro
            gris = arr[..., 0] * arr[..., 1]
            convertida_a_gris = True
        elif canales_orig == 3:
            # RGB -> luminancia estándar
            gris = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
            convertida_a_gris = True
        elif canales_orig >= 4:
            # RGBA -> componer sobre fondo negro y pasar a gris
            rgb = arr[..., :3] * arr[..., 3:4]
            gris = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            convertida_a_gris = True
        else:
            raise ValueError(f"Número de canales no soportado: {canales_orig}")
    else:
        raise ValueError(f"Dimensión de imagen no soportada: {forma_original}")

    # Redimensionar a 28x28 únicamente si tiene otro tamaño
    redimensionada = gris.shape != (28, 28)
    tensor = gris[np.newaxis, ..., np.newaxis]
    if redimensionada:
        tensor = tf.image.resize(tensor, [28, 28], method="bilinear").numpy()

    if invertir_colores:
        tensor = 1.0 - tensor

    tensor = np.clip(tensor.astype(np.float32), 0.0, 1.0)
    imagen_28x28 = tensor[0, :, :, 0]

    metadatos = {
        "origen": origen,
        "modo_original": modo_original,
        "dtype_original": dtype_original,
        "forma_original": forma_original,
        "canales_originales": canales_orig,
        "forma_final": (28, 28, 1),
        "rango_original": (round(min_orig, 4), round(max_orig, 4)),
        "rango_final": (round(float(imagen_28x28.min()), 4), round(float(imagen_28x28.max()), 4)),
        "media_intensidad": round(float(imagen_28x28.mean()), 4),
        "convertida_a_gris": convertida_a_gris,
        "redimensionada": redimensionada,
        "normalizada_desde_entero": normalizada,
        "colores_invertidos": bool(invertir_colores),
        "nota_dominio": NOTA_DOMINIO_FASHION_MNIST,
    }
    return tensor, imagen_28x28, metadatos


def predecir_imagen(
    imagen: Any,
    modelo: tf.keras.Model | None = None,
    ruta_modelo: str | Path | None = None,
    ruta_protocolo: str | Path = RUTA_PROTOCOLO_DEFECTO,
    invertir_colores: bool = False,
) -> Dict[str, Any]:
    """Ejecuta la inferencia CNN sobre una imagen y devuelve categoría, confianza y datos procesados."""
    protocolo = cargar_protocolo(ruta_protocolo)
    categorias = protocolo.get("categorias", CATEGORIAS_DEFECTO)
    estantes_json = protocolo.get("estantes", {})

    if modelo is None:
        modelo = cargar_modelo_cnn(ruta_modelo=ruta_modelo, ruta_protocolo=ruta_protocolo)

    inicio = time.perf_counter()
    tensor_entrada, imagen_28x28, datos_procesados = preparar_imagen_cnn(
        imagen,
        invertir_colores=invertir_colores,
    )
    probabilidades = modelo(tensor_entrada, training=False).numpy()[0]
    tiempo_ms = (time.perf_counter() - inicio) * 1000.0

    categoria_id = int(np.argmax(probabilidades))
    confianza = float(probabilidades[categoria_id])

    if str(categoria_id) in estantes_json:
        coord = estantes_json[str(categoria_id)]
        estante_objetivo = (int(coord[0]), int(coord[1]))
    else:
        estante_objetivo = ESTANTES_DEFECTO[categoria_id]

    indices_top3 = np.argsort(probabilidades)[::-1][:3]
    top_3 = [
        {
            "categoria_id": int(idx),
            "categoria": categorias[int(idx)],
            "confianza": round(float(probabilidades[int(idx)]), 6),
        }
        for idx in indices_top3
    ]

    datos_procesados["tiempo_inferencia_ms"] = round(tiempo_ms, 3)

    return {
        "categoria_id": categoria_id,
        "categoria": categorias[categoria_id],
        "confianza": round(confianza, 6),
        "confianza_porcentaje": round(confianza * 100.0, 2),
        "estante_objetivo": estante_objetivo,
        "top_3": top_3,
        "probabilidades": {categorias[i]: round(float(probabilidades[i]), 6) for i in range(len(categorias))},
        "imagen_procesada": imagen_28x28,
        "tensor_entrada": tensor_entrada,
        "datos_procesados": datos_procesados,
        "nota_dominio": NOTA_DOMINIO_FASHION_MNIST,
    }


def ejecutar_pruebas_cnn(
    ruta_salida: str | Path = "resultados/entrega4/pruebas_cnn.csv",
    ruta_modelo: str | Path | None = None,
    ruta_protocolo: str | Path = RUTA_PROTOCOLO_DEFECTO,
) -> pd.DataFrame:
    """Ejecuta batería de pruebas (correctas, distinto tamaño y en color) y exporta el CSV."""
    protocolo = cargar_protocolo(ruta_protocolo)
    categorias = protocolo.get("categorias", CATEGORIAS_DEFECTO)
    modelo = cargar_modelo_cnn(ruta_modelo=ruta_modelo, ruta_protocolo=ruta_protocolo)

    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Seleccionar índices fijos de prueba (una prenda de distintas categorías conocidas)
    rng = np.random.default_rng(int(protocolo.get("semilla_demo", 2026)))
    muestras: List[Tuple[int, np.ndarray, int]] = []
    for clase in [0, 1, 2, 3, 5, 7, 8, 9]:
        idx = int(rng.choice(np.where(y_test == clase)[0], size=1)[0])
        muestras.append((idx, x_test[idx], clase))

    casos: List[Dict[str, Any]] = []

    # 1. Imágenes correctas de Fashion-MNIST (28x28 escala de grises)
    for k, (idx, img_u8, clase_real) in enumerate(muestras[:4]):
        entrada = img_u8.astype(np.float32) / 255.0 if k % 2 == 0 else img_u8
        tipo_txt = "float32 [0,1]" if k % 2 == 0 else "uint8 [0,255]"
        casos.append(
            {
                "id_prueba": f"correcta_{k + 1}",
                "tipo_prueba": "Imagen correcta (28x28 gris)",
                "descripcion": f"Fashion-MNIST índice {idx} en {tipo_txt}",
                "imagen": entrada,
                "categoria_real": categorias[clase_real],
                "observacion": (
                    "Formato nativo de Fashion-MNIST (28x28 gris). " + NOTA_DOMINIO_FASHION_MNIST
                ),
            }
        )

    # 2. Imágenes de diferente tamaño (más chicas, más grandes y rectangulares)
    tamanos_prueba = [
        (16, 16, "Reducción a 16x16 y reescalado a 28x28"),
        (64, 64, "Imagen cuadrada mediana 64x64"),
        (224, 224, "Imagen grande estándar 224x224"),
        (300, 180, "Imagen rectangular 300x180"),
    ]
    for k, (alto, ancho, desc) in enumerate(tamanos_prueba):
        idx, img_u8, clase_real = muestras[k % len(muestras)]
        img_resized = np.asarray(
            Image.fromarray(img_u8, mode="L").resize((ancho, alto), Image.Resampling.BILINEAR)
        )
        casos.append(
            {
                "id_prueba": f"tamano_{k + 1}",
                "tipo_prueba": "Diferente tamaño",
                "descripcion": f"{desc} (origen índice {idx})",
                "imagen": img_resized,
                "categoria_real": categorias[clase_real],
                "observacion": (
                    f"Redimensionada automáticamente de {alto}x{ancho} a 28x28 sin errores. "
                    + NOTA_DOMINIO_FASHION_MNIST
                ),
            }
        )

    # 3. Imágenes en color (RGB, RGBA, archivo BytesIO estilo Streamlit y foto sintética color)
    idx_rgb1, img_rgb1, clase_rgb1 = muestras[4]
    rgb_28 = np.stack([img_rgb1, (img_rgb1 * 0.85).astype(np.uint8), (img_rgb1 * 0.6).astype(np.uint8)], axis=-1)
    casos.append(
        {
            "id_prueba": "color_1",
            "tipo_prueba": "Imagen en color (RGB)",
            "descripcion": f"Prenda RGB tintada 28x28x3 (origen índice {idx_rgb1})",
            "imagen": rgb_28,
            "categoria_real": categorias[clase_rgb1],
            "observacion": (
                "Convertida de RGB a escala de grises 28x28 sin errores. " + NOTA_DOMINIO_FASHION_MNIST
            ),
        }
    )

    idx_rgb2, img_rgb2, clase_rgb2 = muestras[5]
    img_big_rgb = np.asarray(
        Image.fromarray(
            np.stack([img_rgb2, np.clip(img_rgb2 * 0.7, 0, 255).astype(np.uint8), img_rgb2], axis=-1),
            mode="RGB",
        ).resize((200, 200), Image.Resampling.BILINEAR)
    )
    casos.append(
        {
            "id_prueba": "color_2",
            "tipo_prueba": "Imagen en color y diferente tamaño (RGB 200x200)",
            "descripcion": f"Imagen RGB 200x200x3 (origen índice {idx_rgb2})",
            "imagen": img_big_rgb,
            "categoria_real": categorias[clase_rgb2],
            "observacion": (
                "Convertida de RGB 200x200 a 28x28 gris sin errores. " + NOTA_DOMINIO_FASHION_MNIST
            ),
        }
    )

    idx_rgba, img_rgba_base, clase_rgba = muestras[6]
    alpha = np.full_like(img_rgba_base, 255)
    rgba_128 = np.asarray(
        Image.fromarray(
            np.stack([img_rgba_base, img_rgba_base, (img_rgba_base * 0.5).astype(np.uint8), alpha], axis=-1),
            mode="RGBA",
        ).resize((128, 128), Image.Resampling.BILINEAR)
    )
    casos.append(
        {
            "id_prueba": "color_3",
            "tipo_prueba": "Imagen en color con transparencia (RGBA 128x128)",
            "descripcion": f"PNG RGBA 128x128x4 (origen índice {idx_rgba})",
            "imagen": rgba_128,
            "categoria_real": categorias[clase_rgba],
            "observacion": (
                "Canal alfa compuesto sobre fondo oscuro y convertido a 28x28 gris sin errores. "
                + NOTA_DOMINIO_FASHION_MNIST
            ),
        }
    )

    # Simulación exacta de archivo subido en Streamlit (PNG en memoria BytesIO)
    idx_st, img_st_base, clase_st = muestras[7]
    buffer_png = io.BytesIO()
    Image.fromarray(
        np.stack([img_st_base, img_st_base, img_st_base], axis=-1), mode="RGB"
    ).resize((150, 150), Image.Resampling.BILINEAR).save(buffer_png, format="PNG")
    buffer_png.seek(0)
    casos.append(
        {
            "id_prueba": "color_4_streamlit",
            "tipo_prueba": "Archivo PNG RGB en memoria (flujo Streamlit)",
            "descripcion": f"BytesIO PNG RGB 150x150 (origen índice {idx_st})",
            "imagen": buffer_png,
            "categoria_real": categorias[clase_st],
            "observacion": (
                "Lectura directa desde flujo binario compatible con st.file_uploader. "
                + NOTA_DOMINIO_FASHION_MNIST
            ),
        }
    )

    filas_csv = []
    for caso in casos:
        try:
            res = predecir_imagen(caso["imagen"], modelo=modelo, ruta_protocolo=ruta_protocolo)
            meta = res["datos_procesados"]
            coincide = res["categoria"] == caso["categoria_real"]
            filas_csv.append(
                {
                    "id_prueba": caso["id_prueba"],
                    "tipo_prueba": caso["tipo_prueba"],
                    "descripcion": caso["descripcion"],
                    "forma_original": str(meta["forma_original"]),
                    "forma_procesada": str(meta["forma_final"]),
                    "convertida_a_gris": meta["convertida_a_gris"],
                    "redimensionada": meta["redimensionada"],
                    "categoria_real": caso["categoria_real"],
                    "categoria_predicha": res["categoria"],
                    "categoria_id": res["categoria_id"],
                    "confianza": res["confianza"],
                    "coincide_real": coincide,
                    "estante_objetivo": str(res["estante_objetivo"]),
                    "tiempo_ms": meta["tiempo_inferencia_ms"],
                    "estado": "OK",
                    "observacion": caso["observacion"],
                }
            )
        except Exception as err:
            filas_csv.append(
                {
                    "id_prueba": caso["id_prueba"],
                    "tipo_prueba": caso["tipo_prueba"],
                    "descripcion": caso["descripcion"],
                    "forma_original": "error",
                    "forma_procesada": "error",
                    "convertida_a_gris": False,
                    "redimensionada": False,
                    "categoria_real": caso["categoria_real"],
                    "categoria_predicha": "",
                    "categoria_id": -1,
                    "confianza": 0.0,
                    "coincide_real": False,
                    "estante_objetivo": "",
                    "tiempo_ms": 0.0,
                    "estado": f"ERROR: {err}",
                    "observacion": caso["observacion"],
                }
            )

    df_pruebas = pd.DataFrame(filas_csv)
    salida = Path(ruta_salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df_pruebas.to_csv(salida, index=False, encoding="utf-8")
    return df_pruebas
    
def predecir_prenda(
    imagen: Any,
    modelo: tf.keras.Model | None = None,
    ruta_modelo: str | Path | None = None,
    invertir_colores: bool = False,
) -> Dict[str, Any]:
    """Adaptador oficial para E4_StyleBot.ipynb y app/app.py (Streamlit).

    Recibe directamente un archivo de st.file_uploader, ruta, PIL.Image, bytes o ndarray
    y devuelve los cuatro campos requeridos por el protocolo de integración.
    """
    resultado = predecir_imagen(
        imagen,
        modelo=modelo,
        ruta_modelo=ruta_modelo,
        invertir_colores=invertir_colores,
    )
    return {
        "indice_categoria": int(resultado["categoria_id"]),
        "categoria": str(resultado["categoria"]),
        "confianza": float(resultado["confianza"]),
        "imagen_procesada": resultado["imagen_procesada"],
    }

if __name__ == "__main__":
    df = ejecutar_pruebas_cnn()
    ok = int((df["estado"] == "OK").sum())
    aciertos = int(df["coincide_real"].sum())
    print(f"Pruebas ejecutadas: {len(df)} | Sin errores: {ok}/{len(df)} | Coincidencias: {aciertos}/{len(df)}")
    print(df[["id_prueba", "tipo_prueba", "forma_original", "categoria_real", "categoria_predicha", "confianza", "estado"]])
