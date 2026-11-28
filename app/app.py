"""Interfaz Streamlit de StyleBot.
Preparada por Krysto como apoyo a la integración de Jacqueline.
Reutiliza la CNN de Juan José y la simulación de Luis Ángel.
Ejecutar desde la raíz: python -m streamlit run app/app.py
"""
from pathlib import Path
import hashlib
import json
import sys
import threading

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

st.set_page_config(page_title="StyleBot | Almacén inteligente", page_icon="👕", layout="wide")


@st.cache_resource
def preparar_servicios():
    from src.entrega4 import inferencia_cnn as cnn
    from src.entrega4 import simulacion_final as rl
    from src.entrega3.entorno_almacen import MAPA, ESTANTES

    # El módulo publicado de RL calcula su raíz dentro de src.
    # Configuramos sus rutas absolutas desde la aplicación sin editar su archivo.
    rl.RAIZ = RAIZ
    rl.RUTA_PROTOCOLO = RAIZ / "datos/entrega3/protocolo_entrega3.json"
    rl.RUTAS_TABLAS = {
        "q_learning": RAIZ / "modelos/entrega3/q_table_qlearning.npz",
        "dyna_q": RAIZ / "modelos/entrega3/q_table_dynaq.npz",
    }
    protocolo_path = RAIZ / "datos/entrega3/protocolo_entrega3.json"
    protocolo = json.loads(protocolo_path.read_text(encoding="utf-8"))
    modelo = cnn.cargar_modelo_cnn(
        ruta_modelo=RAIZ / "modelos/stylebot_cnn_final.keras",
        ruta_protocolo=protocolo_path,
    )
    return cnn, rl, modelo, protocolo, MAPA, ESTANTES, threading.Lock()


@st.cache_data
def cargar_muestras():
    from tensorflow.keras.datasets import fashion_mnist
    (_, _), (imagenes, etiquetas) = fashion_mnist.load_data()
    tabla = pd.read_csv(RAIZ / "datos/entrega3/predicciones_50_prendas.csv")
    indices = tabla["indice_test"].to_numpy(dtype=int)
    if len(indices) != 50 or np.any(indices < 0) or np.any(indices >= len(imagenes)):
        raise ValueError("La demostración requiere 50 índices válidos de Fashion-MNIST.")
    return imagenes[indices], etiquetas[indices], indices


def ejecutar(imagen, agente, semilla, categoria_real, invertir):
    cnn, rl, modelo, protocolo, _, _, bloqueo = preparar_servicios()
    # Se usa la función publicada de Juan y se adapta su salida al contrato E4.
    with bloqueo:
        percepcion = cnn.predecir_imagen(
            imagen, modelo=modelo,
            ruta_protocolo=RAIZ / "datos/entrega3/protocolo_entrega3.json",
            invertir_colores=invertir,
        )
        indice = int(percepcion["categoria_id"])
        confianza = float(percepcion["confianza"])
        procesada = np.asarray(percepcion["imagen_procesada"])
        if not 0 <= indice < 10 or not np.isfinite(confianza) or not 0 <= confianza <= 1:
            raise ValueError("La CNN devolvió una categoría o confianza inválida.")
        if procesada.shape != (28, 28) or not np.isfinite(procesada).all():
            raise ValueError("La imagen procesada no cumple el formato 28x28.")
        # La categoría real solo evalúa el resultado; la política usa la predicción.
        simulacion = rl.simular_acomodo(
            indice, agente=agente, semilla=semilla, categoria_real=categoria_real,
        )
    return {
        "indice_categoria": indice,
        "categoria": protocolo["categorias"][indice],
        "confianza": confianza,
        "imagen_procesada": procesada,
        "simulacion": simulacion,
        "categoria_real": categoria_real,
    }


def dibujar_recorrido(resultado):
    _, _, _, _, mapa, estantes, _ = preparar_servicios()
    sim = resultado["simulacion"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for fila, linea in enumerate(mapa):
        for columna, simbolo in enumerate(linea):
            color = {"#": "#334155", "E": "#bfdbfe", "R": "#bbf7d0"}.get(simbolo, "#f8fafc")
            ax.add_patch(Rectangle((columna - .5, fila - .5), 1, 1,
                                   facecolor=color, edgecolor="#cbd5e1"))
    for (fila, columna), categoria in estantes.items():
        ax.text(columna, fila, str(categoria), ha="center", va="center", fontsize=10)
    recorrido = np.asarray([sim["posicion_inicial"]] + sim["trayectoria"])
    ax.plot(recorrido[:, 1], recorrido[:, 0], color="#7c3aed", linewidth=2, label="Recorrido")
    ax.scatter(recorrido[0, 1], recorrido[0, 0], color="#15803d", s=90, label="Inicio", zorder=4)
    ax.scatter(recorrido[-1, 1], recorrido[-1, 0], color="#7c3aed", marker="X", s=90, label="Final", zorder=5)
    destino = sim["estante"]
    ax.scatter(destino[1], destino[0], s=300, facecolors="none", edgecolors="#f97316", linewidths=2, label="Destino", zorder=6)
    ax.set(xlim=(-.5, 7.5), ylim=(7.5, -.5), xlabel="Columna", ylabel="Fila", title="Recorrido en el almacén")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
    fig.tight_layout()
    return fig


st.title("👕 StyleBot")
st.markdown("Clasifica una prenda y sigue su recorrido hasta el estante del almacén.")
st.caption("Proyecto académico · CNN congelada + Q-Learning / Dyna-Q")

with st.sidebar:
    st.header("Configura la prueba")
    origen = st.radio("Imagen de entrada", ["Muestra de Fashion-MNIST", "Subir una imagen"])
    agente_nombre = st.selectbox("Agente del almacén", ["Dyna-Q", "Q-Learning"])
    agente = {"Dyna-Q": "dyna_q", "Q-Learning": "q_learning"}[agente_nombre]
    semilla = int(st.number_input("Semilla", min_value=0, max_value=2147483647, value=2026, step=1))
    st.caption("Cada prueba inicia un almacén vacío con una sola prenda. Los modelos no se reentrenan.")

imagen = None
categoria_real = None
invertir = False
clave_imagen = None
try:
    if origen == "Muestra de Fashion-MNIST":
        with st.spinner("Cargando muestras de demostración…"):
            muestras, etiquetas, indices = cargar_muestras()
        numero = st.selectbox("Prenda de demostración", range(50), format_func=lambda n: f"Prenda {n + 1} · índice de prueba {indices[n]}")
        imagen = muestras[numero]
        categoria_real = int(etiquetas[numero])
        clave_imagen = f"fashion:{indices[numero]}"
        st.caption("Se utilizan los mismos 50 índices de la Entrega 3. La CNN calcula una predicción nueva en cada ejecución.")
    else:
        archivo = st.file_uploader("Sube una prenda", type=["png", "jpg", "jpeg"])
        st.warning("La CNN fue entrenada con Fashion-MNIST: prendas centradas sobre fondo oscuro. En fotografías externas puede equivocarse incluso con confianza alta.")
        invertir = st.checkbox("Invertir claro y oscuro antes de clasificar", value=False)
        if archivo is not None:
            if archivo.size > 10 * 1024 * 1024:
                raise ValueError("Utiliza una imagen de hasta 10 MB.")
            with Image.open(archivo) as original:
                if original.width * original.height > 20_000_000:
                    raise ValueError("Utiliza una imagen de hasta 20 megapíxeles.")
                imagen = original.convert("RGBA").copy()
            clave_imagen = hashlib.sha256(archivo.getvalue()).hexdigest()
except Exception as error:
    st.error("No se pudo preparar la imagen. Revisa el archivo o la conexión para descargar Fashion-MNIST.")
    with st.expander("Detalle del problema"):
        st.code(str(error))

if imagen is not None:
    st.image(imagen, caption="Imagen de entrada", width=220, clamp=True)

clave = (origen, clave_imagen, agente, semilla, invertir)
if st.session_state.get("clave_resultado") != clave:
    st.session_state.pop("resultado_stylebot", None)

if st.button("Clasificar y simular", type="primary", disabled=imagen is None):
    st.session_state.pop("resultado_stylebot", None)
    try:
        with st.spinner("Clasificando y ejecutando el agente…"):
            resultado = ejecutar(imagen, agente, semilla, categoria_real, invertir)
        st.session_state["resultado_stylebot"] = resultado
        st.session_state["clave_resultado"] = clave
    except Exception as error:
        st.error("No se pudo completar la prueba. Comprueba los modelos, las tablas Q y las dependencias del proyecto.")
        with st.expander("Detalle para el equipo"):
            st.code(str(error))

resultado = st.session_state.get("resultado_stylebot")
if resultado is not None:
    sim = resultado["simulacion"]
    st.divider()
    st.subheader("Resultado de la prueba")
    cols = st.columns(4)
    cols[0].metric("Categoría predicha", resultado["categoria"])
    cols[1].metric("Confianza CNN", f"{resultado['confianza']:.1%}")
    cols[2].metric("Pasos", sim["pasos"])
    cols[3].metric("Recompensa", sim["recompensa"])
    st.write(f"Agente: {agente_nombre} · Estante objetivo (fila, columna): {tuple(sim['estante'])}")
    if resultado["categoria_real"] is None:
        if sim["exito"]:
            st.success("El agente llegó al estante de la categoría predicha.")
        else:
            st.warning(f"El acomodo no se completó: {sim['tipo_fallo']}")
        st.info("La etiqueta real de esta foto es desconocida; este resultado no confirma que la CNN haya clasificado correctamente la prenda.")
    else:
        _, _, _, protocolo, _, _, _ = preparar_servicios()
        real = protocolo["categorias"][resultado["categoria_real"]]
        st.write("Categoría real de referencia:", real)
        if sim["exito"]:
            st.success("Prenda clasificada y colocada correctamente.")
        else:
            st.warning(f"Prueba sin éxito completo. Tipo de fallo: {sim['tipo_fallo']}")
    izquierda, derecha = st.columns([1, 2])
    with izquierda:
        st.image(resultado["imagen_procesada"], caption="Entrada de la CNN · 28 × 28", width=220, clamp=True)
        st.caption("Azul: estantes (0–9). Verde: recepción. Gris oscuro: obstáculos.")
    with derecha:
        fig = dibujar_recorrido(resultado)
        st.pyplot(fig)
        plt.close(fig)
    with st.expander("Acciones paso a paso"):
        pasos = pd.DataFrame({
            "paso": range(1, len(sim["acciones"]) + 1),
            "accion": sim["acciones"],
            "posicion": [str(tuple(p)) for p in sim["trayectoria"]],
            "evento": sim["eventos"],
        })
        st.dataframe(pasos, hide_index=True)
    exportable = {k: v for k, v in resultado.items() if k != "imagen_procesada"}
    st.download_button("Descargar resultado JSON", json.dumps(exportable, ensure_ascii=False, indent=2),
                       file_name="stylebot_resultado.json", mime="application/json")

with st.expander("Cómo funciona StyleBot"):
    st.write("La CNN recibe una imagen en escala de grises de 28 × 28. Su categoría predicha determina el destino. El agente utiliza una tabla Q guardada para seleccionar movimientos y acomodar la prenda.")
    st.write("La confianza es una puntuación del modelo, no una garantía. La web ejecuta una prueba individual; la demostración acumulada de 50 prendas se valida en el notebook final.")
    st.caption("Equipo: Krysto · Ángel Uriel · Juan José · Luis Ángel · Jacqueline")
