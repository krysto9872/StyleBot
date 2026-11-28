"""Simulación final de acomodo con las tablas Q de StyleBot (Entrega 4).

Responsable: Luis Ángel Mendoza Gaytán (simulación y aprendizaje por refuerzo).

Carga las tablas Q ya entrenadas en la Entrega 3 y simula cómo el robot lleva una
prenda desde la recepción hasta su estante. No entrena ni modifica las tablas: se
abren en modo de solo lectura y la política es codiciosa (sin exploración).

Uso desde la raíz del repositorio (notebook E4 o aplicación):

    from src.entrega4.simulacion_final import simular_acomodo
    resultado = simular_acomodo("Suéter", agente="dyna_q", semilla=2026)

Prueba independiente del módulo:

    python src/entrega4/simulacion_final.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.entrega3.entorno_almacen import (  # noqa: E402
    CATEGORIAS, ESTANTES, NOMBRES_ACCIONES, AlmacenStyleBot,
)

RUTA_PROTOCOLO = RAIZ / "datos" / "entrega3" / "protocolo_entrega3.json"
RUTAS_TABLAS = {
    "q_learning": RAIZ / "modelos" / "entrega3" / "q_table_qlearning.npz",
    "dyna_q": RAIZ / "modelos" / "entrega3" / "q_table_dynaq.npz",
}
ALIAS_AGENTES = {
    "dyna_q": "dyna_q", "dynaq": "dyna_q", "dyna-q": "dyna_q", "dyna q": "dyna_q",
    "q_learning": "q_learning", "qlearning": "q_learning", "q-learning": "q_learning", "q learning": "q_learning",
}
POSICION_ESTANTE = {indice: [fila, columna] for (fila, columna), indice in ESTANTES.items()}
CAPACIDAD_ESTANTE = 8
MAX_PASOS_POR_PRENDA = 60
SEMILLA_DEMO = 2026

_tablas = {}


def _protocolo():
    try:
        return json.loads(RUTA_PROTOCOLO.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _recompensas():
    return _protocolo().get("recompensas")


def normalizar_agente(agente):
    """Acepta "dyna_q", "Dyna-Q", "q_learning", "Q-Learning", etc."""
    clave = ALIAS_AGENTES.get(str(agente).strip().lower())
    if clave is None:
        raise ValueError(f"Agente no reconocido: {agente!r}. Usa 'dyna_q' o 'q_learning'.")
    return clave


def indice_categoria(categoria):
    """Devuelve el índice 0-9 de una categoría dada por nombre o por número."""
    if isinstance(categoria, (int, np.integer)) and not isinstance(categoria, bool):
        if 0 <= int(categoria) < len(CATEGORIAS):
            return int(categoria)
    elif isinstance(categoria, str):
        texto = categoria.strip().lower()
        for i, nombre in enumerate(CATEGORIAS):
            if texto == nombre.lower():
                return i
    raise ValueError(f"Categoría no reconocida: {categoria!r}. Opciones: {', '.join(CATEGORIAS)} o 0-9.")


def cargar_tabla_q(agente="dyna_q"):
    """Carga una tabla Q guardada (1408 x 6) en modo de solo lectura.

    La tabla se guarda en memoria después de la primera carga. Si el archivo no
    existe o tiene otra forma, se lanza un error con un mensaje claro.
    """
    agente = normalizar_agente(agente)
    if agente not in _tablas:
        ruta = RUTAS_TABLAS[agente]
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró la tabla Q de {agente}: {ruta.relative_to(RAIZ)}")
        with np.load(ruta, allow_pickle=False) as datos:
            tabla = np.array(datos["q_table"], dtype=np.float64)
            metadatos = json.loads(str(datos["metadatos"])) if "metadatos" in datos.files else {}
        if tabla.shape != (1408, 6):
            raise ValueError(f"La tabla Q de {agente} debe tener forma (1408, 6); tiene {tabla.shape}")
        tabla.setflags(write=False)  # garantiza que la evaluación no modifique la tabla
        _tablas[agente] = (tabla, metadatos)
    return _tablas[agente]


def _accion_codiciosa(tabla, estado, rng):
    valores = tabla[int(estado)]
    return int(rng.choice(np.flatnonzero(valores == valores.max())))


def _numero(valor):
    valor = float(valor)
    return int(valor) if valor.is_integer() else round(valor, 4)


def _ejecutar(tabla, entorno, semilla, opciones):
    """Corre un episodio sin exploración ni aprendizaje y registra cada paso."""
    estado, info = entorno.reset(seed=semilla, options=opciones)
    rng = np.random.default_rng(semilla)
    posicion_inicial = [int(info["posicion"][0]), int(info["posicion"][1])]
    trayectoria, acciones, eventos = [], [], []
    total, terminado, truncado = 0.0, False, False
    while not (terminado or truncado):
        accion = _accion_codiciosa(tabla, estado, rng)
        estado, recompensa, terminado, truncado, info = entorno.step(accion)
        total += recompensa
        trayectoria.append([int(info["posicion"][0]), int(info["posicion"][1])])
        acciones.append(NOMBRES_ACCIONES[accion])
        eventos.append(info["evento"])
    return posicion_inicial, trayectoria, acciones, eventos, total, terminado, truncado, info


def simular_acomodo(categoria, agente="dyna_q", semilla=SEMILLA_DEMO, categoria_real=None,
                    estantes_llenos=None, max_pasos=MAX_PASOS_POR_PRENDA):
    """Simula el acomodo de UNA prenda con la tabla Q guardada del agente elegido.

    Parámetros
    ----------
    categoria : str | int
        Categoría predicha por la CNN (nombre como en CATEGORIAS o índice 0-9).
        Es lo único que el robot "ve".
    agente : "dyna_q" | "q_learning"
    semilla : int
        Fija el desempate entre acciones con el mismo valor Q (resultado reproducible).
    categoria_real : str | int | None
        Etiqueta real, solo para evaluar. Si se omite, se asume igual a la predicha.
    estantes_llenos : lista de categorías cuyos estantes empiezan llenos (opcional).
    max_pasos : límite de pasos del episodio (60 por prenda según el protocolo).

    Devuelve un diccionario con los campos del contrato de integración:
    agente, categoria, estante, trayectoria, acciones, pasos, recompensa, exito y
    tipo_fallo, además de campos de apoyo para la demostración.

    tipo_fallo: None si la prenda quedó en su estante correcto; "cnn" si el robot la
    llevó al estante que indicó la CNN pero la predicción era incorrecta; "politica"
    si el agente la dejó en otro estante; "estante_lleno" si el destino estaba lleno
    y la prenda se devolvió a recepción; "limite_pasos" si no terminó a tiempo.
    """
    agente = normalizar_agente(agente)
    tabla, metadatos = cargar_tabla_q(agente)
    pred = indice_categoria(categoria)
    real = pred if categoria_real is None else indice_categoria(categoria_real)
    ocupacion = [0] * len(CATEGORIAS)
    for lleno in estantes_llenos or []:
        ocupacion[indice_categoria(lleno)] = CAPACIDAD_ESTANTE

    entorno = AlmacenStyleBot(
        etiquetas=np.array([real]), predicciones=np.array([pred]), prendas_por_episodio=1,
        capacidad=CAPACIDAD_ESTANTE, max_pasos=int(max_pasos), recompensas=_recompensas(),
    )
    inicio, trayectoria, acciones, eventos, total, terminado, truncado, info = _ejecutar(
        tabla, entorno, int(semilla), {"indices": [0], "ocupacion": ocupacion})

    colocada = entorno.historial_colocadas[0][3] if entorno.historial_colocadas else None
    exito = bool(terminado and info["correctas"] == 1)
    if exito:
        tipo_fallo = None
    elif info["fallos_cnn"]:
        tipo_fallo = "cnn"
    elif info["fallos_politica"]:
        tipo_fallo = "politica"
    elif info["devueltas"]:
        tipo_fallo = "estante_lleno"
    else:
        tipo_fallo = "limite_pasos"

    return {
        "agente": agente,
        "categoria": CATEGORIAS[pred],
        "indice_categoria": pred,
        "categoria_real": CATEGORIAS[real],
        "estante": list(POSICION_ESTANTE[pred]),
        "estante_colocado": None if colocada is None else list(POSICION_ESTANTE[colocada]),
        "posicion_inicial": inicio,
        "trayectoria": trayectoria,
        "acciones": acciones,
        "eventos": eventos,
        "pasos": int(info["pasos"]),
        "recompensa": _numero(total),
        "exito": exito,
        "tipo_fallo": tipo_fallo,
        "terminado": bool(terminado),
        "truncado": bool(truncado),
        "acciones_invalidas": int(info["acciones_invalidas"]),
        "semilla": int(semilla),
        "tabla_q": str(RUTAS_TABLAS[agente].relative_to(RAIZ)).replace("\\", "/"),
        "configuracion": metadatos.get("corrida"),
    }


def simular_prendas(categorias, agente="dyna_q", semilla=SEMILLA_DEMO, categorias_reales=None):
    """Simula varias prendas seguidas en un solo episodio (por ejemplo, las 50 de la demo).

    Las prendas llegan en el orden dado y los estantes se van llenando (capacidad 8).
    Devuelve un resumen con correctas, fallos de CNN y de política, pasos y recompensa.
    """
    agente = normalizar_agente(agente)
    tabla, _ = cargar_tabla_q(agente)
    pred = np.array([indice_categoria(c) for c in categorias])
    real = pred.copy() if categorias_reales is None else np.array([indice_categoria(c) for c in categorias_reales])
    if len(real) != len(pred):
        raise ValueError("categorias y categorias_reales deben tener la misma longitud")
    entorno = AlmacenStyleBot(
        etiquetas=real, predicciones=pred, prendas_por_episodio=len(pred), capacidad=CAPACIDAD_ESTANTE,
        max_pasos=MAX_PASOS_POR_PRENDA * len(pred), recompensas=_recompensas(),
    )
    _, trayectoria, acciones, _, total, terminado, truncado, info = _ejecutar(
        tabla, entorno, int(semilla), {"indices": np.arange(len(pred))})
    return {
        "agente": agente,
        "prendas": int(len(pred)),
        "correctas": int(info["correctas"]),
        "incorrectas": int(info["incorrectas"]),
        "fallos_cnn": int(info["fallos_cnn"]),
        "fallos_politica": int(info["fallos_politica"]),
        "devueltas": int(info["devueltas"]),
        "acciones_invalidas": int(info["acciones_invalidas"]),
        "pasos": int(info["pasos"]),
        "recompensa": _numero(total),
        "exito_prendas": round(info["correctas"] / len(pred), 4) if len(pred) else 0.0,
        "terminado": bool(terminado),
        "truncado": bool(truncado),
        "trayectoria": trayectoria,
        "acciones": acciones,
    }


def _huella(tabla):
    return hashlib.sha256(np.ascontiguousarray(tabla).tobytes()).hexdigest()[:16]


def probar_modulo(guardar_csv=True):
    """Pruebas independientes: las 10 categorías con ambos agentes, un error de la CNN,
    un estante lleno y la demostración de 50 prendas de la Entrega 3."""
    filas = []
    huellas = {a: _huella(cargar_tabla_q(a)[0]) for a in RUTAS_TABLAS}

    for agente in RUTAS_TABLAS:
        for categoria in CATEGORIAS:
            r = simular_acomodo(categoria, agente=agente)
            filas.append({"prueba": "categoria", "agente": agente, "categoria": categoria,
                          "categoria_real": r["categoria_real"], "estante": r["estante"],
                          "pasos": r["pasos"], "recompensa": r["recompensa"], "exito": r["exito"],
                          "tipo_fallo": r["tipo_fallo"], "esperado": "exito", "ok": r["exito"]})
        r = simular_acomodo("Camiseta/top", agente=agente, categoria_real="Suéter")
        filas.append({"prueba": "error_cnn", "agente": agente, "categoria": r["categoria"],
                      "categoria_real": r["categoria_real"], "estante": r["estante"], "pasos": r["pasos"],
                      "recompensa": r["recompensa"], "exito": r["exito"], "tipo_fallo": r["tipo_fallo"],
                      "esperado": "cnn", "ok": r["tipo_fallo"] == "cnn"})
        r = simular_acomodo("Bolso", agente=agente, estantes_llenos=["Bolso"])
        filas.append({"prueba": "estante_lleno", "agente": agente, "categoria": r["categoria"],
                      "categoria_real": r["categoria_real"], "estante": r["estante"], "pasos": r["pasos"],
                      "recompensa": r["recompensa"], "exito": r["exito"], "tipo_fallo": r["tipo_fallo"],
                      "esperado": "estante_lleno", "ok": r["tipo_fallo"] == "estante_lleno"})

    ruta_pred = RAIZ / "datos" / "entrega3" / "predicciones_50_prendas.csv"
    if ruta_pred.exists():
        with open(ruta_pred, encoding="utf-8") as archivo:
            datos = list(csv.DictReader(archivo))
        pred = [int(d["prediccion"]) for d in datos]
        real = [int(d["etiqueta_real"]) for d in datos]
        esperado = {"dyna_q": (49, 1, 0), "q_learning": (49, 1, 0)}  # correctas, fallos CNN, fallos política (E3)
        for agente in RUTAS_TABLAS:
            r = simular_prendas(pred, agente=agente, categorias_reales=real)
            obtenido = (r["correctas"], r["fallos_cnn"], r["fallos_politica"])
            filas.append({"prueba": "demo_50_prendas", "agente": agente, "categoria": "50 prendas CNN",
                          "categoria_real": "", "estante": "", "pasos": r["pasos"], "recompensa": r["recompensa"],
                          "exito": f"{r['correctas']}/{r['prendas']}", "tipo_fallo": f"cnn={r['fallos_cnn']}, politica={r['fallos_politica']}",
                          "esperado": "49/50, cnn=1, politica=0", "ok": obtenido == esperado[agente]})

    for agente, huella in huellas.items():
        filas.append({"prueba": "tabla_sin_cambios", "agente": agente, "categoria": "", "categoria_real": "",
                      "estante": "", "pasos": "", "recompensa": "", "exito": "", "tipo_fallo": "",
                      "esperado": huella, "ok": _huella(cargar_tabla_q(agente)[0]) == huella})

    if guardar_csv:
        destino = RAIZ / "resultados" / "entrega4" / "pruebas_simulacion.csv"
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]), lineterminator="\n")
            escritor.writeheader()
            escritor.writerows(filas)
    return filas


if __name__ == "__main__":
    ejemplo = simular_acomodo("Suéter", agente="dyna_q")
    print("Ejemplo:", {k: v for k, v in ejemplo.items() if k not in {"trayectoria", "acciones", "eventos"}})
    print("Acciones:", " -> ".join(ejemplo["acciones"]))
    print("Trayectoria:", ejemplo["trayectoria"])
    resultados = probar_modulo()
    for fila in resultados:
        print(f"[{'OK' if fila['ok'] else 'FALLA'}] {fila['prueba']:<17} {fila['agente']:<10} {fila['categoria']:<14} "
              f"pasos={fila['pasos']!s:<4} recompensa={fila['recompensa']!s:<5} exito={fila['exito']!s:<6} fallo={fila['tipo_fallo']}")
    total_ok = sum(bool(f["ok"]) for f in resultados)
    print(f"\n{total_ok}/{len(resultados)} pruebas correctas. Resultados en resultados/entrega4/pruebas_simulacion.csv")
