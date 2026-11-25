"""Agentes de aprendizaje por refuerzo tabular de StyleBot (Entrega 3).

Responsable: Luis Angel Mendoza Gaytán (simulación y aprendizaje por refuerzo).

Contiene Q-Learning y Dyna-Q para el entorno `AlmacenStyleBot`, el ciclo de
entrenamiento, la evaluación con el protocolo común (semillas 1000-1049, sin
exploración ni actualización de la tabla Q) y el guardado de las tablas Q.

Uso básico:
    from src.entrega3.agentes_rl import AgenteQLearning, entrenar, evaluar_politica
    agente = AgenteQLearning(alpha=0.1, gamma=0.95, semilla=42)
    historial = entrenar(agente, entorno, episodios=3000, semilla=42)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

N_ESTADOS = 1408
N_ACCIONES = 6


class AgenteQLearning:
    """Q-Learning tabular con política epsilon-greedy.

    Actualización (Watkins):
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    Si el episodio terminó, el valor futuro es 0. Si solo se truncó por límite de
    pasos, se conserva el valor futuro, porque el estado no es terminal.
    Epsilon baja multiplicándose por `epsilon_decay` al final de cada episodio.
    Los empates en la tabla Q se rompen al azar para no favorecer siempre la acción 0.
    """

    algoritmo = "Q-Learning"

    def __init__(self, n_estados=N_ESTADOS, n_acciones=N_ACCIONES, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, semilla=42):
        self.n_estados = int(n_estados)
        self.n_acciones = int(n_acciones)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_inicial = float(epsilon)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.semilla = int(semilla)
        self.rng = np.random.default_rng(semilla)
        self.q = np.zeros((self.n_estados, self.n_acciones), dtype=np.float64)

    def accion_codiciosa(self, estado, rng=None):
        """Mejor acción según la tabla Q; los empates se rompen al azar."""
        valores = self.q[int(estado)]
        mejores = np.flatnonzero(valores == valores.max())
        return int((rng or self.rng).choice(mejores))

    def elegir_accion(self, estado, explorar=True):
        if explorar and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_acciones))
        return self.accion_codiciosa(estado)

    def _actualizar_q(self, estado, accion, recompensa, siguiente, terminado):
        objetivo = recompensa if terminado else recompensa + self.gamma * self.q[siguiente].max()
        self.q[estado, accion] += self.alpha * (objetivo - self.q[estado, accion])

    def aprender(self, estado, accion, recompensa, siguiente, terminado):
        self._actualizar_q(int(estado), int(accion), float(recompensa), int(siguiente), bool(terminado))

    def fin_episodio(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def hiperparametros(self):
        return {
            "algoritmo": self.algoritmo, "alpha": self.alpha, "gamma": self.gamma,
            "epsilon_inicial": self.epsilon_inicial, "epsilon_minimo": self.epsilon_min,
            "decaimiento_epsilon": self.epsilon_decay, "pasos_planificacion": 0, "semilla": self.semilla,
        }


class AgenteDynaQ(AgenteQLearning):
    """Dyna-Q tabular: Q-Learning + modelo aprendido + pasos de planeación.

    Después de cada paso real:
      1. Actualiza Q con la experiencia real (igual que Q-Learning).
      2. Guarda la transición en el modelo.
      3. Repite `pasos_planeacion` veces: elige al azar un par (estado, acción) ya
         visitado, simula su resultado con el modelo y actualiza Q.

    El modelo guarda cuántas veces se observó cada resultado de un par (estado, acción)
    y simula en proporción a esas frecuencias. Así representa bien la acción "tomar",
    cuya prenda siguiente es aleatoria; el resto del almacén es determinista.
    """

    algoritmo = "Dyna-Q"

    def __init__(self, *args, pasos_planeacion=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.pasos_planeacion = int(pasos_planeacion)
        self.modelo = {}          # (s, a) -> {(r, s', terminado): conteo}
        self.pares_visitados = []  # lista de (s, a) para muestrear uniformemente
        self.actualizaciones_planeacion = 0

    def aprender(self, estado, accion, recompensa, siguiente, terminado):
        estado, accion, siguiente = int(estado), int(accion), int(siguiente)
        recompensa, terminado = float(recompensa), bool(terminado)
        self._actualizar_q(estado, accion, recompensa, siguiente, terminado)

        par = (estado, accion)
        resultados = self.modelo.get(par)
        if resultados is None:
            resultados = self.modelo[par] = {}
            self.pares_visitados.append(par)
        resultado = (recompensa, siguiente, terminado)
        resultados[resultado] = resultados.get(resultado, 0) + 1

        for _ in range(self.pasos_planeacion):
            s, a = self.pares_visitados[int(self.rng.integers(len(self.pares_visitados)))]
            opciones = self.modelo[(s, a)]
            if len(opciones) == 1:
                r, s2, fin = next(iter(opciones))
            else:
                claves = list(opciones)
                conteos = np.fromiter(opciones.values(), dtype=np.float64)
                r, s2, fin = claves[int(self.rng.choice(len(claves), p=conteos / conteos.sum()))]
            self._actualizar_q(s, a, r, s2, fin)
        self.actualizaciones_planeacion += self.pasos_planeacion

    def hiperparametros(self):
        datos = super().hiperparametros()
        datos["pasos_planificacion"] = self.pasos_planeacion
        return datos


def entrenar(agente, entorno, episodios, semilla=42):
    """Entrena al agente y devuelve el historial por episodio.

    El entorno se reinicia con `semilla` solo en el primer episodio; después sigue su
    propio generador. Así la corrida es reproducible y no reutiliza las semillas
    1000-1049 reservadas para la evaluación final.
    """
    historial = {clave: [] for clave in [
        "episodio", "recompensa", "pasos", "correctas", "total_prendas", "incorrectas",
        "acciones_invalidas", "devueltas", "terminado", "epsilon", "segundos"]}
    inicio = time.perf_counter()
    for episodio in range(1, int(episodios) + 1):
        estado, _ = entorno.reset(seed=semilla if episodio == 1 else None)
        total, terminado, truncado = 0.0, False, False
        while not (terminado or truncado):
            accion = agente.elegir_accion(estado, explorar=True)
            siguiente, recompensa, terminado, truncado, info = entorno.step(accion)
            agente.aprender(estado, accion, recompensa, siguiente, terminado)
            estado = siguiente
            total += recompensa
        historial["episodio"].append(episodio)
        historial["recompensa"].append(total)
        historial["pasos"].append(info["pasos"])
        historial["correctas"].append(info["correctas"])
        historial["total_prendas"].append(entorno.total_prendas)
        historial["incorrectas"].append(info["incorrectas"])
        historial["acciones_invalidas"].append(info["acciones_invalidas"])
        historial["devueltas"].append(info["devueltas"])
        historial["terminado"].append(bool(terminado))
        historial["epsilon"].append(agente.epsilon)
        historial["segundos"].append(time.perf_counter() - inicio)
        agente.fin_episodio()
    return historial


def entrenar_con_validacion(agente, crear_entorno, episodios, semillas_validacion, cada=500, semilla=42):
    """Entrena por bloques de `cada` episodios y evalúa la política codiciosa en validación.

    La validación solo lee la tabla Q (no la modifica) y usa semillas distintas de las
    de evaluación final, así que puede usarse para elegir configuraciones.
    Devuelve (historial_entrenamiento, curva_validacion).
    """
    entorno = crear_entorno()
    historial, curva = None, []
    hechos = 0
    while hechos < episodios:
        bloque = min(cada, episodios - hechos)
        parcial = entrenar(agente, entorno, bloque, semilla=semilla if hechos == 0 else None)
        if historial is None:
            historial = parcial
        else:
            desfase = historial["segundos"][-1]
            for clave, valores in parcial.items():
                if clave == "episodio":
                    valores = [v + hechos for v in valores]
                elif clave == "segundos":
                    valores = [v + desfase for v in valores]
                historial[clave].extend(valores)
        hechos += bloque
        resumen = resumir(evaluar_politica(crear_entorno, politica_tabla_q(agente.q), semillas_validacion))
        curva.append({"episodio": hechos, "segundos": historial["segundos"][-1], "epsilon": agente.epsilon,
                      **{f"val_{k}": v for k, v in resumen.items() if k != "episodios"}})
    return historial, curva


def politica_tabla_q(tabla_q):
    """Política codiciosa con la firma del notebook de evaluación: (estado, entorno, rng) -> acción."""
    tabla_q = np.asarray(tabla_q, dtype=float)

    def politica(estado, _entorno, rng):
        valores = tabla_q[int(estado)]
        return int(rng.choice(np.flatnonzero(valores == valores.max())))

    return politica


def politica_aleatoria(_estado, entorno, rng):
    return int(rng.integers(entorno.action_space.n))


def ejecutar_episodio(entorno, politica, semilla, opciones=None, guardar_trayectoria=False):
    """Un episodio de evaluación, igual al de `03_evaluacion_rl.ipynb`: sin exploración ni aprendizaje."""
    estado, info = entorno.reset(seed=semilla, options=opciones)
    rng = np.random.default_rng(semilla)
    total, terminado, truncado = 0.0, False, False
    trayectoria = []
    while not (terminado or truncado):
        accion = politica(estado, entorno, rng)
        carga_real, carga_percibida = info["carga_real"], info["carga_percibida"]
        estado, recompensa, terminado, truncado, info = entorno.step(accion)
        total += recompensa
        if guardar_trayectoria:
            trayectoria.append({
                "paso": info["pasos"], "accion": int(accion), "fila": info["posicion"][0],
                "columna": info["posicion"][1], "evento": info["evento"], "recompensa": recompensa,
                "carga_real_antes": carga_real, "carga_percibida_antes": carga_percibida,
                "prendas_pendientes": info["prendas_pendientes"], "correctas": info["correctas"],
                "ocupacion": " ".join(str(int(x)) for x in info["ocupacion"]),
            })
    correctas, total_prendas = int(info["correctas"]), int(entorno.total_prendas)
    fila = {
        "semilla": semilla,
        "recompensa_total": total,
        "exito_prendas": correctas / total_prendas if total_prendas else 0.0,
        "exito_episodio": int(terminado and correctas == total_prendas and info["incorrectas"] == 0
                              and info["devueltas"] == 0),
        "pasos": int(info["pasos"]),
        "pasos_por_prenda": info["pasos"] / total_prendas if total_prendas else 0.0,
        "terminado": bool(terminado),
        "truncado": bool(truncado),
        "correctas": correctas,
        "incorrectas": int(info["incorrectas"]),
        "fallos_cnn": int(info["fallos_cnn"]),
        "fallos_politica": int(info["fallos_politica"]),
        "acciones_invalidas": int(info["acciones_invalidas"]),
        "intentos_estante_lleno": int(info["intentos_estante_lleno"]),
        "devueltas": int(info["devueltas"]),
    }
    return (fila, trayectoria) if guardar_trayectoria else fila


def evaluar_politica(crear_entorno, politica, semillas):
    """Evalúa una política en las semillas dadas, con un entorno nuevo por episodio."""
    return [ejecutar_episodio(crear_entorno(), politica, int(s)) for s in semillas]


def resumir(filas):
    """Promedios con los mismos nombres de columnas de `comparacion_agentes.csv`."""
    def media(clave):
        return float(np.mean([f[clave] for f in filas]))

    recompensas = np.array([f["recompensa_total"] for f in filas], dtype=float)
    return {
        "episodios": len(filas),
        "recompensa_promedio": float(recompensas.mean()),
        "recompensa_desv": float(recompensas.std(ddof=0)),
        "tasa_exito_prendas": media("exito_prendas"),
        "tasa_exito_episodios": media("exito_episodio"),
        "pasos_promedio": media("pasos"),
        "incorrectas_promedio": media("incorrectas"),
        "fallos_cnn_promedio": media("fallos_cnn"),
        "fallos_politica_promedio": media("fallos_politica"),
        "acciones_invalidas_promedio": media("acciones_invalidas"),
        "intentos_estante_lleno_promedio": media("intentos_estante_lleno"),
        "devueltas_promedio": media("devueltas"),
    }


def guardar_tabla_q(ruta, agente, **metadatos):
    """Guarda la tabla Q (clave `q_table`, forma 1408 x 6) y sus metadatos en un .npz."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = {**agente.hiperparametros(), **metadatos}
    np.savez_compressed(ruta, q_table=agente.q, metadatos=np.array(json.dumps(datos, ensure_ascii=False)))
    return ruta


def cargar_tabla_q(ruta):
    """Devuelve (tabla_q, metadatos) desde un .npz guardado con `guardar_tabla_q`."""
    with np.load(ruta, allow_pickle=False) as datos:
        tabla = datos["q_table"].copy()
        metadatos = json.loads(str(datos["metadatos"])) if "metadatos" in datos.files else {}
    if tabla.shape != (N_ESTADOS, N_ACCIONES):
        raise ValueError(f"La tabla Q debe tener forma ({N_ESTADOS}, {N_ACCIONES}); tiene {tabla.shape}")
    return tabla, metadatos
