"""Pruebas del entorno del almacén (Entrega 3). Ejecutar: python src/entrega3/probar_entorno.py

Comprueba que el entorno cumple la API de Gymnasium, que todos los estantes son alcanzables,
que cada regla de recompensa funciona y compara un agente aleatorio contra una política
óptima calculada con búsqueda en anchura (sirve como referencia para Q-Learning y Dyna-Q).
"""
import numpy as np
from gymnasium.utils.env_checker import check_env

from entorno_almacen import (AlmacenStyleBot, CATEGORIAS, ESTANTES, MOVIMIENTOS, SIN_CARGA,
                             ABAJO, COLOCAR, TOMAR)


def politica_optima(env):
    """Camino más corto: recepción -> tomar -> estante percibido -> colocar (o devolver si está lleno)."""
    if env.carga_percibida == SIN_CARGA:
        if env.posicion in env.recepcion:
            return TOMAR
        objetivo = env.inicio
    elif env._destino_lleno():
        if env.posicion in env.recepcion:
            return COLOCAR
        objetivo = env.inicio
    else:
        objetivo = env.posicion_estante[env.carga_percibida]
        if env.posicion == objetivo:
            return COLOCAR
    distancias = env.distancias_desde(objetivo)
    f, c = env.posicion
    opciones = [a for a, (df, dc) in MOVIMIENTOS.items() if (f + df, c + dc) in distancias]
    return min(opciones, key=lambda a: distancias[(f + MOVIMIENTOS[a][0], c + MOVIMIENTOS[a][1])])


def evaluar(env, politica, episodios=200, semilla=0):
    recompensas, aciertos = [], []
    for k in range(episodios):
        env.reset(seed=semilla + k)
        total, fin = 0.0, False
        while not fin:
            _, r, terminado, truncado, info = env.step(politica(env))
            total += r
            fin = terminado or truncado
        recompensas.append(total)
        aciertos.append(info["correctas"] / env.total_prendas)
    return float(np.mean(recompensas)), float(np.mean(aciertos))


if __name__ == "__main__":
    # 1. API de Gymnasium
    check_env(AlmacenStyleBot(), skip_render_check=True)
    print("1. check_env de Gymnasium: correcto")

    # 2. Todos los estantes se pueden alcanzar desde recepción
    env = AlmacenStyleBot()
    distancias = env.distancias_desde(env.inicio)
    assert all(p in distancias for p in ESTANTES)
    print("2. Distancia desde recepción a cada estante:",
          {CATEGORIAS[c]: distancias[p] for p, c in sorted(ESTANTES.items(), key=lambda x: x[1])})

    # 3. Reglas de recompensa
    env.reset(seed=0, options={"indices": [3], "ocupacion": [0] * 10})
    assert env.step(COLOCAR)[1] == -2      # colocar sin carga
    assert env.step(ABAJO)[1] == -2        # chocar con la pared
    assert env.step(TOMAR)[1] == -1        # tomar en recepción
    assert env.step(TOMAR)[1] == -2        # tomar con carga
    assert env.step(COLOCAR)[1] == -2      # devolver sin que el estante esté lleno
    ocupacion = [0] * 10
    ocupacion[3] = env.capacidad
    env.reset(seed=0, options={"indices": [3], "ocupacion": ocupacion})
    env.step(TOMAR)
    _, r, terminado, _, info = env.step(COLOCAR)
    assert r == -3 and info["evento"] == "devuelta" and terminado
    env.reset(seed=0, options={"indices": [3], "ocupacion": ocupacion})
    env.step(TOMAR)
    for accion in [0, 0, 0, 0, 0, 0]:
        env.step(accion)               # sube por la columna 3 hasta el estante 3 (lleno)
    assert env.posicion == (1, 3)
    env.step(0)
    assert env.step(COLOCAR)[1] == -3      # estante lleno

    def llevar_al_estante_3(etiqueta, prediccion):
        e = AlmacenStyleBot(etiquetas=[etiqueta], predicciones=[prediccion])
        e.reset(seed=0, options={"indices": [0], "ocupacion": [0] * 10})
        e.step(TOMAR)
        for _ in range(7):
            e.step(0)                  # recepción (7, 3) -> estante 3 en (0, 3)
        return e.step(COLOCAR)

    _, r, terminado, _, info = llevar_al_estante_3(3, 3)
    assert r == 10 and info["correctas"] == 1 and terminado
    _, r, _, _, info = llevar_al_estante_3(4, 3)     # la CNN dijo 3 pero era 4: fallo de la CNN
    assert r == -8 and info["fallos_cnn"] == 1
    _, r, _, _, info = llevar_al_estante_3(4, 4)     # la CNN acertó pero el robot fue a otro estante
    assert r == -8 and info["fallos_politica"] == 1
    print("3. Recompensas: +10 correcto, -8 incorrecto, -3 lleno o devuelta, -2 inválida, -1 paso: correcto")

    # 4. Aleatorio contra política óptima
    rng = np.random.default_rng(0)
    print("4. Recompensa media y aciertos en 200 episodios de 5 prendas:")
    print("   Aleatorio:", evaluar(AlmacenStyleBot(), lambda e: int(rng.integers(6))))
    print("   Óptima:   ", evaluar(AlmacenStyleBot(), politica_optima))
    print("   Óptima con 10% de estantes llenos:", evaluar(AlmacenStyleBot(prob_estante_lleno=0.1), politica_optima))

    # 5. Percepción con errores (CNN de ~93%) y simulación de 50 prendas
    etiquetas = rng.integers(0, 10, 5000)
    predicciones = np.where(rng.random(5000) < 0.07, rng.integers(0, 10, 5000), etiquetas)
    env50 = AlmacenStyleBot(etiquetas=etiquetas, predicciones=predicciones, prendas_por_episodio=50)
    recompensa, aciertos = evaluar(env50, politica_optima, episodios=20)
    print(f"5. 50 prendas con percepción de ~93%: recompensa {recompensa:.1f}, aciertos {aciertos:.1%}, "
          f"último episodio: {env50.contadores}")
