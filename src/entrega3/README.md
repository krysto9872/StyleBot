# Código reutilizable de la Entrega 3

- `entorno_almacen.py`: entorno Gymnasium 8×8 y reglas del MDP. Responsable: Luis Ángel Mendoza Gaytán.
- `probar_entorno.py`: validaciones del entorno y políticas de referencia.
- `validar_integracion.py`: revisión del líder para comprobar el protocolo común y avisar qué entregables faltan. Responsable: Krysto Alberto Adan Jauregui Zúñiga.
- `agentes_rl.py`: Q-Learning y Dyna-Q. Responsable: Luis Ángel Mendoza Gaytán.
- `percepcion_cnn.py`: carga de la CNN congelada y predicción por categoría. Responsable: Juan José Aranda Flores.

Todos los archivos deben usar las constantes definidas en `datos/entrega3/protocolo_entrega3.json` o valores equivalentes comprobados por `00_protocolo_entrega3.ipynb`.

Para revisar la compatibilidad actual sin exigir que la entrega esté terminada:

```bash
python src/entrega3/validar_integracion.py
```

Antes del cierre final, se debe usar el modo estricto:

```bash
python src/entrega3/validar_integracion.py --estricto
```
