# Laberinto IA — proyecto base

Videojuego 2D en Python + Pygame con un jugador, un robot perseguidor y comparación de rutas mediante Dijkstra y A*.

La interfaz usa un estilo tecnológico de IA con consola oscura, acentos neón y panel de estado.
Las paredes cambian de color neón al iniciar cada partida. El usuario controla un tigre con uniforme GTU y el robot perseguidor es el enemigo.

## Requisitos
- Python 3.10 o posterior
- Pygame

## Instalar
Abre una terminal en esta carpeta y ejecuta:

```bash
python -m pip install -r requirements.txt
```

En Windows, si `python` no funciona, prueba `py -m pip install -r requirements.txt`.

## Ejecutar
```bash
python main.py
```

## Controles
- En el menú inicial: iniciar partida, consultar instrucciones o abrir la tienda
- Flechas o WASD: mover al jugador
- R: reiniciar
- M al terminar: volver al menú y acceder a la tienda
- 1: usar Dijkstra para que el robot calcule la ruta
- 2: usar A* para que el robot calcule la ruta
- Clic en el botón Dijkstra: muestra la ruta a la estrella pendiente más cercana y la recalcula al moverte; tras las tres, guía a la salida
- ESC: salir
- Recoge las 3 estrellas y llega a la salida para obtener 100 puntos
- El mapa, las posiciones del jugador, robot y salida, las estrellas y las 5 cajas cambian al reiniciar
- Algunas cajas aparecen agrietadas: empújalas con el jugador dos veces para romperlas y abrir el paso
- El temporizador muestra el tiempo de juego y se detiene al ganar o ser atrapado
- Al terminar aparece una tabla con puntuación, estrellas, celdas recorridas, celdas únicas y tiempo
- Pulsa el botón Dijkstra para seguir las estrellas pendientes por cercanía y después llegar a la salida
- Tienda con personajes que se compran con estrellas; moneda, desbloqueos y personaje equipado se guardan

## Qué implementa
- Laberinto generado al azar en cada partida, con rutas alternativas y salida alcanzable
- Movimiento del jugador con colisiones
- Tigre GTU controlado por el usuario; robot perseguidor como enemigo
- Tres estrellas coleccionables; la puntuación final depende de cuántas recogiste (1 = 33, 2 = 66, 3 = 100)
- Distribución aleatoria de estrellas y cajas en cada partida; el nivel conserva un camino hacia la salida
- Dos cajas rompibles por partida que se distinguen por sus grietas
- Robot que recalcula su ruta hacia el jugador
- Dijkstra: búsqueda de camino de costo mínimo en una cuadrícula sin pesos
- A*: búsqueda informada usando distancia Manhattan como heurística
- Visualización de celdas exploradas y ruta encontrada
- Comparación básica del número de celdas exploradas

## Nota sobre IA
Es una IA basada en búsqueda y toma de decisiones: el robot observa la posición del jugador y calcula una ruta. No es aprendizaje automático; no entrena con ejemplos. Si la consigna exige machine learning, se debe añadir un componente entrenado por separado.
