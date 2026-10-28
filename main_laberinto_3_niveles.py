import heapq
import pygame

# 0 = camino; 1 = pared.
TILE = 40
HUD = 82
FPS = 60

BG = (18, 22, 34)
WALL = (49, 61, 85)
FLOOR = (235, 239, 245)
GRID = (211, 218, 230)
PLAYER_COLOR = (40, 120, 255)
ROBOT_COLOR = (238, 73, 84)
GOAL_COLOR = (45, 180, 110)
PATH_COLOR = (255, 205, 65)
VISITED_COLOR = (120, 190, 255)
TEXT = (245, 247, 252)

# Tres niveles: cada nivel tiene más enemigos que el anterior.
LEVELS = [
    {
        "name": "Nivel 1",
        "maze": [
            "1111111111111111111",
            "1000001000000000001",
            "1011101011110111101",
            "1010001000010100001",
            "1010111111010101101",
            "1010000000010000101",
            "1011110111111100101",
            "1000010000000000101",
            "1111011110111110101",
            "1000000010000010001",
            "1011111010111011101",
            "1000000010000000001",
            "1111111111111111111",
        ],
        "player": (1, 1),
        "goal": (1, 17),
        "enemies": [(11, 17)],
    },
    {
        "name": "Nivel 2",
        "maze": [
            "1111111111111111111",
            "1000001000000000001",
            "1011101011110111101",
            "1010001000010100001",
            "1010111111010101101",
            "1010000000010000101",
            "1011110111111100101",
            "1000010000000000101",
            "1111011110111110101",
            "1000000010000010001",
            "1011111010111011101",
            "1000000010000000001",
            "1111111111111111111",
        ],
        "player": (1, 1),
        "goal": (1, 17),
        "enemies": [(11, 17), (9, 1)],
    },
    {
        "name": "Nivel 3",
        "maze": [
            "1111111111111111111",
            "1000001000000000001",
            "1011101011110111101",
            "1010001000010100001",
            "1010111111010101101",
            "1010000000010000101",
            "1011110111111100101",
            "1000010000000000101",
            "1111011110111110101",
            "1000000010000010001",
            "1011111010111011101",
            "1000000010000000001",
            "1111111111111111111",
        ],
        "player": (1, 1),
        "goal": (1, 17),
        "enemies": [(11, 17), (9, 1), (5, 7)],
    },
]

LEVEL_COUNT = len(LEVELS)
ROWS, COLS = len(LEVELS[0]["maze"]), len(LEVELS[0]["maze"][0])
WIDTH, HEIGHT = COLS * TILE, ROWS * TILE + HUD


def cell_center(cell):
    r, c = cell
    return c * TILE + TILE // 2, HUD + r * TILE + TILE // 2


def walkable(cell, maze):
    r, c = cell
    return 0 <= r < len(maze) and 0 <= c < len(maze[0]) and maze[r][c] == "0"


def neighbors(cell, maze):
    r, c = cell
    for nxt in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
        if walkable(nxt, maze):
            yield nxt


def reconstruct(came_from, start, goal):
    if goal not in came_from:
        return []
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def dijkstra(start, goal, maze):
    """Devuelve la ruta más corta y las celdas exploradas."""
    frontier = [(0, start)]
    came_from = {start: None}
    cost = {start: 0}
    explored = []

    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current_cost != cost.get(current):
            continue
        explored.append(current)
        if current == goal:
            break

        for nxt in neighbors(current, maze):
            new_cost = current_cost + 1
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(frontier, (new_cost, nxt))

    return reconstruct(came_from, start, goal), explored


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, maze):
    """A* usa el costo real y la distancia Manhattan."""
    frontier = [(heuristic(start, goal), 0, start)]
    came_from = {start: None}
    cost = {start: 0}
    explored = []

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost != cost.get(current):
            continue
        explored.append(current)
        if current == goal:
            break

        for nxt in neighbors(current, maze):
            new_cost = cost[current] + 1
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic(nxt, goal), new_cost, nxt)
                )

    return reconstruct(came_from, start, goal), explored


class MazeGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Laberinto IA — 3 niveles, A* y Dijkstra")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)
        self.small = pygame.font.SysFont("arial", 16)
        self.level_index = 0
        self.algorithm = "A*"
        self.reset_level()

    def reset_level(self):
        level = LEVELS[self.level_index]
        self.maze = level["maze"]
        self.player = level["player"]
        self.goal = level["goal"]
        self.robots = list(level["enemies"])
        self.path = []
        self.explored = []
        self.explored_count = 0
        self.won = False
        self.caught = False
        self.all_won = False
        self.replan()

    def reset_game(self):
        self.level_index = 0
        self.reset_level()

    def replan(self):
        search = astar if self.algorithm == "A*" else dijkstra
        # Cada enemigo calcula su ruta para perseguir al jugador.
        self.robot_paths = []
        explored_union = set()
        for robot in self.robots:
            path, explored = search(robot, self.player, self.maze)
            self.robot_paths.append(path)
            explored_union.update(explored)
        # Mostrar la ruta del enemigo más cercano (el primero con ruta).
        self.path = next((p for p in self.robot_paths if p), [])
        self.explored = list(explored_union)
        self.explored_count = len(self.explored)

    def move_player(self, dr, dc):
        if self.won or self.caught or self.all_won:
            return
        nxt = (self.player[0] + dr, self.player[1] + dc)
        if walkable(nxt, self.maze):
            self.player = nxt
            if self.player == self.goal:
                self.won = True
                if self.level_index == LEVEL_COUNT - 1:
                    self.all_won = True
            if self.player in self.robots:
                self.caught = True
            self.replan()

    def move_robots(self):
        if self.won or self.caught or self.all_won:
            return
        # Cada enemigo avanza un paso hacia la posición actual del jugador.
        new_positions = []
        for robot in self.robots:
            path, _ = (astar if self.algorithm == "A*" else dijkstra)(
                robot, self.player, self.maze
            )
            if len(path) > 1:
                new_positions.append(path[1])
            else:
                new_positions.append(robot)

        self.robots = new_positions
        if self.player in self.robots:
            self.caught = True
        self.replan()

    def draw(self):
        self.screen.fill(BG)
        level_name = LEVELS[self.level_index]["name"]
        title = self.font.render(
            f"LABERINTO IA | {level_name} de {LEVEL_COUNT} | Enemigos: {len(self.robots)}",
            True, TEXT
        )
        self.screen.blit(title, (12, 8))
        subtitle = self.small.render(
            f"Algoritmo: {self.algorithm} | Exploradas: {self.explored_count} "
            "| 1: Dijkstra  2: A* | Flechas/WASD: mover",
            True, TEXT
        )
        self.screen.blit(subtitle, (12, 42))

        # Dibujar el laberinto.
        for r, row in enumerate(self.maze):
            for c, value in enumerate(row):
                rect = pygame.Rect(c * TILE, HUD + r * TILE, TILE, TILE)
                if value == "1":
                    pygame.draw.rect(self.screen, WALL, rect)
                else:
                    pygame.draw.rect(self.screen, FLOOR, rect)
                    pygame.draw.rect(self.screen, GRID, rect, 1)

        # Celdas exploradas.
        for r, c in self.explored:
            pygame.draw.rect(
                self.screen, VISITED_COLOR,
                (c * TILE + 4, HUD + r * TILE + 4, TILE - 8, TILE - 8),
                border_radius=5
            )

        # Ruta del enemigo que se muestra en amarillo.
        for r, c in self.path:
            pygame.draw.rect(
                self.screen, PATH_COLOR,
                (c * TILE + 12, HUD + r * TILE + 12, TILE - 24, TILE - 24),
                border_radius=4
            )

        gx, gy = cell_center(self.goal)
        pygame.draw.circle(self.screen, GOAL_COLOR, (gx, gy), TILE // 3)

        px, py = cell_center(self.player)
        pygame.draw.circle(self.screen, PLAYER_COLOR, (px, py), TILE // 3)

        # Todos los enemigos se dibujan en rojo.
        for rx_cell in self.robots:
            rx, ry = cell_center(rx_cell)
            pygame.draw.circle(self.screen, ROBOT_COLOR, (rx, ry), TILE // 3)

        legend = self.small.render(
            "Azul: jugador | Rojo: enemigos | Verde: salida | Amarillo: ruta",
            True, TEXT
        )
        self.screen.blit(legend, (12, HEIGHT - 24))

        if self.won:
            if self.all_won:
                message = "¡FELICIDADES! Completaste los 3 niveles. Pulsa R para jugar otra vez."
            else:
                message = f"¡Nivel completado! Pulsa ENTER para pasar al nivel {self.level_index + 2}."
            self.draw_overlay(message)
        elif self.caught:
            self.draw_overlay("¡Un enemigo te atrapó! Pulsa R para reiniciar el juego.")

        pygame.display.flip()

    def draw_overlay(self, message):
        overlay = pygame.Surface((WIDTH, 58), pygame.SRCALPHA)
        overlay.fill((15, 20, 30, 230))
        y = HEIGHT // 2 - 29
        self.screen.blit(overlay, (0, y))
        msg = self.small.render(message, True, TEXT)
        self.screen.blit(msg, (max(8, WIDTH // 2 - msg.get_width() // 2), y + 20))

    def run(self):
        running = True
        robot_timer = 0

        while running:
            dt = self.clock.tick(FPS)
            robot_timer += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()
                        robot_timer = 0
                    elif event.key == pygame.K_RETURN and self.won and not self.all_won:
                        self.level_index += 1
                        self.reset_level()
                        robot_timer = 0
                    elif event.key == pygame.K_1:
                        self.algorithm = "Dijkstra"
                        self.replan()
                    elif event.key == pygame.K_2:
                        self.algorithm = "A*"
                        self.replan()
                    elif event.key in (pygame.K_UP, pygame.K_w):
                        self.move_player(-1, 0)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.move_player(1, 0)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        self.move_player(0, -1)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        self.move_player(0, 1)

            if robot_timer >= 550:
                robot_timer = 0
                self.move_robots()

            self.draw()

        pygame.quit()


if __name__ == "__main__":
    MazeGame().run()
