import heapq
import json
import math
import random
from pathlib import Path
import pygame

# 0 = camino; 1 = pared. El mapa debe tener ancho/alto impares para este diseño.
MAZE = [
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
]
TILE = 40
HUD = 100
FOOTER = 42
ROWS, COLS = len(MAZE), len(MAZE[0])
WIDTH, HEIGHT = COLS * TILE, ROWS * TILE + HUD + FOOTER
FPS = 60

BG = (6, 11, 25)
PANEL = (12, 22, 42)
WALL = (22, 38, 65)
FLOOR = (12, 23, 41)
GRID = (20, 43, 68)
PLAYER_COLOR = (50, 232, 255)
ROBOT_COLOR = (255, 67, 143)
GOAL_COLOR = (76, 255, 190)
STAR_COLOR = (255, 211, 74)
PATH_COLOR = (19, 105, 132)
VISITED_COLOR = (13, 49, 76)
GOAL_ROUTE_COLOR = (247, 94, 225)
TEXT = (222, 240, 255)
MUTED = (112, 150, 180)
OBSTACLE_COLOR = (42, 62, 87)
OBSTACLE_LIGHT = (91, 135, 171)
WALL_THEMES = (
    ((0, 235, 255), (8, 49, 72)),
    ((255, 65, 198), (65, 18, 61)),
    ((165, 255, 58), (40, 59, 18)),
    ((255, 139, 48), (67, 37, 18)),
    ((173, 112, 255), (39, 24, 72)),
    ((64, 150, 255), (14, 37, 77)),
)
SAVE_FILE = Path(__file__).with_name("save_data.json")
CHARACTERS = (
    {"id": "gtu", "name": "Tigre GTU", "cost": 0,
     "fur": (255, 165, 48), "stripe": (31, 26, 27),
     "shirt": (11, 48, 70), "accent": (50, 232, 255),
     "description": "El campeon original del GTU."},
    {"id": "arctic", "name": "Tigre Artico", "cost": 5,
     "fur": (228, 240, 250), "stripe": (36, 61, 83),
     "shirt": (21, 66, 100), "accent": (114, 226, 255),
     "description": "Pelaje claro y uniforme polar."},
    {"id": "cyber", "name": "Tigre Cyber", "cost": 8,
     "fur": (198, 126, 255), "stripe": (44, 20, 63),
     "shirt": (49, 28, 91), "accent": (225, 154, 255),
     "description": "Una apariencia salida del laboratorio."},
    {"id": "gold", "name": "Tigre Dorado", "cost": 12,
     "fur": (255, 207, 66), "stripe": (66, 41, 15),
     "shirt": (74, 50, 14), "accent": (255, 224, 107),
     "description": "Edicion especial para expertos."},
)


def generate_maze():
    """Crea un laberinto nuevo con pasillos conectados y algunas rutas alternas."""
    grid = [["1"] * COLS for _ in range(ROWS)]
    start = (random.randrange(1, ROWS, 2), random.randrange(1, COLS, 2))
    grid[start[0]][start[1]] = "0"
    stack = [start]

    while stack:
        row, col = stack[-1]
        options = [(row + dr * 2, col + dc * 2, dr, dc)
                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                   if 1 <= row + dr * 2 < ROWS - 1
                   and 1 <= col + dc * 2 < COLS - 1
                   and grid[row + dr * 2][col + dc * 2] == "1"]
        if not options:
            stack.pop()
            continue
        next_row, next_col, dr, dc = random.choice(options)
        grid[row + dr][col + dc] = "0"
        grid[next_row][next_col] = "0"
        stack.append((next_row, next_col))

    # Abre algunos cruces para que existan rutas alternativas.
    for row in range(1, ROWS - 1):
        for col in range(1, COLS - 1):
            if grid[row][col] != "1":
                continue
            vertical = grid[row - 1][col] == grid[row + 1][col] == "0"
            horizontal = grid[row][col - 1] == grid[row][col + 1] == "0"
            if (vertical or horizontal) and random.random() < 0.16:
                grid[row][col] = "0"

    return ["".join(row) for row in grid]


def cell_center(cell):
    r, c = cell
    return c * TILE + TILE // 2, HUD + r * TILE + TILE // 2


def walkable(cell, obstacles=None):
    r, c = cell
    return (0 <= r < ROWS and 0 <= c < COLS and MAZE[r][c] == "0"
            and cell not in (obstacles or set()))


def neighbors(cell, obstacles=None):
    r, c = cell
    for nxt in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
        if walkable(nxt, obstacles):
            yield nxt


def distances_from(start):
    distances = {start: 0}
    frontier = [start]
    while frontier:
        current = frontier.pop(0)
        for nxt in neighbors(current):
            if nxt not in distances:
                distances[nxt] = distances[current] + 1
                frontier.append(nxt)
    return distances


def dijkstra(start, goal, obstacles=None):
    """Devuelve (ruta, explorados). Encuentra ruta de costo mínimo."""
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
        for nxt in neighbors(current, obstacles):
            new_cost = current_cost + 1
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(frontier, (new_cost, nxt))
    return reconstruct(came_from, start, goal), explored


def heuristic(a, b):
    # Distancia Manhattan: adecuada para movimiento en cuatro direcciones.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, obstacles=None):
    """Devuelve (ruta, explorados). A* combina costo real y heurística."""
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
        for nxt in neighbors(current, obstacles):
            new_cost = cost[current] + 1
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(frontier, (new_cost + heuristic(nxt, goal), new_cost, nxt))
    return reconstruct(came_from, start, goal), explored



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


class MazeGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("NEURAL MAZE | AI Pursuit System")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20, bold=True)
        self.title_font = pygame.font.SysFont("consolas", 23, bold=True)
        self.small = pygame.font.SysFont("consolas", 14)
        self.tiny = pygame.font.SysFont("consolas", 11, bold=True)
        self.badge_font = pygame.font.SysFont("consolas", 8, bold=True)
        self.route_button_rect = pygame.Rect(493, 62, 253, 28)
        self.menu_play_rect = pygame.Rect(92, 548, 260, 62)
        self.menu_instructions_rect = pygame.Rect(408, 548, 260, 62)
        self.menu_shop_rect = pygame.Rect(582, 28, 150, 42)
        self.back_button_rect = pygame.Rect(250, 570, 260, 54)
        self.shop_back_rect = pygame.Rect(610, 24, 122, 40)
        self.shop_item_buttons = {}
        self.shop_message = ""
        self.screen_mode = "menu"
        self.wall_theme = None
        self.load_progress()
        self.reset()

    def load_progress(self):
        self.wallet_stars = 0
        self.unlocked_characters = {"gtu"}
        self.selected_character = "gtu"
        try:
            data = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            self.wallet_stars = max(0, int(data.get("wallet_stars", 0)))
            valid_ids = {character["id"] for character in CHARACTERS}
            self.unlocked_characters.update(
                character_id for character_id in data.get("unlocked_characters", [])
                if character_id in valid_ids)
            saved_character = data.get("selected_character", "gtu")
            if saved_character in self.unlocked_characters:
                self.selected_character = saved_character
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def save_progress(self):
        data = {
            "wallet_stars": self.wallet_stars,
            "unlocked_characters": sorted(self.unlocked_characters),
            "selected_character": self.selected_character,
        }
        try:
            SAVE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def draw_button(self, rect, label, accent=PLAYER_COLOR):
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        fill = (15, 49, 68) if hovered else (8, 22, 39)
        pygame.draw.rect(self.screen, fill, rect, border_radius=9)
        pygame.draw.rect(self.screen, accent, rect, 2 if hovered else 1, border_radius=9)
        label_surface = self.small.render(label, True, TEXT if hovered else accent)
        self.screen.blit(label_surface,
                         (rect.centerx - label_surface.get_width() // 2,
                          rect.centery - label_surface.get_height() // 2))

    def draw_wrapped_text(self, text, x, y, max_width, color=TEXT, line_height=22):
        words = text.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if self.small.size(candidate)[0] > max_width and line:
                self.screen.blit(self.small.render(line, True, color), (x, y))
                y += line_height
                line = word
            else:
                line = candidate
        if line:
            self.screen.blit(self.small.render(line, True, color), (x, y))
            y += line_height
        return y

    def buy_or_equip(self, character):
        character_id = character["id"]
        if character_id in self.unlocked_characters:
            self.selected_character = character_id
            self.shop_message = f"{character['name']} equipado"
        elif self.wallet_stars >= character["cost"]:
            self.wallet_stars -= character["cost"]
            self.unlocked_characters.add(character_id)
            self.selected_character = character_id
            self.shop_message = f"{character['name']} comprado y equipado"
        else:
            missing = character["cost"] - self.wallet_stars
            self.shop_message = f"Te faltan {missing} estrellas para {character['name']}"
        self.save_progress()

    def draw_menu(self):
        self.screen.fill(BG)
        pygame.draw.circle(self.screen, (12, 29, 51), (WIDTH - 55, 155), 104, 1)
        pygame.draw.circle(self.screen, (18, 47, 69), (WIDTH - 55, 155), 82, 1)
        pygame.draw.line(self.screen, (25, 70, 101), (0, 98), (WIDTH, 98), 1)
        self.screen.blit(self.title_font.render("NEURAL MAZE", True, TEXT), (48, 26))
        self.screen.blit(self.tiny.render("AVENTURA DE IA  //  GRUPO TECNOLOGICO UNIVERSITARIO", True,
                                          PLAYER_COLOR), (52, 59))

        wallet_rect = pygame.Rect(390, 27, 174, 42)
        pygame.draw.rect(self.screen, PANEL, wallet_rect, border_radius=8)
        pygame.draw.rect(self.screen, (40, 68, 91), wallet_rect, 1, border_radius=8)
        self.draw_star(wallet_rect.x + 22, wallet_rect.centery, 7)
        self.screen.blit(self.small.render(f"{self.wallet_stars} ESTRELLAS", True, STAR_COLOR),
                         (wallet_rect.x + 36, wallet_rect.y + 13))
        self.draw_button(self.menu_shop_rect, "TIENDA", STAR_COLOR)

        story = pygame.Rect(55, 119, 650, 328)
        pygame.draw.rect(self.screen, PANEL, story, border_radius=16)
        pygame.draw.rect(self.screen, (31, 77, 105), story, 1, border_radius=16)
        self.screen.blit(self.tiny.render("ARCHIVO DE MISION  //  ESTUDIANTE GTU", True, PLAYER_COLOR),
                         (story.x + 28, story.y + 25))
        self.screen.blit(self.font.render("La prueba del laberinto", True, TEXT),
                         (story.x + 28, story.y + 56))
        narrative = (
            "Eres el tigre estudiante del Grupo Tecnologico Universitario (GTU). "
            "Durante una practica de inteligencia artificial, el robot de seguridad "
            "toma el control del laberinto y bloquea la salida. Reune tres estrellas, "
            "usa las rutas de Dijkstra para orientarte y llega a la salida antes de que "
            "el robot te atrape. Las estrellas que recuperes tambien sirven para "
            "desbloquear nuevos personajes en la tienda."
        )
        self.draw_wrapped_text(narrative, story.x + 28, story.y + 100, 430,
                               color=(181, 205, 224), line_height=22)
        self.draw_tiger(story.right - 72, story.y + 211, scale=1.55,
                        character_id=self.selected_character)
        equipped = next(character["name"] for character in CHARACTERS
                        if character["id"] == self.selected_character)
        equipped_text = self.tiny.render(f"EQUIPADO: {equipped.upper()}", True, STAR_COLOR)
        self.screen.blit(equipped_text,
                         (story.right - equipped_text.get_width() - 24, story.bottom - 34))

        self.draw_button(self.menu_play_rect, "INICIAR PARTIDA", PLAYER_COLOR)
        self.draw_button(self.menu_instructions_rect, "COMO JUGAR", GOAL_COLOR)
        self.screen.blit(self.tiny.render("GANA ESTRELLAS EN CADA NIVEL Y CANJEALAS POR PERSONAJES",
                                          True, MUTED), (WIDTH // 2 - 188, 626))
        pygame.display.flip()

    def draw_instructions(self):
        self.screen.fill(BG)
        self.screen.blit(self.title_font.render("INSTRUCCIONES", True, TEXT), (54, 33))
        panel = pygame.Rect(55, 94, 650, 440)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=14)
        pygame.draw.rect(self.screen, (31, 77, 105), panel, 1, border_radius=14)
        instructions = (
            ("OBJETIVO", "Guia al tigre GTU hasta la salida mientras evitas al robot enemigo."),
            ("MOVIMIENTO", "Usa las flechas o WASD para recorrer el laberinto."),
            ("ESTRELLAS", "Recoge las tres estrellas para obtener hasta 100 puntos. Cada estrella tambien se suma a tu tienda."),
            ("RUTA DIJKSTRA", "Pulsa el boton para ir a la estrella pendiente mas cercana. Al reunirlas, te guiara a la salida."),
            ("CAJAS", "Las cajas agrietadas se rompen al chocar contra ellas dos veces."),
            ("ALGORITMOS", "Las teclas 1 y 2 cambian entre Dijkstra y A* para la persecucion del robot."),
            ("NUEVO NIVEL", "Pulsa R para generar otro mapa. El tiempo y las estadisticas se muestran al finalizar."),
            ("TIENDA", "Gasta las estrellas guardadas para desbloquear y equipar otros personajes GTU."),
        )
        y = panel.y + 24
        for heading, detail in instructions:
            self.screen.blit(self.tiny.render(heading, True, PLAYER_COLOR), (panel.x + 26, y))
            y = self.draw_wrapped_text(detail, panel.x + 180, y - 2, 430,
                                       color=TEXT, line_height=20) + 11
        self.draw_button(self.back_button_rect, "VOLVER AL MENU", PLAYER_COLOR)
        pygame.display.flip()

    def draw_shop(self):
        self.screen.fill(BG)
        self.screen.blit(self.title_font.render("TIENDA DE PERSONAJES", True, TEXT), (42, 28))
        wallet_label = self.small.render(f"{self.wallet_stars} ESTRELLAS", True, STAR_COLOR)
        self.screen.blit(wallet_label, (395, 38))
        self.draw_star(382, 45, 7)
        self.draw_button(self.shop_back_rect, "VOLVER", PLAYER_COLOR)
        self.shop_item_buttons = {}

        for index, character in enumerate(CHARACTERS):
            col, row = index % 2, index // 2
            card = pygame.Rect(42 + col * 350, 104 + row * 220, 326, 192)
            pygame.draw.rect(self.screen, PANEL, card, border_radius=12)
            pygame.draw.rect(self.screen, character["accent"], card, 1, border_radius=12)
            self.draw_tiger(card.x + 48, card.y + 75, scale=1.0,
                            character_id=character["id"])
            self.screen.blit(self.small.render(character["name"], True, TEXT),
                             (card.x + 93, card.y + 25))
            self.screen.blit(self.tiny.render(character["description"], True, MUTED),
                             (card.x + 93, card.y + 51))
            owned = character["id"] in self.unlocked_characters
            selected = character["id"] == self.selected_character
            if selected:
                label = "EQUIPADO"
            elif owned:
                label = "EQUIPAR"
            else:
                label = f"COMPRAR  //  {character['cost']} ESTRELLAS"
            button = pygame.Rect(card.x + 92, card.y + 123, 214, 42)
            self.shop_item_buttons[character["id"]] = button
            self.draw_button(button, label, character["accent"])

        message_color = GOAL_COLOR if "faltan" not in self.shop_message.lower() else ROBOT_COLOR
        if self.shop_message:
            message = self.small.render(self.shop_message, True, message_color)
            self.screen.blit(message, (WIDTH // 2 - message.get_width() // 2, 558))
        else:
            note = self.tiny.render("LAS COMPRAS SE GUARDAN AUTOMATICAMENTE", True, MUTED)
            self.screen.blit(note, (WIDTH // 2 - note.get_width() // 2, 565))
        pygame.display.flip()

    def reset(self):
        global MAZE
        available_themes = [theme for theme in WALL_THEMES if theme != self.wall_theme]
        self.wall_theme = random.choice(available_themes)
        self.wall_neon, self.wall_dark = self.wall_theme
        MAZE = generate_maze()
        floor_cells = [(r, c) for r in range(ROWS) for c in range(COLS)
                       if MAZE[r][c] == "0"]
        while True:
            player = random.choice(floor_cells)
            player_distances = distances_from(player)
            possible_goals = [cell for cell in floor_cells
                              if player_distances[cell] >= 10]
            if not possible_goals:
                continue
            goal = random.choice(possible_goals)
            goal_distances = distances_from(goal)
            possible_robots = [cell for cell in floor_cells
                               if cell not in (player, goal)
                               and player_distances[cell] >= 8
                               and goal_distances[cell] >= 5]
            if possible_robots:
                self.player = player
                self.goal = goal
                self.robot = random.choice(possible_robots)
                break
        self.algorithm = "A*"
        self.path = []
        self.explored = []
        self.explored_count = 0
        self.won = False
        self.caught = False
        self.started_at = pygame.time.get_ticks()
        self.finished_at = None
        self.show_goal_route = False
        self.goal_route = []
        self.route_target_label = ""
        self.route_target = None
        self.obstacles, reachable = self.create_level()
        breakable_cells = random.sample(list(self.obstacles), 2)
        self.breakable_obstacles = {cell: 2 for cell in breakable_cells}
        star_candidates = [cell for cell in reachable
                           if cell not in (self.player, self.robot, self.goal)
                           and heuristic(cell, self.player) >= 3]
        self.stars = set(random.sample(star_candidates, 3))
        self.stars_collected = 0
        self.moves = 0
        self.visited_cells = {self.player}
        self.score = 0
        self.replan()

    def begin_game(self):
        self.reset()
        self.screen_mode = "game"
        self.started_at = pygame.time.get_ticks()

    def create_level(self):
        """Genera objetos y estrellas al azar sin bloquear la salida."""
        reserved = {self.player, self.robot, self.goal}
        floor_cells = [(r, c) for r in range(ROWS) for c in range(COLS)
                       if MAZE[r][c] == "0" and (r, c) not in reserved]

        while True:
            obstacles = set(random.sample(floor_cells, 5))
            reachable = {self.player}
            frontier = [self.player]
            while frontier:
                for nxt in neighbors(frontier.pop(), obstacles):
                    if nxt not in reachable:
                        reachable.add(nxt)
                        frontier.append(nxt)
            if self.goal in reachable and self.robot in reachable:
                candidates = [cell for cell in reachable
                              if cell not in reserved and heuristic(cell, self.player) >= 3]
                if len(candidates) >= 3:
                    return obstacles, list(reachable)

    def collect_star(self):
        if self.player in self.stars:
            self.stars.remove(self.player)
            self.stars_collected += 1
            self.score = self.stars_collected * 100 // 3
            self.wallet_stars += 1
            self.save_progress()

    def elapsed_ms(self):
        end_time = self.finished_at if self.finished_at is not None else pygame.time.get_ticks()
        return max(0, end_time - self.started_at)

    def formatted_time(self):
        total_seconds = self.elapsed_ms() // 1000
        return f"{total_seconds // 60:02}:{total_seconds % 60:02}"

    def finish_if_needed(self):
        if (self.won or self.caught) and self.finished_at is None:
            self.finished_at = pygame.time.get_ticks()

    def toggle_goal_route(self):
        self.show_goal_route = not self.show_goal_route
        if self.show_goal_route:
            self.update_objective_route()
        else:
            self.goal_route = []
            self.route_target_label = ""
            self.route_target = None

    def update_objective_route(self):
        """Recalcula el objetivo más cercano desde la posición actual del tigre."""
        if self.stars:
            options = []
            for star in self.stars:
                route, _ = dijkstra(self.player, star, self.obstacles)
                if route:
                    options.append((len(route), star))
            if options:
                self.route_target = min(options, key=lambda option: (option[0], option[1]))[1]
            if self.route_target in self.stars:
                self.goal_route, _ = dijkstra(self.player, self.route_target, self.obstacles)
                if self.goal_route:
                    self.route_target_label = f"ESTRELLA {self.stars_collected + 1}/3"
                    return
        self.route_target = self.goal
        self.goal_route, _ = dijkstra(self.player, self.goal, self.obstacles)
        self.route_target_label = "SALIDA"

    def replan(self):
        search = astar if self.algorithm == "A*" else dijkstra
        self.path, self.explored = search(self.robot, self.player, self.obstacles)
        self.explored_count = len(self.explored)
        if self.show_goal_route:
            self.update_objective_route()

    def move_player(self, dr, dc):
        if self.won or self.caught:
            return
        nxt = (self.player[0] + dr, self.player[1] + dc)
        if nxt in self.breakable_obstacles:
            self.breakable_obstacles[nxt] -= 1
            if self.breakable_obstacles[nxt] == 0:
                self.obstacles.remove(nxt)
                del self.breakable_obstacles[nxt]
                self.replan()
            return
        if walkable(nxt, self.obstacles):
            self.player = nxt
            self.moves += 1
            self.visited_cells.add(self.player)
            self.collect_star()
            if self.player == self.goal:
                self.won = True
            if self.player == self.robot:
                self.caught = True
            self.finish_if_needed()
            self.replan()

    def move_robot(self):
        if self.won or self.caught or not self.path:
            return
        # Recalcula el camino a la posición actual del jugador y avanza un paso.
        self.replan()
        if len(self.path) > 1:
            self.robot = self.path[1]
        if self.robot == self.player:
            self.caught = True
            self.finish_if_needed()

    def draw(self):
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, PANEL, (0, 0, WIDTH, HUD))
        pygame.draw.line(self.screen, (25, 70, 101), (0, HUD - 1), (WIDTH, HUD - 1), 2)
        # Emblema y título de la consola.
        pygame.draw.circle(self.screen, PLAYER_COLOR, (25, 27), 14, 2)
        pygame.draw.circle(self.screen, (18, 84, 112), (25, 27), 8, 1)
        pygame.draw.line(self.screen, PLAYER_COLOR, (18, 27), (32, 27), 1)
        pygame.draw.line(self.screen, PLAYER_COLOR, (25, 20), (25, 34), 1)
        self.screen.blit(self.title_font.render("NEURAL MAZE", True, TEXT), (48, 8))
        self.screen.blit(self.tiny.render("GTU TIGER  //  ROBOT HUNTER", True, PLAYER_COLOR),
                         (50, 39))

        status = pygame.Rect(493, 12, 253, 36)
        pygame.draw.rect(self.screen, (8, 17, 34), status, border_radius=8)
        pygame.draw.rect(self.screen, (27, 74, 105), status, 1, border_radius=8)
        status_text = self.small.render(
            f"IA {self.algorithm}   /   NODOS {self.explored_count:03}", True, PLAYER_COLOR)
        self.screen.blit(status_text, (status.x + 12, status.y + 10))

        stats = self.small.render("ESTRELLAS", True, MUTED)
        self.screen.blit(stats, (18, 68))
        for index in range(3):
            color = STAR_COLOR if index < self.stars_collected else (45, 63, 82)
            pygame.draw.circle(self.screen, color, (111 + index * 19, 76), 5)
            pygame.draw.circle(self.screen, STAR_COLOR if index < self.stars_collected else MUTED,
                               (111 + index * 19, 76), 5, 1)
        self.screen.blit(self.small.render(f"PUNTOS {self.score:03}", True, STAR_COLOR), (190, 68))
        self.screen.blit(self.small.render(f"TIEMPO {self.formatted_time()}", True, TEXT), (315, 68))
        self.screen.blit(self.tiny.render(f"CELDAS {self.moves}", True, TEXT), (420, 72))
        button_color = GOAL_ROUTE_COLOR if self.show_goal_route else PLAYER_COLOR
        pygame.draw.rect(self.screen, (8, 22, 39), self.route_button_rect, border_radius=7)
        pygame.draw.rect(self.screen, button_color, self.route_button_rect, 1, border_radius=7)
        button_label = (f"{self.route_target_label}" if self.show_goal_route
                        else "BUSCAR ESTRELLA CERCANA")
        button_text = self.tiny.render(f"DIJKSTRA  //  {button_label}", True, button_color)
        self.screen.blit(button_text,
                         (self.route_button_rect.centerx - button_text.get_width() // 2,
                          self.route_button_rect.y + 9))

        for r, row in enumerate(MAZE):
            for c, value in enumerate(row):
                rect = pygame.Rect(c * TILE + 1, HUD + r * TILE + 1, TILE - 2, TILE - 2)
                if value == "1":
                    pygame.draw.rect(self.screen, self.wall_dark, rect, border_radius=4)
                    pygame.draw.rect(self.screen, self.wall_neon, rect, 2, border_radius=4)
                    pygame.draw.line(self.screen, self.wall_neon,
                                     (rect.x + 7, rect.y + 5), (rect.right - 7, rect.y + 5), 1)
                else:
                    pygame.draw.rect(self.screen, FLOOR, rect, border_radius=3)
                    pygame.draw.rect(self.screen, GRID, rect, 1, border_radius=3)

        # Superpone la exploración y la ruta en tonos neón discretos.
        for r, c in self.explored:
            if MAZE[r][c] == "0":
                pygame.draw.rect(self.screen, VISITED_COLOR,
                                 (c * TILE + 3, HUD + r * TILE + 3, TILE - 6, TILE - 6),
                                 border_radius=4)
        for r, c in self.path:
            pygame.draw.rect(self.screen, PATH_COLOR,
                             (c * TILE + 7, HUD + r * TILE + 7, TILE - 14, TILE - 14),
                             border_radius=5)
            pygame.draw.rect(self.screen, (40, 196, 218),
                             (c * TILE + 11, HUD + r * TILE + 11, TILE - 22, TILE - 22),
                             1, border_radius=4)
        if self.show_goal_route and self.goal_route:
            route_points = [cell_center(cell) for cell in self.goal_route]
            if len(route_points) > 1:
                pygame.draw.lines(self.screen, GOAL_ROUTE_COLOR, False, route_points, 4)
            for point in route_points:
                pygame.draw.circle(self.screen, GOAL_ROUTE_COLOR, point, 4)
            target_x, target_y = cell_center(self.route_target)
            pygame.draw.circle(self.screen, GOAL_ROUTE_COLOR, (target_x, target_y), 17, 2)

        # Cajas tecnológicas; las agrietadas se rompen después de dos golpes.
        for r, c in self.obstacles:
            x, y = c * TILE, HUD + r * TILE
            is_breakable = (r, c) in self.breakable_obstacles
            border = STAR_COLOR if is_breakable else OBSTACLE_LIGHT
            pygame.draw.rect(self.screen, OBSTACLE_COLOR,
                             (x + 5, y + 5, TILE - 10, TILE - 10), border_radius=5)
            pygame.draw.rect(self.screen, border,
                             (x + 5, y + 5, TILE - 10, TILE - 10), 2, border_radius=5)
            pygame.draw.line(self.screen, border, (x + 11, y + 12), (x + 29, y + 12), 2)
            pygame.draw.line(self.screen, OBSTACLE_LIGHT,
                             (x + 12, y + 20), (x + 28, y + 20), 1)
            if is_breakable:
                crack = [(x + 13, y + 10), (x + 20, y + 18),
                         (x + 16, y + 23), (x + 25, y + 30)]
                pygame.draw.lines(self.screen, STAR_COLOR, False, crack, 2)
                if self.breakable_obstacles[(r, c)] == 1:
                    pygame.draw.line(self.screen, STAR_COLOR,
                                     (x + 28, y + 10), (x + 23, y + 17), 2)

        gx, gy = cell_center(self.goal)
        pygame.draw.circle(self.screen, (10, 37, 47), (gx, gy), 15)
        pygame.draw.circle(self.screen, GOAL_COLOR, (gx, gy), 15, 2)
        pygame.draw.circle(self.screen, GOAL_COLOR, (gx, gy), 9, 1)
        pygame.draw.circle(self.screen, GOAL_COLOR, (gx, gy), 3)
        for star in self.stars:
            sx, sy = cell_center(star)
            self.draw_star(sx, sy, 12)
        px, py = cell_center(self.player)
        self.draw_tiger(px, py, character_id=self.selected_character)
        rx, ry = cell_center(self.robot)
        pygame.draw.circle(self.screen, (45, 15, 44), (rx, ry), 17)
        pygame.draw.circle(self.screen, ROBOT_COLOR, (rx, ry), 17, 2)
        robot_shape = [(rx, ry - 12), (rx + 12, ry - 5), (rx + 10, ry + 8),
                       (rx, ry + 13), (rx - 10, ry + 8), (rx - 12, ry - 5)]
        pygame.draw.polygon(self.screen, (92, 23, 67), robot_shape)
        pygame.draw.polygon(self.screen, ROBOT_COLOR, robot_shape, 1)
        pygame.draw.circle(self.screen, (255, 221, 241), (rx - 4, ry), 2)
        pygame.draw.circle(self.screen, (255, 221, 241), (rx + 4, ry), 2)

        footer_y = HEIGHT - FOOTER
        pygame.draw.rect(self.screen, PANEL, (0, footer_y, WIDTH, FOOTER))
        pygame.draw.line(self.screen, (25, 70, 101), (0, footer_y), (WIDTH, footer_y), 1)
        help_text = self.tiny.render(
            "FLECHAS / WASD: MOVER   //   1/2: ALGORITMO ENEMIGO   //   CAJA AGRIETADA: 2 GOLPES",
            True, MUTED)
        self.screen.blit(help_text, (16, footer_y + 16))

        if self.won or self.caught:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((3, 8, 20, 190))
            self.screen.blit(overlay, (0, 0))
            panel = pygame.Rect(105, HEIGHT // 2 - 145, WIDTH - 210, 290)
            accent = GOAL_COLOR if self.won else ROBOT_COLOR
            pygame.draw.rect(self.screen, PANEL, panel, border_radius=14)
            pygame.draw.rect(self.screen, accent, panel, 2, border_radius=14)
            heading = "NIVEL COMPLETADO" if self.won else "TIGRE GTU INTERCEPTADO"
            heading_surface = self.title_font.render(heading, True, accent)
            self.screen.blit(heading_surface,
                             (WIDTH // 2 - heading_surface.get_width() // 2, panel.y + 20))
            self.screen.blit(self.tiny.render("RESUMEN DE LA PARTIDA", True, MUTED),
                             (panel.x + 28, panel.y + 59))

            table = pygame.Rect(panel.x + 22, panel.y + 77, panel.width - 44, 156)
            pygame.draw.rect(self.screen, (7, 16, 32), table, border_radius=8)
            pygame.draw.rect(self.screen, (36, 70, 96), table, 1, border_radius=8)
            self.screen.blit(self.tiny.render("ESTADISTICA", True, MUTED), (table.x + 14, table.y + 9))
            self.screen.blit(self.tiny.render("RESULTADO", True, MUTED), (table.right - 112, table.y + 9))
            pygame.draw.line(self.screen, (36, 70, 96),
                             (table.x + 12, table.y + 27), (table.right - 12, table.y + 27), 1)
            rows = (
                ("Puntuacion", f"{self.score}/100"),
                ("Estrellas", f"{self.stars_collected}/3"),
                ("Celdas recorridas", str(self.moves)),
                ("Celdas unicas", str(len(self.visited_cells))),
                ("Tiempo", self.formatted_time()),
            )
            for index, (label, value) in enumerate(rows):
                row_y = table.y + 34 + index * 23
                self.screen.blit(self.tiny.render(label, True, TEXT), (table.x + 14, row_y))
                value_surface = self.tiny.render(value, True, accent)
                self.screen.blit(value_surface, (table.right - 14 - value_surface.get_width(), row_y))
            self.screen.blit(self.tiny.render("R REPETIR NIVEL   //   M VOLVER AL MENU", True, MUTED),
                             (panel.x + 28, panel.bottom - 32))
        pygame.display.flip()

    def draw_tiger(self, x, y, scale=1.0, character_id=None):
        character_id = character_id or self.selected_character
        character = next((item for item in CHARACTERS if item["id"] == character_id), CHARACTERS[0])
        fur, stripe = character["fur"], character["stripe"]
        shirt, accent = character["shirt"], character["accent"]

        def point(dx, dy):
            return round(x + dx * scale), round(y + dy * scale)

        def radius(value):
            return max(1, round(value * scale))

        def rect(dx, dy, width, height):
            return (round(x + dx * scale), round(y + dy * scale),
                    max(1, round(width * scale)), max(1, round(height * scale)))

        uniform = [point(-12, 5), point(-8, 2), point(8, 2), point(12, 5),
                   point(10, 16), point(-10, 16)]
        pygame.draw.polygon(self.screen, shirt, uniform)
        pygame.draw.polygon(self.screen, accent, uniform, max(1, radius(2)))
        pygame.draw.line(self.screen, accent, point(-8, 8), point(8, 8), max(1, radius(1)))
        badge = self.badge_font.render("GTU", True, (225, 255, 255))
        if scale != 1.0:
            badge = pygame.transform.scale(badge,
                                           (max(1, round(badge.get_width() * scale)),
                                            max(1, round(badge.get_height() * scale))))
        self.screen.blit(badge, (round(x - badge.get_width() / 2), round(y + 10 * scale)))

        pygame.draw.polygon(self.screen, fur,
                            [point(-12, -7), point(-11, -17), point(-3, -11)])
        pygame.draw.polygon(self.screen, fur,
                            [point(12, -7), point(11, -17), point(3, -11)])
        pygame.draw.circle(self.screen, fur, point(0, -4), radius(12))
        pygame.draw.circle(self.screen, stripe, point(0, -4), radius(12), max(1, radius(1)))
        pygame.draw.line(self.screen, stripe, point(-2, -15), point(-5, -10), radius(2))
        pygame.draw.line(self.screen, stripe, point(2, -15), point(5, -10), radius(2))
        pygame.draw.line(self.screen, stripe, point(-10, -7), point(-6, -5), radius(2))
        pygame.draw.line(self.screen, stripe, point(10, -7), point(6, -5), radius(2))
        eye_white = (245, 255, 220)
        eye = (25, 22, 24)
        pygame.draw.circle(self.screen, eye_white, point(-4, -5), radius(2))
        pygame.draw.circle(self.screen, eye_white, point(4, -5), radius(2))
        pygame.draw.circle(self.screen, eye, point(-4, -5), radius(1))
        pygame.draw.circle(self.screen, eye, point(4, -5), radius(1))
        pygame.draw.ellipse(self.screen, (255, 226, 184), rect(-7, 0, 14, 7))
        pygame.draw.polygon(self.screen, stripe,
                            [point(-2, 1), point(2, 1), point(0, 3)])
        pygame.draw.line(self.screen, stripe, point(0, 3), point(0, 5), max(1, radius(1)))
        pygame.draw.line(self.screen, (245, 221, 183), point(-10, 1), point(-6, 2), max(1, radius(1)))
        pygame.draw.line(self.screen, (245, 221, 183), point(10, 1), point(6, 2), max(1, radius(1)))

    @staticmethod
    def draw_star(x, y, radius):
        points = []
        for i in range(10):
            angle = -90 + i * 36
            r = radius if i % 2 == 0 else radius * 0.45
            points.append((x + r * math.cos(math.radians(angle)),
                           y + r * math.sin(math.radians(angle))))
        pygame.draw.polygon(pygame.display.get_surface(), STAR_COLOR, points)

    def run(self):
        running = True
        robot_timer = 0
        while running:
            dt = self.clock.tick(FPS)
            if self.screen_mode == "game":
                robot_timer += dt
            else:
                robot_timer = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if self.screen_mode == "game" or self.screen_mode == "menu":
                        running = False
                    else:
                        self.screen_mode = "menu"
                elif self.screen_mode == "menu" and event.type == pygame.MOUSEBUTTONDOWN \
                        and event.button == 1:
                    if self.menu_play_rect.collidepoint(event.pos):
                        self.begin_game()
                        robot_timer = 0
                    elif self.menu_instructions_rect.collidepoint(event.pos):
                        self.screen_mode = "instructions"
                    elif self.menu_shop_rect.collidepoint(event.pos):
                        self.screen_mode = "shop"
                        self.shop_message = ""
                elif self.screen_mode == "instructions" and event.type == pygame.MOUSEBUTTONDOWN \
                        and event.button == 1 and self.back_button_rect.collidepoint(event.pos):
                    self.screen_mode = "menu"
                elif self.screen_mode == "shop" and event.type == pygame.MOUSEBUTTONDOWN \
                        and event.button == 1:
                    if self.shop_back_rect.collidepoint(event.pos):
                        self.screen_mode = "menu"
                    else:
                        for character in CHARACTERS:
                            button = self.shop_item_buttons.get(character["id"])
                            if button and button.collidepoint(event.pos):
                                self.buy_or_equip(character)
                                break
                elif self.screen_mode == "game" and event.type == pygame.MOUSEBUTTONDOWN \
                        and event.button == 1 and self.route_button_rect.collidepoint(event.pos) \
                        and not (self.won or self.caught):
                    self.toggle_goal_route()
                elif self.screen_mode == "game" and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
                        robot_timer = 0
                    elif event.key == pygame.K_m and (self.won or self.caught):
                        self.screen_mode = "menu"
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

            if self.screen_mode == "game" and robot_timer >= 550:
                robot_timer = 0
                self.move_robot()
            if self.screen_mode == "game":
                self.draw()
            elif self.screen_mode == "menu":
                self.draw_menu()
            elif self.screen_mode == "instructions":
                self.draw_instructions()
            elif self.screen_mode == "shop":
                self.draw_shop()
        pygame.quit()


if __name__ == "__main__":
    MazeGame().run()
