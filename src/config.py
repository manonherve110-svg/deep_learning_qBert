"""
Constantes communes au jeu Q*bert.
Contient les définitions des couleurs et des mouvements possibles pour chaque type d'entité.
"""
# Dictionnaire des couleurs : associe un nom de couleur à sa valeur entière



## ============== COULEUR DES BOTS ============== ##
BOT_COLORS = {
    "neutral": "white",
    "coily": "blue",
    "ugg": "violet",
    "wrong_way": "violet",
    "slick": "yellow",
    "sam": "yellow",
    "red_ball": "red",
    "green_ball": "green"
}

## ============== COULEUR EN INT ============== ##
COLORS_TO_INT = {
    "white": 0,  # neutral
    "blue": 1,  # coily
    "violet": 2,  # ugg et wrong_way
    "green": 3,  # green_ball
    "red": 4,  # red_ball
    "yellow": 5,  # sam et slick
}
## ============== COULEUR AMICAL ============== ##
FRIENDLY_COLORS_INT = {
    0: "white",
    3: "green",
}

## ============== COULEUR DES ENNEMIS ============== ##
ENEMY_COLORS_INT = {
    1: "blue",
    2: "violet",
    4: "red",
    5: "yellow",
}

## ============== COULEUR INT EN HEXA ============== ##
COLORS_INT_HEX = {
    0: "#FFFFFF",   # white
    1: "#0000FF",   # blue
    2: "#8F00FF",   # violet (purple, approximated as "violet" is not a web color)
    3: "#00FF00",   # green
    4: "#FF0000",   # red
    5: "#FFFF00",   # yellow
}


## ==============  mouvements possibles selon le type d'entité ============== ##
MOVES = {
    "qbert": ["UP", "DOWN", "LEFT", "RIGHT"],
    "coily": ["UP", "DOWN", "LEFT", "RIGHT"],  # Coily poursuit Q*bert
    "ugg": ["LEFT", "UP"],  # Spawn en bas à DROITE (col=row), monte (LEFT) ou va vers la gauche (UP)
    "wrong_way": ["UP", "RIGHT"],  # Spawn en bas à GAUCHE (col=0), monte (UP) ou va vers la droite (RIGHT)
    "slick": ["DOWN", "LEFT", "RIGHT"],  # Ennemi vert qui descend et réinitialise les cubes (même comportement que Sam)
    "sam": ["DOWN", "RIGHT"],  # Ennemi vert qui descend et réinitialise les cubes (même comportement que Slick)
    "red_ball": ["DOWN", "RIGHT"],
    "green_ball": ["DOWN", "RIGHT"]
}

## ==============  DIRECTION ============== ##
MOVES_DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (+1, 0),
    "LEFT": (-1, -1),
    "RIGHT": (+1, +1),
    "WAIT": (0, 0)
}

## ==============  INTERVALLES DE SPAWN PAR NIVEAU ============== ##
# Temps entre chaque apparition d'ennemi selon le niveau (en secondes)
# Note: Aucune norme officielle trouvée, valeurs basées sur une progression de difficulté standard
SPAWN_INTERVALS_BY_LEVEL = {
    1: 8.0,   # Niveau 1 : 8 secondes entre chaque apparition
    2: 7.0,   # Niveau 2 : 7 secondes
    3: 6.0,   # Niveau 3 : 6 secondes
    4: 5.0,   # Niveau 4 : 5 secondes
    5: 4.5,   # Niveau 5 : 4.5 secondes
    6: 4.0,   # Niveau 6 : 4 secondes
    7: 3.5,   # Niveau 7 : 3.5 secondes
    8: 3.0,   # Niveau 8 : 3 secondes
    9: 2.5,   # Niveau 9 : 2.5 secondes
    10: 2.0,  # Niveau 10 : 2 secondes
}




## ==============  TAILLE TERMINAL ============== ##
TERMINAL_SIZE = {
    "WIDTH": 100,
    "HEIGHT": 30
}

## ============== CONFIGURATION FENETRE ARCADE ============== ##
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SCREEN_TITLE = "Q*Bert - Python Arcade"

GAME_BACKGROUND_COLOR = (45, 26, 90)

LIFE_WIDTH = 60
LIFE_HEIGHT = 60

CUBE_WIDTH = 120
CUBE_HEIGHT = 90  # Essayons une valeur plus petite pour l'effet écrasé 3D
START_Y = 650

LEVEL_CUBE_DOUBLE_JUMP = 10  # Désactivé : pas de double coloration
LEVEL_CUBE_REVISION = 15

# Profondeur des mouvements futurs pré-calculés pour les bots
BOT_FUTURE_MOVES_DEPTH = 500

# Profondeur de recherche pour l'algorithme MinMax
MINMAX_DEPTH = 7

# Pour compatibilité, MAX_DEPTH pointe vers BOT_FUTURE_MOVES_DEPTH
MAX_DEPTH = BOT_FUTURE_MOVES_DEPTH
