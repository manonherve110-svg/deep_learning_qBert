from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import random


from src.config import *



if TYPE_CHECKING:
    from core import Qbert


class Bot(ABC):
    """
    Classe abstraite représentant un bot générique du jeu.
    """

    position: tuple[float, float]
    color: str
    game: 'Qbert'  # Utilisation de string pour éviter l'import circulaire
    bot_type: str  # "coily", "ugg", "wrong_way", "slick", "sam" TODO ENUM
    is_frozen: bool
    is_alive: bool
    is_moving: bool
    vitesse : float

    target_position: tuple[int, int] | None
    start_position: tuple[float, float] | None
    move_direction: tuple[int, int] | None
    move_progress: float

    moves:  list[str] # liste des mouvements qu'il peut faire
    future_moves: list[str] # Stockage des futurs mouvements
    

    ## ==============  Initialisation des attributs du bot  ============== ##
    def __init__(self,game: 'Qbert', bot_type:str):
        self.game = game
        self.position = (1, random.randint(0,1)) 
        self.is_frozen = False
        self.is_alive = True
        self.is_moving = False
        self.vitesse = 2 # prend 0.5 seconde pour faire un mouvement
        self.target_position = None
        self.start_position = None
        self.move_direction = None
        self.move_progress = 0.0
        self.bot_type = bot_type
        self.moves = MOVES[self.bot_type]
        self.color = BOT_COLORS[self.bot_type]
        self.future_moves = []
        self._cached_sim_position = self.position  # Cache pour position simulée
        self.fill_future_moves()

    def _get_simulated_position(self) -> tuple[float, float]:
        """Retourne la position simulée mise en cache."""
        return self._cached_sim_position

    def _generate_move(self) -> str:
        """Génère un mouvement selon la logique du bot (interne)."""
        sim_r, sim_c = self._cached_sim_position
        
        if self.bot_type not in ["ugg", "wrong_way"] and sim_r < 0:
            return "DOWN"
        
        possible_moves = MOVES[self.bot_type]
        
        # Pour Sam, on vérifie que le mouvement reste sur le plateau
        if self.bot_type == "sam":
            valid_moves = []
            board_len = len(self.game.board)
            for move in possible_moves:
                dr, dc = MOVES_DIRECTIONS[move]
                nr, nc = sim_r + dr, sim_c + dc
                if 0 <= nr < board_len and 0 <= nc <= nr:
                    valid_moves.append(move)
            if valid_moves:
                return random.choice(valid_moves)

        if self.bot_type == "ugg":
            return random.choice([ "UP"]) 
        
        if self.bot_type == "wrong_way":
            return random.choice(["LEFT"])  

        return random.choice(possible_moves)

    def fill_future_moves(self):
        """Remplit la file de mouvements futurs jusqu'à MAX_DEPTH."""
        # Initialise le cache de position simulée si nécessaire
        if not hasattr(self, '_cached_sim_position'):
            self._cached_sim_position = self.position
        
        target_depth = MAX_DEPTH
        while len(self.future_moves) < target_depth:
            move = self._generate_move()
            self.future_moves.append(move)
            # Met à jour la position simulée de manière incrémentale
            if move != "WAIT":
                dr, dc = MOVES_DIRECTIONS[move]
                r, c = self._cached_sim_position
                self._cached_sim_position = (r + dr, c + dc)

    def get_remaining_turns(self, ratio: int) -> int:
        """
        Calcule le nombre de 'tours' restants avant d'arriver à destination.
        Sert à synchroniser l'IA MinMax avec le mouvement continu du jeu.
        """
        if self.is_moving and self.target_position:
            tr, tc = self.target_position
            cr, cc = self.position
            dist_remaining = ((tr - cr)**2 + (tc - cc)**2)**0.5
            turns = int(dist_remaining * ratio)
            return max(0, min(turns, ratio))
        return 0

    ## ============== MOVE  ============== ##
    def move(self, move: str) -> bool:
        """
        Définit le comportement de déplacement du bot.
        """
        if move not in MOVES[self.bot_type]:
            raise ValueError(f"Usage: mouvement non valide {move} pour le bot {self.bot_type}")
        row, col = self.position
        delta_row, delta_col = MOVES_DIRECTIONS[move]
        row += delta_row
        col += delta_col
        self.position = (row, col)
        return True
    
    ## ============== PROCHAIN MOVE DU BOOT  ============== ##
    def get_next_move(self) -> str | None:
        """
        Retourne le prochain mouvement stocké et en régénère un à la fin.
        """
        if not self.future_moves:
            self.fill_future_moves()
        
        next_move = self.future_moves.pop(0)
        self.future_moves.append(self._generate_move())
        return next_move
    
    ## ============== JOUER UN COUP SELON DIRECTION============== ##
    def play_move(self, move: str | None ) -> bool:
        """Initie le mouvement du bot selon la direction donnée."""
        if self.is_moving:
            return False
        
        if move == None:
            return False

        if move not in self.moves:
            raise ValueError(f"Usage: Argument de mouvement non valide: {move}")   

        # On part de la position actuelle arrondie pour garantir l'alignement grille
        current_row, current_col = self.position
        r_int, c_int = round(current_row), round(current_col)
        
        delta_row, delta_col = MOVES_DIRECTIONS[move]
        
        self.start_position = (float(r_int), float(c_int))
        self.target_position = (r_int + delta_row, c_int + delta_col)
        self.move_direction = (delta_row, delta_col)
        
        # On recalibre la position pour éviter les dérives cumulées
        self.position = (float(r_int), float(c_int))
        
        self.is_moving = True
        return True


    ## ============== MET À JOUR LA POSITIION ============== ##

    # méthodes pour tout les bots sauf coily qui doit la surcharger à cause d
    def update_position(self, dt: float) -> None:
        """Met à jour l'état du mouvement du joueur en fonction du temps écoulé."""
        if self.is_moving and self.target_position and self.move_direction:
            cur_r, cur_c = self.position
            dir_r, dir_c = self.move_direction
            target_r, target_c = self.target_position

            # Calcul du déplacement : pos = pos + vitesse * dt * direction
            step = self.vitesse * dt
            new_r = cur_r + dir_r * step
            new_c = cur_c + dir_c * step
            
            # Vérification si on a atteint ou dépassé la cible
            # On vérifie si la distance au carré augmente (ce qui signifie qu'on a dépassé le point le plus proche)
            # ou si on est très proche.
            dist_sq_before = (target_r - cur_r)**2 + (target_c - cur_c)**2
            dist_sq_after = (target_r - new_r)**2 + (target_c - new_c)**2
            
            if dist_sq_after >= dist_sq_before or dist_sq_after < 0.001:
                # On est arrivé
                self.position = (float(target_r), float(target_c))
                self.is_moving = False
                self.target_position = None
                self.start_position = None
                self.move_direction = None
            else:
                self.position = (new_r, new_c)
    
    ## ============== MEURT ============== ##

    def die(self) -> None:
        """Le bot meurt et est supprimé du jeu."""
        self.is_alive = False
        self.is_moving = False
        self.target_position = None
        self.start_position = None
        self.move_direction = None
        self.move_progress = 0.0
        if self in self.game.bots:
            self.game.bots.remove(self)

    def get_speed_multiplier(self) -> float:
        """Calcule le multiplicateur de vitesse basé sur le niveau."""
        if self.game.level < 6:
            return 1.0
        # Augmentation de 10% par niveau à partir du niveau 6, max +100% (x2)
        bonus = (self.game.level - 5) * 0.1
        return 1.0 + min(1.0, bonus)


class Coily(Bot):

    is_egg: bool # Passera à False lorsque Coily éclot
    is_hatching: bool
    hatch_timer: float
    """
    Représente le serpent Coily.
    Commence comme un oeuf puis poursuit Q*bert.
    """


    def __init__(self, game: 'Qbert'):
        self.is_egg = True
        super().__init__(game, "coily")
        self.position = (-4.0, 0.0)
        # Réinitialiser le cache et les mouvements futurs pour la nouvelle position
        self._cached_sim_position = self.position
        self.future_moves = []
        self.fill_future_moves()
        self.vitesse = 0.5 * self.get_speed_multiplier()
        self.is_hatching = False
        self.hatch_timer = 1.5

    def __str__(self) -> str:
        return "🥚" if self.is_egg else "🐍"

    def _generate_move(self) -> str:
        """Surcharge pour gérer le mouvement de l'oeuf dans la file."""
        sim_r, sim_c = self._get_simulated_position()

        if self.is_egg:
             if sim_r < 0: return "DOWN"
             return random.choice(MOVES["red_ball"])
        
        # Logique Serpent : on essaie de se rapprocher du joueur tout en restant valide
        possible_moves = MOVES["coily"]
        valid_moves = []
        board_len = len(self.game.board)
        
        for move in possible_moves:
            dr, dc = MOVES_DIRECTIONS[move]
            nr, nc = sim_r + dr, sim_c + dc
            if 0 <= nr < board_len and 0 <= nc <= nr:
                valid_moves.append(move)
        
        if not valid_moves:
            return "WAIT"

        # Heuristique vers le joueur (position actuelle)
        target_r, target_c = self.game.player.position
        best_move = random.choice(valid_moves)
        min_dist = float('inf')
        
        for move in valid_moves:
            dr, dc = MOVES_DIRECTIONS[move]
            nr, nc = sim_r + dr, sim_c + dc
            dist = (target_r - nr)**2 + (target_c - nc)**2
            if dist < min_dist:
                min_dist = dist
                best_move = move
                
        return best_move

    def update_egg(self, dt: float) -> None:
        """Met à jour le timer d'éclosion."""
        if self.is_egg:
            # Si l'oeuf atteint le bas de la pyramide, l'éclosion commence
            if self.position[0] >= len(self.game.board) - 1:
                self.is_hatching = True

            if self.is_hatching:
                self.hatch_timer -= dt
                if self.hatch_timer <= 0:
                    self.hatch()

    # surcharge de la méthode get_next_move pour Coily
    def get_next_move(self) -> str | None:
        """
        Retourne le prochain mouvement du bot.
        Seul coily aura besoin d'une heuristique pour choisir son prochain mouvement.
        """
        if self.is_egg:
            if self.is_hatching:
                return None
            # Utilise la logique de file du parent pour l'oeuf
            return super().get_next_move()
            
        if self.position[0] < 0:
            return "DOWN"
        else:
            # Heuristique: recherche du chemin le plus court vers le joueur
            target_r, target_c = self.game.player.position
            
            best_moves = []
            min_dist = float('inf')
            
            # Position actuelle sur la grille
            curr_r, curr_c = round(self.position[0]), round(self.position[1])
            
            for move in self.moves:
                dr, dc = MOVES_DIRECTIONS[move]
                next_r, next_c = curr_r + dr, curr_c + dc
                
                # Vérification des limites du plateau
                if (0 <= next_r < len(self.game.board) and 0 <= next_c <= next_r) or (self.game.transport_active and not self.is_egg):
                    # Calcul de la distance au carré vers le joueur
                    dist = (target_r - next_r)**2 + (target_c - next_c)**2
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_moves = [move]
                    elif abs(dist - min_dist) < 0.001:
                        best_moves.append(move)
            
            if best_moves:
                return random.choice(best_moves)
            
            # si aucun mouvement valide trouvé (ne devrait pas arriver)
            return random.choice(self.moves)
    



    def hatch(self) -> None:
        """
        Coily éclot et devient un serpent.
        """
        self.is_egg = False
        self.is_hatching = True
        self.hatch_timer = 0.0

    def die(self) -> None:
        """Surcharge de la mort pour ajouter les points."""
        self.game.player.addScore(500)
        super().die()


class Ugg(Bot):
    """
    Représente Ugg, se déplaçant sur les côtés de la pyramide.
    """
    def __init__(self, game: 'Qbert'):
        super().__init__(game, "ugg")
        self.vitesse = 0.5 * self.get_speed_multiplier()

    def __str__(self) -> str:
        return "😈"


class WrongWay(Bot):
    """
    Représente Wrong-Way, se déplaçant sur les côtés de la pyramide.
    """
    def __init__(self, game: 'Qbert'):
        super().__init__(game, "wrong_way")
        self.vitesse = 0.5 * self.get_speed_multiplier()

    def __str__(self) -> str:
        return "👺"


class Slick(Bot):
    """
    Représente Slick, qui réinitialise la couleur des cubes.
    """
    def __init__(self, game: 'Qbert'):
        super().__init__(game, "slick")
        self.position = (0.0, 0.0)
        # Réinitialiser le cache et les mouvements futurs pour la nouvelle position
        self._cached_sim_position = self.position
        self.future_moves = []
        self.fill_future_moves()
        self.vitesse = 0.5 * self.get_speed_multiplier()

    def __str__(self) -> str:
        return "🕶️"

class Sam(Bot):
    """
    Représente Sam, qui réinitialise la couleur des cubes.
    """
    def __init__(self, game: 'Qbert'):
        super().__init__(game, "sam")
        self.position = (0.0, 0.0)
        # Réinitialiser le cache et les mouvements futurs pour la nouvelle position
        self._cached_sim_position = self.position
        self.future_moves = []
        self.fill_future_moves()
        self.vitesse = 0.5 * self.get_speed_multiplier()
        

    def __str__(self) -> str:
        return "👽"


class RedBall(Bot):
    """
    Représente une balle rouge qui tombe aléatoirement.
    """
    def __init__(self, game: 'Qbert'):
        super().__init__(game, "red_ball")
        self.position = (-4.0, 0.0)
        # Réinitialiser le cache et les mouvements futurs pour la nouvelle position
        self._cached_sim_position = self.position
        self.future_moves = []
        self.fill_future_moves()
        self.vitesse = 0.75 * self.get_speed_multiplier()
    
    def __str__(self) -> str:
        return "🔴"
    
    

class GreenBall(Bot):
    """
    Représente une balle verte qui gèle les ennemis.
    """
    def __init__(self, game: 'Qbert'):
        super().__init__(game, "green_ball")

    def __str__(self) -> str:
        return "🟢"