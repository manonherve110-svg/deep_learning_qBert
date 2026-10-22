from abc import ABC, abstractmethod
import torch

try:
    from src.config import MOVES_DIRECTIONS
    from src.ia.utils.util import load_ai_model
    from src.ia.mlp import QbertMLP
    from src.ia.minmax import find_best_move_ab
    from src.ia_deepQ import dqn as dqn_mod
    import src.ia.mlp as mlp
except ModuleNotFoundError:
    from config import MOVES_DIRECTIONS
    from ia.utils.util import load_ai_model
    from ia.mlp import QbertMLP
    from ia.minmax import find_best_move_ab
    from ia_deepQ import dqn as dqn_mod
    import ia.mlp as mlp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from src.core.game import Qbert
    except ModuleNotFoundError:
        from core.game import Qbert


class Player(ABC):
    """Classe abstraite représentant un joueur (humain ou IA)."""

    mode: str  # 'human' or 'ai'
    position: tuple[float, float]
    color: str
    score: int
    lives: int
    game: 'Qbert'
    is_moving: bool 
    vitesse : float
    
    target_position: tuple[int, int] | None
    start_position: tuple[float, float] | None
    move_direction: tuple[int, int] | None
    move_progress: float

    moves: dict[str, list[str]] # liste des mouvements qu'il peut faire

    def __init__(self, game: 'Qbert', mode:str = 'human', color: str = 'red') -> None:
        """Initialise le joueur avec un mode donné."""
        self.mode = mode
        self.color = color
        self.game = game

        self.score = 0
        self.lives = 3
        self.position = (0.0, 0.0)
        self.is_moving = False
        self.vitesse = 2 # selon la doc, 
        
        self.target_position = None
        self.start_position = None
        self.move_direction = None
        self.move_progress = 0.0
        self.last_move = None

        # Dictionnaire des mouvements: nom du mouvement -> liste d'entrées clavier acceptées
        self.moves = {
            'UP': ['z','Z','up'],    
            'DOWN': ['s', 'S', 'down'],   
            'LEFT': ['q', 'Q', 'left'],     
            'RIGHT': ['d', 'D', 'right']      
        }


    def play_move(self, move: str | None) -> bool:
        """Initie le mouvement du joueur selon la direction donnée."""
        if self.is_moving:
            return False
        
        if move == None:
            return False
        


        if move not in self.moves.keys():
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
        self.last_move = move
        return True
    
    def next_move(self) -> str | None:
        " Retourne le prochaine coups à jouer "
        pass

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

    def addScore(self, points: int) -> None:
        """Ajoute des points au score du joueur."""
        old_score = self.score
        self.score += points
        
        if self.score // 5000 > old_score // 5000:
            self.lives = min(5, self.lives + 1)

    def reset(self) -> None:
        """Réinitialise le joueur à son état initial."""
        self.position = (0.0, 0.0)  # Position de départ
        self.score = 0
        self.lives = 3
        self.is_moving = False
        self.target_position = None
        self.start_position = None
        self.move_direction = None
        self.move_progress = 0.0
        self.last_move = None
    
    def respawn(self) -> None:
        """ Fais réapparaître le joueur au sommet du plateau. """
        self.position = (0.0, 0.0)
        self.is_moving = False
        self.target_position = None
        self.start_position = None
        self.move_direction = None
        self.move_progress = 0.0
        self.last_move = None

class HumanPlayer(Player):
    """Représente un joueur humain."""
    
    def __init__(self, game: 'Qbert'):
        """Initialise un joueur humain."""
        super().__init__(game)
    
    
            
class AIPlayer(Player):
    


    """Représente un joueur IA."""
    
    def __init__(self, game:'Qbert', model =  None) -> None:
        super().__init__(game=game, mode="ai")
        self.model = model
    
    def next_move(self) -> str | None:
        """Retourne le prochain mouvement de l'IA.
        Returns:
            Le mouvement ("up", "down", "left", "right") ou None
        """
        if isinstance(self.model, QbertMLP):
            object.__setattr__(self.model, '_train_sample', True)
            object.__setattr__(self.model, '_train_temperature', 1.3 if (self.game.sim_time is not None) else 0.5)
            state = mlp.encode_state(self.game)
            move = mlp.neural_choice(model=self.model, state=state)
            return move
        else:
            action = dqn_mod.neural_choice(self.model, self.game, testing=True)
            return dqn_mod.ACTION_MAP.get(action)



class MinMaxPlayer(Player):
    """Représente un joueur IA basé sur MinMax."""

    def __init__(self, game: 'Qbert', depth: int | None = None) -> None:
        super().__init__(game=game, mode="ai")
        self.depth = depth

    def next_move(self) -> str | None:
        move = find_best_move_ab(self.game, depth=self.depth) if self.depth is not None else find_best_move_ab(self.game)
        if move == "WAIT":
            return None
        return move