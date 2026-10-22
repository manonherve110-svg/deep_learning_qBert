import math
import random
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict

if TYPE_CHECKING:
    from src.core import Qbert

try:
    from src.config import MOVES_DIRECTIONS, MOVES, MINMAX_DEPTH
except ModuleNotFoundError:
    from config import MOVES_DIRECTIONS, MOVES, MINMAX_DEPTH


class SimState:
    """
    Représente l'état du jeu pour la simulation MinMax.
    """
    def __init__(self, player_pos: Tuple[float, float], board_config: List[List[int]], score: int, step: int, bots: List[Dict], lives: int, cube_level: int, disks: List[Tuple[int, int]]):
        self.player_pos = player_pos
        self.board_config = board_config
        self.score = score
        self.step = step
        self.bots = bots  # Liste de dicts : {'type': str, 'pos': (r, c), 'is_egg': bool}
        self.lives = lives
        self.cube_level = cube_level
        self.disks = disks
    
    def to_hash(self):
        """Génère une clé unique hashable pour cet état (pour la mémorisation)."""
        # Conversion du board en tuple de tuples (immuable)
        board_tuple = tuple(tuple(row) for row in self.board_config)
        # Conversion des bots (on arrondit les positions pour éviter les micro-différences flottantes)
        bots_tuple = tuple(
            (b['type'], round(b['pos'][0], 2), round(b['pos'][1], 2), b.get('is_egg', False), b.get('timer', 0), b.get('move_idx', 0))
            for b in self.bots
        )
        disks_tuple = tuple(sorted(self.disks))
        # On inclut la position du joueur, le score n'est pas nécessaire dans la clé d'état pur mais le board l'est
        return (self.player_pos, board_tuple, bots_tuple, self.lives, self.cube_level, disks_tuple)

def get_valid_moves(state: SimState, board_rows: int) -> List[str]:
    """Génère les mouvements valides pour le joueur."""
    moves = []
    r, c = state.player_pos
    
    # On ajoute WAIT et les directions
    candidates = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
    

    for move in candidates:
        if move == "WAIT":
            moves.append(move)
            continue
            
        dr, dc = MOVES_DIRECTIONS[move]
        nr, nc = r + dr, c + dc
        nr_int, nc_int = int(round(nr)), int(round(nc))
        
        # Vérification des limites de la pyramide
        if (0 <= nr < board_rows and 0 <= nc <= nr) or (nr_int, nc_int) in state.disks:
            moves.append(move)
            
    return moves

def simulate_move(state: SimState, move: str, board_rows: int, bot_future_moves: List[List[str]]) -> SimState:
    """
    Simule un tour de jeu (mouvement joueur + mouvement bots).
    Retourne un nouvel état.
    """
    # 1. Copie de l'état (Board)
    new_board = [row[:] for row in state.board_config]
    new_score = state.score
    new_disks = list(state.disks)
    
    # 2. Mouvement du joueur
    pr, pc = state.player_pos
    prev_pr, prev_pc = pr, pc # On garde l'ancienne position pour détecter les croisements
    disk_used = False
    disk_pos = None
    if move != "WAIT":
        dr, dc = MOVES_DIRECTIONS[move]
        pr += dr
        pc += dc
        
        # Logique de changement de couleur
        r_int, c_int = int(round(pr)), int(round(pc))
        
        # Gestion des disques
        if (r_int, c_int) in new_disks:
            new_disks.remove((r_int, c_int))
            disk_used = True
            disk_pos = (r_int, c_int)
            
            # On vérifie si Coily est proche pour justifier l'utilisation du disque (Kill Coily)
            coily_nearby = False
            for b in state.bots:
                if b['type'] == 'coily' and not b.get('is_egg', False):
                    br, bc = b['pos']
                    if (br - r_int)**2 + (bc - c_int)**2 < 25: # Rayon d'environ 5 cases
                        coily_nearby = True
                        break
            
            if coily_nearby:
                new_score += 500 # Bonus pour tuer Coily
            else:
                new_score -= 50 # Pénalité pour l'utilisation abusive (sauf si c'est le seul moyen de survie)

            # Téléportation au sommet
            pr, pc = 0.0, 0.0
            # Le sommet change de couleur à l'atterrissage
            if new_board[0][0] < state.cube_level:
                new_board[0][0] += 1
                new_score += 25
        else:
            if 0 <= r_int < board_rows and 0 <= c_int <= r_int:
                # On suppose que si la couleur est inférieure au max, on l'augmente
                if new_board[r_int][c_int] < state.cube_level: 
                     new_board[r_int][c_int] += 1
                     new_score += 25

    new_player_pos = (pr, pc)
    
    # 3. Mouvement des bots
    new_bots = []
    current_step = state.step
    
    for i, bot in enumerate(state.bots):
        b_type = bot['type']
        br, bc = bot['pos']
        
        # Si le disque est utilisé, Coily est éliminé SEULEMENT s'il est proche (il saute dans le vide)
        if disk_used and b_type == 'coily':
            if not bot.get('is_egg', False) and disk_pos is not None:
                d_r, d_c = disk_pos
                if (br - d_r)**2 + (bc - d_c)**2 < 3.0: # Si proche (< 2 cases), il meurt
                    continue
            
        is_egg = bot.get('is_egg', False)
        ratio = bot.get('ratio', 1)
        timer = bot.get('timer', 0) # Temps avant le prochain mouvement
        move_idx = bot.get('move_idx', 0)
        step_r, step_c = bot.get('step', (0.0, 0.0))
        
        # Gestion du timing précis
        if timer > 0:
            # Le bot est en mouvement : on met à jour sa position progressivement
            new_bot = bot.copy()
            # On avance le bot selon son vecteur de déplacement
            new_bot['pos'] = (br + step_r, bc + step_c)
            new_bot['prev_pos'] = (br, bc)
            new_bot['timer'] = timer - 1
            new_bots.append(new_bot)
            continue
        
        # Le timer est à 0, le bot bouge ce tour-ci
        # On réinitialise le timer pour le prochain cycle (ratio)
        
        next_r, next_c = br, bc
        bot_move = "WAIT"
        
        # Cas Coily Serpent (Dynamique)
        if b_type == 'coily' and not is_egg:
            best_dist = float('inf')
            best_move = "WAIT"
            possible_moves = MOVES['coily']
            shuffled_moves = possible_moves[:]
            random.shuffle(shuffled_moves)
            
            for m in shuffled_moves:
                mdr, mdc = MOVES_DIRECTIONS[m]
                tr, tc = br + mdr, bc + mdc
                if 0 <= tr < board_rows and 0 <= tc <= tr:
                    dist = (tr - pr)**2 + (tc - pc)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_move = m
            bot_move = best_move

        else:
            # Cas Bots Déterministes (ou Coily Oeuf)
            # On prend le prochain move dispo (simplification : on suppose qu'ils suivent leur file)
            if i < len(bot_future_moves) and move_idx < len(bot_future_moves[i]):
                bot_move = bot_future_moves[i][move_idx]
        
        next_step_r, next_step_c = 0.0, 0.0
        next_timer = 0
        
        if bot_move != "WAIT":
            dr, dc = MOVES_DIRECTIONS[bot_move]
            # On calcule le pas de déplacement pour les prochains tours
            next_step_r = dr / ratio
            next_step_c = dc / ratio
            next_timer = ratio - 1
            
            # On applique le premier pas tout de suite
            next_r += next_step_r
            next_c += next_step_c
        else:
            next_timer = ratio - 1
            
        # Gestion transition Oeuf -> Serpent
        new_is_egg = is_egg
        if b_type == 'coily' and is_egg:
            if next_r >= board_rows - 1:
                new_is_egg = False 
        
        new_bots.append({
            'type': b_type, 
            'pos': (next_r, next_c), 
            'prev_pos': (br, bc),
            'is_egg': new_is_egg, 
            'ratio': ratio, 
            'timer': next_timer, 
            'move_idx': move_idx + 1,
            'step': (next_step_r, next_step_c)
        })

    # 4. Vérification Collisions (Mort)
    alive = True
    final_bots = []
    for bot in new_bots:
        br, bc = bot['pos']
        b_type = bot['type']
        
        # Collision standard (sur la même case)
        # On utilise une marge de 0.6 (un peu plus large que le round qui est 0.5)
        # pour anticiper les collisions limites et éviter le timing trop serré
        if abs(br - pr) < 0.6 and abs(bc - pc) < 0.6:
            if b_type in ["slick", "sam"]:
                new_score += 300
                continue # Le bot est mangé, on ne l'ajoute pas à la liste finale (il disparait)
            else:
                alive = False
                break
        
        prev_br, prev_bc = bot['prev_pos']
        
        # Sécurité supplémentaire : on considère la case précédente du bot comme dangereuse
        # Cela évite que l'IA aille sur une case que le bot est en train de quitter (timing serré)
        if b_type not in ["slick", "sam"]:
            if abs(prev_br - pr) < 0.6 and abs(prev_bc - pc) < 0.6:
                alive = False
                break

        # Collision par croisement (Le joueur et le bot ont échangé leurs places)
        # Si Joueur était à A et va à B, et Bot était à B et va à A -> Mort
        if (round(pr) == round(prev_br) and round(pc) == round(prev_bc)) and \
           (round(prev_pr) == round(br) and round(prev_pc) == round(bc)):
             if b_type not in ["slick", "sam"]:
                 alive = False
                 break
        
        final_bots.append(bot)
            
    if not alive:
        return SimState(new_player_pos, new_board, float('-inf'), current_step + 1, final_bots, state.lives - 1, state.cube_level, new_disks)

    return SimState(new_player_pos, new_board, new_score, current_step + 1, final_bots, state.lives, state.cube_level, new_disks)

def minmax_recursive(state: SimState, depth: int, alpha: float, beta: float, memo: Dict, board_rows: int, bot_future_moves: List[List[str]]) -> Tuple[float, Optional[str]]:
    
    # 1. Vérification Cache (Transposition Table)
    state_key = (state.to_hash(), depth)
    if state_key in memo:
        return memo[state_key]

    if state.score == float('-inf'):
        return float('-inf'), None
        
    # Si le niveau est terminé (toutes les cases à la couleur cible), c'est une victoire
    if all(all(val >= state.cube_level for val in row) for row in state.board_config):
        # On retourne le score actuel + un gros bonus pour prioriser la fin du niveau
        # On ajoute un bonus proportionnel à la profondeur restante (depth) pour favoriser la rapidité
        return float(state.score + 10000 + depth * 1000), None

    if depth == 0:
        # Heuristique : si on ne peut plus calculer, on favorise la proximité avec les cubes restants
        heuristic_bonus = 0.0
        min_dist = float('inf')
        pr, pc = state.player_pos
        found_target = False
        for r, row in enumerate(state.board_config):
            for c, val in enumerate(row):
                if val < state.cube_level:
                    dist = abs(r - pr) + abs(c - pc)
                    if dist < min_dist: min_dist = dist
                    found_target = True
        if found_target:
            # Bonus inverse à la distance. DOIT ETRE INFERIEUR A 25 (valeur d'un cube)
            # Sinon l'IA préfère rester à côté du cube (bonus) plutôt que le manger (25 pts + 0 bonus)
            heuristic_bonus = 20.0 / (min_dist + 1.0)
        return float(state.score + heuristic_bonus), None

    moves = get_valid_moves(state, board_rows)
    best_val = float('-inf')
    best_move = None
    
    for move in moves:
        next_state = simulate_move(state, move, board_rows, bot_future_moves)
        val, _ = minmax_recursive(next_state, depth - 1, alpha, beta, memo, board_rows, bot_future_moves)
        
        if val != float('-inf'):
            # Coef suivant la profondeur (0.99) pour maximiser le gain direct
            val = next_state.score + 0.99 * (val - next_state.score)
        
        if val > best_val:
            best_val = val
            best_move = move
            
        alpha = max(alpha, best_val)
        if beta <= alpha:
            break # Élagage Beta
        
    # 2. Stockage dans le Cache
    memo[state_key] = (best_val, best_move)
    return best_val, best_move

def find_best_move_ab(game: 'Qbert', depth: int = MINMAX_DEPTH) -> Optional[str]:
    bots_data = []
    player_speed = game.player.vitesse
    for b in game.bots:
        if b.is_alive:
            bot_speed = getattr(b, 'vitesse', 1.0)
            if bot_speed <= 0: bot_speed = 0.1
            
            raw_ratio = player_speed / bot_speed
            # On utilise int pour ne pas sous-estimer la vitesse (bot plus rapide en simu = sécurité à l'arrivée)
            # La sécurité "prev_pos" gère déjà le risque de collision au départ (bot qui quitte la case)
            if getattr(b, 'is_egg', False):
                ratio = int(raw_ratio) + 1
            else:
                ratio = max(1, int(raw_ratio))
            
            # Calcul du timer initial via la méthode du bot
            initial_timer = b.get_remaining_turns(ratio)
            
            # Calcul du vecteur de déplacement actuel (si le bot est déjà en mouvement)
            step_r, step_c = 0.0, 0.0
            if b.is_moving and b.move_direction:
                dr, dc = b.move_direction
                step_r = dr / ratio
                step_c = dc / ratio
            
            bots_data.append({'type': b.bot_type, 'pos': b.position, 'prev_pos': b.position, 'is_egg': getattr(b, 'is_egg', False), 'ratio': ratio, 'timer': initial_timer, 'move_idx': 0, 'step': (step_r, step_c)})

    bot_future_moves = [list(b.future_moves) for b in game.bots if b.is_alive]
    disks = list(game.board.disque.keys())
    # On passe le vrai cube_level du jeu (ex: 1 pour le niveau 1) pour que l'IA sache quand un cube est fini
    initial_state = SimState(game.player.position, game.board.board, game.player.score, 0, bots_data, game.player.lives, game.cube_level, disks)
    val, move = minmax_recursive(initial_state, depth, float('-inf'), float('inf'), {}, len(game.board), bot_future_moves)
    
    # Sécurité : si l'IA ne trouve aucun coup (ex: tous mortels), on renvoie un coup au hasard valide plutôt que None (qui bloque)
    if move is None:
        # Si tous les coups mènent à la mort, on préfère ATTENDRE plutôt que de bouger au hasard
        # Cela évite les suicides visuellement stupides
        return "WAIT"
        
    return move
