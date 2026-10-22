from multiprocessing import process
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .utils.torch_utils import set_seed

import pygad 
import pygad.torchga as torchga

import numpy as np
import time

from typing import TYPE_CHECKING 
if TYPE_CHECKING:
    from src.core.game import Qbert

# Détection du device (GPU si dispo, sinon CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du périphérique : {DEVICE}")

# Module pour entraîner et évaluer un réseau MLP pour Q*bert.
# Contient :
# - la définition du modèle `QbertMLP`
# - l'encodage d'état (`encode_state`) et la sélection d'action (`neural_choice`)
# - la fonction de fitness utilisée par PyGAD et un wrapper pour injecter le contexte
# - une callback adaptative qui ajuste la pénalité de mort et la mutation en cas de stagnation




from .utils import torch_utils as torchu
from .utils import util as utl
from .utils import pygad_util as pygadu

# Architecture
N_INPUT = 10  
# goulot d'étranglement
N_HIDDEN_1 = 64
N_HIDDEN_2 = 32
N_HIDDEN_3 = 16
N_OUTPUT = 4


# Entraînement

N_SOLUTIONS = 300 # taille de la population
N_GENERATIONS = 350  # nombre de générations pour l'entraînement
N_SIM = 2  # nombre de simulations pour l'évaluation
N_PARENTS_MATING = round(N_SOLUTIONS * 0.25) # nombre de parents sélectionnés pour la reproduction
KEEP_PARENT = 2 # nombre de parents conservés intacts 
KEEP_ELITISM = 2 # nombre de meilleurs individus conservés intacts
PRINT_EVERY = 10 # nombre de générations entre chaque affichage
MAX_STEPS_PER_SIM = 250 # max pour éviter les boucles infinies, 8 niveaux de 28 cases avec 0.5 seconde par case
MAX_SIM_TIME = 125 # max pour éviter les boucles infinies, 125 secondes = 2 minutes et 5 secondes par niveau

# -------------------- Training adaptatif --------------------

STAGNATION_WINDOW = 5  # fenêtre (générations) pour détecter la stagnation
MIN_IMPROV_REL = 0.01  # amélioration relative minimale (0.01 = 1%)
POOR_FITNESS_THRESHOLD = 15000.0  # seuil sous lequel la fitness est considérée "faible" | dépend de la fitness
COOLDOWN_BOOST_GEN = 2 * STAGNATION_WINDOW  # générations minimales entre deux boosts
MUTATION_PROBA_INIT = 0.10 # probabilité de mutation initiale
MUTATION_BOOST_FACTOR = 1.5  # facteur multiplicatif pour la probabilité de mutation lors d'un boost
MUTATION_MAX = 0.25  # probabilité de mutation maximale autorisée (augmentée)
MUTATION_PERCENT_INIT = 2  # pourcentage de gènes mutés initial 
MUTATION_PERCENT_INCR = 2  # augmentation en points (%) du pourcentage de gènes mutés lors d'un boost
MUTATION_PERCENT_MAX = 10  # pourcentage maximal de gènes pouvant être mutés
# -----------------------------------------------------------------------------

# -------------------- Exploration / champions --------------------
SEED_STRIDE = 1_000_000  # variation de seed entre générations
HOF_K = 2  # Nombre de "champions" (meilleurs historiques) conservés en mémoire
HOF_INJECT_FRAC = 0.05  # Proportion de la population remplacée par des variantes de champions
HOF_INJECT_MIN = max(1, round(N_SOLUTIONS * HOF_INJECT_FRAC))  # nombre minimum de champion réinjectés
HOF_NOISE_STD = 0.02  # bruit (gaussien) ajouté aux champions lors de la ré-injection
# -----------------------------------------------------------------------------

# -------------------- Accélération fitness (early-stop) --------------------
NO_SCORE_STEPS = 20  # stop si aucun gain de score pendant N steps simulées
EARLYSTOP_MIN_STEPS = 20  # évite de couper trop tôt
# -----------------------------------------------------------------------------


from src.config import MOVES, MOVES_DIRECTIONS


class QbertMLP(nn.Module):
    """Modèle de réseau de neurones pour l'IA.
    
    Architecture:
    - 3 couches cachées
    - LayerNorm après chaque couche pour stabiliser l'apprentissage
    - Initialisation Xavier pour une meilleure convergence
    """
    def __init__(self, n_inputs: int = N_INPUT, n_outputs: int = N_OUTPUT) -> None:
        super().__init__()
        
        # Couches linéaires
        self.fc1 = nn.Linear(n_inputs, N_HIDDEN_1)
        self.fc2 = nn.Linear(N_HIDDEN_1, N_HIDDEN_2)
        self.fc3 = nn.Linear(N_HIDDEN_2, N_HIDDEN_3)
        self.fc_out = nn.Linear(N_HIDDEN_3, n_outputs)
        
        # Layer Normalization pour stabiliser l'apprentissage
        self.ln1 = nn.LayerNorm(N_HIDDEN_1)
        self.ln2 = nn.LayerNorm(N_HIDDEN_2)
        self.ln3 = nn.LayerNorm(N_HIDDEN_3)
        
        # Initialisation Xavier pour une meilleure convergence
        self._init_weights()
        
        self.to(DEVICE)
    
    def _init_weights(self):
        """Initialise les poids du réseau avec la méthode Xavier uniforme.
        
        Les poids sont initialisés selon une distribution uniforme Xavier,
        et les biais sont mis à zéro pour une meilleure convergence.
        """
        for module in [self.fc1, self.fc2, self.fc3, self.fc_out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant du réseau de neurones.
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, n_inputs).
            
        Returns:
            Tenseur de sortie de forme (batch_size, n_outputs) avec probabilités
            normalisées via softmax pour chaque action possible.
        """
        # Couche 1
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        
        # Couche 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        # Couche 3
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        
        # sortie softmax pour classification multi-classe
        return F.softmax(self.fc_out(x), dim=1)
    
    @staticmethod
    def load_model(model_path: str) -> 'QbertMLP':
        """Charge un modèle sauvegardé depuis un fichier.
        
        Args:
            model_path: Chemin vers le fichier de sauvegarde du modèle (.pth).
            
        Returns:
            Instance de QbertMLP avec les poids chargés depuis le fichier.
        """
        return torchu.load_model(model_path, QbertMLP, device=str(DEVICE))


def encode_state(game: 'Qbert') -> list[float]:
    """
    Encode l'état du jeu en un vecteur de taille 10.

    La structure du vecteur est la suivante:

    1. Vision locale (4): 
        Pour chaque direction ["UP", "DOWN", "LEFT", "RIGHT"]:
            - -1.0 si la case voisine est hors terrain ou occupée/dangereuse (ennemi dessus ou la vise en mouvement)
            -  1.0 si la case voisine (valide) est non coloriée
            -  0.0 si la case voisine (valide) est coloriée
    2. Vecteur vers la case non coloriée la plus proche (2): 
        - dx, dy (normalisés)
    3. Vecteur vers l'ennemi le plus proche (2): 
        - dx, dy (normalisés)
    4. Vecteur vers la dernière case visitée (2): 
        - dx, dy (normalisés) pointant vers la position précédente (inverse du dernier move)

    Returns:
        list[float]: Vecteur d'état de taille 10 représentant la situation courante.
    """
    
    state = []
    px, py = game.player.position
    px_int, py_int = round(px), round(py)
    
    moves_order = ["UP", "DOWN", "LEFT", "RIGHT"] # Ordre fixe pour bien faire correspondre le mouvement au delta x,y correspondant
    
    # --- 1: Vision Locale (case) ---
    for move in moves_order:
        dr, dc = MOVES_DIRECTIONS[move]
        nx, ny = px_int + dr, py_int + dc
        
        # 1. case valide (pyramide) ou disque
        is_valid = ((0 <= nx < len(game.board)) and (0 <= ny <= nx)) or game.board.is_disque(((nx, ny)))
        
        is_safe = True
        for bot in game.bots:
            if bot.is_alive:
                if bot.is_moving and bot.target_position == (round(nx), round(ny)):
                    is_safe = False
                    break
                elif (round(bot.position[0]), round(bot.position[1])) == (round(nx), round(ny)):
                    is_safe = False
                    break
        
        if not is_valid or not is_safe:
            state.append(-1.0)
        else:
            # disque: case valide mais pas une tuile à colorier
            if game.board.is_disque((nx, ny)):
                state.append(0.75)
            elif game.board[nx, ny] == 0:
                state.append(1.0)
            else:
                state.append(0.0) 
        


    # case non coloriée la plus proche
    closest_uncolored = None
    min_dist = float('inf')
    
    # Optimisation: accès direct à la liste sous-jacente et distance au carré
    # On utilise enumerate car on a besoin des coordonnées (r, c) pour calculer la distance
    for r, row_vals in enumerate(game.board):
        for c, val in enumerate(row_vals):
            if val == 0:
                dx = r - px_int
                dy = c - py_int
                dist = dx*dx + dy*dy
                if dist < min_dist:
                    min_dist = dist
                    closest_uncolored = (r, c)

    if closest_uncolored:
        tr, tc = closest_uncolored
        vec_x, vec_y = tr - px, tc - py
        norm = (vec_x**2 + vec_y**2)**0.5
        if norm > 1e-6:
            state.extend([vec_x/norm, vec_y/norm])
        else:
            state.extend([0.0, 0.0])
    else:
        state.extend([0.0, 0.0])

    # ennemi le plus proche
    closest_enemy = None
    min_dist_enemy = float('inf')
    
    for bot in game.bots:
        if bot.is_alive:
            bx, by = bot.position
            dx, dy = bx - px, by - py
            dist = (dx**2 + dy**2)**0.5
            if dist < min_dist_enemy:
                min_dist_enemy = dist
                closest_enemy = (bx, by)
                
    if closest_enemy:
        ex, ey = closest_enemy
        vec_x, vec_y = ex - px, ey - py
        norm = (vec_x**2 + vec_y**2)**0.5
        # normalisation pour éviter les problèmes de calculs
        if norm > 1e-6:
             state.extend([vec_x/norm, vec_y/norm])
        else:
             state.extend([0.0, 0.0])
    else:
        state.extend([0.0, 0.0])

    # vecteur vers la dernière case visitée
    last_visited_vec = [0.0, 0.0]
    if game.player.last_move in moves_order:
        dr, dc = MOVES_DIRECTIONS[game.player.last_move]
        # Le vecteur retour est l'opposé du mouvement
        vec_x, vec_y = -float(dr), -float(dc)
        norm = (vec_x**2 + vec_y**2)**0.5
        if norm > 1e-6:
            last_visited_vec = [vec_x/norm, vec_y/norm]
    state.extend(last_visited_vec)
    
    return state


def neural_choice(model:QbertMLP, state:list[float])->str:
    """Sélectionne une action à partir de l'état encodé du jeu.
    
    Args:
        model: Modèle QbertMLP à utiliser pour la prédiction.
        state: Vecteur d'état encodé de taille N_INPUT (10).
        
    Returns:
        Chaîne de caractères représentant le mouvement sélectionné
        (parmi "UP", "DOWN", "LEFT", "RIGHT").
        
    Note:
        En mode entraînement (si `_train_sample=True`), utilise une politique
        stochastique avec température pour favoriser l'exploration.
        Sinon, utilise argmax pour une sélection déterministe.
    """
    with torch.no_grad():
        # unsqeeze: ajoute une dimension à la tensor pour qu'elle soit de la forme (1, n)
        # car le modèle attend une tensor de la forme (batch_size, n_features)
        x = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # state est déjà encodé
        z = model(x)[0]
        # mode entraînement: politique stochastique pour casser les plateaux de fitness (démo: argmax)
        if getattr(model, "_train_sample", False):
            temp = float(getattr(model, "_train_temperature", 1.3))
            if temp != 1.0:
                z = torch.pow(z, 1.0 / max(1e-6, temp))
                z = z / z.sum()
            imax = int(torch.multinomial(z, 1).item())
        else:
            imax = int(torch.argmax(z).item())
        move = MOVES["qbert"][imax]
    return move




def fitness_func_b(solution:np.ndarray, model:QbertMLP, n_sim:int, base_seed: int | None = None, deterministic: bool = False, sol_idx: int = 0) -> float:
    """Évalue la fitness d'une solution en simulant plusieurs parties.
    
    Crée une copie locale du modèle et charge les poids de la solution dedans,
    puis simule n_sim parties de Q*bert. 
    
    Args:
        solution: Vecteur numpy contenant les poids du réseau à évaluer.
        model: Instance du modèle QbertMLP (architecture de référence, non modifié).
        n_sim: Nombre de simulations à effectuer pour évaluer la solution.
        base_seed: Graine aléatoire de base pour la reproductibilité.
        deterministic: Si True, utilise argmax au lieu de sampling stochastique.
        sol_idx: Index de la solution dans la population (pour diversifier les seeds).
                  
    Returns:
        Fitness moyenne sur les n_sim simulations (float).
    """
    
    from src.core.game import Qbert
    from src.models.bots import Coily
    from collections import deque

    # Crée une copie locale du modèle pour éviter les race conditions en mode thread
    local_model = QbertMLP(N_INPUT, N_OUTPUT)
    model_weights_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model_weights_dict = {k: v.to(DEVICE) for k, v in model_weights_dict.items()}
    local_model.load_state_dict(model_weights_dict)
    local_model.eval()
    
    # Mode déterministe : désactive le sampling stochastique
    if deterministic:
        setattr(local_model, '_train_sample', False)

    total_fitness = 0
    qbert = Qbert(player_mode='ai', model=local_model)

    # simule ,_sim parties avec le modèle
    for i in range(n_sim):
        # Fixe la graine par (solution, simulation) pour assurer diversité ET reproductibilité.
        # Chaque solution voit des scénarios différents, évitant l'overfitting à un run chanceux.
        # Formule: base_seed + sol_idx * 1000 + i (1000 >> n_sim pour éviter les collisions)
        if base_seed is not None:
            s = base_seed + sol_idx * 1000 + i
        else:
            s = int(time.time()) + sol_idx * 1000 + i
        set_seed(s)

        # On réinitialise l'environnement et on lance la simulation
        qbert.reset()
        # Il faut positionner sim_time avant start() (sinon lst part en time_ns()).
        qbert.sim_time = 0.0
        qbert.start()
        steps = 0
        
        recent_positions = deque(maxlen=20)
        start_pos = (round(qbert.player.position[0]), round(qbert.player.position[1]))
        recent_positions.append(start_pos)

        

        # --- Métriques de fitness (événements) ---
        last_level = int(qbert.level)
        last_lives = int(qbert.player.lives)
        last_disques = int(len(qbert.board.disque))
        last_colored = int(sum(cell == 1 for row in qbert.board for cell in row))

        # "stagnation" = pas de progrès (case/niveau/disque) depuis un certain temps
        last_progress_step = steps
        stagnation_penalty = 0.0
        last_penalty_step = steps  


        while qbert.state == qbert.states['RUNNING'] and steps < MAX_STEPS_PER_SIM and qbert.sim_time < MAX_SIM_TIME:

            # Pendant le transport disque, ne pas buffer des moves (sinon états incohérents).
            if not qbert.transport_active and not qbert.player.is_moving:
                move = qbert.player.next_move()
                if move:
                    qbert.player.play_move(move) 
                    steps += 1


            prev_level = qbert.level
            qbert.update(0.125)
            qbert.sim_time += 0.125

            if prev_level != qbert.level:
                recent_positions.clear()

            # --- Détection d'événements (deltas) ---
            curr_level = qbert.level
            curr_lives = qbert.player.lives
            curr_disques = len(qbert.board.disque)
            curr_colored = sum(cell == 1 for row in qbert.board for cell in row)

            colored_delta = max(0, curr_colored - last_colored)     # nouvelles cases coloriées
            level_delta = max(0, curr_level - last_level)           # niveau(x) terminé(s)
            death_delta = max(0, last_lives - curr_lives)           # mort(s)
            disque_delta = max(0, last_disques - curr_disques)      # disque(s) pris
            disque_bonus = 0.0
            if disque_delta > 0:
                px, py = round(qbert.player.position[0]), round(qbert.player.position[1])
                
                # Cherche les Coilys vivants
                coilys_alive = [b for b in qbert.bots if isinstance(b, Coily) and b.is_alive and not b.is_egg]
                
                if coilys_alive:
                    # Vérifie si un Coily est sur une case adjacente
                    coily_adjacent = False
                    for coily in coilys_alive:
                        cx, cy = round(coily.position[0]), round(coily.position[1])
                        # Cases adjacentes dans Q*bert (diagonales)
                        if abs(cx - px) <= 1 and abs(cy - py) <= 1 and (cx, cy) != (px, py):
                            coily_adjacent = True
                            break
                    
                    if coily_adjacent:
                        disque_bonus = 500.0 * disque_delta  # Coily adjacent = max bonus
                    else:
                        disque_bonus = 300.0 * disque_delta  # Coily présent mais pas adjacent
                else:
                    disque_bonus = 100.0 * disque_delta  # Pas de Coily = bonus minimal


            progress = (colored_delta > 0) or (level_delta > 0) or (disque_delta > 0)
            if progress:
                last_progress_step = steps
            else:
                # malus si on ne progresse pas depuis 4 mouvements
                step_no_prog = steps - last_progress_step
                if step_no_prog > 4 and steps != last_penalty_step:
                    stagnation_penalty += 150  
                    last_penalty_step = steps

            # early-stop: aucun progrès (case/niveau/disque) depuis un moment
            if steps >= EARLYSTOP_MIN_STEPS and (steps - last_progress_step) >= NO_SCORE_STEPS:
                break
            

            total_fitness += (
                100.0 * colored_delta         
                + 2000.0 * level_delta        
                + disque_bonus        
                - 1000.0 * death_delta    
            )

            last_colored = curr_colored
            last_level = curr_level
            last_disques = curr_disques
            last_lives = curr_lives

        # Pénalités continues (faibles) en fin de simulation
        total_fitness -= stagnation_penalty

    return total_fitness / n_sim


def fitness_func_wrapper(ga_instance:pygad.GA, solution:np.ndarray, sol_idx:int):
    """Wrapper de la fonction de fitness pour PyGAD.
    
    Extrait le contexte nécessaire (modèle, nombre de simulations, seed) depuis
    l'instance PyGAD et appelle fitness_func_b avec ces paramètres.
    
    Args:
        ga_instance: Instance PyGAD contenant le contexte dans ses attributs _ctx.
        solution: Vecteur numpy représentant une solution candidate.
        sol_idx: Index de la solution dans la population courante.
        
    Returns:
        Fitness de la solution (float).
        
    Note:
        Affiche la progression tous les 10 individus évalués.
    """
    model = getattr(ga_instance, 'model_ctx', None)
    n_sim = getattr(ga_instance, 'n_sim_ctx', N_SIM)
    n_solutions = getattr(ga_instance, 'n_solutions_ctx', N_SOLUTIONS)

    base_seed = getattr(ga_instance, 'seed_ctx', None)
    if model is None:
        raise ValueError("model_ctx manquant dans ga_instance")
    if (sol_idx + 1) % 10 == 0:
        print(f"  Éval {sol_idx + 1}/{n_solutions}...", end='\r', flush=True)
    return fitness_func_b(solution, model, n_sim, base_seed=base_seed, sol_idx=sol_idx)


def callback_generation_wrapper(ga_instance):
    """Callback appelé à chaque génération par PyGAD.
    
    Effectue plusieurs tâches à chaque génération :
    1. Affiche la progression de l'entraînement
    2. Met à jour la seed pour chaque génération (stables intra-gen, variables inter-gen)
    3. Gère le Hall of Fame (top-K champions) et réinjecte des variantes bruitées
    4. Détecte la stagnation de la fitness et ajuste dynamiquement les paramètres
       de mutation (probabilité et pourcentage de gènes mutés)
    5. Détecte les dégradations et réduit les mutations en conséquence
    
    Args:
        ga_instance: Instance PyGAD contenant l'état de l'algorithme génétique.
        
    Note:
        Les paramètres de mutation sont ajustés selon :
        - Stagnation : augmentation si fitness faible et amélioration < MIN_IMPROV_REL
        - Dégradation : réduction si fitness baisse de >100 points
        - Amélioration : reset aux valeurs par défaut
        - Sinon : décroissance progressive vers les valeurs par défaut
    """


    # Callback exécuté à chaque génération : affiche l'état et ajuste dynamiquement certains paramètres
    n_generations = getattr(ga_instance, 'n_generations_ctx', N_GENERATIONS)  # nombre total de générations prévues
    t0 = getattr(ga_instance, 't0_ctx', time.time())  # temps de début de l'entraînement
    # Affichage / logging standard via l'utilitaire pygadu
    pygadu.callback_generation_b(ga_instance, n_generations, t0, print_every=PRINT_EVERY)

    gen = ga_instance.generations_completed

    stop_event = getattr(ga_instance, 'stop_event_ctx', None)
    if stop_event is not None and stop_event.is_set():
        ga_instance.stop_generation = True
        window = getattr(ga_instance, 'window_ctx', None)
        if window is not None:
            window.training_info = "Stopping..."
        return

    # --- Seeds par génération (stables intra-gen, variables inter-gen) ---
    if not hasattr(ga_instance, "seed_base_ctx"):
        ga_instance.seed_base_ctx = getattr(ga_instance, "seed_ctx", 0)
    ga_instance.seed_ctx = int(ga_instance.seed_base_ctx) + (gen + 1) * SEED_STRIDE

    # --- Hall of Fame (top-K global) + ré-injection de champions bruités ---
    if not hasattr(ga_instance, "hof_ctx"):
        ga_instance.hof_ctx = []  # list[(fitness: float, solution: np.ndarray)]

    try:
        pop = np.asarray(ga_instance.population)
        fits = np.asarray(ga_instance.last_generation_fitness)
        if pop.ndim == 2 and len(fits) == pop.shape[0] and pop.shape[0] > 0:
            top_n = min(3, pop.shape[0])
            for idx in np.argsort(fits)[-top_n:]:
                ga_instance.hof_ctx.append((float(fits[idx]), pop[idx].copy()))
            ga_instance.hof_ctx = sorted(ga_instance.hof_ctx, key=lambda x: x[0], reverse=True)[:HOF_K]

            keep_el = int(getattr(ga_instance, "keep_elitism", 0) or 0)
            n_inject = max(HOF_INJECT_MIN, int(pop.shape[0] * HOF_INJECT_FRAC))
            n_inject = min(n_inject, max(0, pop.shape[0] - keep_el))
            if n_inject > 0 and ga_instance.hof_ctx:
                rng = np.random.default_rng(int(getattr(ga_instance, "seed_ctx", 0)) + 12345)
                for j, pop_idx in enumerate(range(pop.shape[0] - n_inject, pop.shape[0])):
                    _, champ = ga_instance.hof_ctx[j % len(ga_instance.hof_ctx)]
                    ga_instance.population[pop_idx, :] = champ + rng.normal(0.0, HOF_NOISE_STD, size=champ.shape)
    except Exception as e:
        # Évite un crash, mais ne masque pas complètement l'erreur (debug).
        ga_instance.hof_error_ctx = str(e)


    # --- Historique des fitness et détection de stagnation ---
    if not hasattr(ga_instance, "fitness_history_ctx"):
        # initialisation au premier appel : historique + sauvegarde des valeurs par défaut de mutation
        ga_instance.fitness_history_ctx = []
        ga_instance.mutation_probability_default = getattr(ga_instance, 'mutation_probability', 0.08)
        ga_instance.mutation_percent_genes_default = getattr(ga_instance, 'mutation_percent_genes', MUTATION_PERCENT_INIT)

    # récupère la meilleure fitness de la génération courante et l'ajoute à l'historique
    best_fit = ga_instance.best_solution()[1]
    ga_instance.fitness_history_ctx.append(best_fit)

    window = getattr(ga_instance, 'window_ctx', None)
    if window is not None:
        try:
            window.training_history.append(float(best_fit))
            if len(window.training_history) > 500:
                window.training_history = window.training_history[-500:]
            avg_fit = None
            if hasattr(ga_instance, 'last_generation_fitness') and ga_instance.last_generation_fitness is not None:
                avg_fit = float(np.mean(ga_instance.last_generation_fitness))
            if avg_fit is not None:
                window.training_info = f"Gen {gen}/{n_generations} | Best {best_fit:.0f} | Avg {avg_fit:.0f}"
            else:
                window.training_info = f"Gen {gen}/{n_generations} | Best {best_fit:.0f}"

            prev_best = getattr(ga_instance, 'best_demo_fit_ctx', float('-inf'))
            if best_fit > prev_best:
                model = getattr(ga_instance, 'model_ctx', None)
                best_sol, best_sol_fitness, _ = ga_instance.best_solution()
                if model is not None:
                    weights = torchga.model_weights_as_dict(model=model, weights_vector=best_sol)
                    weights = {k: v.detach().cpu().clone() for k, v in weights.items()}
                    window.demo_checkpoint = {
                        "weights": weights,
                        "generation": gen,
                        "fitness": float(best_sol_fitness),
                    }
                    ga_instance.best_demo_fit_ctx = float(best_fit)
        except Exception:
            pass

    # Si on observe une amélioration du best, on "reset" immédiatement la mutation
    # aux valeurs de départ (et on réarme le cooldown). Si ça restagne, ça remontera.
    if len(ga_instance.fitness_history_ctx) >= 2:
        prev_best = ga_instance.fitness_history_ctx[-2]
        if best_fit > prev_best + 1.0:
            ga_instance.mutation_probability = ga_instance.mutation_probability_default
            ga_instance.mutation_percent_genes = ga_instance.mutation_percent_genes_default
            # "Ré-arme" vraiment le cooldown : empêche un boost immédiat après une amélioration.
            ga_instance.last_boost_gen = ga_instance.generations_completed

    # si on a assez de générations, on vérifie la stagnation sur la fenêtre définie
    if len(ga_instance.fitness_history_ctx) >= STAGNATION_WINDOW:
        recent = ga_instance.fitness_history_ctx[-STAGNATION_WINDOW:]
        improvement_abs = recent[-1] - recent[0]  # amélioration absolue sur la fenêtre
        last_boost = getattr(ga_instance, 'last_boost_gen', -1)

        # Détection de dégradation : si la fitness baisse, réduire immédiatement les mutations
        if improvement_abs < -150.0:  # fitness a baissé de plus de 150 points
            old_mp = ga_instance.mutation_probability
            old_pct = ga_instance.mutation_percent_genes
            # réduction plus agressive si dégradation
            ga_instance.mutation_probability = max(ga_instance.mutation_probability_default, ga_instance.mutation_probability * 0.85)
            ga_instance.mutation_percent_genes = max(ga_instance.mutation_percent_genes_default, ga_instance.mutation_percent_genes - 2)
            if ga_instance.mutation_probability != old_mp or ga_instance.mutation_percent_genes != old_pct:
                print(f"📉 Dégradation détectée ; réduction mutation (gen {ga_instance.generations_completed}): prob {old_mp:.3f}->{ga_instance.mutation_probability:.3f}, percent {old_pct}->{ga_instance.mutation_percent_genes}")
        # condition de stagnation : faible amélioration absolue sur la fenêtre
        elif improvement_abs < 150.0:
            # n'appliquer un boost que si le cooldown est écoulé
            if ga_instance.generations_completed - last_boost >= COOLDOWN_BOOST_GEN:
                old_mp = ga_instance.mutation_probability
                old_pct = ga_instance.mutation_percent_genes
                # augmenter la probabilité de mutation (avec plafond)
                ga_instance.mutation_probability = min(MUTATION_MAX, ga_instance.mutation_probability * MUTATION_BOOST_FACTOR)
                # augmenter le pourcentage de gènes mutés (avec plafond)
                ga_instance.mutation_percent_genes = min(MUTATION_PERCENT_MAX, ga_instance.mutation_percent_genes + MUTATION_PERCENT_INCR)
                ga_instance.last_boost_gen = ga_instance.generations_completed
                # n'affiche que si un changement effectif a eu lieu
                if ga_instance.mutation_probability != old_mp or ga_instance.mutation_percent_genes != old_pct:
                    print(f"⚠️ Stagnation détectée ; boost mutation (gen {ga_instance.last_boost_gen}): prob {old_mp:.3f}->{ga_instance.mutation_probability:.3f}, percent {old_pct}->{ga_instance.mutation_percent_genes}")
            else:
                # cooldown en cours — on signale parfois que le boost est sauté
                if ga_instance.generations_completed % PRINT_EVERY == 0:
                    print(f"(Boost skip à cause du cooldown) last_boost={last_boost}, gen={ga_instance.generations_completed}")
        else:
            # si pas de stagnation, on fait décroître progressivement les paramètres de mutation vers leurs valeurs par défaut
            # décroissance plus rapide (0.90 au lieu de 0.95)
            if ga_instance.mutation_probability > ga_instance.mutation_probability_default:
                ga_instance.mutation_probability = max(ga_instance.mutation_probability_default, ga_instance.mutation_probability * 0.90)
            if ga_instance.mutation_percent_genes > ga_instance.mutation_percent_genes_default:
                ga_instance.mutation_percent_genes = max(ga_instance.mutation_percent_genes_default, ga_instance.mutation_percent_genes - 1)

    # affichage périodique des paramètres adaptatifs
    if gen % PRINT_EVERY == 0:
        print(f"  Gen {gen}: probabilité de mutation={ga_instance.mutation_probability:.3f}, pourcentage de gènes mutés={ga_instance.mutation_percent_genes}")


def train(n_solutions=N_SOLUTIONS, n_generations=N_GENERATIONS, n_sim=N_SIM, n_parents_mating=N_PARENTS_MATING, seed=None, save=True, window=None, stop_event=None):
    """
    Lance l'entraînement de l'IA Q*bert par algorithme génétique.

    Args:
        n_solutions (int): Nombre de solutions par génération (taille de la population).
        n_generations (int): Nombre maximal de générations d'évolution.
        n_sim (int): Nombre de parties simulées pour chaque solution (sert à évaluer la fitness).
        n_parents_mating (int): Nombre de parents sélectionnés à chaque génération pour la reproduction.
        seed (int, optionnel): Graine aléatoire pour la reproductibilité. Si None, une seed aléatoire est générée.
        save (bool): Si True, enregistre le meilleur modèle trouvé dans le fichier 'model.pth'.

    Returns:
        dict: Un dictionnaire contenant les informations suivantes :
            - "model": le meilleur modèle entraîné (QbertMLP)
            - "fitness": le score de fitness associé à ce modèle
            - "solution": le vecteur de poids associé
            - "ga_instance": l'objet pygad.GA pour analyses ultérieures
            - "total_time": durée totale de l'entraînement en secondes
    """
    
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    torchu.set_seed(seed)

    # Create the PyTorch model.
    args = [N_INPUT, N_OUTPUT]
    model = QbertMLP(*args) # * désémplie les arguments dans la fonction QbertMLP

    torch_ga = torchga.TorchGA(model=model, num_solutions=n_solutions) # créer n_solutions individus (solutions candidates) 
    initial_population = torch_ga.population_weights

    ga_instance = pygad.GA(

        num_generations=n_generations,  # Nombre de générations à faire évoluer
        num_parents_mating=n_parents_mating,  # Nombre de parents sélectionnés pour la reproduction à chaque génération
        initial_population=initial_population,  # Population initiale (poids des réseaux générés par torchga)
        fitness_func=fitness_func_wrapper,  # Fonction d'évaluation de la fitness
        on_generation=callback_generation_wrapper,  # Callback appelé à chaque nouvelle génération (affichage/log)
        sol_per_pop=n_solutions,  # Nombre de solutions (d'individus) dans la population

        # sélection / élitisme
        parent_selection_type="tournament",
        K_tournament=4,
        keep_parents=KEEP_PARENT,
        keep_elitism=KEEP_ELITISM,

        crossover_type="uniform",  # chaque gène a 50% de chance de venir de chaque parent
        crossover_probability=0.5,  # réduit car uniform est moins destructeur

        # mutation (poids continus)
        mutation_type="random",
        mutation_probability=MUTATION_PROBA_INIT,
        mutation_percent_genes=MUTATION_PERCENT_INIT,  # type: ignore
        mutation_by_replacement=False,
        random_mutation_min_val=-0.10,
        random_mutation_max_val=0.10,

        # reproductibilité
        random_seed=seed,
        parallel_processing=["thread", 6]  # thread (pas process) pour que seed_ctx soit partagé

    )
    

    # initialise le temps de début de l'entraînement
    t0 = time.time()
    # Injecte dans `ga_instance` le contexte utilisé par `fitness_func_wrapper`
    # et `callback_generation_wrapper`. Ces attributs permettent de partager
    # des paramètres (seed, nombre de sim, facteur de pénalité adaptatif, ...)
    # sans modifier l'API de PyGAD.
    setattr(ga_instance, 'model_ctx', model)
    setattr(ga_instance, 'n_sim_ctx', n_sim)
    setattr(ga_instance, 'n_solutions_ctx', n_solutions)
    setattr(ga_instance, 'n_generations_ctx', n_generations)
    setattr(ga_instance, 't0_ctx', t0)
    # seed pour reproductibilité des simulations (utilisée dans fitness_func_b)
    setattr(ga_instance, 'seed_ctx', seed)
    setattr(ga_instance, 'seed_base_ctx', seed)
    setattr(ga_instance, 'window_ctx', window)
    setattr(ga_instance, 'stop_event_ctx', stop_event)

    # état pour éviter boosts répétés (cooldown)
    setattr(ga_instance, 'last_boost_gen', -1)

    print(f"\n{'='*60}")
    print(f"🧬 Entraînement IA Q*bert")
    print(f"{'='*60}")
    print(f"  Solutions: {n_solutions} | Générations: {n_generations}")
    print(f"  Simulations/solution: {n_sim} | Parents: {n_parents_mating}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}\n")
    
    ga_instance.run()
    
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Entraînement terminé en {total_time:.1f}s")

    # After the generations complete, some plots are shown that summarize how the outputs/fitness values evolve over generations.
    # ga_instance.plot_fitness(title="Disk prediction - Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    best_sol, best_sol_fitness, best_sol_idx = ga_instance.best_solution()
    print(f"Fitness (entraînement, {n_sim} sims): {best_sol_fitness}")

    # Fetch the parameters of the best solution.
    best_sol_weights = torchga.model_weights_as_dict(model=model, weights_vector=best_sol)
    model.load_state_dict(best_sol_weights)
    
    n_sim_validation = 10
    # Ré-évaluation finale avec plus de simulations pour une mesure fiable
    print(f"Ré-évaluation finale ({n_sim_validation} simulations)...")
    final_fitness = fitness_func_b(best_sol, model, n_sim=n_sim_validation, base_seed=seed + 999999, deterministic=True)
    print(f"Fitness finale (validation): {final_fitness}")
    best_sol_fitness = final_fitness  # Utiliser la fitness validée pour la sauvegarde
    
    if save:
        best_model_path = "models/genetics/best_genetic.pth"
        is_best = False
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        # Vérifier si un meilleur modèle existe déjà
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, weights_only=False)
                best_fitness = checkpoint.get("eval", float('-inf'))
                if best_sol_fitness > best_fitness:
                    is_best = True
            except Exception:
                is_best = True  # Si erreur de chargement, on considère comme meilleur
        else:
            is_best = True
        
        # Sauvegarder le meilleur modèle ou avec un nom explicite
        if is_best:
            model_file_path = best_model_path
            print(f" Nouveau meilleur modèle ! Sauvegarde sous {model_file_path}")
        else:
            # Nom basé sur les paramètres
            model_file_path = f"models/genetics/genetic_{n_solutions}_{n_generations}_{n_sim}_{n_parents_mating}_{seed}.pth"
            print(f"Sauvegarde sous {model_file_path}")
        
        torchu.save_model(model_file_path, best_sol_weights, args, best_sol_fitness, n_generations)
    
    return {
        "model": model,
        "fitness": best_sol_fitness,
        "solution": best_sol,
        "ga_instance": ga_instance,
        "total_time": total_time
    }



