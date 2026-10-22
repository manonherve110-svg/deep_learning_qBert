import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
import sys

# Path project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ia_deepQ import dqn_util as dqnu
from src.config import MOVES_DIRECTIONS


SEED = 42
N_EPISODES = 1800
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000

TARGET_UPDATE = 1000
MEMORY_SIZE = 50000

N_ACTIONS = 4
N_HIDDEN = 256

PRINT_EVERY = 10
DT_SIM = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_MAP = {0:'UP',1:'DOWN',2:'LEFT',3:'RIGHT'}

GLOBAL_T = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


def create_model(n_inputs, n_hidden, n_outputs):
    return nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_outputs)
    )

def get_valid_moves_mask(game):
    """Masque des actions valides : Pas de vide (sauf disque), pas d'ennemi MORTEL."""
    r, c = game.player.position
    board_size = len(game.board)
    mask = []
    
    # 1. Repérer les zones de danger (Code inchangé)
    danger_zones = set()
    for bot in game.bots:
        if bot.is_alive:
            if type(bot).__name__ in ['Sam', 'Slick']:
                continue 
            e_r, e_c = int(round(bot.position[0])), int(round(bot.position[1]))
            danger_zones.add((e_r, e_c))
            is_red_ball = (type(bot).__name__ == 'RedBall')
            is_coily_egg = (type(bot).__name__ == 'Coily' and bot.is_egg)
            if is_red_ball or is_coily_egg:
                danger_zones.add((e_r + 1, e_c))
                danger_zones.add((e_r + 1, e_c + 1))

    # 2. Calcul des mouvements
    for m in ['UP','DOWN','LEFT','RIGHT']:
        dr, dc = MOVES_DIRECTIONS[m]
        nr, nc = round(r+dr), round(c+dc)
        valid = True
        
      
        if hasattr(game.board, 'is_disque') and game.board.is_disque((nr, nc)):
            valid = True
            
        elif nr < 0 or nr >= board_size or nc < 0 or nc > nr:
            valid = False
            
        elif (nr, nc) in danger_zones:
            valid = False

        mask.append(1.0 if valid else 0.0)
    return mask
    
def get_dist_to_nearest_zero(game):
    """Distance Manhattan vers la case non coloriée la plus proche."""
    r, c = game.player.position
    min_dist = float('inf')
    for ir in range(len(game.board)):
        for ic in range(len(game.board.board[ir])):
            if game.board.board[ir][ic] != game.cube_level:
                dist = abs(ir-r)+abs(ic-c)
                if dist<min_dist:
                    min_dist=dist
    return min_dist if min_dist!=float('inf') else 0.0

def get_dist_to_nearest_green(game):
    """Distance vers le Sam ou Slick le plus proche."""
    r, c = game.player.position
    min_dist = float('inf')
    found = False
    for bot in game.bots:
        if bot.is_alive and type(bot).__name__ in ['Slick', 'Sam']:
            dist = abs(bot.position[0] - r) + abs(bot.position[1] - c)
            if dist < min_dist:
                min_dist = dist
                found = True
    return min_dist if found else float('inf')

def encode_state(game):
    """State simple : plateau(28)+ennemis(28)+position(2)+vision(4)+mask(4)+disques(2) = 68"""
    state=[]

 # plateau
    board_flat=[]
    for r in range(7):
        if r<len(game.board.board):
            for cell in game.board.board[r]:
                board_flat.append(float(cell))
    board_flat += [1.0]*(28-len(board_flat))
    state.extend(board_flat[:28])

 # ennemy 
    enemy_grid = []
    
    # Dictionnaires pour stocker positions ennemis vs bonus
    deadly_positions = set()
    bonus_positions = set()

    for bot in game.bots:
        if bot.is_alive:
            e_r, e_c = int(round(bot.position[0])), int(round(bot.position[1]))
            
            if type(bot).__name__ in ['Sam', 'Slick']:
                bonus_positions.add((e_r, e_c))
            else:
                deadly_positions.add((e_r, e_c))
            
    count_cells = 0
    for r in range(7):
        if r < len(game.board.board):
            for c in range(len(game.board.board[r])):
                
                # --- MODIFICATION ICI ---
                if (r, c) in deadly_positions:
                    enemy_grid.append(1.0)  # Danger !
                elif (r, c) in bonus_positions:
                    enemy_grid.append(-1.0) # Miam ! (Valeur différente pour que l'IA apprenne)
                else:
                    enemy_grid.append(0.0)
                # ------------------------
                
                count_cells += 1
                
    enemy_grid += [0.0] * (28 - len(enemy_grid))
    
    state.extend(enemy_grid)

# position
    r, c = game.player.position
    state.append(r/7.0)
    state.append(c/7.0)

# vision locale
    r_i, c_i = int(round(r)), int(round(c))
    for m in ['UP','DOWN','LEFT','RIGHT']:
        dr, dc = MOVES_DIRECTIONS[m]
        nr, nc = r_i+dr, c_i+dc
        val=-1.0
        if 0<=nr<len(game.board.board) and 0<=nc<=nr:
            val=float(game.board.board[nr][nc])
        state.append(val)
# masque actions
    state.extend(get_valid_moves_mask(game))

# disques (Optimisé pour Board actuel : fixés à la ligne 3)
    disks_encoding = []
    
    # Gauche
    if game.board.is_disque((3, -1)):
        disks_encoding.append(1.0)
    else:
        disks_encoding.append(0.0)

    # Droit
    if game.board.is_disque((3, 4)):
        disks_encoding.append(1.0)
    else:
        disks_encoding.append(0.0)
            
    state.extend(disks_encoding)

    return np.array(state,dtype=np.float32)

# =====================
# POLICY
# =====================
def neural_choice(model, game, testing=False):
    global GLOBAL_T
    state = torch.tensor([encode_state(game)], device=DEVICE)
    epsilon = dqnu.epsilon_exponential_decay(EPS_START, EPS_END, EPS_DECAY, GLOBAL_T)
    if not testing and random.random()<epsilon:
        return random.randrange(N_ACTIONS)
    with torch.no_grad():
        qvals = model(state)
        return qvals.argmax(dim=1).item()


def evaluate(model, n_episodes=3, stop_event=None):
    from ..core import Qbert
    model.eval()
    total_score = 0
    max_lvl_reached = 1  # Pour stocker le niveau max atteint
    
    for _ in range(n_episodes):
        if stop_event and stop_event.is_set(): break
        
        game = Qbert(player_mode='ai', model="training")
        game.sim_time = 0.0
        game.start()
        
        steps = 0
        # On ajoute une sécurité steps < 4000 pour laisser le temps de finir un niveau si elle est forte
        while game.state == game.states['RUNNING'] and steps < 4000:
            if stop_event and stop_event.is_set(): break
            
            a = neural_choice(model, game, testing=True)
            moved = game.player.play_move(ACTION_MAP[a])
            
            if moved:
                while game.player.is_moving and game.state == game.states['RUNNING']:
                    game.update(DT_SIM)
                    game.sim_time += DT_SIM
            else:
                game.update(DT_SIM)
                game.sim_time += DT_SIM
            steps += 1
        
        total_score += game.player.score
        
        # On regarde le niveau atteint dans cette partie
        if game.level > max_lvl_reached:
            max_lvl_reached = game.level

    model.train()
    
    # On retourne le score moyen ET le niveau max
    return total_score / n_episodes, max_lvl_reached


def train(window=None, stop_event=None):
    global GLOBAL_T
    from ..core import Qbert

    save_dir = os.path.join(os.path.dirname(__file__),'../../models/dqn')
    os.makedirs(save_dir,exist_ok=True)
    model_path = os.path.join(save_dir,"best_dqn.pt")

    dummy = Qbert(player_mode='ai', model="dummy")
    n_inputs = encode_state(dummy).shape[0]

    online = create_model(n_inputs, N_HIDDEN, N_ACTIONS).to(DEVICE)
    target = create_model(n_inputs, N_HIDDEN, N_ACTIONS).to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = optim.Adam(online.parameters(), lr=LR)
    memory = dqnu.ReplayMemory(MEMORY_SIZE)

    best_eval = -1e9
    start_time = time.time()

    print(f"State dim: {n_inputs} | Device: {DEVICE}")
    print("=== TRAINING START (PURE SCORE) ===")

    for ep in range(N_EPISODES):
        if stop_event and stop_event.is_set():
            print("Training stopped by user.")
            break

        game = Qbert(player_mode='ai', model="training")
        game.sim_time = 0.0  # Utiliser le temps simulé pour le spawn des bots
        
        if window:
            window.game_engine = game
            window.training_info = f"Ep: {ep+1}/{N_EPISODES} | Best: {best_eval:.1f}"
            
        game.start()
        state = encode_state(game)
        total_reward = 0

        # Compteur pour détecter si l'IA tourne en rond trop longtemps
        steps_without_score = 0
        
        # Pour le shaping reward
        prev_min_dist = get_dist_to_nearest_zero(game)
        prev_min_green_dist = get_dist_to_nearest_green(game)

        while game.state == game.states['RUNNING']:
            GLOBAL_T += 1
            
            if stop_event and stop_event.is_set(): break

            prev_score = game.player.score
            prev_lives = game.player.lives

        # pour savoir la stratégie du disque
            was_transporting = game.transport_active

            top_cube_needed = (game.board.board[0][0] != game.cube_level)
            min_danger_dist = float('inf')
            player_r, player_c = game.player.position
            for bot in game.bots:
                if bot.is_alive and type(bot).__name__ not in ['Sam', 'Slick', 'GreenBall']:
                    # Distance de Manhattan
                    d = abs(bot.position[0] - player_r) + abs(bot.position[1] - player_c)
                    if d < min_danger_dist:
                        min_danger_dist = d


            action = neural_choice(online, game)
            moved = game.player.play_move(ACTION_MAP[action])


            # Gestion de la physique (attendre la fin du saut)
            if moved:
                while game.player.is_moving and game.state == game.states['RUNNING']:
                    game.update(DT_SIM)
                    game.sim_time += DT_SIM
            else:
                game.update(DT_SIM)
                game.sim_time += DT_SIM
            
            if window:
                time.sleep(0.001) 

            done = game.state != game.states['RUNNING']

            # rewards ======================================
            
            reward = 0.0

            # 1. Pénalité de temps 
            reward -= 0.1

            curr_min_dist = get_dist_to_nearest_zero(game)
            curr_min_green_dist = get_dist_to_nearest_green(game)
            
            # Aimant vers Sam/Slick (Si présent)
            if curr_min_green_dist != float('inf') and prev_min_green_dist != float('inf'):
                if curr_min_green_dist < prev_min_green_dist:
                    reward += 10.0 
                elif curr_min_green_dist > prev_min_green_dist:
                    reward -= 10.0 
            
            # IMise à jour pour le prochain tour
            prev_min_green_dist = curr_min_green_dist

            # 2. case colorié
            if game.player.score > prev_score:
                reward += 40.0     
                steps_without_score = 0 
                
                # Si coin
                r_curr = int(round(game.player.position[0]))
                r_curr_c = int(round(game.player.position[1]))
                
                if r_curr == 6 and (r_curr_c == 0 or r_curr_c == 6):
                    reward += 50.0
                # SAM 
                if (game.player.score - prev_score) >= 300:
                    reward += 1500.0 
            else:
                steps_without_score += 1
            
            # 3. Pénalité pas de case colorié
            if steps_without_score > 50:
                reward -= 5

            # 4 disque 
            if game.transport_active and not was_transporting:
                
             
                
                # B. Récompense "SAUVETAGE" : Si un monstre était à moins de 2 cases
                if min_danger_dist <= 2.5: 
                    reward += 250.0 
                    #print(f"{GREEN}Sauvetage Disque ! Dist: {min_danger_dist:.1f}{RESET}") 
                
                # C. Récompense "STRATÉGIQUE" : Si le sommet n'est pas colorié
                if top_cube_needed:
                    reward += 250.0 

            # 4. Mort
            if game.player.lives < prev_lives:
                reward -= 350.0
                steps_without_score = 0 

            # --- Shaping Reward (Manget) ---
            curr_min_dist = get_dist_to_nearest_zero(game)
            if curr_min_dist != float('inf') and prev_min_dist != float('inf'):
                if curr_min_dist < prev_min_dist:
                    reward += 1.0
                elif curr_min_dist > prev_min_dist:
                    reward -= 1.0
        
            prev_min_dist = curr_min_dist
            
            # 5. VICTOIRE 
            if done and game.state == game.states['WIN']:
                reward += 1000.0 
                print("victoire")

            # ==========================================

            total_reward += reward
            next_state = encode_state(game)

            memory.push(
                torch.tensor([state], device=DEVICE),
                torch.tensor([[action]], device=DEVICE),
                None if done else torch.tensor([next_state], device=DEVICE),
                torch.tensor([reward], device=DEVICE)
            )

            state = next_state

            dqnu.optimize_online_qnet(
                online, target, memory, optimizer, BATCH_SIZE, GAMMA, DEVICE
            )

            if GLOBAL_T % TARGET_UPDATE == 0:
                target.load_state_dict(online.state_dict())
            
            # Sécurité : Si l'épisode est trop long (IA bloquée), on coupe
            if steps_without_score > 1500:
                break

        # Evaluation
        if ep % PRINT_EVERY == 0:
            eval_score, eval_max_lvl = evaluate(online, stop_event=stop_event)
            # On considère que le modèle est "meilleur" s'il a un meilleur score d'évaluation
            if eval_score >= best_eval:
                best_eval = eval_score
                torch.save(online.state_dict(), model_path)
                print(f"{GREEN}{BOLD}Ep {ep} | Eval {eval_score:.1f} | Lvl Max {eval_max_lvl} | Saved (Best){RESET}")
            else:
                print(f"Ep {ep} | Score {game.player.score} | Eval {eval_score:.1f} | Lvl Max {eval_max_lvl} | R: {total_reward:.1f}")

    print("=== TRAINING END ===")

if __name__=="__main__":
    train()
