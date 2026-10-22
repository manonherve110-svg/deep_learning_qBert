import random
import pickle # sauvegarde le dictionnaire dans un fichier
import os
import time 

try:
    from src.config import MOVES, MOVES_DIRECTIONS
except ModuleNotFoundError:
    from config import MOVES, MOVES_DIRECTIONS

class QTableAgent:
    def __init__(self, actions=MOVES["qbert"]):
        self.q_table = {}  #initialistion du dictionnaire de mémoire 
        self.actions = actions

        # param d'app
        self.lr = 0.1 
        self.gamma = 0.95
        self.epsilon = 1.0 # au debut full hasard
        self.epsilon_decay = 0.999 # si epsilon reduit trop vite , tombe dans un minimum local
        self.epsilon_min = 0.05

    def get_nearest_target_direction(self, game, p_r, p_c):
        """
        Retourne une direction grossière vers la case non coloriée la plus proche.
        0: Pas de cible (Gagné)
        1: Haut-Gauche
        2: Haut-Droite
        3: Bas-Gauche
        4: Bas-Droite
        """
        min_dist = float('inf')
        target_r, target_c = -1, -1
        found = False

        # On parcourt tout le plateau pour trouver les cases à faire (valeur 0)
        for r in range(len(game.board)):
            for c in range(r + 1):
                # Si la case n'est pas encore faite (ou couleur de départ)
                if game.board.board[r][c] == 0: 
                    dist = abs(r - p_r) + abs(c - p_c)
                    if dist < min_dist:
                        min_dist = dist
                        target_r, target_c = r, c
                        found = True
        
        if not found:
            return 0 
            
        # Déterminer la direction relative simple
        dr = target_r - p_r
        dc = target_c - p_c
        
        if dr <= 0 and dc <= 0: return 1 # Haut-Gauche 
        if dr <= 0 and dc > 0:  return 2 # Haut-Droite
        if dr > 0 and dc <= 0:  return 3 # Bas-Gauche
        if dr > 0 and dc > 0:   return 4 # Bas-Droite
        
        return 4 

    def get_state(self, game):
        """
        Simplifie le jeu , le mettre en forme de tuple 
        pour commencer je vais mettre que la position du joueur
        """
        pos_l , pos_c = round(game.player.position[0]), round(game.player.position[1])  # round: arrondie au plus proche 

        ## ajouter d'autre caracteristique pour ameliorer l'ia
        try:
            current_color = game.board.get_color((pos_l, pos_c))
        except:
            current_color = 0 

        danger = 0 # 0 = Sûr, 1 = Danger
        
        for bot in game.bots:
            if not bot.is_alive: continue
            b_r, b_c = bot.position
            dist = abs(b_r - pos_l) + abs(b_c - pos_c)
            if dist < 2.0: # Si très proche
                danger = 1
                break

        target_dir = self.get_nearest_target_direction(game, pos_l, pos_c)

        state =(pos_l , pos_c, current_color, danger, target_dir)
        return state

    def get_q_value(self , state , action):
        """
        recupere la note du q-value 0.0 si on ne la connait pas encore"""
        return self.q_table.get((state, action), 0.0) 

    def choose_action(self, game):
        state = self.get_state(game)

        ## --- strat exploration (epsilon -greedy) ---
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        else:
            q_values = [self.get_q_value(state,a) for a in self.actions]
            max_q = max(q_values)
            best_actions = [self.actions[i] for i in range(len(self.actions)) if q_values[i] == max_q ]

            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """ mise a jour du dictionnaire de memoire"""

        old_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            next_max = max([self.get_q_value(next_state,a) for a in self.actions])
            target = reward + self.gamma * next_max

        new_q = old_q + self.lr * (target - old_q)
        
        # On écrit la nouvelle note dans le dictionnaire
        self.q_table[(state, action)] = new_q

    def end_epsilon(self):
        """ reduire le epsilon pour reduire l'exploration , appeler en fin de partie"""
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def save(self, filename="models/qlearning/best_qlearning.pkl"):
        os.makedirs(os.path.dirname(filename) , exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Table Q sauvegardée ({len(self.q_table)} états connus).")


    def load(self, filename="models/qlearning/best_qlearning.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            self.epsilon = 0.05 
            print("Table Q chargée.")

def train_qlearning_loop():
    """Fonction entrainement de l'ia"""
    from ..core import Qbert


    print(" Demarrage de l'entrainement Qlearning...")

    agent = QTableAgent()
    game = Qbert(player_mode='ai', model=agent, target_fps=9999)

    EPISODES = 3000
    t0 = time.time()

    for e in range(EPISODES):
        game.reset()
        game.start()

        state = agent.get_state(game)
        total_reward = 0
        steps = 0

        # anti-minimum local : compteur de visites
        visited_positions = {}

        while game.state == game.states['RUNNING']:

            # 1. Choix action
            action = agent.choose_action(game)

            # 2. Sauvegarde avant action
            score_before = game.player.score
            lives_before = game.player.lives

            # 3. Exécution
            game.player.play_move(action)

            while game.player.is_moving and game.state == game.states['RUNNING']:
                game.update(dt=0.05)

            # 4. Nouvelle position
            new_r, new_c = round(game.player.position[0]), round(game.player.position[1])
            pos = (new_r, new_c)

            # mise à jour visites
            visited_positions[pos] = visited_positions.get(pos, 0) + 1

            reward = 0

            #  mort
            if game.player.lives < lives_before or game.state == game.states['GAME_OVER']:
                reward = -100

            #  victoire
            elif game.state == game.states['WIN']:
                reward = 100

            #  score
            elif game.player.score > score_before:
                reward = 50

            # mouvement normal
            else:
                reward = -1

            

            # 5. Apprentissage
            next_state = agent.get_state(game)
            done = (game.state != game.states['RUNNING'])

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if steps > 3000:
                break

        agent.end_epsilon()

        if (e + 1) % 50 == 0:
            print(
                f"Episode {e+1}/{EPISODES} | "
                f"Score: {game.player.score} | "
                f"Reward: {total_reward} | "
                f"Epsilon: {agent.epsilon:.2f}"
            )

    print(f"Entraînement terminé en {time.time() - t0:.2f}s")
    agent.save("models/qlearning/best_qlearning.pkl")
