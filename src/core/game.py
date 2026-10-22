import time
import curses
import argparse
import random
import arcade
import json
import os
import sys
import torch



# Ajout de la racine du projet au path pour permettre l'exécution directe
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)


from src.models import Board,Bot,Player,HumanPlayer,Coily,RedBall,AIPlayer,MinMaxPlayer, Slick,Sam, Ugg, WrongWay
from src.config import *
from src.ia import mlp_train, QbertMLP
from src.ia_deepQ.qtable import QTableAgent, train_qlearning_loop
from src.ia_deepQ.dqn import train, create_model, DEVICE



## ==============  CLASS PRINCIPAL POUR LE JEU ============== ##
class Qbert:
    """Classe principale représentant le jeu Q*bert."""
    
    board: Board
    player: Player
    mode: str
    state: int
    states = {
        'GAME_OVER': 0,
        'RUNNING': 1,
        'WIN': 2,
        'PAUSE': 3,
        'IN_MENU': 4
    }  
    bots: list[Bot]
    target_fps: int # le nombre de fps qu'on souhaite afficher
    start_time: float # temps de démarrage de la partie
    lst: float # last spawn time de l'ennemi
    sim_time: float | None # temps simulé pour le spawn des bots
    level: int
    spwint: float # spawn_interval --> temps de spawn entre deux bots pourra varier en fonction des niveaux
    coily_timer: float
    slick_sam_timer: float
    ugg_wrongway_timer: float
    next_side_bot: int # 0 pour Ugg, 1 pour WrongWay
    next_move: str
    invincibility_timer: float
    screen: curses.window
    cube_level: int
    coily_spawned: bool

    transport_active = False
    transport_side = None
    rescue_active = False
    high_score: int = 0
    high_score_genetic: int = 0
    high_score_dqn: int = 0
    high_score_minmax: int = 0

    
    def __init__(self, board_size: int = 7, player_mode: str = 'human', model = None, target_fps:int = 60) -> None:
        """Initialise le jeu Q*bert.
        
        Args:
            player_mode: Mode du joueur ('human' ou 'ai')
            board_size: Taille du plateau de jeu
            target_fps: Images par seconde ciblées
            model: Modèle de l'IA

        """
        
        self.board = Board(board_size)

        if player_mode == 'human':
            self.player = HumanPlayer(game=self) 
        elif player_mode == 'minmax':
            self.player = MinMaxPlayer(game=self)
        elif player_mode == 'ai':
            if model is None:
                try:
                    model = QbertMLP.load_model("best_model.pth")
                except FileNotFoundError:
                    raise ValueError("Un modèle doit être spécifié pour jouer en mode IA ou bien un modèle doit avoir été enregistré ('model.pth').")
            self.player = AIPlayer(game=self, model=model)
        self.mode = player_mode
        self.state = self.states['IN_MENU'] # On commencera dans le menu
        self.bots = list[Bot]()
        self.target_fps = target_fps
        self.level = 1
        self.spwint = random.uniform(0, 1)
        # self.spwint = SPAWN_INTERVALS_BY_LEVEL[1]
        self.coily_timer = random.uniform(10, 20)
        self.slick_sam_timer = random.uniform(15, 30)
        self.ugg_wrongway_timer = random.uniform(20, 40)
        self.next_side_bot = random.choice([0, 1])
        self.invincibility_timer = 0.0
        self.start_time = None # c'est au PREMIER start qu'on lance le chrono 
        self.sim_time = None # pour l'ia 
        self.transport_active = False
        self.screen = None # le screen est géné par le main
        self.transport_side = None
        self.rescue_active = False
        self.high_score = 0
        self.high_score_genetic = 0
        self.high_score_dqn = 0
        self.high_score_minmax = 0
        self.load_high_scores()
        self.cube_level = 1
        self.coily_spawned = False

    ## ==============  TO STRING ============== ##
    def __str__(self) -> str: 
        """Retourne une représentation string complète du jeu."""
        
        # --- MENU ---

        def get_menu_str(self) -> str:
            """Helper pour l'affichage du menu."""
            header = ["", "", "==== Q*bert =====".center(TERMINAL_SIZE["WIDTH"])]
            options = [
                "", "Bienvenue dans Q*bert !".center(TERMINAL_SIZE["WIDTH"]), "",
                "[1] Nouvelle partie".center(TERMINAL_SIZE["WIDTH"]),
                "[2] Mode IA".center(TERMINAL_SIZE["WIDTH"]),
                "[x] Quitter".center(TERMINAL_SIZE["WIDTH"]), ""
            ]
            return "\n".join(header + options)

        if self.state == self.states['IN_MENU']:
            return get_menu_str(self)

        # --- JEU / PAUSE / GAME OVER ---
        lines = []
        
        # 1. En-tête (Score, Vies)
        lines.append("")
        lines.append(f"Score: {self.player.score} | High Score: {self.high_score} | Vies: {self.player.lives}".center(TERMINAL_SIZE["WIDTH"]))
        lines.append("-" * TERMINAL_SIZE["WIDTH"])

        # 2. Plateau de base
        board_lines = self.board.__str__().split('\n')



        # Fonction locale pour dessiner une entité sur le plateau textuel
        def _draw_entity(row: int, col: int, symbol: str):
            """Dessine un symbole à la position donnée sur board_lines."""
            if not (0 <= row < len(board_lines)):
                return
            
            line = board_lines[row]
            bracket_start = line.find('[')
            bracket_end = line.find(']')
            
            if bracket_start == -1: return

            char_index = -1
            
            # --- Cas 1 : Disque GAUCHE ---
            if col == -1:
                char_index = line.find('@', 0, bracket_start)
                if char_index == -1: char_index = bracket_start - 2

            # --- Cas 2 : Disque DROITE ---
            elif col == row + 1:
                char_index = line.find('@', bracket_end)
                if char_index == -1: char_index = bracket_end + 2

            # --- Cas 3 : Pyramide ---
            elif 0 <= col <= row:
                char_index = bracket_start + 2 + (col * 2)
            
            # Application
            if 0 <= char_index < len(line):
                line_list = list(line)
                line_list[char_index] = symbol
                board_lines[row] = "".join(line_list)
        

        # 3. Superposition des entités (si le jeu n'est pas fini)
        
        if self.state != self.states['GAME_OVER']:
            # Joueur
            p_r, p_c = self.player.position
            _draw_entity(int(round(p_r)), int(round(p_c)), 'J')
            
            # Bots
            for bot in self.bots:
                if bot.is_alive:
                    b_r, b_c = bot.position
                    _draw_entity(int(round(b_r)), int(round(b_c)), str(bot))

        lines.extend(board_lines)

        # 4. Footer (Pause ou Game Over)
        if self.state == self.states['PAUSE']:
            lines.append("")
            lines.append("PAUSE - Appuyez sur 'r' pour reprendre".center(TERMINAL_SIZE["WIDTH"]))
            lines.append("\n" + "PAUSE".center(TERMINAL_SIZE["WIDTH"]))
            
        elif self.state == self.states['GAME_OVER']:
            lines.append("\n" + "GAME OVER - [1] Rejouer [X] Quitter".center(TERMINAL_SIZE["WIDTH"]))

        return '\n'.join(lines)

            
    ## ==============  DESSINE SUR L'ÉCRAN ============== ##
    def draw(self, screen:curses.window) -> None:
        """Dessine l'état actuel du jeu sur l'écran en utilisant __str__."""
        height, width = screen.getmaxyx()
        
        # Mettre à jour la largeur globale pour que __str__ s'adapte (centrage)
        TERMINAL_SIZE["WIDTH"] = width
        # TERMINAL_SIZE["HEIGHT"] = height # Pas utilisé pour l'instant

        try:
            # Générer tout le texte
            full_text = self.__str__()
            lines = full_text.split('\n')
            
            # Afficher ligne par ligne
            for i, line in enumerate(lines):
                if i >= height: # Ne pas dépasser l'écran
                    break
                
                # Tronquer si la ligne est plus large que l'écran pour éviter crash curses
                display_line = line[:width-1]
                
                try:
                    screen.addstr(i, 0, display_line)
                except curses.error:
                    # Ignorer les erreurs de bordure (ex: écrire dans le coin bas-droit)
                    pass

        except Exception:
            pass

    ## ==============  Demarre une partie  ============== ##
    def start(self) -> None:
        """Démarre une nouvelle partie."""
        self.board.reset()

        self.player.reset()
        self.bots = []  # Vide la liste des bots
        self.state = self.states['RUNNING']  # Passe en mode jeu
        self.level = 1  # Remet le niveau à 1
        self.cube_level = 1
        self.coily_spawned = False
        self.spwint = random.uniform(0, 3)  # Premier spawn entre 0 et 3 sec
        # self.spwint = SPAWN_INTERVALS_BY_LEVEL[1]  # Premier spawn selon le niveau
        self.coily_timer = random.uniform(10, 20)
        self.slick_sam_timer = random.uniform(15, 30)
        self.ugg_wrongway_timer = random.uniform(20, 40)
        self.lst = 0 if self.sim_time is not None else time.time_ns()
        if self.start_time is None:
            self.start_time = time.time()


    
    ## ==============  Retourne a l'état du menu  ============== ##
    def reset(self) -> None:
        """Réinitialise le jeu à son état initial (retour au menu)."""
        self.player.reset()
        self.board.reset()
        self.bots = list[Bot]()  # Vide la liste des bots
        self.state = self.states['IN_MENU']  # Retour au menu
        self.level = 1  # Remet le niveau à 1
        self.coily_spawned = False
        self.spwint = random.uniform(0, 3) # Réinitialise l'intervalle de spawn
        self.coily_timer = random.uniform(10, 20)
        self.slick_sam_timer = random.uniform(15, 30)
        self.ugg_wrongway_timer = random.uniform(20, 40)
        self.invincibility_timer = 0.0
        self.transport_active = False
        self.transport_side = None
        self.rescue_active = False
        # Initialise sim_time AVANT lst pour que lst soit cohérent
        if self.mode == 'ai':
            self.sim_time = 0.0
        else:
            self.sim_time = None
        # Initialise lst selon le mode : 0 pour IA (sim_time), time_ns() pour humain
        self.lst = 0 if self.sim_time is not None else time.time_ns()

    ## ==============  Traite l'entrée du joueur  ============== ##
    def handle_player_entry(self, key: int) -> None:
        """Traite l'entrée de du player et met à jour les positions si besoin."""
        move = None
        if key == ord('z') or key == ord('Z') or key == curses.KEY_UP: move = 'UP'
        elif key == ord('s') or key == ord('S') or key == curses.KEY_DOWN: move = 'DOWN'
        elif key == ord('q') or key == ord('Q') or key == curses.KEY_LEFT: move = 'LEFT'
        elif key == ord('d') or key == ord('D') or key == curses.KEY_RIGHT: move = 'RIGHT'
    
        if move:
            self.player.play_move(move) 
        
        if key == ord('p') or key == ord('P'): self.pause()
        if (key == ord('r') or key == ord('R')) and self.state == self.states['PAUSE']: self.resume()


    ## ==============  Gère apparition des bots  ============== ##
    def spawn_bot(self, dt: float) -> None:
        """Gère l'apparition des bots. Utilise sim_time si disponible (entraînement IA)."""

        current_ns = time.time_ns() if self.sim_time is None else int(self.sim_time * 1e9)
        # Facteur de difficulté : réduit les délais de 8% par niveau à partir du niveau 3, min 40% du temps original
        if self.level < 2:
            level_factor = 1.0
        else:
            level_factor = max(0.4, 1.0 - (self.level - 3) * 0.08)

        if (current_ns - self.lst) / 1e9 >= self.spwint and self.level >= 1:
                self.bots.append(RedBall(self))
                
                self.lst = current_ns
                self.spwint = random.uniform(6 * level_factor, 8 * level_factor)
        
        # Gestion de Coily (Level 2+, 1 par vague, 8-15s délai)
        if self.level >= 2 and not self.coily_spawned and not any(isinstance(b, Coily) for b in self.bots):
                self.coily_timer -= dt
                if self.coily_timer <= 0:
                    self.bots.append(Coily(self))
                    self.coily_spawned = True
                    self.coily_timer = random.uniform(6 * level_factor, 8 * level_factor)
        
        # Gestion de Slick/Sam (Level 5+, 20-35s délai)
        if self.level >= 4 :
            if(not any(isinstance(b, (Slick, Sam)) for b in self.bots if b.is_alive)):
                self.slick_sam_timer -= dt
                if self.slick_sam_timer <= 0:
                        if random.choice([True, False]):
                            self.bots.append(Slick(self))
                        else:
                            self.bots.append(Sam(self))
                        self.slick_sam_timer = random.uniform(8 * level_factor, 15 * level_factor)
            elif not any(isinstance(b, ( Sam)) for b in self.bots if b.is_alive):
                    self.slick_sam_timer -= dt
                    if self.slick_sam_timer <= 0:
                        self.bots.append(Slick(self))
                        self.slick_sam_timer = random.uniform(20 * level_factor, 30 * level_factor)
            elif not any(isinstance(b, ( Slick)) for b in self.bots if b.is_alive):
                    self.slick_sam_timer -= dt
                    if self.slick_sam_timer <= 0:
                        self.bots.append(Sam(self))
                        self.slick_sam_timer = random.uniform(20 * level_factor, 30 * level_factor)
        
        if self.level >= 6:
            self.ugg_wrongway_timer -= dt
            if self.ugg_wrongway_timer <= 0:
                spawn_row = float(len(self.board) + 3)
                if self.next_side_bot == 0:
                    bot = Ugg(self)
                    bot.position = (spawn_row, spawn_row)
                    bot._cached_sim_position = bot.position
                    bot.future_moves = []
                    bot.fill_future_moves()
                    self.bots.append(bot)
                    self.next_side_bot = 1
                else:
                    bot = WrongWay(self)
                    bot.position = (spawn_row, 0.0)
                    bot._cached_sim_position = bot.position
                    bot.future_moves = []
                    bot.fill_future_moves()
                    self.bots.append(bot)
                    self.next_side_bot = 0
                self.ugg_wrongway_timer = random.uniform(15 * level_factor, 30 * level_factor)
            
                 
            



    ## ==============  METTRE À JOUR LES BOTS  ============== ##
    
    def update_bots(self, dt: float) -> None:
        """Met à jour les bots et en fait spawn si besoin"""

        self.spawn_bot(dt)
        
        for bot in list(self.bots):
            if not bot.is_alive:
                continue
                
            bot.update_position(dt)
            
            if isinstance(bot, Coily) and bot.is_egg:
                bot.update_egg(dt)
            
            if not bot.is_moving:
                if not isinstance(bot, (Ugg, WrongWay)) and not self.is_in_bounds(bot.position):
                    if bot.position[0] < 0:
                        pass
                    else:
                        bot.die()
                        continue
                
                if isinstance(bot, (Ugg, WrongWay)) and bot.position[0] < -4:
                        bot.die()
                        continue
                
                if isinstance(bot, (Slick, Sam)):
                    r, c = int(round(bot.position[0])), int(round(bot.position[1]))
                    if self.is_in_bounds((r, c)):
                        self.board[r, c] = 0
                
                try:
                    move = bot.get_next_move()
                    
                    if isinstance(bot, (Slick, Sam)):
                        r, c = bot.position
                        dr, dc = MOVES_DIRECTIONS[move]
                        target_pos = (r + dr, c + dc)
                        
                        if not self.is_in_bounds(target_pos):
                            possible_moves = list(bot.moves)
                            random.shuffle(possible_moves) 
                            
                            found = False
                            for alt_move in possible_moves:
                                adr, adc = MOVES_DIRECTIONS[alt_move]
                                alt_target = (r + adr, c + adc)
                                if self.is_in_bounds(alt_target):
                                    move = alt_move
                                    found = True
                                    break
                            
                            if not found:
                                all_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
                                random.shuffle(all_moves)
                                for alt_move in all_moves:
                                    adr, adc = MOVES_DIRECTIONS[alt_move]
                                    alt_target = (r + adr, c + adc)
                                    if self.is_in_bounds(alt_target):
                                        move = alt_move
                                        if move not in bot.moves:
                                            bot.moves = list(bot.moves)
                                            bot.moves.append(move)
                                        break

                    bot.play_move(move)
                except Exception:
                    pass

            # Vérification Collision
            b_r, b_c = int(round(bot.position[0])), int(round(bot.position[1]))
            p_r, p_c = int(round(self.player.position[0])), int(round(self.player.position[1]))
            
            if (b_r, b_c) == (p_r, p_c) and bot.is_alive:
                if isinstance(bot, (Slick, Sam)):
                    self.player.addScore(300)
                    bot.die()
                    continue
                
                if not self.rescue_active and not self.transport_active and self.invincibility_timer <= 0:
                    self.handle_death()
                    return

    ## ==============  METTRE À JOUR LE JEU  ============== ##


    def update(self, dt: float = 0.0) -> None:
        """Met à jour le jeu.
        Vérifie les collisions, met à jour les positions, vérifie l'état du jeu (game over, win, pause, etc. )
        Args:
            dt: Temps écoulé depuis la dernière frame (en secondes)
        """

        if self.invincibility_timer > 0:
            self.invincibility_timer -= dt

        if self.rescue_active:
            self.update_rescue(dt)
            self.update_bots(dt)
            return

        if self.transport_active:
            self.update_transport(dt)
            self.update_bots(dt)
            return

        if self.mode in ['ai', 'minmax'] and not self.player.is_moving:
            move = self.player.next_move()
            if move:
                self.player.play_move(move)

        was_moving = self.player.is_moving
        self.player.update_position(dt)
        # attention: fait aussi spawn les bots si besoin
        self.update_bots(dt)
        if self.rescue_active:
            return

        r, c = int(round(self.player.position[0])), int(round(self.player.position[1]))

        if not self.is_valid_position():
            if self.board.is_disque((r, c)):
                self.disque_activation((r, c))
            else:
                self.handle_death()
            return

        if was_moving and not self.player.is_moving:
            if 0 <= r < len(self.board) and 0 <= c < len(self.board.board[r]):
                if self.board[r, c] < self.cube_level or self.level > LEVEL_CUBE_REVISION:
                    self.board[r, c] = (self.board[r, c] + 1) % (self.cube_level + 1)

                    if self.cube_level == self.board[r, c]:
                        self.player.addScore(25)
                    else:
                        self.player.addScore(15)

            if all(cell == self.cube_level for row in self.board for cell in row):
                self.state = self.states['WIN']
                self.handle_win()



    
    def disque_activation(self, pos : tuple[(int,int)]):
        "Gère l'activation des disques"
        self.transport_active = True
        
        # Détermination du côté
        r, c = pos
        if c == -1:
            self.transport_side = 'LEFT'
        else:
            self.transport_side = 'RIGHT'
            
        self.board.remove_disque(pos)
        self.player.addScore(100)
        # On ne change pas self.player.position ici, on laisse le transport faire
        # self.player.position = (0.0 ,0.0)

        self.player.is_moving = False
        self.player.target_position = None
        self.player.start_position = None
        self.player.move_direction = None


    def update_transport(self, dt: float) -> None:
        """Gère le déplacement du joueur sur le disque."""
        speed = 2  # Vitesse de montée
        r, c = self.player.position
        
        # Déplacement vers le haut (diminution de r)
        new_r = r - (speed * dt)
        
        # On monte jusqu'à -1.0 pour être visuellement au-dessus
        if new_r <= -1.0:
            # Arrivé au sommet, on "tombe" sur la case (0,0)
            self.player.position = (0.0, 0.0)
            
            if self.board[0, 0] < self.cube_level or self.level > LEVEL_CUBE_REVISION :
                    self.board[0, 0] =  (self.board[0,0] + 1) % (self.cube_level+1)
                    
                    if self.cube_level == self.board[0, 0] :  self.player.addScore(25)
                    else : self.player.addScore(15)

            # Vérification Victoire
            if all(cell == self.cube_level for row in self.board for cell in row):
                self.state = self.states['WIN']
                self.handle_win()
                return
          

            self.transport_active = False
            self.transport_side = None
            self.invincibility_timer = 1.0
            return

        # Mise à jour de la colonne en fonction du côté pour rester sur le bord
        if self.transport_side == 'LEFT':
            new_c = -1.0
        else: # RIGHT
            new_c = new_r + 1.0
            
        self.player.position = (new_r, new_c)

    def update_rescue(self, dt: float) -> None:
        """Gère le déplacement du joueur lors du sauvetage (retour en haut)."""
        speed = 3.0  # Vitesse de retour
        target = (0.0, 0.0)
        curr = self.player.position
        
        # Calcul du vecteur vers la cible
        dr = target[0] - curr[0]
        dc = target[1] - curr[1]
        dist = (dr**2 + dc**2)**0.5
        
        # Si on est arrivé (ou très proche)
        if dist < 0.1:
            self.player.position = target
            self.rescue_active = False
            self.player.respawn() # Réinitialise les états du joueur
            # Relance le timer de spawn selon le mode
            self.lst = 0 if self.sim_time is not None else time.time_ns()
            self.invincibility_timer = 2.0
        else:
            move_r = (dr / dist) * speed * dt
            move_c = (dc / dist) * speed * dt
            self.player.position = (curr[0] + move_r, curr[1] + move_c)



    def pause(self) -> None:
        """Met le jeu en pause."""
        self.state = self.states['PAUSE']
    
    def resume(self) -> None:
        """Reprend le jeu depuis la pause."""
        self.state = self.states['RUNNING']
    


    def next_level(self) -> None:
        """Passe au niveau suivant."""
        current_lives = self.player.lives  # Sauvegarde des vies
        current_score = self.player.score  # Sauvegarde du score
        current_sim_time = self.sim_time  # Sauvegarde de sim_time pour l'IA
        
        ancien_level = self.level
        ancien_score = self.player.score
        ancien_lives = self.player.lives
        self.reset() 
        
        # Restaurer sim_time après reset pour continuer la simulation IA
        self.sim_time = current_sim_time
        
        self.level = ancien_level + 1
        if(self.level == LEVEL_CUBE_DOUBLE_JUMP) : self.cube_level += 1

        self.player.score = ancien_score
        self.player.lives = ancien_lives
        self.player.addScore(((1000 + 250*ancien_level-1)%5000))

        self.spwint = random.uniform(0, 3)
        # self.spwint = SPAWN_INTERVALS_BY_LEVEL.get(self.level, SPAWN_INTERVALS_BY_LEVEL[10])
        # Relance le timer de spawn selon le mode
        self.lst = 0 if self.sim_time is not None else time.time_ns()
        self.state = self.states['RUNNING']
        
    def is_in_bounds(self, position: tuple[float, float]) -> bool:
        """Vérifie si une position est dans les limites du plateau."""
        r, c = position
        r_int, c_int = round(r), round(c)
        
        if r_int < 0 or r_int >= len(self.board):
            return False
            
        if c_int < 0 or c_int > r_int:
            return False
        
        return True

    def is_valid_position(self) -> bool:
        """Vérifie si une position est à l'intérieur du plateau et s'il n'y a pas de collision avec un bot."""

        if self.transport_active or self.rescue_active:
            return True

        if not self.is_in_bounds(self.player.position):
            return False
        
        # Vérification des collisions avec les bots 
        if self.invincibility_timer <= 0:
            r_int, c_int = round(self.player.position[0]), round(self.player.position[1])
            for bot in list(self.bots):
                if not bot.is_alive:
                    continue
                bot_r, bot_c = bot.position
                bot_r_int, bot_c_int = round(bot_r), round(bot_c)
                
                # Si le joueur est sur la même case qu'un bot
                if r_int == bot_r_int and c_int == bot_c_int:
                    if isinstance(bot, (Slick, Sam)):
                        self.player.addScore(300)
                        bot.die()
                        continue
                    return False
            
        return True

    def handle_death(self) -> None:
        """Gère la mort du joueur (sortie de plateau)."""
        self.player.lives -= 1
        
        # Affichage du message "T'es mort"
        if self.screen:
            height, width = self.screen.getmaxyx()
            msg = "T'ES MORT !"
            try:
                # On essaye de centrer le message
                y = height // 2
                x = max(0, (width - len(msg)) // 2)
                self.screen.addstr(y, x, msg, curses.A_BOLD | curses.A_REVERSE)
                self.screen.refresh()
            except curses.error:
                pass
            
            # Pause de 1.5 secondes
            curses.napms(1500)

        if self.player.lives > 0:
            # Réinitialisation du joueur au sommet
            self.rescue_active = True
            
        else:
            self.state = self.states['GAME_OVER']
            self.update_high_score()



    def handle_input(self, key: int) -> None:
        """Gère les entrées utilisateur globales."""
        if key == ord('x') or key == ord('X'):
            raise KeyboardInterrupt

        if self.state == self.states['IN_MENU']:
            if key == ord('1'):
                self.start()
            elif key == ord('2'):
                self.main_cli(stdscr=self.screen)
    
        elif self.state == self.states['RUNNING']:
            if key != -1:
                
                move = None
                if key == ord('z') or key == ord('Z') or key == curses.KEY_UP: move = 'UP'
                elif key == ord('s') or key == ord('S') or key == curses.KEY_DOWN: move = 'DOWN'
                elif key == ord('q') or key == ord('Q') or key == curses.KEY_LEFT: move = 'LEFT'
                elif key == ord('d') or key == ord('D') or key == curses.KEY_RIGHT: move = 'RIGHT'
            
                if move:
                    self.player.play_move(move) 
                
                if key == ord('p') or key == ord('P'): self.pause()
                if (key == ord('r') or key == ord('R')) and self.state == self.states['PAUSE']: self.resume()
                

        elif self.state == self.states['PAUSE']:
            if key == ord('r') or key == ord('R'):
                self.resume()
        
        elif self.state == self.states['GAME_OVER']:
            if key == ord('1'):
                self.start()


    



    def handle_win(self, human: bool = True) -> None:
        """Gère la victoire (niveau terminé)."""
        # Petite pause
        if self.screen and human:
            curses.napms(1500)
        

        self.next_level()

    def load_high_scores(self) -> None:
        """Charge les meilleurs scores depuis un fichier JSON."""
        if os.path.exists("data.json"):
            try:
                with open("data.json", "r") as f:
                    data = json.load(f)
                    self.high_score = data.get("high_score", 0)
                    self.high_score_genetic = data.get("high_score_genetic", 0)
                    self.high_score_dqn = data.get("high_score_dqn", 0)
                    self.high_score_minmax = data.get("high_score_minmax", 0)
            except (json.JSONDecodeError, IOError):
                pass

    def save_high_score(self) -> None:
        """Sauvegarde le meilleur score dans un fichier JSON."""
        try:
            with open("data.json", "w") as f:
                json.dump({
                    "high_score": self.high_score, 
                    "high_score_genetic": self.high_score_genetic,
                    "high_score_dqn": self.high_score_dqn,
                    "high_score_minmax": self.high_score_minmax
                }, f)
        except IOError:
            print("Erreur lors de la sauvegarde du score.")

    def update_high_score(self) -> None:
        """Met à jour le high score approprié selon le mode."""
        if self.mode == 'human':
            if self.player.score > self.high_score:
                self.high_score = self.player.score
                self.save_high_score()
        elif self.mode == 'minmax':
            if self.player.score > self.high_score_minmax:
                self.high_score_minmax = self.player.score
                self.save_high_score()
        elif self.mode == 'ai' and isinstance(self.player, AIPlayer):
            # Vérification du type de modèle par nom de classe pour éviter les soucis d'import
            model_name = type(self.player.model).__name__
            if model_name == 'QbertMLP':
                if self.player.score > self.high_score_genetic:
                    self.high_score_genetic = self.player.score
                    self.save_high_score()
            elif model_name == 'Sequential': # DQN (torch.nn.Sequential)
                if self.player.score > self.high_score_dqn:
                    self.high_score_dqn = self.player.score
                    self.save_high_score()

    def handle_game_over(self) -> None:
        """Gère l'état Game Over."""
        self.update_high_score()

        if self.screen:
            height, width = self.screen.getmaxyx()
            msg = "TU AS PERDU"
            try:
                y = height // 2
                x = max(0, (width - len(msg)) // 2)
                self.screen.addstr(y, x, msg, curses.A_BOLD | curses.A_REVERSE)
                self.screen.refresh()
            except curses.error:
                pass
            if self.screen:
                curses.napms(1500)
        self.reset() # Retour au menu

    def handle_game_over_ai(self) -> int:
        """Gère l'état Game Over pour l'IA."""
        score = self.player.score
        self.reset()
        return score


    # le stdscr est passé par curses.wrapper   
    def main_cli(self, stdscr) -> None:
        
        """Lance le jeu avec l'interface curses."""
        self.screen = stdscr
        self.screen.keypad(True)
        curses.noecho()
        curses.cbreak() # évite d'avoir à taper entrer pour validé l'entrée
        curses.curs_set(0)  # Cache le curseur
        self.screen.nodelay(True) # Ne bloque pas l'attente d'une touche
        
        last_time = time.time()

        if self.mode in ['ai', 'minmax']:
            self.start() # autoplay pour l'ia
    
        while True:
            #gestion du temps de boucle
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # nettoie l'ancien affichage et affiche le nouveau
            self.screen.clear() 
            self.draw(self.screen) 
        
            if self.mode == 'human':# Gestion des entrées
                key = self.screen.getch() # Récupère la touche pressée (-1 si aucune)
                self.handle_input(key)

            # Mettre à jour la logique du jeu
            if self.state == self.states['RUNNING']:
                self.update(dt)
        
            # Gestion des états de fin de partie (WIN) ou indéfinis
            if self.state == self.states['WIN']: 
                self.handle_win()
            
            elif self.state == self.states['GAME_OVER']:
                self.handle_game_over()
                if self.mode in ['ai', 'minmax']:
                    print(f"Score: {self.player.score}")
                    return 

            
        
            self.screen.refresh() # Met à jour l'écran
            
            # Petit délai pour respecter le target_fps
            elapsed = time.time() - current_time
            curses.napms(max(0, int((1 / self.target_fps - elapsed) * 1000))) 
        

    # A AJOUTER DANS LA CLASSE Qbert (fichier src/Qbert.py)
    
    def play(self, mode='ai', max_tick=10000, next_move_fn=None, on_step=None, on_start=None):
        """
        Lance une partie complète sans affichage (Headless) pour l'entraînement.
        """
        self.start()
        
        # Initialisation optionnelle
        if on_start:
            on_start(self)
            
        ticks = 0
        dt = 0.05 # On simule 20 FPS pour la physique

        # BOUCLE DU JEU (Tant que c'est RUNNING)
        prev_mode = self.mode
        if next_move_fn is not None:
            self.mode = 'human'

        while self.state == self.states['RUNNING'] and ticks < max_tick:
            
            # 1. Demander à l'IA ce qu'elle veut faire
            if next_move_fn:
                action = next_move_fn(self) # L'IA regarde le jeu et décide
                
                # Si l'IA renvoie un index (0,1,2,3) -> DQN pur
                if isinstance(action, int):
                    moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                    if 0 <= action < 4:
                        self.player.play_move(moves[action])
                
                # Si l'IA renvoie une chaine "UP" -> Player classique
                elif isinstance(action, str):
                    self.player.play_move(action)
            
            # 2. Mettre à jour le jeu (physique, ennemis, score)
            self.update(dt)

            # 3. Callback (utile pour enregistrer les données d'entraînement)
            if on_step:
                on_step(self)

            ticks += 1

        # Retourner le score final pour que le DQN sache s'il a bien joué
        if next_move_fn is not None:
            self.mode = prev_mode
        return self.player.score
    
    



def get_model_path(algo: str) -> str:
    """Retourne le chemin du modèle selon l'algorithme."""
    paths = {
        "genetic": "models/genetics/best_genetic.pth",
        "dqn": "models/dqn/best_dqn.pt"
    }
    return paths.get(algo, "model.pth")

def load_ai_model(algo: str):
    """Charge le modèle IA selon l'algorithme."""

    path = get_model_path(algo)

    if algo == "genetic":
        return QbertMLP.load_model(get_model_path(algo))
    elif algo == "qlearning":
        agent = QTableAgent() 
        try:
            agent.load(get_model_path(algo))
        except FileNotFoundError:
            print("Attention : Aucun fichier de sauvegarde trouvé, l'IA part de zéro.")
        return agent

    elif algo == "dqn":
    
        model = create_model(68, 256, 4).to(DEVICE) 
        
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                print(f"Modèle DQN chargé : {path}")
            except Exception as e:
                print(f"Erreur de chargement du modèle : {e}")
                print("Vérifiez que la taille (5) correspond bien à celle de l'entraînement.")
        else:
            print(f"Pas de fichier trouvé à : {path}. L'IA jouera au hasard.")
            
        return model
    else:
        raise ValueError(f"Algorithme inconnu: {algo}")

def train_ai(algo: str, window=None, stop_event=None):
    """Lance l'entraînement selon l'algorithme."""
    if algo == "genetic":
        mlp_train(window=window, stop_event=stop_event)
    elif algo == "qlearning":
        train_qlearning_loop()
    elif algo == "dqn":
        print("Démarrage de l'entraînement DQN...")
        train(window=window, stop_event=stop_event)
    else:
        raise ValueError(f"Algorithme inconnu: {algo}")
