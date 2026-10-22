import arcade 
import threading

from src.core import Qbert
from src.core.game import load_ai_model, train_ai
from src.config import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, CUBE_WIDTH, CUBE_HEIGHT, START_Y, GAME_BACKGROUND_COLOR,LIFE_WIDTH, LIFE_HEIGHT

## ==============  GAME UI ============== ##
##
## -> Interface d'une partie de Qbert 
##
##
class QbertWindow (arcade.Window):

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
     
        self.game_engine = Qbert()

        self.texture_cube_1 = None
        self.texture_cube_2 = None
        self.texture_cube_3 = None

        self.cube_list = None
        self.disc_list = None
        self.player_list = None
        self.bot_list = None 
        self.rescuer_sprite = None
        
        self.player_sprite = None

        # --- Selection dans le menu --- #
        self.menu_selection = 0
        self.game_over_selection = 0
        self.in_ai_submenu = False
        self.in_train_load_submenu = False
        self.is_training = False
        self.stop_training_event = threading.Event()
        self.training_thread = None
        self.selected_algo = None
        self.training_info = ""
        self.training_history = []
        # --- Visualisation asynchrone pendant l'entraînement --- #
        self.demo_checkpoint = None  # Poids du meilleur modèle à démontrer
        self.demo_game = None        # Partie de démo en cours



        # --- Chargement du fond du menu --- #
        # (une seule fois au démarrage)
        self.menu_background = arcade.Sprite("assets/images/menu_background.png")
        self.menu_background.center_x = SCREEN_WIDTH / 2
        self.menu_background.center_y = SCREEN_HEIGHT / 2
        self.menu_background.width = SCREEN_WIDTH
        self.menu_background.height = SCREEN_HEIGHT

        self.menu_background_list = arcade.SpriteList()
        self.menu_background_list.append(self.menu_background)

        # --- Chargement du fond de la partie --- #
        # (une seule fois au démarrage)
        self.background_color = GAME_BACKGROUND_COLOR

        # --- Chargement de la police ---
        arcade.load_font("assets/fonts/pressStart2P.ttf") 
        self.font_name = "Press Start 2P" 
        
        self.init_ui_elements()

    def init_ui_elements(self):
        """Initialise les objets Text pour éviter de les recréer à chaque frame (PerformanceWarning)."""
        
        # --- MENU ---
        startYTop = SCREEN_HEIGHT * 0.625
        splitXHighScore = (SCREEN_WIDTH / 2) + 146 
        
        self.text_menu_high_score = arcade.Text(
            "HIGH SCORE: 12050 - ", splitXHighScore, startYTop,
            arcade.color.WHITE, font_size=16, anchor_x="right", font_name=self.font_name
        )
        self.text_menu_pseudo = arcade.Text(
            "Q*bert", splitXHighScore, startYTop,
            arcade.color.YELLOW, font_size=16, anchor_x="left", font_name=self.font_name
        )
        
        splitX = (SCREEN_WIDTH / 2) + 90 
        self.text_menu_top_ia = arcade.Text(
            "TOP IA: 9500 - ", splitX, startYTop - 35,
            arcade.color.WHITE, font_size=16, anchor_x="right", font_name=self.font_name
        )
        self.text_menu_minmax = arcade.Text(
            "MINMAX", splitX, startYTop - 35,
            arcade.color.CYAN, font_size=16, anchor_x="left", font_name=self.font_name
        )

        startYBot = SCREEN_HEIGHT * 0.415
        self.text_menu_opt0 = arcade.Text(
            "PLAY CLASSIC", SCREEN_WIDTH / 2, startYBot,
            arcade.color.WHITE, font_size=18, anchor_x="center", font_name=self.font_name
        )
        self.text_menu_opt1 = arcade.Text(
            "PLAY IA", SCREEN_WIDTH / 2, startYBot - 40,
            arcade.color.WHITE, font_size=18, anchor_x="center", font_name=self.font_name
        )
        self.text_menu_ia_details = arcade.Text(
            "GENETIC, DEEPQ, MINMAX", SCREEN_WIDTH / 2, startYBot - 60,
            arcade.color.LIGHT_GRAY, font_size=9, anchor_x="center", font_name=self.font_name
        )
        self.text_menu_opt2 = arcade.Text(
            "BYE BYE BYE", SCREEN_WIDTH / 2, startYBot - 105,
            arcade.color.WHITE, font_size=18, anchor_x="center", font_name=self.font_name
        )
        
        # --- AI SUBMENU ---
        self.text_ai_menu_opt0 = arcade.Text(
            "MINMAX", SCREEN_WIDTH / 2, startYBot ,
            arcade.color.WHITE, font_size=15, anchor_x="center", font_name=self.font_name
        )
        self.text_ai_menu_opt1 = arcade.Text(
            "GENETIC", SCREEN_WIDTH / 2, startYBot - 33,
            arcade.color.WHITE, font_size=15, anchor_x="center", font_name=self.font_name
        )
        self.text_ai_menu_opt2 = arcade.Text(
            "DEEPQ", SCREEN_WIDTH / 2, startYBot - 66,
            arcade.color.WHITE, font_size=15, anchor_x="center", font_name=self.font_name
        )
        self.text_ai_menu_opt3 = arcade.Text(
            "BACK", SCREEN_WIDTH / 2, startYBot - 100,
            arcade.color.WHITE, font_size=15, anchor_x="center", font_name=self.font_name
        )

        self.text_menu_error = arcade.Text(
            "", SCREEN_WIDTH / 2, startYBot - 140,
            arcade.color.RED, font_size=10, anchor_x="center", font_name=self.font_name
        )

        # --- TRAIN/LOAD SUBMENU ---
        self.text_tl_menu_opt0 = arcade.Text(
            "LOAD MODEL", SCREEN_WIDTH / 2, startYBot,
            arcade.color.WHITE, font_size=15, anchor_x="center", font_name=self.font_name
        )
        self.text_tl_menu_opt1 = arcade.Text(
            "TRAIN NEW", SCREEN_WIDTH / 2, startYBot - 33,
            arcade.color.WHITE, font_size=15, anchor_x="center", font_name=self.font_name
        )
        self.text_tl_menu_opt2 = arcade.Text(
            "BACK", SCREEN_WIDTH / 2, startYBot - 66,
            arcade.color.WHITE, font_size=15, anchor_x="center", font_name=self.font_name
        )

        # --- GAME OVER --- #
        self.text_game_over = arcade.Text(
            "GAME OVER", SCREEN_WIDTH/2, SCREEN_HEIGHT*0.7,
            arcade.color.RED, 50, anchor_x="center", font_name=self.font_name
        )
        self.text_score = arcade.Text(
            "FINAL SCORE - "+str(self.game_engine.player.score), SCREEN_WIDTH/2, SCREEN_HEIGHT*0.55,
            arcade.color.WHITE, 24, anchor_x="center", font_name=self.font_name
        )

         # --- PAUSE --- #
        self.text_pause = arcade.Text(
            "PAUSE", SCREEN_WIDTH/2, SCREEN_HEIGHT*0.7,
            arcade.color.RED, 50, anchor_x="center", font_name=self.font_name
        )
        self.text_current_score = arcade.Text(
            "CURRENT SCORE - "+str(self.game_engine.player.score), SCREEN_WIDTH/2, SCREEN_HEIGHT*0.55,
            arcade.color.WHITE, 24, anchor_x="center", font_name=self.font_name
        )
        

        # --- LIFE --- #
        splitXLife = SCREEN_WIDTH * 0.75 - 120
        splitYLife = SCREEN_HEIGHT * 0.94-60

        self.text_running_life = arcade.Text(
                "LIFE: ", splitXLife, splitYLife,
                arcade.color.WHITE, font_size=18, anchor_x="left", font_name=self.font_name
            )
        

        self.life1_alive = arcade.Sprite("assets/images/qbert_life.png")
        self.life1_alive.center_x = splitXLife + 145
        self.life1_alive.center_y = splitYLife + 17
        self.life1_alive.width = LIFE_WIDTH
        self.life1_alive.height = LIFE_HEIGHT

        self.life2_alive = arcade.Sprite("assets/images/qbert_life.png")
        self.life2_alive.center_x = splitXLife + 205
        self.life2_alive.center_y = splitYLife + 17
        self.life2_alive.width = LIFE_WIDTH
        self.life2_alive.height = LIFE_HEIGHT

        self.life3_alive = arcade.Sprite("assets/images/qbert_life.png")
        self.life3_alive.center_x = splitXLife + 265
        self.life3_alive.center_y = splitYLife + 17
        self.life3_alive.width = LIFE_WIDTH
        self.life3_alive.height = LIFE_HEIGHT

        self.life4_alive = arcade.Sprite("assets/images/qbert_life.png")
        self.life4_alive.center_x = splitXLife + 325
        self.life4_alive.center_y = splitYLife + 17
        self.life4_alive.width = LIFE_WIDTH
        self.life4_alive.height = LIFE_HEIGHT

        self.life5_alive = arcade.Sprite("assets/images/qbert_life.png")
        self.life5_alive.center_x = splitXLife + 385
        self.life5_alive.center_y = splitYLife + 17
        self.life5_alive.width = LIFE_WIDTH
        self.life5_alive.height = LIFE_HEIGHT

        self.life1_dead = arcade.Sprite("assets/images/qbert_dead.png")
        self.life1_dead.center_x = splitXLife + 145
        self.life1_dead.center_y = splitYLife + 17
        self.life1_dead.width = LIFE_WIDTH
        self.life1_dead.height = LIFE_HEIGHT

        self.life2_dead = arcade.Sprite("assets/images/qbert_dead.png")
        self.life2_dead.center_x = splitXLife + 205
        self.life2_dead.center_y = splitYLife + 17
        self.life2_dead.width = LIFE_WIDTH
        self.life2_dead.height = LIFE_HEIGHT

        self.life3_dead = arcade.Sprite("assets/images/qbert_dead.png")
        self.life3_dead.center_x = splitXLife + 265
        self.life3_dead.center_y = splitYLife + 17
        self.life3_dead.width = LIFE_WIDTH
        self.life3_dead.height = LIFE_HEIGHT

        self.life4_dead = arcade.Sprite("assets/images/qbert_dead.png")
        self.life4_dead.center_x = splitXLife + 325
        self.life4_dead.center_y = splitYLife + 17
        self.life4_dead.width = LIFE_WIDTH
        self.life4_dead.height = LIFE_HEIGHT

        self.life5_dead = arcade.Sprite("assets/images/qbert_dead.png")
        self.life5_dead.center_x = splitXLife + 385
        self.life5_dead.center_y = splitYLife + 17
        self.life5_dead.width = LIFE_WIDTH
        self.life5_dead.height = LIFE_HEIGHT

       

       
    
    def get_screen_coordinates(self, row, col):
        """ 
        Transforme (Ligne, Colonne) en (Pixel X, Pixel Y).
        """

        x = (SCREEN_WIDTH / 2) + (col * CUBE_WIDTH) - (row * CUBE_WIDTH / 2)
        y = START_Y - (row * CUBE_HEIGHT) 
        return x, y

    ## ==============  INITIALISER LE JEU ET LES SPRITES ============== ##
    def setup(self):
        """
        Configurer le jeu et positionner les sprites
        """
        
        self.cube_list = arcade.SpriteList()
        self.disc_list = arcade.SpriteList()
        self.player_list = arcade.SpriteList()
        self.bot_list = arcade.SpriteList()
        self.rescuer_list = arcade.SpriteList()

        # --- Chargement du sprite sauveteur ---
        self.rescuer_sprite = arcade.Sprite("assets/images/disque.png", scale=0.12)
        self.rescuer_list.append(self.rescuer_sprite)


        self.player_list = arcade.SpriteList()

        self.texture_cube_1 = arcade.load_texture("assets/images/cube1.png")
        self.texture_cube_2 = arcade.load_texture("assets/images/cube2.png")
        self.texture_cube_3 = arcade.load_texture("assets/images/cube3.png")


        for row_index, row_data in enumerate(self.game_engine.board):
            for col_index, value in enumerate(row_data):
                
                x , y = self.get_screen_coordinates(row_index ,col_index)
                cube_sprite = arcade.Sprite( self.texture_cube_1) 

                cube_sprite.center_x = x
                cube_sprite.center_y = y
                cube_sprite.width = 320
                cube_sprite.height = 150


                # Tu peux stocker les infos logiques dans le sprite si besoin
                cube_sprite.logic_pos = (row_index, col_index)
                
                # Ajouter à la liste
                self.cube_list.append(cube_sprite)

        self.player_sprite = arcade.Sprite("assets/images/personnage.png" , scale =0.12)
        self.player_list.append(self.player_sprite)

        for pos , ativate in self.game_engine.board.disque.items():
            if ativate:
                row, col = pos
                x, y = self.get_screen_coordinates(row,col)

                disc_sprite = arcade.Sprite("assets/images/disque.png", scale = 0.128)
                disc_sprite.center_x = x - 10
                disc_sprite.center_y = y 
                disc_sprite.logic_pos = pos
                self.disc_list.append(disc_sprite)

        

       
        
        self.sync_visuals()

    ## ==============  MISE À JOUR DONNÉES QBERT ============== ##
    def sync_visuals(self):
        """met a jour les sprites selon les données de Qbert"""


        ## joueur 
        player_row , player_col = self.game_engine.player.position
        x,y = self.get_screen_coordinates(player_row, player_col )

        if self.game_engine.transport_active:
            x -= 10

        self.player_sprite.center_x=x-4
        offset_y = 0
        if self.game_engine.player.is_moving:
            offset_y = 15

        self.player_sprite.center_y = y + CUBE_HEIGHT - 10 + offset_y

        # Gestion du clignotement pendant l'invincibilité
        if self.game_engine.invincibility_timer > 0:
            # Clignote toutes les 0.1 secondes (alpha 50 = transparent, 255 = opaque)
            self.player_sprite.alpha = 50 if (self.game_engine.invincibility_timer * 10) % 2 < 1 else 255
        else:
            self.player_sprite.alpha = 255

        # Mise à jour du sauveteur
        if self.game_engine.rescue_active and self.rescuer_sprite:
            self.rescuer_sprite.center_x = self.player_sprite.center_x
            self.rescuer_sprite.center_y = self.player_sprite.center_y + 50 # Au-dessus du joueur


        ## cube
        for cube in self.cube_list:
            r,c = cube.logic_pos

            logic_val = self.game_engine.board.get_color((r,c))
            if logic_val == 2:
                if cube.texture != self.texture_cube_3:
                    cube.texture = self.texture_cube_3
                    cube.width = 320
                    cube.height = 150
            elif logic_val == 1:
                if cube.texture != self.texture_cube_2:
                    cube.texture = self.texture_cube_2
                    cube.width = 320
                    cube.height = 150
            else:
                # État normal (0)
                if cube.texture != self.texture_cube_1:
                    cube.texture = self.texture_cube_1
                    cube.width = 320
                    cube.height = 150

        for disc in list(self.disc_list):
            pos = disc.logic_pos
            
            if pos not in self.game_engine.board.disque:
                if self.game_engine.transport_active:
                    disc.center_x = x
                    disc.center_y = y+20
                else:
                    disc.remove_from_sprite_lists()
            else:
                # Si le disque est présent dans la logique (ex: après reset), on force sa position
                # pour corriger le glitch visuel où il reste en haut.
                row, col = pos
                dx, dy = self.get_screen_coordinates(row, col)
                disc.center_x = dx - 10
                disc.center_y = dy

        # Ajout des disques manquants (ex: après un reset de niveau)
        sprite_positions = [d.logic_pos for d in self.disc_list]
        for pos, active in self.game_engine.board.disque.items():
            if active and pos not in sprite_positions:
                row, col = pos
                dx, dy = self.get_screen_coordinates(row, col)
                disc_sprite = arcade.Sprite("assets/images/disque.png", scale=0.128)
                disc_sprite.center_x = dx - 10
                disc_sprite.center_y = dy
                disc_sprite.logic_pos = pos
                self.disc_list.append(disc_sprite)

        # --- GESTION DES BOTS ---
        # On recrée la liste des sprites bots à chaque frame pour suivre la logique du jeu
        self.bot_list.clear()
        for bot in self.game_engine.bots:
            if bot.is_alive:
                r, c = bot.position
                x, y = self.get_screen_coordinates(r, c)
                
                sprite = None
                if bot.bot_type == "red_ball":
                    sprite = arcade.Sprite("assets/images/red_ball.png", scale=0.05)
                elif bot.bot_type == "coily":
                    if getattr(bot, "is_egg", False):
                        sprite = arcade.Sprite("assets/images/violet_ball.png", scale=0.129)
                    else:
                        sprite = arcade.Sprite("assets/images/snake.png", scale=0.06)
                elif bot.bot_type == "slick":
                    sprite = arcade.Sprite("assets/images/slick.png", scale=0.12)
                elif bot.bot_type == "sam":
                    sprite = arcade.Sprite("assets/images/sam.png", scale=0.12)
                elif bot.bot_type == "ugg":
                    sprite = arcade.Sprite("assets/images/ugg.png", scale=0.24)
                elif bot.bot_type == "wrong_way":
                    sprite = arcade.Sprite("assets/images/wrongway.png", scale=0.24)
                
                if sprite:
                    sprite.center_x = x
                    sprite.center_y = y + CUBE_HEIGHT - 45 # Ajustement pour poser la balle sur le cube
                    if bot.bot_type == "coily" and not getattr(bot, "is_egg", False): sprite.center_y += 20
                    if bot.bot_type == "sam": 
                        sprite.center_y += 8
                        sprite.center_x -= 30
                    if bot.bot_type == "slick" : 
                        sprite.center_y += 8
                        sprite.center_x += 30
                    
                    if bot.bot_type == "ugg":
                        sprite.center_x += 100
                        sprite.center_y -= 20
                    elif bot.bot_type == "wrong_way":
                        sprite.center_x -= 100
                        sprite.center_y -= 20

                    # Effet de saut 
                    if bot.is_moving and bot.start_position and bot.target_position and bot.bot_type not in ["ugg", "wrong_way"] and bot.position[0] > 0:
                        start_r, start_c = bot.start_position
                        target_r, target_c = bot.target_position
                        curr_r, curr_c = bot.position

                        total_dist = ((target_r - start_r)**2 + (target_c - start_c)**2)**0.5
                        curr_dist = ((curr_r - start_r)**2 + (curr_c - start_c)**2)**0.5

                        if total_dist > 0:
                            progress = curr_dist / total_dist
                            jump_height = 40
                            sprite.center_y += jump_height * 4 * progress * (1 - progress) 

                    self.bot_list.append(sprite)

    ## ==============  GESTION DES TOUCHES ============== ##
    def on_key_press(self, symbol, modifiers):
        """ 
        Gestion des touches 
        """

        if self.game_engine.state == self.game_engine.states['IN_MENU']:
            if self.in_train_load_submenu:
                limit = 3
            elif self.in_ai_submenu:
                limit = 4
            else:
                limit = 3

            if symbol == arcade.key.DOWN:
                self.menu_selection = (self.menu_selection + 1) % limit
                self.text_menu_error.text = ""
            elif symbol == arcade.key.UP:
                self.menu_selection = (self.menu_selection - 1) % limit
                self.text_menu_error.text = ""
            
            elif symbol == arcade.key.ENTER or symbol == arcade.key.SPACE:

                if self.in_train_load_submenu:
                    if self.menu_selection == 0: # LOAD MODEL
                        try:
                            if self.selected_algo is None:
                                self.text_menu_error.text = "Error: Aucun algo sélectionné"
                                return
                            model = load_ai_model(self.selected_algo)
                            self.game_engine = Qbert(player_mode='ai', model=model)
                            self.game_engine.start()
                            self.setup()
                            self.in_train_load_submenu = False
                            self.in_ai_submenu = False
                            self.menu_selection = 0
                        except Exception as e:
                            self.text_menu_error.text = f"Error: {e}"
                    elif self.menu_selection == 1: # TRAIN NEW
                        self.start_training()
                    elif self.menu_selection == 2: # BACK
                        self.in_train_load_submenu = False
                        self.in_ai_submenu = True
                        self.menu_selection = 0
                    return
                
                if not self.in_ai_submenu:
                    if self.menu_selection == 0:
                        if self.game_engine.mode != 'human':
                            self.game_engine = Qbert(player_mode='human')
                        self.game_engine.start()
                        self.setup()
                        self.menu_selection = 0
                        
                    elif self.menu_selection == 1:
                        self.in_ai_submenu = True
                        self.menu_selection = 0
                        
                    elif self.menu_selection == 2:
                        self.close()
                else:
                    if self.menu_selection == 0: # MINMAX
                        self.game_engine = Qbert(player_mode='minmax')
                        self.game_engine.start()
                        self.setup()
                        self.in_ai_submenu = False
                        self.menu_selection = 0
                        
                    elif self.menu_selection == 1: # GENETIC
                        self.selected_algo = "genetic"
                        self.in_train_load_submenu = True
                        self.in_ai_submenu = False
                        self.menu_selection = 0
                        
                    elif self.menu_selection == 2: # DEEPQ
                        self.selected_algo = "dqn"
                        self.in_train_load_submenu = True
                        self.in_ai_submenu = False
                        self.menu_selection = 0

                    elif self.menu_selection == 3: # BACK
                        self.in_ai_submenu = False
                        self.menu_selection = 0

                  
                
            return

        if self.game_engine.state == self.game_engine.states['PAUSE']:
            if symbol == arcade.key.DOWN:
                self.game_over_selection = (self.game_over_selection + 1) % 2 
            elif symbol == arcade.key.UP:
                self.game_over_selection = (self.game_over_selection - 1) % 2 
            
            elif symbol == arcade.key.ENTER or symbol == arcade.key.SPACE:
               
                if self.game_over_selection == 0:
                    self.game_engine.resume()
                    
                elif self.game_over_selection == 1:
                    self.game_engine.state = self.game_engine.states['IN_MENU']
                
                self.game_over_selection = 0
            return
        

        if self.game_engine.state == self.game_engine.states['GAME_OVER']:
            if symbol == arcade.key.DOWN:
                self.game_over_selection = (self.game_over_selection + 1) % 2 
            elif symbol == arcade.key.UP:
                self.game_over_selection = (self.game_over_selection - 1) % 2 
            
            elif symbol == arcade.key.ENTER or symbol == arcade.key.SPACE:
                if self.game_over_selection == 0:
                    self.game_engine.start()
                    self.setup()
                    
                elif self.game_over_selection == 1:
                    self.game_engine.state = self.game_engine.states['IN_MENU']
                
                self.game_over_selection = 0
            return
        if self.game_engine.state == self.game_engine.states['RUNNING']:
            if self.is_training:
                if symbol == arcade.key.ESCAPE:
                    self.stop_training_event.set()
                return

            if symbol == arcade.key.ESCAPE or symbol == arcade.key.P:
                    self.game_engine.pause()
            elif self.game_engine.mode == 'human':
                move = None
                if symbol == arcade.key.UP:
                    move = 'UP'
                elif symbol == arcade.key.DOWN:
                    move = 'DOWN'
                elif symbol == arcade.key.LEFT:
                    move = 'LEFT'
                elif symbol == arcade.key.RIGHT:
                    move = 'RIGHT'

                if move:
                    self.game_engine.player.play_move(move)

    def start_training(self):
        """Lance l'entraînement dans un thread séparé."""
        self.is_training = True
        self.stop_training_event.clear()
        self.in_train_load_submenu = False
        self.in_ai_submenu = False
        
        def training_task():
            try:
                # On passe la fenêtre et l'event d'arrêt pour la visualisation
                if self.selected_algo is None:
                    self.text_menu_error.text = "Error: Aucun algo sélectionné"
                    return
                train_ai(self.selected_algo, window=self, stop_event=self.stop_training_event)
            except Exception as e:
                print(f"Training error: {e}")
            finally:
                self.is_training = False
                self.stop_training_event.clear()
                # Retour au menu proprement (attention thread-safety, on fait au mieux pour l'exemple)
                self.game_engine = Qbert()
                self.game_engine.state = self.game_engine.states['IN_MENU']
                # On ne peut pas appeler setup() ici car OpenGL context est dans le main thread
                # La logique on_update s'en chargera
        
        self.training_thread = threading.Thread(target=training_task)
        self.training_thread.start()

    ## ==============  BOUCLE DU RENDU ============== ##
    def on_draw(self):
        """Boucle de rendu"""
        self.clear()
        
        # --- A. DESSIN DU JEU EN COURS --
        if (self.game_engine.state == self.game_engine.states['RUNNING'] 
            or  self.game_engine.state == self.game_engine.states['GAME_OVER']
            or  self.game_engine.state == self.game_engine.states['PAUSE']
         ) :
            
           

            # 1. Dessiner les élements non joubable (score, meilleur score, vie, niveau )
            # arcade.draw_text(f"Score: {self.game_engine.player.score}", 
            #                  10, SCREEN_HEIGHT - 30, arcade.color.WHITE, 20)
            # arcade.draw_text(f"lives: {self.game_engine.player.lives}", 
            #                  10, SCREEN_HEIGHT - 60, arcade.color.WHITE, 20)
             # --- RUNNING ---
            startYTopRun = SCREEN_HEIGHT * 0.94
            splitXHighScoreRun = SCREEN_WIDTH * 0.02

            text_running_high_score = arcade.Text(
                "HIGH SCORE: ", splitXHighScoreRun, startYTopRun,
                arcade.color.WHITE, font_size=18, anchor_x="left", font_name=self.font_name
            )
            
            current_high_score = self.game_engine.high_score
            if self.game_engine.mode != 'human':
                current_high_score = max(self.game_engine.high_score_genetic, self.game_engine.high_score_dqn, self.game_engine.high_score_minmax)

            text_running_high_score_number = arcade.Text(
                str(current_high_score), splitXHighScoreRun + 270, startYTopRun,
                arcade.color.GOLD, font_size=18, anchor_x="left", font_name=self.font_name
            )
            text_running_level = arcade.Text(
                "SCORE: ", splitXHighScoreRun, startYTopRun-40,
                arcade.color.WHITE, font_size=18, anchor_x="left", font_name=self.font_name
            )
            text_running_level_number = arcade.Text(
                str(self.game_engine.player.score), splitXHighScoreRun + 145, startYTopRun-40,
                arcade.color.GOLD, font_size=18, anchor_x="left", font_name=self.font_name
            )

            ## LEVEL AND LIFE 
            splitXHighScoreRunLevel = SCREEN_WIDTH * 0.83
            

            text_running_current_score = arcade.Text(
                "LEVEL: ", splitXHighScoreRunLevel, startYTopRun,
                arcade.color.WHITE, font_size=18, anchor_x="left", font_name=self.font_name
            )
            text_running_current_score_number = arcade.Text(
                str(self.game_engine.level), splitXHighScoreRunLevel + 145, startYTopRun,
                arcade.color.GOLD, font_size=18, anchor_x="left", font_name=self.font_name
            )

            life_background_list = arcade.SpriteList()

            match self.game_engine.player.lives:
                case 5:
                    life_background_list.append(self.life5_alive)
                    life_background_list.append(self.life4_alive)
                    life_background_list.append(self.life3_alive)
                    life_background_list.append(self.life2_alive)
                    life_background_list.append(self.life1_alive)
                case 4:
                    life_background_list.append(self.life5_dead)
                    life_background_list.append(self.life4_alive)
                    life_background_list.append(self.life3_alive)
                    life_background_list.append(self.life2_alive)
                    life_background_list.append(self.life1_alive)
                case 3:
                    life_background_list.append(self.life5_dead)
                    life_background_list.append(self.life4_dead)
                    life_background_list.append(self.life3_alive)
                    life_background_list.append(self.life2_alive)
                    life_background_list.append(self.life1_alive)
                case 2:
                    life_background_list.append(self.life2_alive)
                    life_background_list.append(self.life1_alive)
                    life_background_list.append(self.life3_dead)
                    life_background_list.append(self.life5_dead)
                    life_background_list.append(self.life4_dead)
                case 1:
                    life_background_list.append(self.life1_alive)
                    life_background_list.append(self.life2_dead)
                    life_background_list.append(self.life3_dead)
                    life_background_list.append(self.life5_dead)
                    life_background_list.append(self.life4_dead)
                case _:
                    life_background_list.append(self.life1_dead)
                    life_background_list.append(self.life2_dead)
                    life_background_list.append(self.life3_dead)
                    life_background_list.append(self.life5_dead)
                    life_background_list.append(self.life4_dead)
                    
            text_running_high_score.draw()
            text_running_high_score_number.draw()
            text_running_current_score.draw()
            text_running_current_score_number.draw()
            text_running_level.draw()
            text_running_level_number.draw()
            self.text_running_life.draw()
            life_background_list.draw()
            

            # 2. Dessiner les cubes, le joueur et les disques

            if self.cube_list:
                self.cube_list.draw()

            if self.disc_list:
                self.disc_list.draw()

            if self.bot_list:
                self.bot_list.draw()

            if self.player_list:
                self.player_list.draw()

            # Dessiner le sauveteur si actif
            if self.game_engine.rescue_active :
                self.rescuer_list.draw()

            # Afficher le Score

         
        # --- D. ECRAN D'ENTRAINEMENT ---
        if self.is_training:
            # On dessine quand même le jeu (qui est mis à jour par le thread d'entrainement)
            # Et on ajoute un overlay
            arcade.draw_text("TRAINING IN PROGRESS", SCREEN_WIDTH/2, SCREEN_HEIGHT - 50, arcade.color.RED, 20, anchor_x="center", font_name=self.font_name)
            arcade.draw_text("PRESS ESC TO STOP", SCREEN_WIDTH/2, SCREEN_HEIGHT - 80, arcade.color.WHITE, 15, anchor_x="center", font_name=self.font_name)

            if self.training_info:
                arcade.draw_text(
                    self.training_info, 
                    SCREEN_WIDTH - 20, 40, 
                    arcade.color.YELLOW, 
                    12, 
                    anchor_x="right", 
                    anchor_y="bottom", 
                    font_name=self.font_name
                )

            if self.selected_algo == "genetic" and len(self.training_history) >= 2:
                chart_w = 320
                chart_h = 120
                chart_x = SCREEN_WIDTH - chart_w - 20
                chart_y = 80
                x_left = chart_x
                x_right = chart_x + chart_w
                y_bottom = chart_y
                y_top = chart_y + chart_h
                arcade.draw_line(x_left, y_bottom, x_right, y_bottom, arcade.color.YELLOW, 1)
                arcade.draw_line(x_left, y_top, x_right, y_top, arcade.color.YELLOW, 1)
                arcade.draw_line(x_left, y_bottom, x_left, y_top, arcade.color.YELLOW, 1)
                arcade.draw_line(x_right, y_bottom, x_right, y_top, arcade.color.YELLOW, 1)

                history = self.training_history[-200:]
                min_fit = min(history)
                max_fit = max(history)
                span = max(1.0, max_fit - min_fit)

                for i in range(1, len(history)):
                    x0 = chart_x + (i - 1) * (chart_w / (len(history) - 1))
                    x1 = chart_x + i * (chart_w / (len(history) - 1))
                    y0 = chart_y + ((history[i - 1] - min_fit) / span) * chart_h
                    y1 = chart_y + ((history[i] - min_fit) / span) * chart_h
                    arcade.draw_line(x0, y0, x1, y1, arcade.color.YELLOW, 2)


        # --- B. DESSIN DU MENU ---
        elif self.game_engine.state == self.game_engine.states['IN_MENU']:

            self.menu_background_list.draw()

            # --- BOITE DU HAUT (Scores) ---            
            self.text_menu_high_score.text = f"HIGH SCORE: {self.game_engine.high_score} - "
            self.text_menu_high_score.draw()
            self.text_menu_pseudo.draw()
            
            # Détermine le meilleur score IA à afficher
            scores = [
                ("GENETIC", self.game_engine.high_score_genetic),
                ("DEEPQ", self.game_engine.high_score_dqn),
                ("MINMAX", self.game_engine.high_score_minmax)
            ]
            best_name, best_score = max(scores, key=lambda x: x[1])

            self.text_menu_top_ia.text = f"TOP IA: {best_score} - "
            self.text_menu_minmax.text = best_name
            self.text_menu_top_ia.draw()
            self.text_menu_minmax.draw()

            # --- BOITE DU BAS (Menu) ---            

            if self.in_train_load_submenu:
                # Option 1 : LOAD
                self.text_tl_menu_opt0.text = "> LOAD MODEL" if self.menu_selection == 0 else "LOAD MODEL"
                self.text_tl_menu_opt0.color = arcade.color.YELLOW if self.menu_selection == 0 else arcade.color.WHITE
                self.text_tl_menu_opt0.draw()

                # Option 2 : TRAIN
                self.text_tl_menu_opt1.text = "> TRAIN NEW" if self.menu_selection == 1 else "TRAIN NEW"
                self.text_tl_menu_opt1.color = arcade.color.YELLOW if self.menu_selection == 1 else arcade.color.WHITE
                self.text_tl_menu_opt1.draw()

                # Option 3 : BACK
                self.text_tl_menu_opt2.text = "> BACK" if self.menu_selection == 2 else "BACK"
                self.text_tl_menu_opt2.color = arcade.color.YELLOW if self.menu_selection == 2 else arcade.color.WHITE
                self.text_tl_menu_opt2.draw()

            elif not self.in_ai_submenu:
                # Option 1 : Classique
                self.text_menu_opt0.text = "> PLAY CLASSIC" if self.menu_selection == 0 else "PLAY CLASSIC"
                self.text_menu_opt0.color = arcade.color.YELLOW if self.menu_selection == 0 else arcade.color.WHITE
                self.text_menu_opt0.draw()

                # Option 2 : AI
                self.text_menu_opt1.text = "> PLAY IA" if self.menu_selection == 1 else "PLAY IA"
                self.text_menu_opt1.color = arcade.color.YELLOW if self.menu_selection == 1 else arcade.color.WHITE
                self.text_menu_opt1.draw()
                
                # Détail des IAs (plus petit)
                self.text_menu_ia_details.draw()
                
                # Option 3 : QUITTER
                self.text_menu_opt2.text = "> BYE BYE BYE" if self.menu_selection == 2 else "BYE BYE BYE"
                self.text_menu_opt2.color = arcade.color.YELLOW if self.menu_selection == 2 else arcade.color.WHITE
                self.text_menu_opt2.draw()
            else:
                # Option 1 : MINMAX
                self.text_ai_menu_opt0.text = "> MINMAX" if self.menu_selection == 0 else "MINMAX"
                self.text_ai_menu_opt0.color = arcade.color.YELLOW if self.menu_selection == 0 else arcade.color.WHITE
                self.text_ai_menu_opt0.draw()

                # Option 2 : GENETIC
                self.text_ai_menu_opt1.text = "> GENETIC" if self.menu_selection == 1 else "GENETIC"
                self.text_ai_menu_opt1.color = arcade.color.YELLOW if self.menu_selection == 1 else arcade.color.WHITE
                self.text_ai_menu_opt1.draw()


                # Option 3 : DEEPQ
                self.text_ai_menu_opt2.text = "> DEEPQ" if self.menu_selection == 2 else "DEEPQ"
                self.text_ai_menu_opt2.color = arcade.color.YELLOW if self.menu_selection == 2 else arcade.color.WHITE
                self.text_ai_menu_opt2.draw()

                # Option 4 : QLEARNING
                self.text_ai_menu_opt3.text = "> BACK" if self.menu_selection == 3 else "BACK"
                self.text_ai_menu_opt3.color = arcade.color.YELLOW if self.menu_selection == 3 else arcade.color.WHITE
                self.text_ai_menu_opt3.draw()

              

            
            self.text_menu_error.draw()

        # --- C. DESSIN GAME OVER ---
        if self.game_engine.state == self.game_engine.states['GAME_OVER']:
           
            # Dessiner un rectangle noir semi-transparent par-dessus
            arcade.draw_polygon_filled([(0, 0),
                                        (SCREEN_WIDTH, 0),
                                        (SCREEN_WIDTH, SCREEN_HEIGHT),
                                        (0, SCREEN_HEIGHT)],
                                       (0, 0, 0, 150))

            text_restart = arcade.Text(
                "> RESTART ?" if self.game_over_selection == 0 else "RESTART ?", SCREEN_WIDTH/2, SCREEN_HEIGHT*0.4,
                arcade.color.GOLD if self.game_over_selection == 0 else arcade.color.GRAY, 24, anchor_x="center", font_name=self.font_name
            ) 

            text_exit = arcade.Text(
                "> EXIT" if self.game_over_selection == 1 else "EXIT", SCREEN_WIDTH/2, SCREEN_HEIGHT*0.3,
                arcade.color.GOLD if self.game_over_selection == 1 else arcade.color.GRAY, 24, anchor_x="center", font_name=self.font_name
            )

            self.text_game_over.draw()
            self.text_score.text = "FINAL SCORE - " + str(self.game_engine.player.score)
            self.text_score.draw()
            text_restart.draw()
            text_exit.draw()

        if self.game_engine.state == self.game_engine.states['PAUSE']:
           
            # Dessiner un rectangle noir semi-transparent par-dessus
            arcade.draw_polygon_filled([(0, 0),
                                        (SCREEN_WIDTH, 0),
                                        (SCREEN_WIDTH, SCREEN_HEIGHT),
                                        (0, SCREEN_HEIGHT)],
                                       (0, 0, 0, 150))

            text_restart = arcade.Text(
                "> CONTINUES ?" if self.game_over_selection == 0 else "CONTINUES ?", SCREEN_WIDTH/2, SCREEN_HEIGHT*0.4,
                arcade.color.GOLD if self.game_over_selection == 0 else arcade.color.GRAY, 24, anchor_x="center", font_name=self.font_name
            ) 

            text_exit = arcade.Text(
                "> EXIT" if self.game_over_selection == 1 else "EXIT", SCREEN_WIDTH/2, SCREEN_HEIGHT*0.3,
                arcade.color.GOLD if self.game_over_selection == 1 else arcade.color.GRAY, 24, anchor_x="center", font_name=self.font_name
            )

            self.text_pause.draw()
            self.text_current_score.text =  "CURRENT SCORE - "+str(self.game_engine.player.score)
            self.text_current_score.draw()
            text_restart.draw()
            text_exit.draw()

    def on_update(self, delta_time):
        """Boucle logique"""
        if self.is_training:
            # Si le thread est fini mais qu'on est encore marqué en training (transition)
            if self.training_thread and not self.training_thread.is_alive():
                self.is_training = False
                self.demo_game = None
                self.demo_checkpoint = None
                self.game_engine = Qbert()
                self.setup()
                return
            
            # Vérifie si un nouveau checkpoint est disponible pour démarrer une démo
            if self.demo_checkpoint:
                try:
                    from src.ia.mlp import QbertMLP
                    # Crée un nouveau modèle et charge les poids du checkpoint
                    demo_model = QbertMLP()
                    demo_model.load_state_dict(self.demo_checkpoint['weights'])
                    demo_model.eval()
                    
                    # Crée une partie de démo
                    self.demo_game = Qbert(player_mode='ai', model=demo_model)
                    self.game_engine = self.demo_game
                    self.demo_game.start()
                    self.setup()
                    
                    # Met à jour l'info de training avec la génération de la démo
                    gen = self.demo_checkpoint['generation']
                    fit = self.demo_checkpoint['fitness']
                    self.training_info = f"Gen: {gen} | Best Fit: {fit:.0f} | DEMO"
                    
                    self.demo_checkpoint = None  # Checkpoint consommé
                except Exception as e:
                    print(f"Erreur création démo: {e}")
                    self.demo_checkpoint = None
            
            # Si une démo est en cours, la faire jouer
            if self.demo_game:
                if self.demo_game.state == self.demo_game.states['RUNNING']:
                    self.demo_game.update(delta_time)
                    self.sync_visuals()
                else:
                    # Démo terminée (game over ou win), relance une nouvelle partie avec le même modèle
                    current_model = getattr(self.demo_game.player, 'model', None)
                    self.demo_game = Qbert(player_mode='ai', model=current_model)
                    self.game_engine = self.demo_game
                    self.demo_game.start()
                    self.setup()
            return


        if self.game_engine.state == self.game_engine.states['RUNNING']:
            self.game_engine.update(delta_time)
            self.sync_visuals()
