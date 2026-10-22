import argparse
import curses
import arcade
from src.core.game import Qbert, load_ai_model, train_ai
from src.ui.game_ui import QbertWindow


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Q*bert Clone")
    parser.add_argument("--cli", action="store_true", help="Lancer en ligne de commande")
    parser.add_argument("--gui", action="store_true", help="Lancer en mode graphique")
    parser.add_argument("--ai", action="store_true", help="Lance une partie avec en mode ia")
    parser.add_argument("--train", action="store_true", help="Lance une session d'entrainement de l'ia")
    parser.add_argument("--algo", choices=["genetic", "dqn", "minmax"], default="genetic", help="Choisir l'algorithme de l'ia")
    parser.add_argument("--model", type=str, default=None, help="Chemin vers le modèle à utiliser (mode démo uniquement, défaut: models/XXX/best_XXX.pth selon l'algo)")
    args = parser.parse_args()

    
    # Si aucun argument n'est fourni, on lance en CLI/Human par défaut
    if not args.cli and not args.gui:
        args.gui = True
    
    try:
        # Mode training
        if args.train:
            train_ai(args.algo)
        # Mode démo
        elif args.cli:
            if args.ai:
                if args.algo == 'minmax':
                    qbert = Qbert(player_mode='minmax')
                else:
                    model = load_ai_model(args.algo)
                    qbert = Qbert(player_mode='ai', model=model)
                curses.wrapper(qbert.main_cli)
            else:
                qbert = Qbert()
                curses.wrapper(qbert.main_cli)
        elif args.gui:
            window = QbertWindow()
            window.setup()
            arcade.run()
            
    except KeyboardInterrupt:
        print("\nAu revoir !")
    except Exception as e:
        print(f"Erreur: {e}")
        exit(1)
