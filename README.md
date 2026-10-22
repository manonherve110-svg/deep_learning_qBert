# Projet Q*bert - Trinome 16

Implémentation du jeu d'arcade classique **Q*bert** en Python. Ce projet utilise une interface en ligne de commande (Curses) et intègre des agents intelligents (IA).

## 📋 Prérequis

*   **Python 3.10**
*   **Conda** (recommandé pour la gestion de l'environnement)
*   **Carte graphique NVIDIA** (optionnel, pour l'accélération GPU lors de l'entraînement)


## 🛠 Installation

1.  Créer l'environnement Conda (cela installera automatiquement PyTorch avec support CUDA si disponible) :
    ```bash
    conda env create -f environment.yml
    ```

2.  Activer l'environnement :
    ```bash
    conda activate qbert
    ```

## 🚀 Lancement

Vous pouvez lancer le jeu manuellement (peut nécessiter un redimensionnement du terminal) :
```bash
python src/Qbert.py [OPTIONS]
```

### Options disponibles

| Argument | Description | Par défaut |
| :--- | :--- | :--- |
| `--cli` | Lance le jeu en mode terminal (interface Curses). | Activé si aucun mode n'est spécifié |
| `--gui` | Lance le jeu en mode graphique. | - |
| `--ai` | Active le mode intelligence artificielle (le joueur est contrôlé par l'IA). Nécessite un modèle entraîné. | - |
| `--train` | Lance une session d'entraînement de l'IA. | - |
| `--algo` | Choisit l'algorithme d'IA : `genetic` (algorithme génétique) ou `deepq` (Q-Learning). | `genetic` |

### Exemples d'utilisation

#### Mode jeu classique

*   Jouer en mode graphique (par défaut) :
    ```bash
    python main.py
    ```
    ou en mode terminal :
    ```bash
    python main.py --cli
    ```

#### Mode IA

*   Regarder l'IA jouer en mode terminal (avec algorithme génétique par défaut) :
    ```bash
    python main.py --cli --ai
    ```

*   Regarder l'IA jouer avec un algorithme spécifique :
    ```bash
    python main.py --cli --ai --algo genetic
    python main.py --cli --ai --algo minmax
    python main.py --cli --ai --algo dqn
    ```

*   Regarder l'IA jouer en mode graphique :
    ```bash
    python main.py --gui --ai
    ```

#### Mode entraînement

*   Entraîner l'IA avec l'algorithme génétique (par défaut) :
    ```bash
    python main.py --train
    ```

*   Entraîner l'IA avec un algorithme spécifique :
    ```bash
    python main.py --train --algo genetic
    python main.py --train --algo dqn
    ```

**Note :** Le mode `--train` lance uniquement l'entraînement et ne démarre pas le jeu. Les modèles entraînés sont sauvegardés dans le dossier `models/`.

## 🎮 Commandes

Le jeu se joue avec les touches directionnelles ou **ZQSD** :

| Action | Touches (Azerty) | Flèches |
| :--- | :---: | :---: |
| Haut | **Z** | ↑ |
| Bas | **S** | ↓ |
| Gauche | **Q** | ← |
| Droite | **D** | → |
| Pause | **P** | - |

## 🤖 Fonctionnalités IA

Le projet inclut des fonctionnalités d'intelligence artificielle (voir `src/Bots.py` et `src/Player.py`) utilisant notamment :
*   **PyTorch**
*   **PyGAD** (Algorithmes génétiques)

## 📂 Structure

*   `src/Qbert.py` : Point d'entrée et boucle principale du jeu.
*   `src/Board.py` : Logique du plateau et des tuiles.
*   `src/Player.py` : Gestion du joueur (Humain et IA).
*   `src/Bots.py` : Comportement des ennemis (Coily, Ugg, etc.).
*   `src/CSTS.py` : Constantes globales du jeu.

