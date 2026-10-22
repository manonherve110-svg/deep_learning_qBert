import datetime
import time


def format_time(elapsed:float)->str:
    """
    Takes a time in seconds and returns a string hh:mm:ss
    :param elapsed: time as a float
    :return: time as a formatted string
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def report_progress(step:int, n_steps:int, t0:float, print_every:int=40, step_str:str="Step", eval_str:str="Eval", eval_val:float=None):
    # Progress update every `print_every` steps.
    if step % print_every == 0 and not step == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        # Report progress.
        progress_str = "  {:} {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step_str, step, n_steps, elapsed)
        if eval_val is not None:
            progress_str += " {:}: {:>5,}.".format(eval_str, eval_val)
        print(progress_str)


def get_timestamp()->str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_best_model_path(algo: str) -> str:
    """Retourne le chemin du meilleur modèle selon l'algorithme."""
    paths = {
        "genetic": "models/genetics/best_genetic.pth",
        "qlearning": "models/qlearning/best_qlearning.pth"
    }
    return paths.get(algo, "model.pth")

def load_ai_model(algo: str, model_path: str = None):
    """Charge le modèle IA selon l'algorithme.
    
    Args:
        algo: Type d'algorithme ('genetic', 'qlearning', ...)
        model_path: Chemin vers le modèle. Si None, utilise le chemin par défaut pour l'algo.
    
    Returns:
        Le modèle chargé
    """
    if model_path is None:
        model_path = get_best_model_path(algo)
    
    if algo == "genetic":
        from ..mlp import QbertMLP
        return QbertMLP.load_model(model_path)
    elif algo == "qlearning":
        # TODO: Implémenter le chargement pour Q-Learning
        raise NotImplementedError(f"Algorithme {algo} non implémenté")
    else:
        raise ValueError(f"Algorithme inconnu: {algo}")

def train_ai(algo: str):
    """Lance l'entraînement selon l'algorithme."""
    if algo == "genetic":
        from ..mlp import train as mlp_train
        mlp_train()
    elif algo == "qlearning":
        # TODO: Implémenter l'entraînement pour Q-Learning
        raise NotImplementedError(f"Entraînement pour {algo} non implémenté")
    else:
        raise ValueError(f"Algorithme inconnu: {algo}")
