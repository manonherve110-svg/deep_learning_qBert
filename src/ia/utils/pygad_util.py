import time
import numpy as np
import pygad


def callback_generation_b(ga_instance:pygad.GA, n_generations:int, t0:float, print_every:int):
    """Callback appelé à chaque génération pour afficher la progression."""
    gen = ga_instance.generations_completed
    if gen % print_every != 0:
        return
    
    # Stats fitness
    # ga_instance.best_solution() retourne (solution, fitness, index)
    # On ne garde que la fitness ici car c'est ce qu'on veut afficher
    best_fitness = ga_instance.best_solution()[1]
    mean_fitness = np.mean(ga_instance.last_generation_fitness)
    
    # Temps
    elapsed = time.time() - t0
    avg_per_gen = elapsed / gen if gen > 0 else 0
    remaining = avg_per_gen * (n_generations - gen)
    
    # Barre de progression
    progress = gen / n_generations
    bar_len = 20
    filled = int(bar_len * progress)
    bar = '█' * filled + '░' * (bar_len - filled)
    
    print(f"[{bar}] {gen}/{n_generations} | Best fitness: {best_fitness:.0f} | Avg: {mean_fitness:.0f} | ETA: {remaining:.0f}s")

