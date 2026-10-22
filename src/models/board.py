
try :
    from src.config import *

except ModuleNotFoundError:
    from config import *


## ============== PLATEAU DE JEU ============== ##

class Board:
    
    """Représente le plateau de jeu (pyramide de cubes)."""
    
    board: list[list[int]]
    disque: dict[tuple[int,int],bool]
    size: int
    
    def __init__(self, size: int = 7) -> None:
        
        self.board = [[0 for i in range(j)] for j in range(1,size+1)]
        self.disque ={}
    

    def add_disque(self, side : str) ->None:
        """ajoute un disque a gauche"""

        if side == 'left':
            self.disque[(3,-1)] = True

        if side == 'right':
            self.disque[(3,4)] = True

    def is_disque(self, pos: tuple[int,int])->bool :
        if pos in self.disque:
            return True
        else:
            return False

    def remove_disque(self, pos:tuple[(int,int)]) ->None :
        if pos in self.disque:
            del self.disque[pos]




    ## ============== TO STRING ============== ##

    def __str__(self) -> str:
        """
        Retourne une représentation string du plateau sans le joueur.
        """
        lines = list[str]()

        for i in range(len(self)):

            left_str=" "
            has_left = self.is_disque((i,-1))
            if has_left:
                left_str ='@'

            center_part =  '[ '
            center_part +=  ' '.join(str(j) for j in self.board[i]) + ' ]'

            right_str=" "
            has_right= self.is_disque((i,i+1))
            if has_right:
                right_str ='@'

            line = left_str + center_part + right_str
            lines.append(line)
        
        for i in range(len(lines)):
            lines[i] = lines[i].center(TERMINAL_SIZE['WIDTH'])
        return '\n'.join(lines)

    
    ## ============== ACCÈS ÉLÉMENT ============== ##
    def __getitem__(self, key: int | tuple[int, int]) -> int | list[int]:
        """Accès aux éléments du plateau."""
        try:
            if isinstance(key, tuple):
                return self.board[key[0]][key[1]]
            elif isinstance(key, int):
                return self.board[key]
        except IndexError:
            raise IndexError("Argument de position non valide")
    
    ## ============== MODIF ÉLÉMENT ============== ##
    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """Modification d'un élément du plateau."""
        try:
            if isinstance(key, tuple):
                self.board[key[0]][key[1]] = value
            else:
                raise TypeError("L'index pour l'assignation doit être un tuple (row, col)")
        except IndexError:
            raise ValueError("Argument de position non valide")


    def __iter__(self):
        """Parcourt les lignes du plateau."""
        return iter(self.board)
    
    def __len__(self) -> int:
        """Nombre de lignes du plateau."""
        return len(self.board)
    
    ## ============== COULEUR D'UN CUBE À LA POISITION ============== ##
    def get_color(self, pos:tuple[int, int]) -> int:
        """Retourne la couleur actuelle d'un cube à la position pos."""
        try:
            return self.board[pos[0]][pos[1]]
        except IndexError:
            raise ValueError(f"Position {pos} is out of bounds")
    
    ## ============== DÉFINIT LA COULEUR À LA POISITION ============== ##
    def set_color(self, pos:tuple[int,int], color: int) -> None:
        """Définit la couleur d'un cube à la position (row, col)."""
        try:
            self.board[pos[0]][pos[1]] = color
        except IndexError:
            raise ValueError(f"Position {pos} is out of bounds")

    
    ## ============== RESET À 0 ============== ##
    def reset(self) -> None:
        """Réinitialise le plateau à son état initial."""
        self.board = [[0 for _ in range(j)] for j in range(1, len(self) + 1)]
        self.disque = dict[tuple[int,int],bool]()
        self.add_disque('left')
        self.add_disque('right')
