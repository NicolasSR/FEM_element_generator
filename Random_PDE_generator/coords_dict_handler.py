from seeded_random_generator import SeededRandomGenerator as Utilities

coords_dict = {
    "classic": ["x","y","z"],
    "var_1": ["e_{1}","e_{2}","e_{3}"],
    "var_2": ["e_{i}","e_{j}","e_{k}"]
}

class CoordsDictHandler():
    
    def __init__(self, coords_dict):
        self.chosen_system = Utilities.get_random_element(list(coords_dict.keys()))[0]
        self.coords_candidates = coords_dict[self.chosen_system]

    def reset(self):
        self.chosen_system = Utilities.get_random_element(list(coords_dict.keys()))[0]
        self.coords_candidates = coords_dict[self.chosen_system]
    
    def get_latex_coord(self, idx):
        return self.coords_candidates[idx]
