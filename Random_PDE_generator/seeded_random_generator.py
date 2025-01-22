import numpy as np

class SeededRandomGenerator:
    # Class-level attribute to store the seed
    _seed = None
    _rand_generator = None

    @classmethod
    def set_seed(cls, seed):
        """Set the class-level seed."""
        cls._seed = seed
        cls._rand_generator = np.random.default_rng(seed)  # Set the seed for the random module

    @classmethod
    def get_random_element(cls, list, ):
        i = cls._rand_generator.integers(0,len(list),1)[0]
        return list[i], i

    @classmethod
    def shuffle_elements(cls, list):
        shuffled_list = list.copy()
        cls._rand_generator.shuffle(shuffled_list)
        return shuffled_list
    
    @classmethod
    def get_chance(cls, probability):
        if cls._rand_generator.random(1)[0] <= probability:
            return True
        else:
            return False
    
    @classmethod
    def get_random_choice(cls, prob_list):
        choice = cls._rand_generator.choice(list(range(len(prob_list))), 1, p=prob_list)[0]
        return choice