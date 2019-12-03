import numpy as np

class BaseGene(object):
    """
    This is just going to be a class to represent the actions/weights/etc. of an agent
    """
    def __init__(self):
        self.NUM_ACTIONS = 17
        self.sequence = np.array([0]) #instantiate something, this should be reassigned

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return str(self.sequence)

    def __getitem__(self, key):
        try:
            return self.sequence[key]
        except KeyError:
            raise KeyError
        except IndexError:
            raise IndexError
        except TypeError:
            raise TypeError
        except Exception:
            raise
    
    def __setitem__(self, key, value):
        try:
            self.sequence[key] = value
        except KeyError:
            raise KeyError
        except IndexError:
            raise IndexError
        except TypeError:
            raise TypeError
        except Exception:
            raise

    def __iter__(self):
        return iter(self.sequence)

class RandomGene(BaseGene):
    """
    This is a class to represent the actions of a random agent
    """
    def __init__(self):
        super().__init__()
        self.length = np.random.randint(low=6000, high=6500)
        self._generate_actions()
    
    def _generate_actions(self):
        """
        Create a random gene structure. These are actions
        """
        self.sequence = np.random.randint(self.NUM_ACTIONS, size=(self.length,))

class ParallelGene(BaseGene):

    def __init__(self):
        super().__init__()
        self.length = 6000
        self._generate_actions()
    
    def _generate_actions(self):
        """
        Create a random gene structure. These are actions
        """
        self.sequence = np.random.randint(self.NUM_ACTIONS, size=(self.length,))