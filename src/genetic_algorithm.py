import numpy as np
from algorithm_object_base import AlgorithmBase
"""
This might end up as a script to run a genetic algorithm on SMW as opposed to typical gradient approaches

pseudo-code:

agent = organism
fitness = total rewards
genes = policy for getting rewards

initialize n random agents
agents each do 3 or 5 random runs, their final fitness is the average of the runs
keep top 10% agents (scored on fitness)
kill off bottom 90%
create copies of the 10% agents until you have n agents again
each new agent has some mutation chance to have their policy (genes) rewritten
keep an unedited copy of the best agent
"""
np.random.seed(seed=42)

class GeneticLearning(AlgorithmBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

class Mutator(object):
    def __init__(self):
        pass

class Genes(object):
    def __init__(self):
        pass

class Organism(Genes):
    def __init__(self):
        pass

class Generation(object):
    def __init__(self):
        pass