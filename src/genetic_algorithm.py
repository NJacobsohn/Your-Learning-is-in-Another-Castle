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
    """
    This might be the main class for running various genetic algorithms that I design

    Random Agents or NN Agents
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def create_random_genepool(self, n):
        """
        Creates Genepool of size n comprised of random agents
        """
        pass

    def create_new_genepool(self, old_genepool, top_percent):
        """
        Creates new genepool from some top_percent of old_genepool
        This might need to fill the genepool with agents that are children (shared genes with two agents)
        """
        pass

    def run(self):
        """
        Starts and optimizes agents on the environment
        """
        pass

class Mutator(object):
    """
    This will be a class for goofing around with child agents
    """
    def __init__(self):
        pass

    def create_child(self):
        """
        This'll likely take 2 agents as an input (either to this method or to this class) and return a child agent from them
        """
        pass

class Genes(object):
    """
    This is just going to be a class to represent the actions/weights/etc. of an agent
    """
    def __init__(self):
        pass

class Agent(Genes):
    """
    The actual members of a genepool, each have their own genes associated with them
    """
    def __init__(self):
        pass
    
    def update_genes(self):
        """
        Create genes for the agent/Change the genetic structure (useful for mutations and whatnot)
        """
        pass

    def get_action(self):
        """
        Choose what action will be chosen, so random or possibly through a NN
        """
        pass

class Genepool(object):
    """
    This is a group of agents, so each genepool should be better than the last (ideally)
    """
    def __init__(self):
        pass
    
    def random_genes(self, n):
        """
        Probably a method to actually create a pool of random agents?

        This method might be better placed somewhere else
        """
        pass