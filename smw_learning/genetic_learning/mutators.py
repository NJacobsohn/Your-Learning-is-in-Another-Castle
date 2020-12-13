import sys
import numpy as np
from genetic_agents import BlankAgent

class Mutator(object):
    """
    This will be a class for goofing around with child agents
    """
    def __init__(self, agent, num_children):
        """
        Takes in an agent (or two) to create children from.
        """
        self.agent = agent
        self.NUM_CHILDREN = num_children
        self._compatibility_manifest = {
            "half_split" : True,
            "random_split" : True,
            "overlap_split" : True,
            "mutation_overlap_split" : True,
            "mutation_split" : False,
            "random_half_split" : False}

    def mutation_split(self, mutation_chance=0.15):
        total_actions = len(self.agent.genes)
        mutated_actions = np.random.rand(total_actions,) <= mutation_chance
        new_genes = np.where(mutated_actions, np.random.randint(0, 17), self.agent.genes.sequence)
        return BlankAgent(gene_sequence=new_genes)
    
    def random_half_split(self, parent1, gene_slice):
        pass
    
    def _get_alg(self, algorithm):
        alg_manifest = {
            "mutation_split" : self.mutation_split,
            "random_half_split" : self.random_half_split}
        return alg_manifest.get(algorithm, False)
        
    def create_children(self, children_alg="mutation_split", children_alg_param=None):
        """
        This will take agents from self.agents and create children based on some algorithm
        Algorithm options are:
        'half_split' = First half of genes from parent1 + second half of genes from parent2
        'random_split', 'x' = Randomly build the child's genes with samples of size 'x' from each parent
        'mutation', 'x' = Create a child who is a copy of a parent, with some mutation chance 'x' on the genes.
        'overlap' = Where the parent1 and parent2 share genes, those are passed down. Differences are chosen randomly
        'mutation_overlap' = overlap method with a passive mutation chance on each action
        'random_half', 'gene_slice' = Keep the genes of a parent defined by the gene_slice param and randomly generate the rest

        Things to think about:
            - When a mutation occurs, should the new action be picked randomly or be a different but similar action?
                i.e. a 'RIGHT' movement gets a mutation, should it be replaced randomly or replaced with either 'UP', 'DOWN', or 'LEFT'?
                Should certain movements even be replaced? Right movements are basically always good
        """
        mutate_alg = self._get_alg(children_alg)
        if not mutate_alg:
            sys.exit("No mutation algorithm of type: '{0}' found".format(children_alg))
        new_children = []
        for _ in range(self.NUM_CHILDREN):
            new_children.append(mutate_alg(children_alg_param))
        return np.array(new_children)


class OverlapMutator(object):

    def __init__(self, agent, num_children):
        """
        Takes in an agent (or two) to create children from.
        """
        self.agent1 = agent[0]
        self.agent2 = agent[1]
        self.NUM_CHILDREN = num_children

    def half_split(self, parent1, parent2):
        pass

    def random_split(self, parent1, parent2, chunk_size):
        pass

    def overlap_split(self, param=None): #this sucks
        parent1_genes = self.agent1.genes.sequence
        parent2_genes = self.agent2.genes.sequence
        new_genes = np.where(parent1_genes == parent2_genes, parent1_genes, np.random.randint(0, 17))
        return BlankAgent(gene_sequence=new_genes)

    def mutation_overlap_split(self, parent1, parent2, mutation_chance):
        pass

    def create_children(self, children_alg_param=None):
        """
        This will take agents from self.agents and create children based on some algorithm
        Algorithm options are:
        'half_split' = First half of genes from parent1 + second half of genes from parent2
        'random_split', 'x' = Randomly build the child's genes with samples of size 'x' from each parent
        'mutation', 'x' = Create a child who is a copy of a parent, with some mutation chance 'x' on the genes.
        'overlap' = Where the parent1 and parent2 share genes, those are passed down. Differences are chosen randomly
        'mutation_overlap' = overlap method with a passive mutation chance on each action
        'random_half', 'gene_slice' = Keep the genes of a parent defined by the gene_slice param and randomly generate the rest

        Things to think about:
            - When a mutation occurs, should the new action be picked randomly or be a different but similar action?
                i.e. a 'RIGHT' movement gets a mutation, should it be replaced randomly or replaced with either 'UP', 'DOWN', or 'LEFT'?
                Should certain movements even be replaced? Right movements are basically always good
        """
        new_children = []
        for _ in range(self.NUM_CHILDREN):
            new_children.append(self.overlap_split(children_alg_param))
        return np.array(new_children)

if __name__ == "__main__":
    pass