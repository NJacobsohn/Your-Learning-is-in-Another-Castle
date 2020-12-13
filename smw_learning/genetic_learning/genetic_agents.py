from smw_learning.genetic_learning.genes import BaseGene, RandomGene, ParallelGene


class BaseAgent(object):

    def __init__(self):
        self.genes = BaseGene()

    def update_genes(self, genes):
        self.genes.sequence = genes


class BlankAgent(BaseAgent):

    def __init__(self, gene_sequence=None):
        super().__init__()
        self.update_genes(gene_sequence)


class RandomAgent(BaseAgent):

    def __init__(self):
        self.genes = RandomGene()


class ParallelAgent(BaseAgent):

    def __init__(self):
        self.genes = ParallelGene()
