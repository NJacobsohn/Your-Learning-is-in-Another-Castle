from genes import BaseGene, RandomGene

class BlankAgent(object):

    def __init__(self, gene_sequence=None):
        self.genes = BaseGene()
        self.update_genes(gene_sequence)

    def update_genes(self, genes):
        self.genes.sequence = genes

class RandomAgent(object):
    """
    The actual members of a genepool, each have their own genes associated with them
    """
    def __init__(self, fitness_function=None):
        """
        init string
        """
        self.genes = RandomGene()

    def update_genes(self, genes):
        self.genes = genes

if __name__ == "__main__":
    pass