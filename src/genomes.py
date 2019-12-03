import numpy as np
from mutators import Mutator
from genetic_agents import RandomAgent

class Genome(object):
    """
    This is a group of agents, so each genepool should be better than the last (ideally)
    """
    def __init__(self, size, full=True, starting_agents=None):
        self.size = size
        self.BEST_AGENT = {}
        self.agents = np.empty(shape=(size,), dtype=RandomAgent)
        self.fitness_levels = np.empty(shape=(size,), dtype=float)
        if full:
            self._populate_genome()
        if starting_agents:
            self._initialize_genome_with_agents(starting_agents)
    
    def _initialize_genome_with_agents(self, starting_agents):
        """
        Initialize genome with array of chosen agents (doesn't have to be the size of the genome)
        """
        num_starting_agents = len(starting_agents)
        for idx, agent in enumerate(starting_agents):
            self.agents[idx] = agent
        self._fill_remainder(num_starting_agents)

    def _fill_remainder(self, num_starting_agents):
        """
        Fills genome when starting agents are given
        """
        # Use NUM_AGENTS and math to decide how many copies of each starting agent are made
        # Create appropriate number of children from each agent
        # Put agents starting at the num_starting_agents index
        num_children_per_agent = int((self.size - num_starting_agents) // num_starting_agents)
        current_idx = num_starting_agents
        for agent in self.agents[:num_starting_agents]:
            laboratory = Mutator(agent=agent, num_children=num_children_per_agent)
            children = laboratory.create_children(children_alg="mutation_split", children_alg_param=None)
            for child in children:
                self.agents[current_idx] = child
                current_idx += 1
        while current_idx < self.size: #this loop is to fill any gaps there might be due to the floor divison above
            agent_to_copy = np.random.randint(0, current_idx)
            laboratory = Mutator(agent=self.agents[agent_to_copy], num_children=1)
            child = laboratory.create_children(children_alg="mutation_split", children_alg_param=None)
            self.agents[current_idx] = child[0]
            current_idx += 1

    def _populate_genome(self):
        """
        Populates the genome with random agents
        """
        for i in range(self.size):
            self.agents[i] = RandomAgent()

    def get_top(self, top_percent):
        """
        top_percent is a float to pull the top percentage of a genome or int to pull the top x agents
        """
        if top_percent < 1:
            num_agents = int(self.size * top_percent)
        else:
            num_agents = int(top_percent)
        strong_indices = np.argsort(self.fitness_levels)[::-1]
        self.BEST_AGENT[self.agents[strong_indices[0]]] = self.fitness_levels[strong_indices[0]]
        agents_to_keep = self.agents[strong_indices[:num_agents]]
        top_agents_fitness = self.fitness_levels[strong_indices[:num_agents]]
        return dict(zip(agents_to_keep, top_agents_fitness))
        #returns dictionary of agents and their fitness for use in GeneticLearning class

if __name__ == "__main__":
    pass