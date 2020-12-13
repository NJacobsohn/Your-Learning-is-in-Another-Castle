import numpy as np
from math import factorial
from itertools import combinations
from smw_learning.genetic_learning.mutators import Mutator, OverlapMutator
from smw_learning.genetic_learning.genetic_agents import RandomAgent, ParallelAgent

class Genome(object):
    """
    This is a group of agents, so each genepool should be better than the last (ideally)
    """
    def __init__(self, size, full=True, starting_agents=None, multi_mutate=True):
        self.size = size
        self.BEST_AGENT = {}
        self.multi_mutate = multi_mutate
        self.agents = np.empty(shape=(self.size,), dtype=RandomAgent)
        self.fitness_levels = np.empty(shape=(self.size,), dtype=float)
        if full:
            self._populate_genome()
        if starting_agents is not None:
            self._initialize_genome_with_agents(starting_agents)

    def _permutations(self, pool_size, perm_size=2):
        return (factorial(pool_size)/factorial(pool_size - perm_size))
    
    def _initialize_genome_with_agents(self, starting_agents):
        """
        Initialize genome with array of chosen agents (doesn't have to be the size of the genome)
        """
        num_starting_agents = len(starting_agents)
        for idx, agent in enumerate(starting_agents):
            self.agents[idx] = agent
        if self.multi_mutate:
            self._fill_remainder_multiple_parent(num_starting_agents)
        else:
            self._fill_remainder_single_parent(num_starting_agents)

    def _fill_remainder_single_parent(self, num_starting_agents):
        """
        Fills genome when starting agents are given, creating children from 1 parent at a time
        """
        # Use NUM_AGENTS and math to decide how many copies of each starting agent are made
        # Create appropriate number of children from each agent
        # Put agents starting at the num_starting_agents index
        num_children_per_agent = int((self.size - num_starting_agents) // num_starting_agents)
        current_idx = num_starting_agents
        for agent in self.agents[:num_starting_agents]:
            laboratory = Mutator(agent=agent, num_children=num_children_per_agent)
            children = laboratory.create_children(children_alg="mutation_split", children_alg_param=0.25)
            for child in children:
                self.agents[current_idx] = child
                current_idx += 1
        while current_idx < self.size: #this loop is to fill any gaps there might be due to the floor divison above
            agent_to_copy = np.random.randint(0, current_idx)
            laboratory = Mutator(agent=self.agents[agent_to_copy], num_children=1)
            child = laboratory.create_children(children_alg="mutation_split", children_alg_param=0.25)
            self.agents[current_idx] = child[0]
            current_idx += 1

    def _fill_remainder_multiple_parent(self, num_starting_agents):
        """
        Fills genome when starting agents are given, creating children from 1 parent at a time
        """
        # Use NUM_AGENTS and math to decide how many copies of each starting agent are made
        # Create appropriate number of children from each agent
        # Put agents starting at the num_starting_agents index
        num_children_per_agent = int((self.size - num_starting_agents) // (self._permutations(num_starting_agents)/2))
        parent_pool = self.agents[:num_starting_agents]
        current_idx = num_starting_agents
        for agent1, agent2 in combinations(parent_pool, 2):
            laboratory = OverlapMutator(agent=np.array([agent1, agent2]), num_children=num_children_per_agent)
            children = laboratory.create_children(children_alg_param=None)
            for child in children:
                self.agents[current_idx] = child
                current_idx += 1
        while current_idx < self.size: #this loop is to fill any gaps there might be due to the floor divison above
            agents_to_copy = np.random.randint(0, current_idx, size=(2,))
            laboratory = OverlapMutator(agent=self.agents[agents_to_copy], num_children=1)
            child = laboratory.create_children(children_alg_param=None)
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
        self.agents = self.agents.reshape(self.size,)
        self.fitness_levels = self.fitness_levels.reshape(self.size,)
        if top_percent < 1:
            num_agents = int(self.size * top_percent)
        else:
            num_agents = int(top_percent)
        strong_indices = np.argsort(self.fitness_levels)[::-1]
        self.BEST_AGENT[self.agents[strong_indices[0]]] = self.fitness_levels[strong_indices[0]]
        agents_to_keep = self.agents[strong_indices[:num_agents]]
        top_agents_fitness = self.fitness_levels[strong_indices[:num_agents]]
        return dict(zip(agents_to_keep, top_agents_fitness))

class ParallelGenome(Genome):

    def __init__(self, size, full=True, starting_agents=None, multi_mutate=True, envs=1):
        self.NUM_ENVS = envs
        super().__init__(size, full, starting_agents, multi_mutate)
        self.agents = self.agents.reshape(-1, self.NUM_ENVS)
        self.fitness_levels = self.fitness_levels.reshape(-1, self.NUM_ENVS)

    def _populate_genome(self):
        """
        Populates the genome with random agents
        """
        for i in range(self.size):
            self.agents[i] = ParallelAgent()