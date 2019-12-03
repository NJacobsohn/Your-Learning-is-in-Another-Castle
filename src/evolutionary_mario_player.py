import sys
import retro
import argparse
import operator
import numpy as np
import keras.backend as K
from algorithm_object_base import AlgorithmBase
"""
Things to consider

- NN agents can either predict/save a set of actions and mutate/share those actions when making a new genome, or share weights and predict the actions actively
- The NN agents might need to be trained slightly before creating a new genome
- Maybe some algorithm for agents could be put into play where:
    It takes in a set of actions and the sum reward from those actions
    Predicts a new action set to get a higher reward

- Is this going to produce good mario players with a NN level or is it going to memorize levels?
- Is memorizing the level fine if it's computationally quick?

Assorted TO-DOs:

- Break script into agent/gene, mutation, genome scripts (might mean moving them to their own folder)
- Add multiprocessing functionality (maybe done through that retro wrapper + multiple defined environments?)
- Add more mutation functions
- Improve interactibility with parameters
- Save metrics (genome rewards, agent rewards, overall rewards, etc.)
    - try and use same format as the ppo_base.py metrics
- Clean/refactor code, good amounts of un-used lines + duplicate metrics being tracked
    - Part of this might involve splitting the script up a little
"""
class GeneticLearning(AlgorithmBase):
    """
    This might be the main class for running various genetic algorithms that I design

    Random Agents or NN Agents
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = self.make_env()
        self.NUM_AGENTS = 20
        self.PERC_AGENTS = 5
        self.NUM_GENOMES = 25
        self.BEST_AGENT = RandomAgent()
        self.BEST_FITNESS = 0
        self.CURRENT_GENOME_IDX = 0
        self.current_episode = 0
        self.EPISODE_REWARDS = {}
        self.genomes = np.empty(shape=(self.NUM_GENOMES+1,), dtype=Genome)
        self._create_initial_genome()

    def _create_initial_genome(self):
        self.genomes[0] = Genome(size=self.NUM_AGENTS)
        self._update_active_genome()

    def _update_active_genome(self):
        self.active_genome = self.genomes[self.CURRENT_GENOME_IDX]
        self.CURRENT_GENOME_IDX += 1

    def update_genome(self, old_genome, top_percent):
        """
        Updates a genepool from some top_percent of old_genepool
        This might need to fill the genepool with agents that are children (shared genes with two agents)
        """
        genetic_winners = old_genome.get_top(top_percent)
        top_agent = max(genetic_winners.items(), key=operator.itemgetter(1))[0]
        top_fitness = genetic_winners[top_agent]
        if top_fitness > self.BEST_FITNESS:
            self.BEST_AGENT = top_agent
            self.BEST_FITNESS = top_fitness
        genetic_winners[self.BEST_AGENT] = self.BEST_FITNESS
        new_genome = Genome(size=self.NUM_AGENTS, full=False, starting_agents=np.array(list(genetic_winners.keys())))
        self.genomes[self.CURRENT_GENOME_IDX] = new_genome
        self._update_active_genome()
        
    def run(self):
        """
        Starts and optimizes agents on the environment
        """
        for n in range(self.NUM_GENOMES):
            print("Genome #{0} Best Reward: {1}".format(n, self.BEST_FITNESS))
            for idx, agent in enumerate(self.active_genome.agents):
                fitness = 0
                done = False
                _ = self.env.reset()
                for action in agent.genes.sequence:
                    if done:
                        break
                    _, reward, done, _ = self.env.step(action)
                    fitness += reward
                self.active_genome.fitness_levels[idx] = fitness
                self.EPISODE_REWARDS[self.current_episode] = fitness
                self.current_episode += 1
                print("Agent: {0} \nFitness: {1}\n".format(idx, fitness))
            self.update_genome(self.active_genome, self.PERC_AGENTS)
        for episode, reward in self.EPISODE_REWARDS.items():
            if reward >= 350:
                print(episode)

class Mutator(object):
    """
    This will be a class for goofing around with child agents
    """
    def __init__(self, agent, num_children):
        """
        Takes in an agent (or two) to create children from.
        """
        self.MULTI_MUTATE = isinstance(agent, (list, np.ndarray))
        self.agent = agent
        self.NUM_CHILDREN = num_children
        self._compatibility_manifest = {
            "half_split" : True,
            "random_split" : True,
            "overlap_split" : True,
            "mutation_overlap_split" : True,
            "mutation_split" : False,
            "random_half_split" : False}

    def half_split(self, parent1, parent2):
        pass

    def random_split(self, parent1, parent2, chunk_size):
        pass

    def mutation_split(self, mutation_chance=0.15):
        total_actions = len(self.agent.genes)
        mutated_actions = np.random.rand(total_actions,) <= mutation_chance
        new_genes = np.where(mutated_actions, np.random.randint(0, 17), self.agent.genes.sequence)
        return BlankAgent(gene_sequence=new_genes)

    def overlap_split(self, parent1, parent2):
        pass

    def mutation_overlap_split(self, parent1, parent2, mutation_chance):
        pass
    
    def random_half_split(self, parent1, gene_slice):
        pass

    def _check_compatibility(self, algorithm):
        compatible = self.MULTI_MUTATE == self._compatibility_manifest[algorithm]
        if not compatible:
            sys.exit("The agent(s) {0}, are not compatible with the algorithm {1}.".format(self.agent, algorithm))
    
    def _get_alg(self, algorithm):
        alg_manifest = {
            "half_split" : self.half_split,
            "random_split" : self.random_split,
            "overlap_split" : self.overlap_split,
            "mutation_overlap_split" : self.mutation_overlap_split,
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
        self._check_compatibility(children_alg)
        mutate_alg = self._get_alg(children_alg)
        if not mutate_alg:
            sys.exit("No mutation algorithm of type: '{0}' found".format(children_alg))
        new_children = []
        for _ in range(self.NUM_CHILDREN):
            new_children.append(mutate_alg())
        return np.array(new_children)

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
        Create a random gene structure. These are actions to take
        """
        self.sequence = np.random.randint(self.NUM_ACTIONS, size=(self.length,))

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
        self.fitness = 0
        self.fitness_function = fitness_function

    def update_genes(self, genes):
        self.genes = genes

    def update_fitness(self, reward):
        if self.fitness_function:
            self.fitness += self.fitness_function(reward)
        else:
            self.fitness += reward

    def get_action(self):
        """
        This is a method for getting/predicting actions.
        """
        pass

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", default="my_project",
        help="""Define the name of your project here. 
                This will create a directory and save
                training records/models/weights to it.
                """)
    parser.add_argument("-g", '--game', default='SuperMarioWorld-Snes', help="This is the name of the game to learn on", type=str)
    parser.add_argument("-t", "--state", default="YoshiIsland2", help="If specified, pick the name of the state to train on", type=str)
    parser.add_argument("-s", '--scenario', default="scenarios/scenario.json", help="Try out a custom scenario", type=str)
    parser.add_argument("-o", "--observations", default=0, help="Either 0 or 1, 0 for screen observation, 1 for numerical observation", type=int)
    parser.add_argument("-r", "--record", default="learning_movies/", help="Choose a directory to record the training session to", type=str)
    parser.add_argument("-v", "--variables", default="variables/data.json", help="Path to reward variable json", type=str)
    args = parser.parse_args()
    learning = GeneticLearning(args.project, args.game, args.state, args.scenario, args.observations, args.record, args.variables)
    learning.run()