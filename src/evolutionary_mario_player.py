import argparse
import operator
import numpy as np
from genomes import Genome
from mutators import Mutator
from genetic_agents import RandomAgent
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
        self.NUM_AGENTS = 8
        self.PERC_AGENTS = 2
        self.NUM_GENOMES = 10
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
        Takes ~4.8ish seconds per agent with 6000 actions
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