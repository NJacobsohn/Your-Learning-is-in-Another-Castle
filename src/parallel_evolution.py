import sys
import retro
import argparse
import operator
import numpy as np
from mutators import Mutator
from genomes import ParallelGenome
from genetic_agents import ParallelAgent
from algorithm_object_base import AlgorithmBase
from action_discretizer import MarioDiscretizer
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

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

class ParallelGeneticLearning(AlgorithmBase):
    """
    This might be the main class for running various genetic algorithms that I design

    Random Agents or NN Agents
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NUM_ENVS = 4
        self.envs = self._parallelize()
        self.NUM_AGENTS = 8
        self._check_math()
        self.PERC_AGENTS = .25
        self.NUM_GENOMES = 10
        self.best_agent = ParallelAgent()
        self.best_fitness = 0
        self.current_episode = 0
        self.episode_rewards = {}
        self.genomes = np.empty(shape=(self.NUM_GENOMES+1,), dtype=ParallelGenome)
        self._create_initial_genome()

    def _check_math(self):
        if self.NUM_AGENTS % self.NUM_ENVS != 0:
            sys.exit("Please choose a number of environments and agents that are divisible")

    def _make_vec_envs(self):
        def _f():
            env = retro.make(
            game=self.game,
            info=self.variables, 
            obs_type=self.observation_type,
            scenario=self.scenario,
            record=self.record_path)
            env.load_state(self.state)
            env = MarioDiscretizer(env)
            return env
        return _f

    def _parallelize(self):
        envs = [self._make_vec_envs() for _ in range(self.NUM_ENVS)]
        envs = SubprocVecEnv(envs)
        _ = envs.reset()
        return envs

    def _create_initial_genome(self):
        self.genomes[0] = ParallelGenome(size=self.NUM_AGENTS, envs=self.NUM_ENVS)
        self._update_active_genome(0)

    def _update_active_genome(self, idx):
        self.active_genome = self.genomes[idx]

    def _create_action_matrix(self, agent_pool):
        return np.stack([agent.genes.sequence for agent in agent_pool]).T

    def update_genome(self, old_genome, top_percent, genome_idx):
        """
        Updates a genepool from some top_percent of old_genepool
        This might need to fill the genepool with agents that are children (shared genes with two agents)
        """
        genetic_winners = old_genome.get_top(top_percent)
        top_agent = max(genetic_winners.items(), key=operator.itemgetter(1))[0]
        top_fitness = genetic_winners[top_agent]
        if top_fitness > self.best_fitness:
            self.best_agent = top_agent
            self.best_fitness = top_fitness
        genetic_winners[self.best_agent] = self.best_fitness
        new_genome = ParallelGenome(size=self.NUM_AGENTS, full=False, starting_agents=np.array(list(genetic_winners.keys())), envs=self.NUM_ENVS)
        self.genomes[genome_idx] = new_genome
        self._update_active_genome(genome_idx)
        
    def run(self):
        """
        Starts and optimizes agents on the environment
        Parallel runtime is ~13 seconds for 4 agents with 6k actions. (almost 7 seconds saved!)
        """
        for n in range(self.NUM_GENOMES):
            print("Genome #{0} Best Reward: {1}".format(n, self.best_fitness))
            for idx, agent_row in enumerate(self.active_genome.agents):
                _ = self.envs.reset()
                fitness = np.zeros(shape=(self.NUM_ENVS,))
                action_matrix = self._create_action_matrix(agent_row)
                for action_row in action_matrix:
                    _, rewards, _, _ = self.envs.step(action_row)
                    fitness += rewards
                self.active_genome.fitness_levels[idx] = fitness
                self.episode_rewards[self.current_episode] = fitness
                self.current_episode += self.NUM_ENVS
                print("Agents: {0}-{1} \nFitness: {2}\n".format(idx*self.NUM_ENVS, (idx*self.NUM_ENVS)+self.NUM_ENVS, fitness))
            self.update_genome(self.active_genome, self.PERC_AGENTS, n+1)
        for episode, reward in self.episode_rewards.items():
            if sum(reward >= 350) > 0:
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
    learning = ParallelGeneticLearning(args.project, args.game, args.state, args.scenario, args.observations, args.record, args.variables)
    learning.run()