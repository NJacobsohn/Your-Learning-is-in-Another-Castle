# Your Learning is in Another Castle

To jump to seeing a model in action, [click here.](#the-numerical-model)

## Table of Contents

1. [Overview](#overview)
2. [Questions](#questions)
3. [Environment](#environment)
4. [Environment Setup](#environment-setup)
5. [Model Decisions](#model-decisions)
6. [Genetic Learning](#genetic-learning)
7. [PPO](#ppo)
    - [Image vs. Numerical Models](#image-vs.-numerical-models)
        - [The Numerical Model](#the-numerical-model)
        - [The Image Model](#the-image-model)
8. [Conclusion](#conclusion)
9. [What's Next?](#what's-next?)
10. [References and Credits](#references-and-credits)

## **Overview**

Videogames have always held a near and dear place in my heart. Whether it's something relaxing like running a farming village or something more intense like a turn-based strategy game where you build an empire and conquer the world, they're all special to me. One game series that holds a very special place in my heart is the Mario franchise, but more specifically within it, Super Mario World. I have so many childhood memories playing this game on my Game Boy on long car rides and sneaking it into school to play during recess. But the best memories have to be all the times I finally beat some level that I was stuck on for days. Which led me to wonder, just how much better could my computer be than me at playing Mario? This project aims to create a phenomenal AI Mario player through reinforcement learning.

## **Questions**

- Can I teach an AI to be actually good at Mario? Not just memorizing the level it's on, a genuinely strong player.

- Do skills learned from seeing the screen differ from not seeing it?

- Do skills translate across multiple levels, worlds, even games?

- Can a model be trained to essentially recreate a TAS (Tool-assisted speedrun)?

## **Environment**

I used a addon for gym called retrogym which has support for quite a few older emulators as the base for my training environment. Retrogym has a lot of built in integrations for game ROMS, Super Mario World included. Unfortunately, their integration for it was very barebones and not at all ideal for what I wanted to do. The only variables you could use a rewards were your score, coins (which affects score anyways, so really just one is needed), and lives. I wanted so much more, as I was hoping to teach my model to beat a level in one try (any score, coins, bonus lives, etc. should just be a bonus).

## **Environment Setup**

In order to setup my custom environment, I had quite a few things to work out. First, how would I actually give incremental rewards to push the model to try and beat a level? Second, how could I define if a level was beaten? Third, how do I stop the model from doing environment-breaking actions like pausing the game, accidentally quitting, etc.? To tackle the first problem, I had to look into if there was a variable in the game that tracks the player's position in a level. To find this and any other variables, I had to follow memory maps (locations in RAM on where these variables are set to save to) and then actually pull those values from the emulator using retrogym's handy game integration tool. I needed the location in memory in base 16 (e.g. 7E13CE for if the midway flag was crossed), which then is converted to its base 10 value automatically by the integration tool (for the earlier example, it's now 8262606) to be used by the python environment. Another important part of these variables which had to be specified was their endianness, how many bytes they were, and whether or not it was a signed integer. These values can be found in [this .json file I built.](https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/variables/data.json)

After much searching and sifting, I finally found what I think to be a pretty good set of variables to work with. Each variable in the above link had to also be defined in the training scenario for the networks. The [scenario.json file](https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/scenarios/scenario.json) I built uses the variables I defined in the data.json to define the training environment. The training session is considered done if Mario loses a life or completes the level. The primary reward is distance traveled on the x axis within a level, with considerably large rewards given to reaching the midway checkpoint and completing the whole level. This was done to ensure that the best way to achieve the highest possible score involved beating the level, not using shenanigans like leaving and entering an area to spawn more enemies.

There were many ways I tried to define if a level was beaten, as there's no 0/1 value that just means "Finish line crossed y/n". At first I used a countdown to when your final score is calculated after beating a level, but I found that to be very unreliable. I also tried using a value which has Mario's status in the overworld (level selection screen), but that wouldn't apply until a level was beaten and Mario had returned to the level selection screen. This became an issue as sometimes the network would very quickly be able to enter a new level before the done condition would trigger, then it wouldn't ever trigger due to being inside of a level again. The final decision was a conditional variable that represents how the curren level was exited. This works not only for quickly telling the network it's done when the final calculations were complete, but also properly rewards the network for beating a level.

So we've got all the variables defined and all their locations in memory have been tracked down, now it's time to limit the action space of the environment to prevent the aforementioned idiot actions from occurring. I decided on the following set of 17 actions for the network to be allowed to use:

The following actions have a regular version, and a sprinting version (12 total actions):

- Move Left
- Move Right
- Regular Jump Left
- Regular Jump Right
- Spin Jump Left
- Spin Jump Right

The following actions only exist in these explicitly defined forms (5 actions):

- Regular Jump in place
- Spin Jump in place (allows for bouncing on certain enemies, I wanted to keep it in to see if it'd learn more advanced tactics)
- Look Up
- Look Down (necessary for entering pipes)
- Jump in place and Look Up (necessary for entering pipes on the ceiling)

These actions are defined in a custom wrapper for the environment, which can be explored in my [action discretizer script.](https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/src/action_discretizer.py)

## **Model Decisions**

When choosing what type of model I wanted to use, many thoughts went through my head. I knew for sure I wanted to try using some sort of multilayer perceptron with an optimization algorithm, so that was my first path that I went down. After some research on popular optimization methods, I settled on trying out PPO as it seemed relatively easy to implement and (according to the paper introducing it) was pretty computationally efficient. Computational efficiency is very important to me as I like to do as much tuning and training locally as I can. But training even a small CNN on the screen of the game isn't an easy feat, despite having good results on many levels. This model and optimization worked well for most levels that I tried, sometimes beating levels in as little as 13 episodes. The main levels that caused issues with this algorithm though was the water levels. These levels required entirely different control schemes and strategies that aren't easily apprent through just looking at the screen of the game. The CNN struggled greatly making any progress, which led me to research potential alternatives such as genetic learning.

## **PPO**

In order to properly optimize how these models learn, I adapted a popular reinforcement learning algorithm called Proximal Policy Optimization. I adapted my own version of it to work with keras and retrogym from other projects people have done using PPO and gym. The basics of my implementation of PPO is there are 2 neural networks, an actor and a critic. The actor looks at the state of the game (141312 inputs for numerical observation or a 256x224x3 image for visual observation) and makes a prediction of an action to take at the current timestep. The critic looks at the state of the game and predicts what the reward will be for the next timestep. It subtracts the predicted reward from the actual reward to calculate what's called the advantage, which, along with the previous prediction, is used as the parameters of the loss function of the actor network. Basically the actor picks an action at a given timestep, the critic evaluates what the reward *should* be given the best action was chosen, then optimizes the networks based on how wrong they were.

To read a much more intensive explanation of PPO, check out the [paper written about it.](https://arxiv.org/abs/1707.06347)

To check out a more digestible explanation, (not quite layman's terms, but with a lexicon less nestled in academia) [OpenAI has a good blog post about it.](https://openai.com/blog/openai-baselines-ppo/)

## **Genetic Learning**

The general idea behind genetic learning is simple, can you have an agent learn an environment through evolution? Obviously computers don't have genes or anything like that, but it's easy to simulate. If you create a genepool of say, 100 agents, all of which have 1000 randomly generated actions assigned to them as their genes, then run all those agents through the environment, you've created 100 bad models. But if you track the rewards for each agent as it goes through the environment, then kill off the bottom 90% of agents, now you've got the best of the worst. But the whole point of evolution isn't being perfect immediately, it's adapting to your environment slowly, generation by generation. So now you need to create the next generation of agents, so to speak. You do this by creating children from the top 10% of the agents of the previous generation. The most basic algorithm would be creating 9 copies of each agent (assuming 100 agent genepools) but with a mutation chance. Meaning each action of the agent has some chance to mutate into a different, random action. You end up with the previous best agents + 90 children in this new generation. Then we run them through the environment again and repeat until happy. This basic model provides many things that the CNN + PPO version doesn't, which is faster computation times and the ability to run all these agnets in parallel.

## **Image vs. Numerical Models**

I touched on this briefly in the PPO explanation, but retrogym offers two different observation types for its emulators. The first and default being visual observations. What this means (this is default settings and can be changed in many ways) is that every frame of the game is a timestep, and each timestep gives a screenshot of the current frame to the network to make a prediction. The network only sees the screen and the varios parts of it. For numerical observation, each time step returns (instead of an image) an array of 16 bit integers of the memory state of the game. That means the model never sees what the game actually looks like, it predicts all of its actions and rewards completely blindly.

There are pitfalls of each of these approaches, for the numerical analysis, it's very easy for the network to memorize exactly how to beat a given level, but the skills aren't transferrable to other levels. It learns stuff like "at x units through the level, a spin jump action doesn't get me killed", but not things like "When I see a gap/goomba/koopa/etc., that means I should jump". So for making a model that's objectively good at Mario (regardless of level), this isn't a good approach. The image analysis side isn't free of issues either though. Using a CNN (even at a small level) can be VERY computationally expensive and drastically increase training times. The image approach, (I'll update this when I tune a CNN more with this game) while at the end of the day produces a better Mario player, takes a lot longer to get to the point of beating levels as it's learning how to play Mario, not necessarily just exactly beat the level it's on as quickly as possible.

## **The Numerical Model**

This is what I feel is a silly model. While it was the easiest to setup, it loves to memorize levels rather than actually just being good at Mario. But ultimately you're this far down this readme because you want to see a neural network play Mario. Well this is the correct section for it!

The following gifs are from a 2 layer network, the first layer had 24 neurons and the second layer had 48. It used tanh activation with the Adam optimizer. (This is the layout of both the actor and critic network). Each episode is an attempt at beating the level, so on episode 0 it has no idea what it's doing. As you can see, by episode 50 it started to learn to traverse the level rather than jump around randomly. Episode 120 here is an example of showing that the model doesn't just get better each time. It still dies at parts that it had beaten before as it's trying new actions/different jump timings, etc.

But on episode 241, it finally happened. The very first level completion by my own model (might I add, the very first time any iteration of any of my models beat a level)

Episode 0                  |  Episode 50               | Episode 120               |  Episode 241
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![nn_episode0][nn_episode0]  |  ![nn_episode50][nn_episode50]  |  ![nn_episode120][nn_episode120]  |  ![nn_episode241][nn_episode241]

## **The Image Model**

The image model (so far) has brought some roadblocks into the equation. This thing takes a LONG time to train, even with very small parameters. Below are some gifs of it's attempts on a 100 episode training session. The images of the screen were downsampled to 128x112 and converted from color to greyscale before being given as inputs. That being said, it did suprisingly well given the augmented data. As you can see, even on the first episode we're seeing movement towards the right, and by episode 2 it's already cleared the midway point. Unfortunately, the farthest this model ever made it during its 100 episodes was on episode 88. It never crossed that gap unfortunately, and the immediate next episode went very poorly. This I feel is due to one of two things. Either the models aren't fitting and updating their weights often enough, or they're doing it way too often.

Episode 0                  |  Episode 2                | Episode 88                |  Episode 89
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![cnn_episode0][cnn_episode0]  |  ![cnn_episode2][cnn_episode2]  |  ![cnn_episode88][cnn_episode88]  |  ![cnn_episode89][cnn_episode89]

## **Conclusion**

After initial testing it would appear the CNN has a much higher potential than the NN. The CNN not only is moving with the level right out of the gate, but also did pretty well consistently. The NN felt almost like a random action type algorithm where when it finds a sequence that works, it'll try similar things to it. That's good on a level by level basis but not good for overall skill.

Both network types were trained with relatively small parameters and architecture, so I hope to retrain both of them with larger set and save the models to do more work locally on them.

## **What's Next?**

- Convert repo into easily pip-installable repository
- Update code to use tensorflow 2.0 (and more recent versions of all libraries)
- Improve command line usage so model/alg parameters can be updated and customized with the script call
- Optimize the way the gifs are hosted in the repo/README because they add so much load time (sorry!)
- Build out docker/headless script for training and saving episodes to a cloud bucket for use on an AWS/GCP/Azure instance
- GPU Support!
- Plot relative learning rates (by looking at rewards) of NN and CNN
- Implement other optimization algorithms and/or models
- View how model(s) performance changes from level to level
- Train a "master" model of each type on various levels in hopes of creating a TAS machine

## **References and Credits**

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [OpenAI PPO Blog Post](https://openai.com/blog/openai-baselines-ppo/)
- [PPO Framwork I Adapted](https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py)
- [Super Mario World Memory Map](https://www.smwcentral.net/?p=memorymap&game=smw&region=ram)

[nn_episode0]:https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/img/nn_episode0.gif
[nn_episode50]:https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/img/nn_episode50.gif
[nn_episode120]:https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/img/nn_episode120.gif
[nn_episode241]:https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/img/nn_episode241.gif

[cnn_episode0]:https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/img/cnn_episode0.gif
[cnn_episode2]:https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/img/cnn_episode2.gif
[cnn_episode88]:https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/img/cnn_episode88.gif
[cnn_episode89]:https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/img/cnn_episode89.gif