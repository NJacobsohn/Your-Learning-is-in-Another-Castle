# Your Learning is in Another Castle

## Table of Contents

1. [Overview](#overview)
2. [Questions](#questions)
3. [Environment](#environment)
4. [Environment Setup](#environment-setup)
5. [PPO](#ppo)
6. [Image vs. Numerical Models](#image-vs.-numerical-models)
7. [Conclusion](#conclusion)
8. [What's Next?](#what's-next?)
9. [Photo and Code Credits](#photo-and-code-credits)

## **Overview**

Videogames have always held a near and dear place in my heart. Whether it's something relaxing like running a farming village or something more intense like a turn-based strategy game where you build an empire and conquer the world, they're all special to me. One game series that holds a very special place in my heart is the Mario franchise, but more specifically within it, Super Mario World. I have so many childhood memories playing this game on my Game Boy on long car rides and sneaking it into school to play during recess. But the best memories have to be all the times I finally beat some level that I was stuck on for days. Which led me to wonder, just how much better could my computer be than me at playing Mario? This project aims to create a phenomenal AI Mario player through reinforcement learning.

## **Questions**

- Can I teach an AI to be actually good at Mario? Not just memorizing the level it's on, a genuinely strong player.

- Do skills learned from seeing the screen differ from not seeing it?

- Do skills translate across multiple levels, worlds, even games?

## **Environment**

I used a addon for gym called retrogym which has support for quite a few older emulators as the base for my training environment. Retrogym has a lot of built in integrations for game ROMS, Super Mario World included. Unfortunately, their integration for it was very barebones and not at all ideal for what I wanted to do. The only variables you could use a rewards were your score, coins (which affects score anyways, so really just one is needed), and lives. I wanted so much more, as I was hoping to teach my model to beat a level in one try (any score, coins, bonus lives, etc. should just be a bonus).

## **Environment Setup**

In order to setup my custom environment, I had quite a few things I had to work out. First, how would I actually give incremental rewards to push the model to try and beat a level? Second, how could I define if a level was beaten? Third, how do I stop the model from doing what I call "idiot actions" like pausing the game, accidentally quitting, etc.? To tackle the first problem, I had to look into if there was a variable in the game that tracks the player's position in a level. To find this and any other variables, I had to follow memory maps (locations in RAM on where these variables are set to save to) and then actually pull those values from the emulator using retrogym's handy game integration tool. I needed the location in memory in base 16 (e.g. 7E13CE for if the midway flag was crossed), which then is converted to its base 10 value automatically by the integration tool (for the earlier example, it's now 8262606) to be used by the python environment. Another important part of these variables which had to be specified was their endiannesses, how many bytes they were, and whether or not it was a signed int. These values can be found in [this .json file I built.](https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/variables/data.json)

After much searching and sifting, I finally found what I think to be a pretty good set of variables to work with. Each variable in the above link had to also be defined in the training scenario for the networks. The [scenario.json file](https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/scenarios/scenario.json) I built uses the variables I defined in the data.json to define the training environment. The training session is considered done if Mario loses a life or completes the level. The primary reward is distance traveled on the x axis within a level, with considerably large rewards given to reaching the midway checkpoint and completing the whole level. This was done to ensure that the best way to achieve the highest possible score involved beating the level, not using shenanigans like leaving and entering an area to spawn more enemies.

There were many ways I tried to define if a level was beaten, as there's no 0/1 value that just means "Finish line crossed y/n". At first I used a countdown to when your final score is calculated after beating a level, but I found that to be very unreliable. I also tried using a value which has Mario's status in the overworld (level selection screen), but that wouldn't apply until a level was beaten and Mario had returned to the level selection screen. This became an issue as sometimes the network would very quickly be able to enter a new level before the done condition would trigger, then it wouldn't ever trigger due to being inside of a level again. The final decision was a conditional variable that represents how the curren level was exited. This works not only for quickly telling the network it's done when the final calculations were complete, but also properly rewards the network for beating a level.

So we've got all the variables defined and all their locations in memory have been tracked down, now it's time to limit the action space of the environment to prevent the aforementioned idiot actions from occurring. I decided on the following set of 17 actions for the network to be allowed to use:

The following actions have a regular version, and a sprinting version (12 actions):

- Move Left
- Move Right
- Regular Jump Left
- Regular Jump Right
- Spin Jump Left
- Spin Jump Right

The following actions only exist in their explicitly defined forms (5 actions):

- Regular Jump in place
- Spin Jump in place (allows for bouncing on certain enemies, I wanted to keep it in to see it it'd learn more advanced tactics)
- Look Up
- Look Down (necessary for entering pipes)
- Jump in place and Look Up (necessary for entering pipes on the ceiling)

These actions are defined in a custom wrapper for the environment, which can be explored in my [action discretizer script.](https://github.com/NJacobsohn/Your-Learning-is-in-Another-Castle/blob/master/src/action_discretizer.py)

## **PPO**

In order to properly optimize how these models learn, I adapted a very popular reinforcement learning algorithm called Proximal Policy Optimization. I made my own adaptation of it to work with keras models, as I vastly prefer using keras over just tensorflow and I wasn't able to find any implementations of it already made for keras and retrogym.

## **Image vs. Numerical Models**

## **Conclusion**

## **What's Next?**

## **Photo and Code Credits**
