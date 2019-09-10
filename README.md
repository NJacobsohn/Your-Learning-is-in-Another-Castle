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

In order to setup my custom environment, I had quite a few things I had to work out. First, how would I actually give incremental rewards to push the model to try and beat a level? Second, how could I define if a level was beaten? Third, how do I stop the model from doing what I call "idiot actions" like pausing the game, accidentally quitting, etc.? To tackle the first problem, I had to look into if there was a variable in the game that tracks the player's position in a level.

## **PPO**

## **Image vs. Numerical Models**

## **Conclusion**

## **What's Next?**

## **Photo and Code Credits**
