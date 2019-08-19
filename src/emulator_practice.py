import retro

def random_actions(env):
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    env = retro.make(game='SuperMarioWorld-Snes')#, state="roms/super_mario_world/save_states/yoshis_island1_start.state")
    random_actions(env)