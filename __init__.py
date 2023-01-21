from gym.envs.registration import register

register(
    id='rl_gym/GridWorld-v0',
    entry_point='rl_gym.envs:GridWorldEnv',
    max_episode_steps=300,
)

register(
    id='rl_gym/PuckWorld-v0',
    entry_point='rl_gym.envs:PuckWorldEnv',
    #max_episode_steps=300,
)

