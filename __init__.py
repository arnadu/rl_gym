from gym.envs.registration import register

register(
    id='gym_examples/GridWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=300,
)

register(
    id='gym_examples/PuckWorld-v0',
    entry_point='gym_examples.envs:PuckWorldEnv',
    #max_episode_steps=300,
)
