import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='GridWorld4-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'n_items': 4},
    max_episode_steps=500
)

register(
    id='GridWorld2-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'n_items': 2},
    max_episode_steps=500
)

register(
    id='GridWorld1-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'n_items': 1},
    max_episode_steps=500
)

register(
    id='GridWorldRandom4-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'map_type': 'random_maze', 'n_items': 4},
    max_episode_steps=500
)

register(
    id='GridWorldRandom2-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'map_type': 'random_maze', 'n_items': 2},
    max_episode_steps=500
)

register(
    id='GridWorldRandom1-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'map_type': 'random_maze', 'n_items': 1},
    max_episode_steps=500
)

register(
    id='GridWorldRandomShape4-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'map_type': 'random_shape_maze', 'n_items': 4},
    max_episode_steps=500
)

register(
    id='GridWorldRandomShape2-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'map_type': 'random_shape_maze', 'n_items': 2},
    max_episode_steps=500
)

register(
    id='GridWorldRandomShape1-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    kwargs={'map_type': 'random_shape_maze', 'n_items': 1},
    max_episode_steps=500
)