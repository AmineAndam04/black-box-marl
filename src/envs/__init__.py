import os
import sys

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
from .smaclite_wrapper import SMACliteWrapper


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    if True:
        from gymnasium.envs.registration import register
        register(
            id="Foraging-1s-8x8-3p-2f-v3",
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": 3,
                "min_player_level": 1,
                "max_player_level": 2,
                "field_size": (8, 8),
                "min_food_level": 1,
                "max_food_level": None,
                "max_num_food": 2,
                "sight": 1,
                "max_episode_steps": 50,
                "force_coop": False,
                'grid_observation': False,
                'penalty': 0.0
            },
        )
        # Register Foraging-1s-10x10-4p-2f
        register(
            id="Foraging-1s-10x10-4p-2f-v3",
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": 4,
                "min_player_level": 1,
                "max_player_level": 2,
                "field_size": (10, 10),
                "min_food_level": 1,
                "max_food_level": None,
                "max_num_food": 2,
                "sight": 1,
                "max_episode_steps": 50,
                "force_coop": False,
                'grid_observation': False,
                'penalty': 0.0
            },
        )
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper
    #from .smac_sgwrapper import SMACSGWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)
        #return SMACSGWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper
    #from .smacv2_sgwrapper import SMACv2SGWrapper
    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)
        #return SMACv2SGWrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
