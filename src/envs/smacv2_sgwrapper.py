from pathlib import Path
import yaml
import types
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from .multiagentenv import MultiAgentEnv


SMACv2_CONFIG_DIR = Path(__file__).parent.parent / "config" / "envs" / "smacv2_configs"


def get_scenario_names():
    return [p.name for p in SMACv2_CONFIG_DIR.iterdir()]

class CustomSightRangeWrapper(StarCraftCapabilityEnvWrapper):
    """Wrapper that allows customizing the sight range of units."""
    
    def __init__(self, **kwargs):
        # Extract the custom sight range parameter if provided, otherwise default to 9
        self.custom_sight_range = kwargs.pop("custom_sight_range", 9)
        super().__init__(**kwargs)
        self.obs_space = None
        # Store the original unit_sight_range method
        original_unit_sight_range = self.env.unit_sight_range
        
        # Define a new method that uses our custom sight range when use_unit_ranges is False
        def patched_unit_sight_range(self_env, agent_id):
            if self_env.use_unit_ranges:
                # Use the original behavior with unit-specific ranges when use_unit_ranges is True
                return original_unit_sight_range(agent_id)
            else:
                # Return our custom value when use_unit_ranges is False
                return self.custom_sight_range
        
        # Replace the method in the environment
        self.env.unit_sight_range = types.MethodType(patched_unit_sight_range, self.env)
    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["sight_range"] =  [self.unit_sight_range(agent_id) for agent_id in range(self.n_agents)]
        return env_info

def load_scenario(map_name, custom_sight_range=9, **kwargs):
    """Load a scenario with a custom sight range."""
    # Find and load the scenario file
    scenario_path = SMACv2_CONFIG_DIR / f"{map_name}.yaml"
    with open(scenario_path, "r") as f:
        scenario_args = yaml.load(f, Loader=yaml.FullLoader)
    
    # Update with any additional kwargs
    scenario_args.update(kwargs)
    
    # Add the custom sight range to the environment args
    env_args = scenario_args["env_args"]
    env_args["custom_sight_range"] = custom_sight_range
    
    # Create and return the environment using our custom wrapper
    return CustomSightRangeWrapper(**env_args)

class SMACv2SGWrapper(MultiAgentEnv):
    """Wrapper for SMACv2 environments with customizable sight range."""
    
    def __init__(self, map_name, seed, custom_sight_range=9, **kwargs):
        # Load the scenario with our custom sight range
        self.env = load_scenario(map_name, seed=seed, custom_sight_range=custom_sight_range, **kwargs)
        self.episode_limit = self.env.episode_limit
        
    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False
        return obss, rews, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        obss, _ = self.env.reset()
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()

if __name__ == "__main__":
    for scenario in get_scenario_names():
        env = load_scenario(scenario)
        env_info = env.get_env_info()
        # print name of config, number of agents, state shape, observation shape, action shape
        print(
            scenario,
            env_info["n_agents"],
            env_info["state_shape"],
            env_info["obs_shape"],
            env_info["n_actions"],
            env_info["sight_range"]
        )
        print()
