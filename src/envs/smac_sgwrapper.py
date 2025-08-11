import types
from smac.env import StarCraft2Env
from .multiagentenv import MultiAgentEnv

class CustomSightRangeStarCraft2Env(StarCraft2Env):
    """StarCraft2Env with customizable sight range."""
    
    def __init__(self, custom_sight_range=9, **kwargs):
        
        self.custom_sight_range = custom_sight_range
        super().__init__(**kwargs)
    
    def unit_sight_range(self, agent_id):
        return self.custom_sight_range
    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["sight_range"] =  [self.unit_sight_range(agent_id) for agent_id in range(self.n_agents)]
        return env_info
    

class SMACSGWrapper(MultiAgentEnv):
    """Wrapper for SMAC environments with customizable sight range."""
    
    def __init__(self, map_name, seed, custom_sight_range=9, **kwargs):
        # Create the environment with our custom sight range
        self.env = CustomSightRangeStarCraft2Env(
            map_name=map_name, 
            seed=seed, 
            custom_sight_range=custom_sight_range,
            **kwargs
        )
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