
from gym import Env

from multiagent.environment import MultiAgentEnv, BatchMultiAgentEnv

class EnvironmentSpawn(Env):
    def __init__(self, id):
        self.id = id
        
    def step(self, action):
    
    def reset(self):
        pass
    
    def render(self):
        pass


class EnvironmentInSync(MultiAgentEnv):
    '''A class to use OpenAi algorithms.'''
    
    
    def __init__(self, *args, **kwargs):
        self.instance_ids = []
        self.id_counter = 0
        super().__init__(*args, **kwargs)
        
    def spawn(self):
        