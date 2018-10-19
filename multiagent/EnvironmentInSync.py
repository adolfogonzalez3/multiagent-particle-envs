
from collections import namedtuple

from gym import Env

from multiagent.environment import MultiAgentEnv, BatchMultiAgentEnv

from multiagent.MailboxInSync import MailboxInSync

SpawnAction = namedtuple('SpawnAction', ['ID', 'action'])
        

class EnvironmentSpawn(Env):
    def __init__(self, observation_space, action_space, mailbox):
        self._mailbox = mailbox
        self.action_space = action_space
        self.observation_space = observation_space
        print(action_space)
        print(observation_space)
        
    def step(self, action):
        print('Here...')
        self._mailbox.append(action)
        print('Step')
        return self._mailbox.get()
    
    def reset(self):
        return self._mailbox.get()
    
    def render(self):
        pass


class EnvironmentInSync(MultiAgentEnv):
    '''A class to use OpenAi algorithms.'''
    
    
    def __init__(self, *args, **kwargs):
        self.instance_ids = []
        self.id_counter = 0
        self.mailbox = MailboxInSync()
        super().__init__(*args, **kwargs)
        
    def spawn(self):
        new_id = self.id_counter
        self.id_counter += 1
        print(self.observation_space)
        return EnvironmentSpawn(self.observation_space[new_id], self.action_space[new_id], self.mailbox.spawn())
        
    def step(self):
        action_n = self.mailbox.get()
        obs, rewards, dones, info = super().step(action_n)
        self.mailbox.append((obs, rewards, dones, info))
        
    def reset(self):
        obs = super().reset()
        self.mailbox.append(obs)
        
        
        
    