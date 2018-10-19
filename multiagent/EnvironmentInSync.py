
from collections import namedtuple

from gym import Env

from multiagent.environment import MultiAgentEnv, BatchMultiAgentEnv

from multiagent.MailboxInSync import MailboxInSync

SpawnAction = namedtuple('SpawnAction', ['ID', 'action'])

class EnvironmentSpawn(Env):
    def __init__(self, mailbox):
        self._mailbox = mailbox
        
    def step(self, action):
        self._mailbox.append(action)
        return self._mailbox.get()
    
    def reset(self):
        pass
    
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
        return EnvironmentSpawn(mailbox.spawn())
        
    def step(self):
        action_n = self.mailbox.get()
        super().step(action_n)
        
    def reset(self):
        self.mailbox.get()
        super().reset()
        
        
        
    