
from collections import namedtuple
from enum import Enum

from gym import Env

from multiagent.environment import MultiAgentEnv, BatchMultiAgentEnv

from multiagent.MailboxInSync import MailboxInSync

class EnvRequestType(Enum):
    STEP = 0
    RESET = 1
    RENDER = 2
    

EnvRequest = namedtuple('EnvRequest', ['type', 'data'])


class EnvironmentSpawn(Env):
    def __init__(self, observation_space, action_space, mailbox):
        self._mailbox = mailbox
        self.action_space = action_space
        self.observation_space = observation_space
        print(action_space)
        print(observation_space)
        
    def step(self, action):
        request = EnvRequest(EnvRequestType.STEP, action)
        self._mailbox.append(request)
        response = self._mailbox.get()
    
    def reset(self):
        request = EnvRequest(EnvRequestType.RESET, None)
        self._mailbox.append(request)
        response = self._mailbox.get()
        
    
    def render(self):
        request = EnvRequest(EnvRequestType.RENDER, None)
        self._mailbox.append(request)


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
        return EnvironmentSpawn(self.observation_space[new_id], self.action_space[new_id], self.mailbox.spawn())
        
    def handle_requests(self):
        requests = self.mailbox.get()
        if all([r.type == EnvRequestType.RESET for r in requests]):
            observations = self.reset()
            self.mailbox.append(observations)
        else:
            actions = [r.data for r in requests]
            obs, rewards, dones, info = self.mailbox.step(actions)
            agent_data = list(zip(obs, rewards, dones, info))
            self.mailbox.append(agent_data)
            
        
        
        
    