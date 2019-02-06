
import itertools
import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

        self.__current_force = 0.

    @property
    def mass(self):
        return self.initial_mass

    def accumulate_force(self, force):
        self.__current_force += force

    def reset_force(self):
        self.__current_force = 0

    @property
    def current_force(self):
        return self.__current_force

    def dampen_velocity(self, damping):
        self.state.p_vel *= (1 - damping)

    def apply_force(self, dt):
        self.state.p_vel += (self.current_force / self.mass) * dt
        speed = np.sqrt(np.sum(np.square(self.state.p_vel)))
        if self.max_speed is not None and speed > self.max_speed:
            self.state.p_vel *= self.max_speed / speed

    def update_position(self, dt):
        self.state.p_pos += self.state.p_vel * dt

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

    @property
    def u_shape(self):
        return self.action.u.shape

    def c_shape(self):
        return self.action.c.shape

    def generate_u_noise(self):
        if self.u_noise is not None:
            return np.random.randn(self.u_shape) * self.u_noise
        else:
            return 0

    def apply_noise(self):
        self.apply_force(self.action.u + self.generate_u_noise())

    def update_comm(self):
        if self.silent:
            self.state.c = np.zeros(self.dim_c)
        else:
            if self.c_noise is not None:
                noise = np.random.randn(*self.c_shape) * self.c_noise
                self.state.c += self.action.c + noise      


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        self.current_step = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        
        # apply agent physical controls
        self.apply_action_force()
        # apply environment forcess
        self.apply_environment_force()
        # integrate physical state
        self.integrate_state()
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

        self.current_step += 1

    # gather agent action forces
    def apply_action_force(self):
        # set applied forces
        for agent in itertools.chain(self.policy_agents, self.scripted_agents):
            agent.apply_noise()

    # gather physical forces acting on entities
    def apply_environment_force(self):
        # simple (but inefficient) collision response
        for entity_a, entity_b in itertools.combinations(self.entities, 2):
            f_a, f_b = self.get_collision_force(entity_a, entity_b)
            entity_a.accumulate_force(f_a)
            entity_b.accumulate_force(f_b)

    # integrate physical state
    def integrate_state(self):
        for entity in self.entities:
            entity.dampen_velocity(self.damping)
            entity.apply_force(self.dt)
            entity.update_position(self.dt)

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        for agent in self.policy_agents:
            agent.update_comm()

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        assert(entity_a is not entity_b)
        if not entity_a.collide or not entity_b.collide:
            return None, None
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else 0
        force_b = -force if entity_b.movable else 0
        return force_a, force_b