import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment
import inspect

class Scenario(BaseScenario):
    def __init__(self, num_agents=3, dist_threshold=0.2, arena_size=1, identity_size=0, num_steps=100, diff_reward=True, same_color = False, random_leader_name=False, goal_at_top = False, hide_goal=False):
        self.num_agents = num_agents
        self.rewards = np.zeros(self.num_agents)
        self.temp_done = False
        self.dist_threshold = dist_threshold
        self.arena_size = arena_size
        self.identity_size = identity_size
        self.num_steps = num_steps
        self.start_loc = np.array([0,-0.9], dtype = float)
        self.diff_reward = diff_reward
        self.same_color = same_color
        self.random_leader_name = random_leader_name
        self.goal_at_top = goal_at_top
        self.hide_goal = hide_goal

    def make_world(self):
        world = World('simple_waypoints')
        # set any world properties first
        world.dim_c = 0
        num_agents = self.num_agents
        num_landmarks = 1
        world.collaborative = False
        # add agents
        world.agents = [Agent(iden=i) for i in range(num_agents)]
        
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.adversary = False
            if i == 0:
                agent.is_leader = True
            else:
                agent.is_leader = False

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        world.dists = []
        world.dist_thres = self.dist_threshold
        return world

    def reset_world(self, world):
        # -------- giving an ID to the leader -------- #
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
        
        world.leader_name = 0    
        if self.random_leader_name:
            world.leader_name = np.random.randint(self.num_agents)
            world.agents[world.leader_name].name = 'agent %d' % 0
            world.agents[0].name = 'agent %d' % world.leader_name

        # random goal loc
        size = self.arena_size
        goal_loc = np.array([-0.8,0])
        world.current_waypoint_id = 0
        # if self.goal_at_top:
        #     goal_loc = np.random.uniform([-size, 0.9*size], [size,0.9*size])
        # else:
        #     goal_loc = np.random.uniform([-size, 0], [size,0.9*size])
            
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if not self.same_color and agent.is_leader:
                agent.color = np.array([0.15, 0.15, 0.95])
            else:
                agent.color = np.array([0.15, 0.95, 0.15])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if self.hide_goal:
                landmark.color = np.array([1, 1, 1, 0])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
        
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.start_loc.copy()
            agent.state.p_pos = np.random.uniform([-0.25*size, -size], [0.25*size,-size*0.9])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.prev_dist = None

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = goal_loc
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.steps = 0
        world.dists = []

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # if agent.iden == 0: # compute this only once when called with the first agent
        #     # each column represents distance of all agents from the respective landmark
        #     world.dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
        #                            for a in world.agents])
        #     # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
        #     self.min_dists = self._bipartite_min_dists(world.dists) 
        #     # the reward is normalized by the number of agents
        #     joint_reward = np.clip(-np.mean(self.min_dists), -15, 15) 
        #     self.rewards = np.full(self.num_agents, joint_reward)
        #     world.min_dists = self.min_dists
        goal_loc = world.landmarks[0].state.p_pos
        dist = np.linalg.norm(agent.state.p_pos-goal_loc)
        if self.diff_reward:
            if agent.prev_dist is None:
                reward = 0
            else:
                reward = 1e2*(agent.prev_dist-dist)
        else:
            reward = -dist
        agent.prev_dist = dist.copy()
        
        # -------- success reward -------- #
        success_list = [self.is_success(a,world) for a in world.agents]
        reward += 50*np.all(success_list)

        # reward += int(self.is_success(agent, world))*500
        return reward
        # return self.rewards.mean()

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):
        # positions of all entities in this agent's reference frame, because no other way to bring the landmark information
        # for entity in world.landmarks:
        #     print(entity.state.p_pos, agent.state.p_pos)
        # entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        goal_pos = [[0,0]]
        if agent.is_leader:
            goal_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        # print('agent', float(agent.is_leader))
        default_obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + goal_pos)
        # print(default_obs)
        if self.identity_size != 0: 
            identified_obs = np.append(np.eye(self.identity_size)[agent.iden],default_obs)
            return identified_obs
        return default_obs

    def done(self, agent, world):
        timeup = world.steps >= world.max_steps_episode
        is_success = self.is_success(agent, world)
        # if timeup:
        #     print('timeup for agent')
        # if is_success:
        #     print('is_success for agent')
            # print('world time', world.steps, world.max_steps_episode)
        return timeup# or is_success
    
    def is_success(self, agent, world):
        dist = np.linalg.norm(agent.state.p_pos-world.landmarks[0].state.p_pos)
        return dist < world.dist_thres
        
    # def done(self,world):
    #     done = 1
    #     for agent in world.agents:
    #         done *= self.done_agent(agent, world)
    #     return(done)

    def info(self, agent, world):
        info = {'is_success': self.is_success(agent, world), 'world_steps': world.steps,
                'reward':self.rewards.mean()}#, 'dists':self.min_dists.mean()}
        return info
