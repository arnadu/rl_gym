"""
PuckWorld Environment for OpenAI gym
http://cs.stanford.edu/people/karpathy/reinforcejs/puckworld.html

Code derived from: 
https://github.com/qqiang00/reinforce/blob/master/reinforce/puckworld.py
https://www.gymlibrary.dev/content/environment_creation/

"""
import pygame
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from collections import deque

from tqdm import tqdm
import gc

RAD2DEG = 57.29577951308232     # 弧度与角度换算关系1弧度=57.29..角度

class PuckWorldEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        "fps": 30
        }

    def __init__(self, render_mode=None, fps=0, reward1=True, reward2=True):
        self.width = 600            # screen width
        self.height = 600           # 
        self.l_unit = 1.0           # pysical world width
        self.v_unit = 1.0           # velocity 
        self.max_speed = 0.025      # max agent velocity along a axis
        
        self.reward1 = reward1      #take distance to target 1 into reward calc
        self.reward2 = reward2      #take distance to target 2 into reward calc

        self.score = deque(maxlen=60)
        self.episode_score = 0

        #self.re_pos_interval = 30   # 
        self.accel = 0.002          # agent
        self.rad = 0.05             # agent radius
        self.target_rad = 0.01      # good target radius.
        self.target_rad2 = 0.25     # bad target radius.
        #self.goal_dis = self.rad    # expected goal distance
        self.t = 0                  # puck world clock
        self.update_time = 100      # time for target randomize its position
        self.target_speed = 0.001   # speed of the bad target
        
        #bounds for the observation space
        self.low = np.array([-self.l_unit/2,                 # agent position x
                            -self.l_unit/2,                  # agent position y
                            -10*self.max_speed,    # agent velocity x
                            -10*self.max_speed,    # agent velocity y
                            -self.l_unit,                  # good target position x
                            -self.l_unit,                  # good target position y
                            -self.l_unit,                  # bad target position x
                            -self.l_unit,                  # bad target position y
                            ])   
        self.high = np.array([self.l_unit/2,                 
                            self.l_unit/2,
                            10*self.max_speed,    
                            10*self.max_speed,    
                            self.l_unit,    
                            self.l_unit,
                            self.l_unit,    
                            self.l_unit,
                            ])   
        self.observation_space = spaces.Box(self.low, self.high)    

        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_space = spaces.Discrete(5)  

        self.reward = 0         # for rendering
        self.action = None      # for rendering

        #self._seed()    

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.fps = fps

        self.quit = False  #becomes true if user quits the display window
        self.reset()

    #def _seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)  
    #    return [seed]

    def _get_obs(self):
        (ppx, ppy, pvx, pvy, tx, ty, tx2, ty2) = self.state
        obs = (ppx - self.l_unit/2, ppy-self.l_unit/2, pvx*10, pvy*10, tx-ppx, ty-ppy, tx2-ppx, ty2-ppy)
        return np.array(obs)

    def _get_info(self):
        return {'state':np.array(self.state)}

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        
        self.action = action    # action for rendering
        ppx,ppy,pvx,pvy,tx,ty, tx2, ty2 = self.state 
        ppx, ppy = ppx+pvx, ppy+pvy         # update agent position
        pvx, pvy = pvx*0.95, pvy*0.95       # damping 

        if action == 0: pvx -= self.accel   # accelerate downward
        if action == 1: pvx += self.accel   # accelerate upward
        if action == 2: pvy += self.accel   # accelerate right
        if action == 3: pvy -= self.accel   # accelerate left
        if action == 4: pass                # no change in velocity

        if ppx < self.rad:              # encounter left bound
            pvx *= -0.5
            ppx = self.rad
        if ppx > self.l_unit - self.rad:          # right bound
            pvx *= -0.5
            ppx = self.l_unit - self.rad
        if ppy < self.rad:              # bottom bound
            pvy *= -0.5
            ppy = self.rad
        if ppy > self.l_unit - self.rad:          # right bound
            pvy *= -0.5
            ppy = self.l_unit - self.rad
        
        #relocate the good target from time to time
        self.t += 1
        if self.t % self.update_time == 0:  # update good target's position
            tx = self._random_pos()         # randomly
            ty = self._random_pos()

        #compute distances from agent to targets
        dx, dy = ppx - tx, ppy - ty         # calculate distance from
        dis1 = self._compute_dis(dx, dy)     # agent to good target

        dx2, dy2 = ppx - tx2, ppy - ty2         # calculate distance from
        dis2 = self._compute_dis(dx2, dy2)     # agent to good target

        #make the bad target moves towards the agent
        dxnorm = dx2/dis2
        dynorm = dy2/dis2
        tx2 += dxnorm * self.target_speed
        ty2 += dynorm * self.target_speed

        #compute reward
        self.reward=0
        if self.reward1:
            self.reward = -dis1     #lose points when far from good target

        if self.reward2:
            if (dis2<self.target_rad2): #also lose points if within reach of the bad target
                self.reward += 2*(dis2 - self.target_rad2)/self.target_rad2

        self.score.append(self.reward)
        self.episode_score += self.reward

        self.state = (ppx, ppy, pvx, pvy, tx, ty, tx2, ty2)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self.reward, False, self.quit, self._get_info()

    def _random_pos(self):
        return self.np_random.uniform(low = 0, high = self.l_unit)

    def _random_velocity(self):
        r = self.np_random.uniform(low = 0, high = self.v_unit)
        return 2*self.max_speed*(r - 0.5)

    def _compute_dis(self, dx, dy):
        return math.sqrt(math.pow(dx,2) + math.pow(dy,2))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([ self._random_pos(),         #agent x ppx
                                self._random_pos(),         #agent y ppy
                                self._random_velocity(),    #agent vx 
                                self._random_velocity(),    #agent vy
                                self._random_pos(),         #good target x
                                self._random_pos(),         #good target y
                                self._random_pos(),         #bad target x
                                self._random_pos()          #bad target y
                               ])

        self.episode_score = 0
        self.quit = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return self.state ,info   # np.array(self.state)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(size=(self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        #canvas.fill((255, 255, 255))

        scale = self.width/self.l_unit      
        rad = self.rad * scale              
        t_rad = self.target_rad * scale     
        t_rad2 = self.target_rad2 * scale   

        ppx,ppy,pvx,pvy,tx,ty,tx2,ty2 = self.state

        font = pygame.font.SysFont(None,40) # Font object
        score = np.sum(self.score)
        text = font.render(f'Reward: {score:.2f}', True, (127,127,127)) 
        canvas.blit(text, (10, 20)) # draw the text to the screen
        text = font.render(f'Score: {self.episode_score:.0f}', True, (127,127,127)) 
        canvas.blit(text, (300, 20)) # draw the text to the screen

        pygame.draw.circle(canvas, (0,0,255), center=(ppx*scale, ppy*scale), radius=rad, width=5)
        
        if self.reward1:
            pygame.draw.circle(canvas, (0,255,0), center=(tx*scale, ty*scale), radius=t_rad, width=5)
        
        if self.reward2:
            pygame.draw.circle(canvas, (255,0,0), center=(tx2*scale, ty2*scale), radius=t_rad2, width=5)

        if self.render_mode == "human":

            for e in pygame.event.get(eventtype=pygame.QUIT): 
                self.quit=True
                
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            if self.fps>0:
                self.clock.tick(self.fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window=None

if __name__ =="__main__":
    
    
    render_mode = 'human'
    env = PuckWorldEnv(render_mode=render_mode, fps=60)
    env.reset()
    nfs = env.observation_space.shape[0]
    nfa = env.action_space
    print("nfs:%s; nfa:d"%(nfs))
    print(env.observation_space)
    print(env.action_space)

    for episode in tqdm(range(10)):
        for _ in range(300):  #event in pygame.event.get(): #
            #if event.type == pygame.QUIT:
            #   break
            env.render()
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()
                break
        env.reset()
        if truncated:
            break
    env.close()
    print("env closed")
