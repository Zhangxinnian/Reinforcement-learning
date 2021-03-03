import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Grid(object):
    def __init__(self, x:int = None,
                       y:int = None,
                       type:int = 0,
                       reward:int = 0,
                       value:float = 0.0): #value attribute alternate
        self.x = x  #Coordinate x
        self.y = y
        self.type = value #Category value (0: empty; 1: obstacle or boundary)
        self.reward = reward #Instant rewards for this grid
        self.value = value #The value of this grid is temporarily useless
        self.name = None # The name of the grid
        self._update_name()

    def _update_name(self):
        self.name = 'X{0}-Y{1}'.format(self.x, self.y)

    def __str__(self):
        return 'name:{4}, x:{0}, y{1}, type:{2}, value:{3}'.format(self.x,
                                                                   self.y,
                                                                   self.type,
                                                                   self.reward,
                                                                   self.value,
                                                                   self.name)

class GrideMatrix(object):
    '''
    Grid matrix, through different settings, simulate different grid world environment
    '''
    def __init__(self, n_width:int, #Horizontal grid number
                       n_height:int, #Number of grids in vertical direction
                       default_type: int = 0, #Default type
                       default_reward:float = 0.0, #Default instant reward value
                       default_value:float = 0.0 #Default value (this is a bit redundant)
                       ):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()

    def reset(self):
        self.grids = []
        for x in  range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x, y, self.default_type, self.default_reward, self.default_value))

    def get_grid(self, x, y= None):
        '''
        Get a grid information
        Args: coordinate information, represented by x, y or only one x of type tuple
        Return: grid object
        '''
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert (xx >= 0 and yy >= 0 and xx < self.n_width and yy < self.n_height),"任意坐标值应在合理区间"
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise ("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise("gride doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise("grid doesn;t exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid =self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type

class GridWorldEnv(gym.Env):
    '''
    The grid world environment can simulate various grid worlds
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, n_width:int = 10,
                       n_height:int = 7,
                       u_size = 40 ,
                       default_reward:float = 0,
                       default_type = 0,
                       windy = False):
        self.u_size = u_size  #Current grid drawing size
        self.n_width = n_width #Grid world width (in the number of grids)
        self.n_height = n_height #Height
        self.width = u_size * n_width  #Scene width screen width
        self.height = u_size * n_height  #Scene length
        self.default_reward = default_reward
        self.default_type = default_type
        self._adjust_size()

        self.grids = GrideMatrix(n_width= self.n_width,
                                 n_height= self.n_height,
                                 default_reward= self.default_reward,
                                 default_type= self.default_type,
                                 default_value= 0.0
                                 )
        self.reward = 0 # For rendering
        self.action = None #For rendering
        self.windy = windy #Is there a style sub-world

        #0,1,2,3,4 represent left, right, up, down, -, five moves
        self.action_space = spaces.Discrete(4)
        #The observation space is determined by low and high
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)
        #The origin of the coordinates is the lower left corner, this pyglet is consistent
        #By setting the starting point, ending point, and special rewards and types of grids,
        # you can build a variety of different types of grid world environments
        # For example: Random walking, car rental, cliff walking and other examples in David Silver's open class
        self.ends = [(7,3)] #End grid coordinates, there can be multiple
        self.start = (0,3) #Start grid coordinate, only one
        self.types = [] #Special types of grids are set here. [(3,2,1)] means that the value at (3,2) is 1
        self.rewards = [] #The grid of special rewards is set here, and the grid reward is terminated 0
        self.refresh_setting()
        self.viewer = None #Graphical Interface Object
        self.seed()
        self.reset()

    def _adjust_size(self):
        '''
        Adjust the scene size to fit the maximum width and height not exceeding 800
        '''
        pass

    def seed(self, seed=None):
        '''
        Generate a seed needed for randomization and return an np_random object to support subsequent randomization generation operations
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _windy_effect(self, x, y):
        new_x, new_y = x, y
        if self.windy:
            if new_x in [3,4,5,8]:
                new_y += 1
            elif new_x in [6,7]:
                new_y += 2
        return new_x, new_y

    def _action_effect(self, x, y, action):
        new_x, new_y = x, y
        if action == 0: new_x -= 1 #Left
        elif action == 1: new_x += 1 #Right
        elif action == 2: new_y += 1 #Up
        elif action == 3: new_y -= 1 #Down
        elif action == 4: new_x, new_y = new_x - 1 , new_y - 1
        elif action == 5:new_x, new_y = new_x + 1, new_y - 1
        elif action == 6:new_x, new_y = new_x + 1, new_y - 1
        elif action == 7:new_x, new_y = new_x + 1, new_y + 1
        return new_x, new_y

    def _boundary_effect(self, x, y):
        new_x, new_y = x, y
        if new_x < 0:new_x =0
        if new_x >= self.n_width:new_x = self.n_width - 1
        if new_y < 0: new_y = 0
        if new_y >= self.n_height: new_y = self.n_height - 1
        return new_x, new_y

    def step(self, action):
        assert self.action_space.contains(action),\
        '%r (%s) invalid ' % (action, type(action))
        self.action = action #Action for rendering
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y
        #wind effect:
        #There is a wind effect,
        # the number indicates the number of cells that the individual will be blown away from the cell when they leave (rather than enter) the cell.
        new_x, new_y = self._windy_effect(new_x, new_y)
        #Behavioral effect
        new_x, new_y = self._action_effect(new_x, new_y, action)
        #boundary effect
        new_x, new_y = self._boundary_effect(new_x, new_y)

        #Wall effect:
        #Type 1 grids are barrier grids and cannot be entered
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y
        self.reward = self.grids.get_reward(new_x, new_y)
        done = self._is_end_state(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)
        #Provide all the information of Grid World in info
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        return self.state, self.reward, done, info

    #Change status to horizontal and vertical coordinates
    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  # Unknown status

    def refresh_setting(self):
        '''
        Users may modify certain grid types or reward values of the grid world after using this class to create a grid world
        After modifying the settings, call this method to make the settings effective.
        '''
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state

    #Determine whether it is a terminal state
    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert (isinstance(x, tuple)), "坐标数据不完整"
            xx, yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    #Graphical interface
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2  # Gap size between grids

        # If the screen object has not been set, initialize the elements of the entire screen.
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # 在Viewer里绘制一个几何图像的步骤如下：
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
            #    性化的方法来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的
            #    属性就是变换属性，
            #    该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制
            ''' 绘制水平竖直格子线，由于设置了格子之间的间隙，可不用此段代码
            for i in range(self.n_width+1):
                line = rendering.Line(start = (i*u_size, 0), 
                                      end =(i*u_size, u_size*self.n_height))
                line.set_color(0.5,0,0)
                self.viewer.add_geom(line)
            for i in range(self.n_height):
                line = rendering.Line(start = (0, i*u_size),
                                      end = (u_size*self.n_width, i*u_size))
                line.set_color(0,0,1)
                self.viewer.add_geom(line)
            '''

            # Draw grid
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [(x * u_size + m, y * u_size + m),
                         ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m),
                         (x * u_size + m, (y + 1) * u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 10
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(1.0, 1.0, 1.0)
                    self.viewer.add_geom(rect)
                    # Draw border
                    v_outline = [(x * u_size + m, y * u_size + m),
                                 ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                 (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x, y):
                        # Add a golden yellow border to the end square
                        outline.set_color(0.9, 0.9, 0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x, y) == 1:  # Obstacle grids are shown in dark gray
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass
            # Draw individual
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

        # Update individual location
        x, y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


def LargeGridWorld():
    '''
    A grid world environment of 10*10, setting reference:
    http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
    '''
    env = GridWorldEnv(n_width=10,
                       n_height = 10,
                       u_size = 40,
                       default_reward = 0,
                       default_type = 0,
                       windy=False)
    env.start = (0,9)
    env.ends = [(5,4)]
    env.types = [(4,2,1),(4,3,1),(4,4,1),(4,5,1),(4,6,1),(4,7,1),
                 (1,7,1),(2,7,1),(3,7,1),(4,7,1),(6,7,1),(7,7,1),
                 (8,7,1)]
    env.rewards = [(3,2,-1),(3,6,-1),(5,2,-1),(6,2,-1),(8,3,-1),
                   (8,4,-1),(5,4,1),(6,4,-1),(5,5,-1),(6,5,-1)]
    env.refresh_setting()
    return env

def SimpleGridWorld():
    '''
    No wind 10*7 grid, setting reference: David Silver Intensive Learning Open Class Video Lecture 3
    '''
    env = GridWorldEnv(n_width=10,
                       n_height = 7,
                       u_size = 40,
                       default_reward = -1,
                       default_type = 0,
                       windy=False)
    env.start = (0,3)
    env.ends = [(7,3)]
    env.rewards = [(7,3,0)]
    env.refresh_setting()
    return env

def WindyGridWorld():
    '''
    Windy 10*7 grid, setting reference: David Silver Intensive Learning Open Class Video Lecture 5
    '''
    env = GridWorldEnv(n_width=10,
                       n_height = 7,
                       u_size = 40,
                       default_reward = -1,
                       default_type = 0,
                       windy=True)
    env.start = (0,3)
    env.ends = [(7,3)]
    env.rewards = [(7,3,0)]

    env.refresh_setting()
    return env


def RandomWalk():
    '''
    Random walk example environment
    '''
    env = GridWorldEnv(n_width=7,
                       n_height=1,
                       u_size=80,
                       default_reward=0,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(2)  # left or right
    env.start = (3, 0)
    env.ends = [(6, 0), (0, 0)]
    env.rewards = [(6, 0, 1)]
    env.refresh_setting()
    return env


def CliffWalk():
    env = GridWorldEnv(n_width=12,
                       n_height=4,
                       u_size=40,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)  # left or right
    env.start = (0, 0)
    env.ends = [(11, 0)]
    env.rewards = [(11, 0, 0)]
    for i in range(10):
        env.rewards.append((i + 1, 0, -100))
        env.ends.append((i + 1, 0))
    env.refresh_setting()
    return env


class CliffWalk2(GridWorldEnv):
    def __init__(self, n_width=12,
                 n_height=4,
                 u_size=40,
                 default_reward=-1,
                 default_type=0,
                 windy=False):
        super(CliffWalk2, self).__init__(n_width=12,
                                         n_height=4,
                                         u_size=40,
                                         default_reward=-1,
                                         default_type=0,
                                         windy=False)
        self.start = (0, 0)
        self.ends = [(11, 0)]
        self.rewards = [(11, 0, 0)]
        for i in range(10):
            self.rewards.append((i + 1, 0, -100))
            # self.ends.append((i+1,0))
        self.refresh_setting()

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        self.action = action  # action for rendering
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        # wind effect:
        # There is a wind effect, the number indicates the number of cells that the individual will be blown away from the cell when they leave (rather than enter) the cell.
        new_x, new_y = self._windy_effect(new_x, new_y)
        # Behavioral effect
        new_x, new_y = self._action_effect(new_x, new_y, action)
        # boundary effect
        new_x, new_y = self._boundary_effect(new_x, new_y)

        # wall effect:
        # Type 1 grids are barrier grids and cannot be entered
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)
        # Cliff effect
        if 0 < new_x < 11 and new_y == 0:
            new_x, new_y = 0, 0

        done = self._is_end_state(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)
        # Provide all the information of Grid World in info
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        return self.state, self.reward, done, info


def SkullAndTreasure():
    '''
    Examples of skeletons and coins to explain the effectiveness of stochastic strategies David Silver Reinforcement Learning Open Lecture 6 Strategy Gradient
    '''
    env = GridWorldEnv(n_width=5,
                       n_height=2,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)  # left or right
    env.start = (0, 1)
    env.ends = [(2, 0)]
    env.rewards = [(0, 0, -100), (2, 0, 100), (4, 0, -100)]
    env.types = [(1, 0, 1), (3, 0, 1)]
    env.refresh_setting()
    return env


if __name__ == "__main__":
    env = GridWorldEnv()
    print("hello")
    env.reset()
    nfs = env.observation_space
    nfa = env.action_space
    print("nfs:%s; nfa:%s" % (nfs, nfa))
    print(env.observation_space)
    print(env.action_space)
    print(env.state)
    env.render()
    # x = input("press any key to exit")
    for _ in range(20000):
        env.render()
        a = env.action_space.sample()
        state, reward, isdone, info = env.step(a)
        print("{0}, {1}, {2}, {3}".format(a, reward, isdone, info))

    print("env closed")