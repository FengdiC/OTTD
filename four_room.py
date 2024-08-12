import numpy as np
import pickle
import matplotlib.pyplot as plt

BLOCK_NORMAL, BLOCK_WALL, BLOCK_HALLWAY, BLOCK_AGENT, BLOCK_TERMINAL = 0, 1, 2, 3, 4
RGB_COLORS = {
    'red': np.array([240, 52, 52]),
    'black': np.array([0, 0, 0]),
    'green': np.array([77, 181, 33]),
    'blue': np.array([29, 111, 219]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([217, 213, 104]),
    'grey': np.array([192, 195, 196]),
    'light_grey': np.array([230, 230, 230]),
    'white': np.array([255, 255, 255])
}
four_room_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 4, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

human_policy = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 3, 3, 3],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3],
    [1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 3, 3, 3],
    [1, 2, 1, 0, 0, 0, 1, 2, 1, 1, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 3, 3, 3],
    [1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 3, 3, 3],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3],
    [1, 0, 0, 0, 0, 0, 1, 2, 2, 0, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
]


class FourRoomGridWorld:
    def __init__(self, stochasticity_fraction=0.0, random_jump_fraction=0.0, **kwargs):
        self._grid = np.transpose(np.flip(np.array(four_room_map, dtype=np.uint8), axis=0)[1:-1, 1:-1])
        self._max_row, self._max_col = self._grid.shape
        self._normal_tiles = np.where(self._grid == BLOCK_NORMAL)
        self._hallways_tiles = np.where(self._grid == BLOCK_HALLWAY)
        self._terminal = np.where(self._grid == BLOCK_TERMINAL)
        self._walls_tiles = np.where(self._grid == BLOCK_WALL)
        self.num_states = self._grid.size

        self._state = None
        self.ACTION_UP, self.ACTION_DOWN, self.ACTION_RIGHT, self.ACTION_LEFT = 0, 1, 2, 3
        self.num_actions = 4
        self._stochasticity_fraction = stochasticity_fraction
        self._random_jump_fraction = random_jump_fraction
        self.hallways = {
            0: (5, 1),
            1: (1, 5),
            2: (5, 8),
            3: (8, 4)
        }
        self._window, self._info = None, None

    def random_jump(self):
        if self._random_jump_fraction > np.random.uniform():
            target_tile_idx = np.random.choice(len(self._hallways_tiles[0]))
            x = self._hallways_tiles[0][target_tile_idx]
            y = self._hallways_tiles[1][target_tile_idx]
            self._state = (x, y)
            jumped = True
        else:
            jumped = False
        return jumped

    def reset(self):
        self._state = (np.random.choice(self._max_row), np.random.choice(self._max_col))
        x, y = self._state
        while self._grid[x,y]!=BLOCK_NORMAL:
            self._state = (np.random.choice(self._max_row), np.random.choice(self._max_col))
            x, y = self._state
        return x, y

    def step(self, action):
        x, y = self._state
        is_stochastic_selected = False
        if self._stochasticity_fraction > np.random.uniform():
            action_probability = [1 / self.num_actions for i in range(self.num_actions)]
            action = np.random.choice(self.num_actions, 1, p=action_probability)[0]
            is_stochastic_selected = True
        x_p, y_p = self._next(action, *self._state)
        is_done = self._grid[x_p, y_p] == BLOCK_TERMINAL
        reward = 1 if is_done else 0
        self._state = (x_p, y_p)
        jumped = self.random_jump()
        return x_p,y_p, reward, is_done, {
            'x': x, 'y': y,
            'x_p': x_p, 'y_p': y_p,
            'is_stochastic_selected': is_stochastic_selected,
            'jumped': jumped,
            'selected_action': action}

    def get_xy(self, state):
        return (state % self._max_row), (state // self._max_col)

    def get_state_index(self, x, y):
        return y * self._max_col + x

    def _next(self, action, x, y):

        def move(current_x, current_y, next_x, next_y):
            if next_y < 0 or next_x < 0:
                return current_x, current_y
            if next_y >= self._max_col or next_x >= self._max_row:
                return current_x, current_y
            if self._grid[next_x, next_y] == BLOCK_WALL:
                return current_x, current_y
            return next_x, next_y

        switcher = {
            self.ACTION_DOWN: lambda pox_x, pos_y: move(pox_x, pos_y, pox_x, pos_y - 1),
            self.ACTION_RIGHT: lambda pox_x, pos_y: move(pox_x, pos_y, pox_x + 1, pos_y),
            self.ACTION_UP: lambda pox_x, pos_y: move(pox_x, y, pox_x, pos_y + 1),
            self.ACTION_LEFT: lambda pox_x, pos_y: move(pox_x, pos_y, pox_x - 1, pos_y),
        }
        move_func = switcher.get(action)
        return move_func(x, y)

    def render(self, mode='human'):
        import sys
        from Environments.utils import colorize
        color = {
            BLOCK_NORMAL: lambda c: colorize(c, "white", highlight=True),
            BLOCK_WALL: lambda c: colorize(c, "gray", highlight=True),
            BLOCK_HALLWAY: lambda c: colorize(c, "green", highlight=True),
            BLOCK_TERMINAL: lambda c: colorize(c, "purple", highlight=True)
        }
        if mode == 'human':
            outfile = sys.stdout
            img = [
                [color[b]('  ')
                 for x, b
                 in enumerate(line)]
                for y, line in enumerate(four_room_map)]
            img[self._max_row - self._state[1]][self._state[0] + 1] = colorize('  ', "red",
                                                                                     highlight=True)
            for line in img:
                outfile.write(f'{"".join(line)}\n')
            outfile.write('\n')
        if mode == "rgb" or mode == "screen":
            x, y = self._state
            img = np.zeros((*self._grid.shape, 3), dtype=np.uint8)
            img[self._normal_tiles] = RGB_COLORS['light_grey']

            # if render_cls is not None:
            #     assert render_cls is not type(Render), "render_cls should be Render class"
            #     img = render_cls.render(img)

            img[self._walls_tiles] = RGB_COLORS['black']
            img[self._hallways_tiles] = RGB_COLORS['green']
            img[self._terminal] = RGB_COLORS['purple']
            img[x, y] = RGB_COLORS['red']

            ext_img = np.zeros((self._max_row + 2, self._max_col + 2, 3), dtype=np.uint8)
            ext_img[1:-1, 1:-1] = np.transpose(img, (1, 0, 2))
            if mode == "screen":

                from pyglet.window import Window
                from pyglet.text import Label
                from pyglet.gl import GLubyte
                from pyglet.image import ImageData
                zoom = 20
                if self._window is None:
                    self._window = Window((self._max_row + 2) * zoom, (self._max_col + 2) * zoom)
                    self._info = Label('Four Room Grid World', font_size=10, x=5, y=5)
                # self._info.text = f'x: {x}, y: {y}'
                dt = np.kron(ext_img, np.ones((zoom, zoom, 1)))
                dt = (GLubyte * dt.size)(*dt.flatten().astype('uint8'))
                texture = ImageData(self._window.width, self._window.height, 'RGB', dt).get_texture()
                self._window.clear()
                self._window.switch_to()
                self._window.dispatch_events()
                texture.blit(0, 0)
                # self._info.draw()
                self._window.flip()
            return np.flip(ext_img, axis=0)

def get_transition(random=0.08):
    transition = np.zeros((11*11*4,11*11*4)).astype(np.float32)
    env = FourRoomGridWorld(stochasticity_fraction=0.0)
    policy = np.transpose(np.flip(np.array(human_policy, dtype=np.uint8), axis=0)[1:-1, 1:-1])
    for x in range(11):
        for y in range(11):
            for a in range(4):
                if x==env._terminal[0] and y==env._terminal[1]:
                    idx = np.ravel_multi_index((env._terminal[0], env._terminal[1], a), (11, 11, 4))
                    transition[idx, idx] = 1
                    continue
                x_p, y_p = env._next(a, *(x,y))
                a_p = policy[x_p,y_p]
                idx = np.ravel_multi_index((x, y, a), (11, 11, 4))
                next_idx = np.ravel_multi_index((x_p,y_p,a_p), (11,11,4))
                transition[idx,next_idx]=1-random
                for a_p in range(4):
                    next_idx = np.ravel_multi_index((x_p, y_p, a_p), (11, 11, 4))
                    transition[idx, next_idx] +=random/4
    return transition

def get_reward_idx():
    env = FourRoomGridWorld(stochasticity_fraction=0.0)
    idx = np.ravel_multi_index((env._terminal[0], env._terminal[1]-1, 0), (11, 11, 4))
    return idx

def get_terminal():
    env = FourRoomGridWorld(stochasticity_fraction=0.0)
    return env._terminal

def get_reward():
    reward = np.zeros(11*11*4)
    env = FourRoomGridWorld(stochasticity_fraction=0.0)
    idx = np.ravel_multi_index((env._terminal[0], env._terminal[1]-1, 0), (11, 11, 4))
    reward[idx]=1
    idx = np.ravel_multi_index((env._terminal[0], env._terminal[1] + 1, 1), (11, 11, 4))
    reward[idx] = 1
    # idx = np.ravel_multi_index((1,4,0), (11, 11, 4))
    # reward[idx] = -1
    # idx = np.ravel_multi_index((1,6,1), (11, 11, 4))
    # reward[idx] = -1
    # idx = np.ravel_multi_index((4,1,2), (11, 11, 4))
    # reward[idx] = -1
    # idx = np.ravel_multi_index((6, 1, 3), (11, 11, 4))
    # reward[idx] = -1
    # idx = np.ravel_multi_index((4, 8, 2), (11, 11, 4))
    # reward[idx] = -1
    # idx = np.ravel_multi_index((6, 8, 3), (11, 11, 4))
    # reward[idx] = -1
    return reward

if __name__ == '__main__':
    mode = 'human'
    mode = 'screen'
    stochasticity_fraction = 1.0

    state_x = []
    state_y = []
    reward = []
    action = []
    next_state_x =[]
    next_state_y = []
    next_action=[]
    env = FourRoomGridWorld(stochasticity_fraction=0.0)
    policy = np.transpose(np.flip(np.array(human_policy, dtype=np.uint8), axis=0)[1:-1, 1:-1])
    x,y = env.reset()
    # a = np.random.randint(0, 4)
    a = policy[x, y]
    if stochasticity_fraction > np.random.uniform():
        action_probability = [1/4 for i in range(4)]
        a = np.random.choice(4, 1, p=action_probability)[0]
    step=0
    done= False
    while step <2500 or not done:
        if done:
            x,y=env.reset()
            # a = np.random.randint(0, 4)
            a = policy[x, y]
            if stochasticity_fraction > np.random.uniform():
                action_probability = [1/4 for i in range(4)]
                a = np.random.choice(4, 1, p=action_probability)[0]
            print('env reset')
            done=False
            continue
        step += 1
        state_x.append(x)
        state_y.append(y)
        action.append(a)
        x ,y, r, done, _ = env.step(action=a)
        # a = np.random.randint(0, 4)
        a = policy[x, y]
        if stochasticity_fraction > np.random.uniform():
            action_probability = [1/4 for i in range(4)]
            a = np.random.choice(4, 1, p=action_probability)[0]
        reward.append(r)
        next_state_x.append(x)
        next_state_y.append(y)
        next_action.append(a)
        env.render(mode=mode)

        di = {'state_x':state_x,'state_y':state_y,'reward':reward,"action":action,
                'next_state_x':next_state_x,'next_state_y':next_state_y,'next_action':next_action}
        with open('./four_room_data/random.pkl', 'wb') as f:
            pickle.dump(di, f)
