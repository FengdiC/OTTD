import numpy as np
import matplotlib.pyplot as plt
import pickle
from four_room import get_transition,get_reward,get_reward_idx,get_terminal,FourRoomGridWorld

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 12.0
    plt.rcParams['ytick.labelsize'] = 13.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['ytick.minor.pad'] = 50.0


def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
                   labelsize=12, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=12, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(getxticklabelsize())

gamma = 0.95
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
policy = np.transpose(np.flip(np.array(human_policy, dtype=np.uint8), axis=0)[1:-1, 1:-1])
full_transition = get_transition(random=0.08)
behaviour_transition = get_transition(random=1.0)
full_reward = get_reward()
# print(Phi[100])
# print(np.unravel_index(100,(11,11,4)))
def compute_IS(random=0.8):
    IS = np.zeros(11*11*4)
    for x in range(11):
        for y in range(11):
            a_human = policy[x, y]
            for a in range(4):
                idx = np.ravel_multi_index((x,y, a), (11, 11, 4))
                if a==a_human:
                     IS[idx]= 0.94/((1-random)+random/4)
                else:
                    IS[idx] = 0.08 / random
    return np.diag(IS),IS

def sample_data(number=300):
    stochasticity_fraction = 1.0

    state_x = []
    state_y = []
    reward = []
    action = []
    next_state_x = []
    next_state_y = []
    next_action = []
    env = FourRoomGridWorld(stochasticity_fraction=0.0)
    policy = np.transpose(np.flip(np.array(human_policy, dtype=np.uint8), axis=0)[1:-1, 1:-1])
    x, y = env.reset()
    # a = np.random.randint(0, 4)
    a = policy[x, y]
    if stochasticity_fraction > np.random.uniform():
        action_probability = [1 / 4 for i in range(4)]
        a = np.random.choice(4, 1, p=action_probability)[0]
    step = 0
    done = False
    while step < number or not done:
        if done:
            x, y = env.reset()
            # a = np.random.randint(0, 4)
            a = policy[x, y]
            if stochasticity_fraction > np.random.uniform():
                action_probability = [1 / 4 for i in range(4)]
                a = np.random.choice(4, 1, p=action_probability)[0]
            print('env reset')
            done = False
            continue
        step += 1
        state_x.append(x)
        state_y.append(y)
        action.append(a)
        x, y, r, done, _ = env.step(action=a)
        # a = np.random.randint(0, 4)
        a = policy[x, y]
        if stochasticity_fraction > np.random.uniform():
            action_probability = [1 / 4 for i in range(4)]
            a = np.random.choice(4, 1, p=action_probability)[0]
        reward.append(r)
        next_state_x.append(x)
        next_state_y.append(y)
        next_action.append(a)

    di = {'state_x': state_x, 'state_y': state_y, 'reward': reward, "action": action,
          'next_state_x': next_state_x, 'next_state_y': next_state_y, 'next_action': next_action}
    return di

class TD():
    def __init__(self,number):
        self.number =number

    def load_over_four_room(self):
        Phi = []
        env = FourRoomGridWorld(stochasticity_fraction=0.0)
        for x in range(11):
            for y in range(11):
                for a in range(4):
                    feature = np.zeros(26)
                    feature[x] = 1
                    feature[11 + y] = 1
                    feature[22 + a] = 1
                    # feature[-2] = float(x)/11
                    # feature[-1] = float(y)/ 11
                    # for action in range(4):
                    #     x_p,y_p = env._next(action,x,y)
                    #     feature[(action+1)*26+x_p]=1
                    #     feature[(action + 1) * 26 + 11+y_p] = 1
                    #     feature[(action + 1) * 26 + 22+ action] = 1
                    Phi.append(feature)
        Phi = np.array(Phi).astype(np.float32)

        # random consists of 2500 transitions
        di = sample_data(self.number)
        IS, _ = compute_IS(random=1.0)

        grid=np.zeros((11,11,4))
        grid_reward = np.zeros((11, 11, 4))
        transition = np.zeros((11*11*4,11*11*4))
        transition_target = np.zeros((11 * 11 * 4, 11 * 11 * 4))
        W = np.zeros((11 * 11 * 4, 11 * 11 * 4))
        for i in range(len(di['state_x'])):
            x = di['state_x'][i]
            y = di['state_y'][i]
            a = di['action'][i]
            grid[x,y,a]+=1
            grid_reward[x,y,a] +=di['reward'][i]
            idx = np.ravel_multi_index((x,y,a), (11,11,4))
            x_p = di['next_state_x'][i]
            y_p = di['next_state_y'][i]
            a_p = di['next_action'][i]

            next_idx = np.ravel_multi_index((x_p,y_p,a_p), (11,11,4))
            W[idx,next_idx] += IS[next_idx,next_idx]
            transition[idx,next_idx]+=1

            a = policy[x, y]
            if 0.08 > np.random.uniform():
                action_probability = [1 / 4 for i in range(4)]
                a = np.random.choice(4, 1, p=action_probability)[0]
            next_idx_target = np.ravel_multi_index((x_p, y_p, a), (11, 11, 4))
            transition_target[idx,next_idx_target]+=1

            if x_p==get_terminal()[0] and y_p==get_terminal()[1]:
                grid[x_p, y_p, a_p] += 1
                transition[next_idx, next_idx] += 1
                W[next_idx, next_idx] += IS[next_idx, next_idx]
                transition_target[next_idx, next_idx_target]+=1
        show_up = grid.ravel() > 0
        H = np.eye(11 * 11 * 4)[show_up, :]
        # plt.imshow(np.sum(grid,axis=2))
        # plt.show()

        R = grid_reward.ravel()/ (grid.ravel()+0.01)
        R = H@R

        grid = grid.ravel()/np.sum(grid)

        transition = H@transition
        transition = transition / np.sum(transition, axis=1)[:, np.newaxis]

        transition_target = H@transition_target
        transition_target = transition_target/np.sum(transition_target, axis=1)[:, np.newaxis]

        W = H@W
        W = W / np.expand_dims(np.sum(W, axis=1), axis=1)

        print("WIS corr: ", np.sum(np.abs(H @ full_transition - W)))
        print("sampling: ", np.sum(np.abs(H @ full_transition - transition_target)))
        # print(np.max(np.abs(np.linalg.eig(transition@IS@H.T)[0])))

        D_k = np.diag(H@grid)
        Phi = np.concatenate((Phi,H.T+np.random.normal(0,0.01,(H.shape[1],H.shape[0]))), axis=1)
        M = H@Phi
        ########### NO IS CORRECTION #############
        N_off = transition @ Phi
        N = W@Phi
        N_target = transition_target@Phi

        N_IS = transition @ IS@Phi
        return D_k,M,N,H,R,None,Phi, N_off, N_IS,N_target

    def compute_D(self):
        n = full_transition.shape[0]
        d = (1-gamma)*np.linalg.inv(np.eye(n) - gamma * full_transition.T) @ np.ones(n)/n
        print(np.sum(d))
        return np.diag(d)

    # # TEST (8,7,3) Action 3--LEFT
    # idx = np.ravel_multi_index((8,7,3), (11,11,4))
    # feat_test = Phi[idx]
    # coef = feat_test@M.T@np.linalg.inv(M@M.T)@H
    # coef = coef.reshape((11,11,4))
    # plt.imshow(np.max(coef,axis=2))
    # plt.show()

    # print(np.linalg.eig(np.eye(M.shape[1]) - 0.5*M.T@D_k@(M-gamma*N))[0])
    # print(np.linalg.eig(np.eye(M.shape[1]) - 0.5*(M-gamma*N).T@D_k@(M-gamma*N))[0])

    def target_q(self):
        q=np.linalg.inv(np.eye(11*11*4)-gamma*full_transition) @ full_reward
        q_behaviour =np.linalg.inv(np.eye(11*11*4)-gamma*behaviour_transition) @ full_reward
        grid_q = np.max(np.reshape(q,(11,11,4)),axis=2)
        # print(grid_q)
        # plt.imshow(grid_q)
        # plt.show()
        return q,q_behaviour

    def train(self,agent,eta=0.5,beta=0.001,checkpoint=3):
        D_k, M, N, H, R, q_hat, Phi, N_off, N_IS, N_target = self.load_over_four_room()
        print(M.shape)
        steps = 0
        train_err=[]
        full_err=[]
        values=[]
        idx = get_reward_idx()
        q, q_b= self.target_q()
        D = self.compute_D()
        theat_star = np.linalg.inv(Phi.T@D@Phi+0.01*np.eye(Phi.shape[1]))@Phi.T@D@q
        print("ERRPR: ",(q-Phi@theat_star).T@D@(q-Phi@theat_star))
        theta = 0.05*np.ones(M.shape[1]).astype(np.float32)
        w = np.copy(theta)
        theta_target = np.copy(theta)
        values.append(Phi[idx]@theta)
        while steps<30000:
            steps+=1
            delta = (M-gamma * N) @ theta-R
            if agent=='WIS':
                theta -= eta * np.transpose(M)@ D_k@delta
                train_err.append(delta.T @ D_k @ delta)
            elif agent == 'no correction':
                delta = (M-gamma * N_off) @ theta-R
                theta -= eta * np.transpose(M) @ D_k @ delta
                train_err.append(delta.T @ D_k @ delta)
            elif agent == 'IS correction':
                delta = (M-gamma * N_IS) @ theta-R
                theta -= eta * np.transpose(M) @ D_k @ delta
                train_err.append(delta.T @ D_k @ delta)
            elif agent == 'target sample':
                delta = (M-gamma * N_target) @ theta-R
                theta -= eta * np.transpose(M) @ D_k @ delta
                train_err.append(delta.T @ D_k @ delta)
            elif agent=='target_TD':
                if steps % checkpoint == 0:
                    theta_target = np.copy(theta)
                theta -= eta *np.transpose(M) @ D_k @ M @ theta -eta * np.transpose(M) @D_k @((gamma * N) @theta_target+R)
            elif agent=='RM':
                theta -= eta * np.transpose(M - gamma * N) @ D_k @ delta
            elif agent=='TDC':
                # theta += eta * (M - gamma  * N).T @ D_k @ M @ w
                theta -= eta * M.T@ D_k @ delta + eta* gamma *N.T@D_k@M@w
                w -= eta * M.T @ D_k @ (delta + M @ w)
            elif agent =='GTD2':
                theta += eta * (M - gamma * N).T @ D_k @ M @ w
                # theta -= eta * M.T@ D_k @(M - gamma * N)@ theta + eta* gamma *N.T@D_k@M@w
                w -= beta * M.T @ D_k @ (delta + M @ w)
            elif agent=='Baird_RM':
                RM_delta = eta * np.transpose(M - gamma * N) @ D_k @ delta
                TD_delta = eta * np.transpose(M) @ D_k @ delta
                if np.dot(RM_delta, TD_delta) - np.dot(RM_delta, RM_delta) == 0:
                    phi = 0
                else:
                    phi = np.clip(np.dot(RM_delta, TD_delta) / (np.dot(RM_delta, TD_delta) - np.dot(RM_delta, RM_delta)), 0,
                                  1)
                theta -= (1 - phi) * TD_delta + phi * RM_delta
            else:
                print("Unknown Agent!!!")
            full_err.append(np.max(np.abs(q-Phi@theta)))
            values.append(Phi[idx] @ theta)
        # plt.plot(range(len(parameters)),parameters)
        # plt.plot(range(len(values)), values)
        # plt.show()
        return values,train_err,full_err

    def plot_MSVE(self,lr=0.001):
        setsizes()
        # agent = 'no correction'
        # eta = 0.97
        # train_err, full_err, values = train(agent, eta=eta)
        # # values = values[50:]
        # print(values[-1])
        # plt.plot(range(len(values)), values, 'b',label='Off-Policy')

        agent = 'IS correction'
        eta = lr
        results = []
        for seed in range(10):
            np.random.seed(seed)
            v, train_err, values = self.train(agent, eta=eta)
            # values = values[50:]
            print(values[-1])
            results.append(values)

        results = np.array(results)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0) / np.sqrt(10)
        plt.plot(range(len(mean)), mean , 'tab:red', label='IS')
        plt.fill_between(range(len(mean)), mean + std, mean - std, color='tab:red', alpha=0.2, linewidth=0.9)

        agent = 'WIS'
        eta = 0.97
        results = []
        for seed in range(10):
            np.random.seed(seed)
            v, train_err, values = self.train(agent, eta=eta)
            # values = values[50:]
            print(values[-1])
            results.append(values)
        results = np.array(results)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0) / np.sqrt(10)
        plt.plot(range(len(mean)), mean,'tab:orange', label='NIS')
        plt.fill_between(range(len(mean)), mean + std, mean - std, color='tab:orange', alpha=0.2, linewidth=0.9)

        agent = 'target sample'
        eta = 0.97
        results = []
        for seed in range(10):
            np.random.seed(seed)
            v, train_err, values = self.train(agent, eta=eta)
            # values = values[50:]
            print(values[-1])
            results.append(values)

        results = np.array(results)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0) / np.sqrt(10)
        plt.plot(range(len(mean)), mean, 'g', label='Target Actions')
        plt.fill_between(range(len(mean)), mean + std, mean - std, color='g', alpha=0.2, linewidth=0.9)

        # plt.yscale('log')
        setaxes()
        plt.ylabel("MAX-VE", fontsize=14)
        plt.ylim((0.6, 1.2))
        plt.xlabel("Training Steps", fontsize=14)
        plt.xticks(rotation=25)

    def plot_EMSBE(self):
        setsizes()
        # agent = 'no correction'
        # eta = 0.97
        # v, train_err, values = train(agent, eta=eta)
        # # values = values[50:]
        # print(values[-1])
        # plt.plot(range(len(train_err)), train_err, 'b',label='Off-Policy')

        agent = 'IS correction'
        eta = 0.1
        results = []
        for seed in range(10):
            np.random.seed(seed)
            v, train_err, values = self.train(agent, eta=eta)
            # values = values[50:]
            print(values[-1])
            results.append(train_err)

        results = np.array(results)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0) / np.sqrt(10)
        plt.plot(range(len(mean)), mean, 'tab:red', label='IS')
        plt.fill_between(range(len(mean)), mean + std, mean - std, color='tab:red', alpha=0.2, linewidth=0.9)

        agent = 'WIS'
        eta = 0.97
        results = []
        for seed in range(10):
            np.random.seed(seed)
            v, train_err, values = self.train(agent, eta=eta)
            # values = values[50:]
            print(values[-1])
            results.append(train_err)
        results = np.array(results)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0) / np.sqrt(10)
        plt.plot(range(len(mean)), mean, 'tab:orange', label='NIS')
        plt.fill_between(range(len(mean)), mean + std, mean - std, color='tab:orange', alpha=0.2, linewidth=0.9)

        agent = 'target sample'
        eta = 0.97
        results = []
        for seed in range(10):
            np.random.seed(seed)
            v, train_err, values = self.train(agent, eta=eta)
            # values = values[50:]
            print(values[-1])
            results.append(train_err)

        results = np.array(results)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0) / np.sqrt(10)
        plt.plot(range(len(mean)), mean, 'g', label='Target Actions')
        plt.fill_between(range(len(mean)), mean + std, mean - std, color='g', alpha=0.2, linewidth=0.9)

        # plt.yscale('log')
        setaxes()
        plt.ylabel("EMSBE", fontsize=14)
        plt.ylim((-0.0002, 0.005))
        plt.xlabel("Training Steps", fontsize=14)
        plt.xticks(rotation=25)

file = './four_room_data/random_300.pkl'
TD300 = TD(300)
plt.figure(figsize=(15, 4.2))
setsizes()
plt.gcf().subplots_adjust(bottom=0.15)
plt.subplot(131)
plt.tight_layout()
# plt.title('Empirical Mean Squared Bellman Error',fontsize=12)
TD300.plot_EMSBE()
plt.legend(fontsize=11)

plt.subplot(132)
plt.tight_layout()
TD300.plot_MSVE()
# plt.title('Max Norm of Value Estimation Error', fontsize=12)


file = './four_room_data/random.pkl'
TD1250 = TD(1250)
plt.subplot(133)
plt.tight_layout()
TD1250.plot_MSVE(0.002)



# plt.figure(figsize=(12,4))
# plt.title('Max Norm of Value Estimation Error',fontsize=12)
# plt.subplot(131)
# file = './four_room_data/random_300.pkl'
# TD(file)
# plt.subplot(132)
# file = './four_room_data/random_1250.pkl'
# TD(file)
# plt.subplot(133)
# file = './four_room_data/random.pkl'
# TD(file)
# plt.legend(fontsize=10)
plt.show()
