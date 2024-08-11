import numpy as np
import matplotlib.pyplot as plt

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3

    plt.rcParams['xtick.labelsize'] = 12.0
    plt.rcParams['ytick.labelsize'] = 12.0
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

#BAIRD COUNTEREXAMPLE DATA

M = np.array([[1,2,0,0,0,0,0,0],[1,0,2,0,0,0,0,0],[1,0,0,2,0,0,0,0],[1,0,0,0,2,0,0,0],
              [1,0,0,0,0,2,0,0],[1,0,0,0,0,0,2,0],[2,0,0,0,0,0,0,1]])
N = np.array([[2,0,0,0,0,0,0,1],[2,0,0,0,0,0,0,1],[2,0,0,0,0,0,0,1],[2,0,0,0,0,0,0,1],
              [2,0,0,0,0,0,0,1],[2,0,0,0,0,0,0,1],[2,0,0,0,0,0,0,1]])
D_k = 1/7 * np.eye(7)
gamma = 0.95

def TD(eta=0.5,alpha=0.8):
    steps = 0
    parameters = []
    values = []
    theta = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    theta_targ = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    parameters.append(theta[0])
    # values.append((part_M @ theta).T @ D_k @ (part_M @ theta))
    values.append((M @ theta).T @ D_k @ (M @ theta))
    # N = 0.5*P @ M
    # N = N[:-1,:]
    while steps<50000:
        steps+=1
        # theta -= eta * np.transpose(M)@ D_k@(M-gamma * N) @ theta
        # theta -= eta * np.transpose(part_M) @ D_k @ part_M @ theta - eta * np.transpose(part_M) @ D_k @ (gamma * N) @ theta_targ
        theta -= eta * np.transpose(M) @ D_k @ M @ theta - eta * np.transpose(M) @ D_k @ ( gamma * N) @ theta_targ
        theta_targ = (1-alpha)*theta_targ+alpha*theta
        parameters.append(theta[0])
        # values.append((part_M@theta).T@D_k@(part_M@theta))
        values.append((M @ theta).T @ D_k @ (M @ theta))
    # plt.plot(range(len(parameters)),parameters)
    # plt.plot(range(len(values)), values)
    # plt.show()
    return parameters, values

def TD_target(eta=0.5,checkpoint=10):
    steps = 0
    parameters = []
    values= []
    theta = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    theta_target =np.copy(theta)
    alpha=1
    parameters.append(theta[0])
    values.append((M@theta).T@D_k@(M@theta))
    while steps<50000:
        if steps%checkpoint==0:
            theta_target = (1-alpha)*theta_target + alpha*np.copy(theta)
        steps+=1
        theta -= eta * np.transpose(M)@ D_k@M @ theta - eta *np.transpose(M)@ D_k@ (gamma*N) @ theta_target
        parameters.append(theta[0])
        values.append((M@theta).T@D_k@(M@theta))
    # plt.plot(range(len(parameters)),parameters)
    # plt.plot(range(len(values)), values)
    # plt.show()
    return parameters, values

def RM(eta=0.5):
    steps = 0
    parameters = []
    values = []
    theta = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    parameters.append(theta[0])
    values.append((M@theta).T@D_k@(M@theta))
    while steps<50000:
        steps+=1
        theta -= eta * np.transpose(M-gamma * N)@ D_k@(M-gamma * N) @ theta
        parameters.append(theta[0])
        values.append((M@theta).T@D_k@(M@theta))
    # plt.plot(range(len(parameters)),parameters)
    # plt.plot(range(len(values)), values)
    # plt.show()
    return parameters, values

def GTD(eta=0.5,alpha=0.5):
    steps = 0
    parameters = []
    values = []
    theta = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    w = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    parameters.append(theta[0])
    values.append((M@theta).T@D_k@(M@theta))
    while steps<50000:
        steps+=1
        theta += eta *(M-gamma * N).T@ D_k@M @w
        # theta -= eta * M.T@ D_k @(M - gamma * N)@ theta + eta* gamma *N.T@D_k@M@w
        w -= alpha * M.T @ D_k @ (M @ theta - gamma * N @ theta + M @ w)
        parameters.append(theta[0])
        values.append((M@theta).T@D_k@(M@theta))
    # plt.plot(range(len(parameters)),parameters)
    # plt.plot(range(len(values)), values)
    # plt.show()
    return parameters, values

def TDC(eta=0.5,alpha=0.5):
    steps = 0
    parameters = []
    values = []
    theta = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    w = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    parameters.append(theta[0])
    values.append((M@theta).T@D_k@(M@theta))
    while steps<50000:
        steps+=1
        # theta += eta *(M-gamma * N).T@ D_k@M @w
        theta -= eta * M.T@ D_k @(M - gamma * N)@ theta + eta* gamma *N.T@D_k@M@w
        w -= alpha * M.T @ D_k @ (M @ theta - gamma * N @ theta + M @ w)
        parameters.append(theta[0])
        values.append((M@theta).T@D_k@(M@theta))
    # plt.plot(range(len(parameters)),parameters)
    # plt.plot(range(len(values)), values)
    # plt.show()
    return parameters, values

def Baird_RM(eta=0.5):
    steps = 0
    parameters = []
    values = []
    theta = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    parameters.append(theta[0])
    values.append((M@theta).T@D_k@(M@theta))
    while steps<50000:
        steps+=1
        RM_delta =  eta * np.transpose(M-gamma * N)@ D_k@(M-gamma * N) @ theta
        TD_delta = eta * np.transpose(M) @ D_k @ (M - gamma * N) @ theta
        if np.dot(RM_delta,TD_delta)-np.dot(RM_delta,RM_delta)==0:
            phi=0
        else:
            phi = np.clip(np.dot(RM_delta,TD_delta)/ (np.dot(RM_delta,TD_delta)-np.dot(RM_delta,RM_delta)),0,1)
        theta -= (1-phi)*TD_delta+phi*RM_delta
        parameters.append(theta[0])
        values.append((M@theta).T@D_k@(M@theta))
    # plt.plot(range(len(parameters)),parameters)
    # plt.plot(range(len(values)), values)
    # plt.show()
    return parameters, values

def RM_target(eta=0.1):
    steps = 0
    parameters = []
    values = []
    theta = np.array([1, 1, 1, 1, 1, 1, 1, 10]).astype(np.float32)
    parameters.append(theta[0])
    values.append(np.max(np.abs(M@theta)))
    theta_target = theta
    checkpoint = 3
    alpha = 1
    while steps<50000:
        if steps % checkpoint == 0:
            theta_target = (1 - alpha) * theta_target + alpha * theta
        steps += 1
        theta -= eta * np.transpose(M - gamma * N) @ D_k @ (M - gamma * N) @ theta
        parameters.append(theta[0])
        values.append(np.max(np.abs(M@theta)))
    # plt.plot(range(len(parameters)),parameters)
    # plt.plot(range(len(values)), values)
    # plt.show()
    return parameters, values

def tune_hyper_param():
    for eta in [0.01]:
        for alpha in [0.3]:
            parameters, values = TD(eta=eta,alpha=alpha)
            plt.plot(range(len(values)), values, label=str(eta)+'-'+str(alpha))
            parameters, values = TD_target(eta=0.01)
            plt.plot(range(len(values)), values, label=str(eta) + 'target' + str(alpha))
    plt.ylabel("Value Function V(7)")
    plt.xlabel("Training Steps")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# tune_hyper_param()

plt.figure(figsize=(12,6))
setsizes()
plt.subplot(231)
parameters, values = TD()
# plt.plot(range(len(parameters)),parameters)
plt.plot(range(len(values)), values,label='TD')
plt.yscale('log')
plt.ylabel("MAX-VE", fontsize=14)
plt.xlabel("Training Steps", fontsize=14)
plt.xscale('log')
setaxes()
plt.xticks(rotation=45)
plt.title('TD', fontsize=15)

plt.subplot(232)
parameters, values = TD_target(eta=0.999,checkpoint=3)
# plt.plot(range(len(parameters)),parameters)
plt.plot(range(len(values)), values,label='TD target')
plt.yscale('symlog')
plt.ylim((-0.4,1000))
plt.ylabel("MAX-VE", fontsize=14)
plt.xlabel("Training Steps", fontsize=14)
plt.xscale('log')
setaxes()
plt.xticks(rotation=45)
plt.title('TD target', fontsize=15)

plt.subplot(233)
parameters, values = RM(eta=0.8)
# plt.plot(range(len(parameters)),parameters)
plt.plot(range(len(values)), values,label='RM')
plt.yscale('symlog')
plt.ylim((-0.4,1000))
plt.ylabel("MAX-VE", fontsize=14)
plt.xlabel("Training Steps", fontsize=14)
plt.xscale('log')
setaxes()
plt.xticks(rotation=45)
plt.title('RM', fontsize=15)

plt.subplot(234)
parameters, values = Baird_RM(eta=0.95)
# plt.plot(range(len(parameters)),parameters)
plt.plot(range(len(values)), values,label='RM')
plt.yscale('symlog')
plt.ylim((-0.4,1000))
plt.ylabel("MAX-VE", fontsize=14)
plt.xlabel("Training Steps", fontsize=14)
plt.xscale('log')
setaxes()
plt.xticks(rotation=45)
plt.title('Baird RM', fontsize=15)

plt.subplot(235)
parameters, values = GTD(eta=0.6,alpha=0.6)
# plt.plot(range(len(parameters)),parameters)
plt.plot(range(len(values)), values,label='GTD')
plt.yscale('symlog')
plt.ylim((-0.4,1000))
plt.ylabel("MAX-VE", fontsize=14)
plt.xlabel("Training Steps", fontsize=14)
plt.xscale('log')
setaxes()
plt.xticks(rotation=45)
plt.title('GTD2', fontsize=15)

plt.subplot(236)
parameters, values = TDC(eta=0.6,alpha=0.4)
plt.yscale('symlog')
plt.ylim((-0.4,1000))
# plt.plot(range(len(parameters)),parameters)
plt.plot(range(len(values)), values,label='TDC')
plt.ylabel("MAX-VE", fontsize=14)
plt.xlabel("Training Steps", fontsize=14)
plt.xscale('log')
setaxes()
plt.xticks(rotation=45)
plt.title('TDC', fontsize=15)

plt.tight_layout()
plt.show()
