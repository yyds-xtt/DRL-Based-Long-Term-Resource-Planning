import os
import random
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from Environment import *
import copy

class MultiAgentReplayBuffer:  # 每个actor有partial state； 而中央控制器有全部state
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.actor_dims = actor_dims
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.action_memory = []

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))

    def store_tansition(self, raw_obs, state, action, reward, raw_obs_, state_):
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]

        if self.mem_cntr <= self.mem_size:
            self.action_memory.append(action)
        else:
            self.action_memory[index] = action
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self):
        x = random.randrange(0, self.mem_size - self.batch_size)

        states = self.state_memory[x: x + self.batch_size]
        states_ = self.new_state_memory[x: x + self.batch_size]
        rewards = self.reward_memory[x: x + self.batch_size]
        actions = self.action_memory[x: x + self.batch_size]

        actor_states = []
        actor_new_states = []

        for agent_index in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_index][x: x + self.batch_size]) # 从第i个agent的50000个选300个
            actor_new_states.append(self.actor_new_state_memory[agent_index][x: x + self.batch_size])

        return actor_states, states, actions, rewards, actor_new_states, states_

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name,
                 chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        state = state.view(-1, 40)
        action = action.view(-1, 16)
        x = F.relu(self.fc1(T.cat((state, action), dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def init(self):
        self.fc1.weight.data = T.abs(0.01 * T.randn(self.fc1.weight.size()))
        self.fc2.weight.data = T.abs(0.01 * T.randn(self.fc2.weight.size()))
        self.q.weight.data = T.abs(0.01 * T.randn(self.q.weight.size()))

        self.fc1.bias.data = (T.zeros(self.fc1.bias.size()) * 0.01* -1/3)
        self.fc2.bias.data = (T.zeros(self.fc2.bias.size()) * 0.01* -1/3)
        self.q.bias.data = (T.zeros(self.q.bias.size()) * 0.01* -1/3)

    def save_checkpoint(self):
        # print('...saving checkpoint...')
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        # print('...loading checkpoint...')
        self.load_state_dict(T.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)  # 输出的大小是n_actions

        self.optimizerr = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        state = state.view(-1, 7)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.tanh(self.pi(x))  # 视频50分50秒对此有解释

        return pi

    def init(self):
        self.fc1.weight.data = T.abs(0.01 * T.randn(self.fc1.weight.size()))
        self.fc2.weight.data = T.abs(0.01 * T.randn(self.fc2.weight.size()))
        self.pi.weight.data = T.abs(0.01 * T.randn(self.pi.weight.size()))

        self.fc1.bias.data = (T.zeros(self.fc1.bias.size()) * 0.01 * -1/3)
        self.fc2.bias.data = (T.zeros(self.fc2.bias.size()) * 0.01* -1/3)
        self.pi.bias.data = (T.zeros(self.pi.bias.size()) * 0.01* -1/3)

    def save_checkpoint(self):
        # print('...saving checkpoint...')
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        # print('...loading checkpoint...')
        self.load_state_dict(T.load(self.chkpt_file))

class Agent:  # 代表了每一个agent
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir, n_agents,
                 alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx  # 百分号 agent_idx会替换s
        self.actor = ActorNetwork(alpha, actor_dims, 32, 16, 2,
                                  chkpt_dir=chkpt_dir, name=self.agent_name + '_actor')
        self.actor.init()
        self.critic = CriticNetwork(beta, critic_dims, 64, 32, n_agents, n_actions,
                                    chkpt_dir=chkpt_dir, name=self.agent_name + '_critic')
        self.critic.init()
        self.target_actor = ActorNetwork(alpha, actor_dims, 32, 16, 2,
                                         chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor')
        self.target_actor.init()
        self.target_critic = CriticNetwork(beta, critic_dims, 64, 32, n_agents, n_actions,
                                           chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic')
        self.target_critic.init()
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)  # type: ignore

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)  # type: ignore

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float)
        action = self.actor.forward(state)
        return action.detach().cpu().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, chkpt_dir):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, agent_idx,
                                     chkpt_dir=chkpt_dir, n_agents=n_agents))

    def initial(self, actor_dims, critic_dims, chkpt_dir):
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, agent_idx,
                                     chkpt_dir=chkpt_dir, n_agents=n_agents))
        for i in self.agents:
            i.actor.init()
            i.critic.init()
            i.target_actor.init()
            i.target_critic.init()

    def save_checkpoint(self):
        print('...saving checkpoint...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('...loading checkpoint...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):  # 共有8个obs
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def obs_list_to_state_vector(self, observation):  # 这个是把所有的和在一起
        state = np.array([])
        for obs in observation:
            state = np.concatenate([state, obs])
        return state

def random_pick(some_list, probabilities):
    item = 0
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

fc1 = 64
fc2 = 64
alpha = 1e-4
beta = 1e-4
chkpt_dir = ''
n_agents = 8
actor_dims = [7,7,7,7,7,7,7,7]  # 每个7 代表两个邻居 + 4个请求信息  只能看到邻居就是partial的
critic_dims = 8 + 4 * 8  # 总共8个节点，每个节点有五个请求信息
n_actions = 2
INITIAL_EPSILON = 0.6
FINAL_EPSILON = 0.001
epsilon = copy.deepcopy(INITIAL_EPSILON)

maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, chkpt_dir)
memory = MultiAgentReplayBuffer(1000, critic_dims, actor_dims, n_actions, n_agents, batch_size=50)


timer = 0
Observe = xx
Explore = xx
Train = xx

Offload = stepgo(301, 10)
Offload.NetStateUpdate(0)
[computing, agents_request] = copy.deepcopy(Offload.pickState())
state = Offload.uniform([computing, agents_request])
state = state[0:8] + state[9:13] + state[14:18] + state[19:23] + state[24:28] + state[29:33] + state[34:38] + state[39:43] + state[44:48]
obs = Offload.getuniobs(computing, agents_request)
rewarddd = []

def random_pick(some_list, probabilities):
    item = 0
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

while timer < 8 * Observe:
    actions = []
    ii = 0
    action_index = []
    reward = []
    statex = None
    while ii < 8:
        action = list(np.zeros(20))
        a = random.randrange(-100, 100) / 100
        b = random.randrange(-100, 100) / 100
        actions.append([a, b])
        i = 0
        j = 0
        if a >= 0:
            j = 1
        if a < 0:
            j = 0
        if -1 < b <= -0.8:
            i = 1
        if -0.8 < b <= -0.6:
            i = 2
        if -0.6 < b <= -0.4:
            i = 3
        if -0.4 < b <= -0.2:
            i = 4
        if -0.2 < b <= 0:
            i = 5
        if 0 < b <= 0.2:
            i = 6
        if 0.2 < b <= 0.4:
            i = 7
        if 0.4 < b <= 0.6:
            i = 8
        if 0.6 < b <= 0.8:
            i = 9
        if 0.8 < b <= 1:
            i = 10

        action[(j * 10 + i) - 1] = 1
        action_idx = action.index(max(action))
        action_index.append(action_idx)
        statex, a = Offload.ActionToReward(action, timer, ii)

        reward.append(a * 10)
        ii += 1
        timer += 1
        Offload.NetStateUpdate(timer)

    computing_ = copy.deepcopy(Offload.compresource)
    rewardd = sum(reward) / 8
    if action_index.count(19) + action_index.count(18) >= 4:
        rewardd = -1

    agents_request_ = copy.deepcopy(Offload.requestinfo)
    state_ = Offload.uniform([computing_, agents_request_])
    state_ = state_[0:8] + state_[9:13] + state_[14:18] + state_[19:23] + state_[24:28] + state_[29:33] + state_[
                                                                                                          34:38] + state_[
                                                                                                                   39:43] + state_[
                                                                                                                            44:48]
    obs_ = Offload.getuniobs(computing_, agents_request_)

    memory.store_tansition(obs, state, actions, rewardd, obs_, state_)

    computing = copy.deepcopy(Offload.compresource)
    agents_request = copy.deepcopy(Offload.requestinfo)
    state = Offload.uniform([computing, agents_request])
    state = state[0:8] + state[9:13] + state[14:18] + state[19:23] + state[24:28] + state[29:33] + state[34:38] + state[
                                                                                                                  39:43] + state[
                                                                                                                           44:48]
    obs = Offload.getuniobs(computing, agents_request)
    timer_real = timer / 8
    if timer_real <= Observe:
        process = 'observe'
    else:
        process = 'train'

    if timer_real % 50 == 0:
        sss = 'time_step {}/ process {}/ action {}/ reward {}/'.format(timer_real, process, action_index, rewardd)
        print(sss)

    if timer_real % 500 == 0:
        maddpg_agents.save_checkpoint()

    if timer_real % 50 == 0:
        rewarddd.append(rewardd)

    if timer_real % 1000 == 0:
        data_reward = {'reward': rewarddd}
        data_reward = pd.DataFrame(data_reward)
        data_reward.to_csv('')


while timer >= 8 * Observe and timer < 8 * (Observe + Explore + Train):
    actions_2 = []
    reward = []
    ex_actions = []
    action_index = []
    rdn = random.random()
    learn = 1
    if rdn <= epsilon:
        learn = 0
        nn = 0
        while nn < 8:
            action = list(np.zeros(20))
            a = random.randrange(-100, 100) / 100
            b = random.randrange(-100, 100) / 100
            actions_2.append([a, b])
            i = 0
            j = 0
            if a >= 0:
                j = 1
            if a < 0:
                j = 0
            if -1 < b <= -0.8:
                i = 1
            if -0.8 < b <= -0.6:
                i = 2
            if -0.6 < b <= -0.4:
                i = 3
            if -0.4 < b <= -0.2:
                i = 4
            if -0.2 < b <= 0:
                i = 5
            if 0 < b <= 0.2:
                i = 6
            if 0.2 < b <= 0.4:
                i = 7
            if 0.4 < b <= 0.6:
                i = 8
            if 0.6 < b <= 0.8:
                i = 9
            if 0.8 < b <= 1:
                i = 10

            action[(j * 10 + i) - 1] = 1
            ex_actions.append(action)

            action_idx = action.index(max(action))
            action_index.append(action_idx)
            statex, a = Offload.ActionToReward(action, timer, nn)
            reward.append(a * 10)
            nn += 1
            timer += 1
            Offload.NetStateUpdate(timer)
    else:
        acts = maddpg_agents.choose_action(obs)  # 所有actor的动作
        acts = list(acts)
        nn = 0
        while nn < 8:
            ac = acts[nn]
            action = list(np.zeros(20))
            ac = list(np.array(ac))
            actions_2.append(ac)
            i = 0
            j = 0
            if ac[0] >= 0:
                j = 1
            if ac[0] < 0:
                j = 0
            if -1 < ac[1] <= -0.8:
                i = 1
            if -0.8 < ac[1] <= -0.6:
                i = 2
            if -0.6 < ac[1] <= -0.4:
                i = 3
            if -0.4 < ac[1] <= -0.2:
                i = 4
            if -0.2 < ac[1] <= 0:
                i = 5
            if 0 < ac[1] <= 0.2:
                i = 6
            if 0.2 < ac[1] <= 0.4:
                i = 7
            if 0.4 < ac[1] <= 0.6:
                i = 8
            if 0.6 < ac[1] <= 0.8:
                i = 9
            if 0.8 < ac[1]:
                i = 10

            action[(j * 10 + i) - 1] = 1
            ex_actions.append(action)
            action_idx = action.index(max(action))
            action_index.append(action_idx)
            statex, a = Offload.ActionToReward(action, timer, nn)
            reward.append(a * 10)
            nn += 1
            timer += 1
            Offload.NetStateUpdate(timer)

    if epsilon > FINAL_EPSILON and timer > 8 * Observe:
        epsilon = epsilon - ((INITIAL_EPSILON - FINAL_EPSILON) / (Explore))

    # print(state)
    computing_ = copy.deepcopy(Offload.compresource)

    rewardd = sum(reward) / 8
    if action_index.count(19) + action_index.count(18) >= 4:
        rewardd = -1


    agents_request_ = copy.deepcopy(Offload.requestinfo)
    state_ = Offload.uniform([computing_, agents_request_])
    state_ = state_[0:8] + state_[9:13] + state_[14:18] + state_[19:23] + state_[24:28] + state_[29:33] + state_[
                                                                                                          34:38] + state_[
                                                                                                                   39:43] + state_[
                                                                                                                            44:48]
    obs_ = Offload.getuniobs(computing_, agents_request_)
    memory.store_tansition(obs, state, actions_2, rewardd, obs_, state_)  # 存的过程，这段时间只会往里存

    actor_states, states, actions, rewards, actor_new_states, states_ = memory.sample_buffer()

    states = T.tensor(states, dtype=T.float)
    actions = T.tensor(actions, dtype=T.float)

    rewards = T.tensor(rewards, dtype=T.float)
    states_ = T.tensor(states_, dtype=T.float)

    all_agents_new_actions = []
    all_agents_new_mu_actions = []

    for agent_idx in range(len(maddpg_agents.agents)):
        new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float)

        new_pi = maddpg_agents.agents[agent_idx].target_actor.forward(new_states)

        all_agents_new_actions.append(new_pi)
        mu_states = T.tensor(actor_states[agent_idx], dtype=T.float)
        pi = maddpg_agents.agents[agent_idx].actor.forward(mu_states)
        all_agents_new_mu_actions.append(pi)

    new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)

    mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)

    xxx = 0

    for agent_idx in range(len(maddpg_agents.agents)):
        critic_value_ = maddpg_agents.agents[agent_idx].target_critic.forward(states_, new_actions).flatten()
        critic_value = maddpg_agents.agents[agent_idx].critic.forward(states, actions).flatten()  # 这就是评论员打的分
        target = rewards[:, agent_idx] + maddpg_agents.agents[agent_idx].gamma * critic_value_
        critic_loss = F.mse_loss(target, critic_value)
        maddpg_agents.agents[agent_idx].critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        maddpg_agents.agents[agent_idx].critic.optimizer.step()

        actor_loss = maddpg_agents.agents[agent_idx].critic.forward(states, mu).flatten()
        actor_loss = -T.mean(actor_loss)

        maddpg_agents.agents[agent_idx].actor.optimizerr.zero_grad()
        actor_loss.backward(retain_graph=True)

        maddpg_agents.agents[agent_idx].actor.optimizerr.step()
        maddpg_agents.agents[agent_idx].update_network_parameters(tau=0.01)

    timer_real = timer / 8
    if timer_real <= Observe:
        process = 'observe'
    else:
        process = 'train'

    if timer_real % 50 == 0:  # 4500的话是真实的第500个
        sss = 'time_step {}/ process {}/ action {}/ reward {}/ learn {}/ epsilon {}/'.format(timer_real, process, action_index, rewardd, learn, epsilon)
        print(sss)

    if timer_real % 500 == 0:
        maddpg_agents.save_checkpoint()

    if timer_real % 5 == 0:
        rewarddd.append(rewardd)

    if timer_real % 1000 == 0:
        data_reward = {'reward': rewarddd}
        data_reward = pd.DataFrame(data_reward)
        data_reward.to_csv('')

check_data_reward = pd.read_csv('')




