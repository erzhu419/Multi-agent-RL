import numpy as np
import visdom
import csv
import os
import threading
from datetime import datetime
import torch, psutil
from tqdm import trange

model_path = '/home/erzhu419/mine_code/Multi-agent-RL/MADDPG_Continous/models/maddpg_models'  # 模型保存路径
def plot(rewards, q_values_episode, path):
    """
    绘制奖励曲线
    :param rewards: 奖励列表
    :param episode_num: 训练的回合数
    :param window_size: 平滑窗口大小
    """
    import matplotlib.pyplot as plt

    # 计算平滑奖励

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.plot(q_values_episode, label="Q-Value")
    plt.legend()

    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Smoothed Rewards over Episodes')

    plt.grid()

    plt.close()
    
class RUNNER:
    def __init__(self, agent, env, args, device, mode ='evaluate'):
        self.agent = agent
        self.env = env
        self.args = args
        # 这里为什么新建而不是直接使用用agent.agents.keys()？
        # 因为pettingzoo中智能体死亡，这个字典就没有了，会导致 td target更新出错。所以这里维护一个不变的字典。
        self.env_agents = [agent_id for agent_id in self.agent.agents.keys()]
        self.done = None

        # 添加奖励记录相关的属性
        self.reward_sum_record = []  # 用于平滑的奖励记录
        self.all_reward_record = []  # 保存所有奖励记录，用于最终统计
        self.all_adversary_avg_rewards = []  # 追捕者平均奖励
        self.all_sum_rewards = []  # 所有智能体总奖励
        self.episode_rewards = 0

        # 将 agent 的模型放到指定设备上
        for agent in self.agent.agents.values():
            agent.actor.to(device)
            agent.target_actor.to(device)
            agent.critic.to(device)
            agent.target_critic.to(device)
        '''
        解决使用visdom过程中，输出控制台阻塞的问题。
        ''' #TODO

        if mode == 'train' and self.args.visdom:
            self.viz = visdom.Visdom()
            self.viz.close()
        else: # evaluate模式下不需要visdom
            pass
    

    def train(self, render):
        # # 使用visdom实时查看训练曲线
        # viz = None
        # if self.par.visdom:
        #     viz = visdom.Visdom()
        #     viz.close()
        step = {agent_id: 0 for agent_id in self.env_agents}
        step_trained = {agent_id: 0 for agent_id in self.env_agents}  # 初始化每个智能体的训练状态

        rewards = []    # 记录奖励
        q_values = []  # 记录 Q 值变化

        q_values_episode = []  # 记录每个 episode 的 Q 值

        # episode循环
        for episode in trange(self.args.episode_num, desc="Training Episodes"):
            action_dict = {key: None for key in self.env_agents}
            # 记录每个智能体在每个episode的奖励
            self.episode_rewards = 0
            q_values = []
            training_steps = 0
            print(f"This is episode {episode}")
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            self.env.reset()
            obs, agent_reward, self.done = self.env.initialize_state(render)
            # trans the value of obs to numpy array
            # obs = {agent_id: obs[agent_id] for agent_id in range(self.env.max_agent_num)}
            # 每个智能体与环境进行交互
            while not self.done:  # 这里的done是一个布尔值，表示所有智能体是否都完成了
                # print(self.env.current_time)
                for key in obs:
                    if len(obs[key]) == 1:
                        action_dict = self.agent.select_action(obs, action_dict)
                        # copy the new action to action_dict
                        
                    elif len(obs[key]) == 2:
                        if obs[key][0][1] != obs[key][1][1]:  # Only process when states are different
                            # Pass the current state_dict directly to the agent.add method
                            # The modified add method will handle the extraction of observations
                            
                            self.agent.add(obs, action_dict, agent_reward, None, self.done)
                            step[key] += 1
                            self.episode_rewards += agent_reward[key]
                        
                        # 状态更新
                        obs[key] = obs[key][1:]
                        action_dict = self.agent.select_action(obs, action_dict)

                # TODO 这里action上面变成了个字典中的字典，有问题，debug
                obs, agent_reward, self.done = self.env.step(action_dict, render=render)
                
                for key in obs:
                    # 开始学习 有学习开始条件 有学习频率
                    if (
                        step[key] >= self.args.random_steps
                        and step[key] % self.args.learn_interval == 0
                        and step[key] != step_trained[key]  # 确保当前 step 尚未训练过
                    ):                    
                        # # 学习
                        q_value = self.agent.learn(self.args.batch_size, self.args.gamma, key)
                        q_values.append(q_value)
                        training_steps += 1
                        step_trained[key] = step[key]  # 更新训练状态
                        # 更新网络
                        self.agent.update_target(self.args.tau)

            rewards.append(self.episode_rewards)
            q_values_episode.append(np.mean(q_values))

            # 绘制所有智能体在当前episode的和奖励
            if episode % self.args.plot_interval == 0:
                plot(rewards, q_values_episode, model_path)
                np.save("rewards.npy", rewards)
                np.save("q_values.npy", q_values_episode)
                # enumerate the agents in the env
                for agent_id in range(self.env.max_agent_num):
                    
                    torch.save(self.agent.agents[agent_id].actor.state_dict(), f"{model_path}_actor_{agent_id}.pth")
                    torch.save(self.agent.agents[agent_id].critic.state_dict(), f"{model_path}_critic_{agent_id}.pth")
            

            print(
                f"Episode: {episode} | Episode Reward: {self.episode_rewards} "
                f"| CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | "
                f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | ")

    def get_running_reward(self, arr):

        if len(arr) == 0:  # 如果传入空数组，使用完整记录
            arr = self.all_reward_record

        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        window = self.args.size_win
        running_reward = np.zeros_like(arr)

        # for i in range(window - 1):
        #     running_reward[i] = np.mean(arr[:i + 1])
        # for i in range(window - 1, len(arr)):
        #     running_reward[i] = np.mean(arr[i - window + 1:i + 1])
            # 确保不会访问超出数组范围的位置
        for i in range(len(arr)):
            # 对每个i，确保窗口大小不会超出数组的实际大小
            start_idx = max(0, i - window + 1)
            running_reward[i] = np.mean(arr[start_idx:i + 1])
        # print(f"running_reward{running_reward}")
        return running_reward

    @staticmethod
    def exponential_moving_average(rewards, alpha=0.1):
        """计算指数移动平均奖励"""
        ema_rewards = np.zeros_like(rewards)
        ema_rewards[0] = rewards[0]
        for t in range(1, len(rewards)):
            ema_rewards[t] = alpha * rewards[t] + (1 - alpha) * ema_rewards[t - 1]
        return ema_rewards

    def moving_average(self, rewards):
        """计算简单移动平均奖励"""
        window_size = self.args.size_win
        sma_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        return sma_rewards
    
    """保存围捕者平均奖励和所有智能体总奖励到 CSV 文件"""
    def save_rewards_to_csv(self, adversary_rewards, sum_rewards, filename = None): # filename="data_rewards.csv"
        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        if filename is None:
            filename = f"data_rewards_{timestamp}.csv"
        # 获取 runner.py 所在目录，并生成与 utils 同级的 plot 目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件（runner.py）的绝对路径
        plot_dir = os.path.join(current_dir, '..', 'plot', 'data')  # 获取与 utils 同级的 plot 文件夹
        os.makedirs(plot_dir, exist_ok=True)  # 创建 plot 目录（如果不存在）

        # 构造完整的 CSV 文件路径
        full_filename = os.path.join(plot_dir, filename)

        header = ['Episode', 'Adversary Average Reward', 'Sum Reward of All Agents']
        data = list(zip(range(1, len(adversary_rewards) + 1), adversary_rewards, sum_rewards))
        # 将数据写入 CSV 文件
        with open(full_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # 写入表头
            writer.writerows(data)  # 写入数据

        print(f"Rewards data saved to {full_filename}")
#============================================================================================================

    def evaluate(self):
        # # 使用visdom实时查看训练曲线
        # viz = None
        # if self.par.visdom:
        #     viz = visdom.Visdom()
        #     viz.close()
        # step = 0
        # 记录每个episode的和奖励 用于平滑，显示平滑奖励函数
        self.reward_sum_record = []
        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.args.episode_num) for agent_id in self.env.agents}
        # episode循环
        for episode in range(self.args.episode_num):
            step = 0  # 每回合step重置
            print(f"评估第 {episode + 1} 回合")
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset()  # 重置环境，开始新回合
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            # 每个智能体与环境进行交互
            while self.env.agents:
                # print(f"While num:{step}")
                step += 1
                # 使用训练好的智能体选择动作（没有随机探索）
                action = self.agent.select_action(obs)
                # 执行动作 获得下一状态 奖励 终止情况
                # 下一状态：字典 键为智能体名字 值为对应的下一状态
                # 奖励：字典 键为智能体名字 值为对应的奖励
                # 终止情况：bool
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}

                # 累积每个智能体的奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                obs = next_obs

                
                if step % 10 == 0:
                    print(f"Step {step}, obs: {obs}, action: {action}, reward: {reward}, done: {self.done}")

            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)