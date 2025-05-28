import numpy as np
import csv
import os, time
from datetime import datetime
import torch, psutil
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor

model_path = '/home/erzhu419/mine_code/Multi-agent-RL/MADDPG_Continous/models/maddpg_models'  # 模型保存路径

def plot(rewards, q_values_episode, path):
    """
    绘制奖励曲线
    :param rewards: 奖励列表
    :param episode_num: 训练的回合数
    :param window_size: 平滑窗口大小
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，避免Tkinter相关问题
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

class RUNNER_PS:
    def __init__(self, agent, env, args, device, mode='evaluate'):
        self.agent = agent
        self.env = env
        self.args = args
        self.device = device
        
        # 记录所有智能体ID（parameter sharing中仍需要知道有哪些智能体）
        self.env_agents = [agent_id for agent_id in range(self.env.max_agent_num)]
        self.done = None

        # 添加奖励记录相关的属性
        self.reward_sum_record = []
        self.all_reward_record = []
        self.all_adversary_avg_rewards = []
        self.all_sum_rewards = []
        self.episode_rewards = 0

        # 将共享模型放到指定设备上
        self.agent.shared_agent.actor.to(device)
        self.agent.shared_agent.target_actor.to(device)
        self.agent.shared_agent.critic.to(device)
        self.agent.shared_agent.target_critic.to(device)

        if mode == 'train' and self.args.visdom:
            import visdom
            self.viz = visdom.Visdom()
            self.viz.close()
        else:
            pass

    def train(self, render):
        """优化的训练循环，使用parameter sharing"""

        # 初始化性能监控
        timing_stats = {
            "select_actions": 0,
            "step_env": 0,
            "add_experiences": 0,
            "train_agents": 0,
            "total": 0
        }

        # 初始化计数器和记录
        transitions_added = 0
        rewards = []
        q_values = []
        q_values_episode = []
        
        # 创建活跃智能体集合
        active_agents = set()
        
        # 持久化的action字典
        action_dict = {key: None for key in range(self.env.max_agent_num)}
        
        # 记录训练步数和智能体步数
        trained_steps = 0  # parameter sharing只需要一个训练步数计数器
        agent_steps = {key: 0 for key in range(self.env.max_agent_num)}

        # 创建线程池
        executor = ThreadPoolExecutor(max_workers=self.args.max_workers)

        # episode循环
        for episode in range(self.args.episode_num):
            # 重置计数和环境
            ep_start_time = time.time()
            self.episode_rewards = 0
            training_steps = 0

            self.env.reset()
            obs, agent_reward, self.done = self.env.initialize_state(render)

            # 重置action_dict
            for key in action_dict:
                action_dict[key] = None

            # 环境交互循环
            step_counter = 0
            while not self.done:
                step_counter += 1

                ### 1. 批量处理动作选择 ###
                t0 = time.time()

                # 第一次到达某个站点的情况
                first_station_states = {k: obs[k] for k in obs if len(obs[k]) == 1 and action_dict[k] is None}
                if first_station_states:
                    # 使用parameter sharing的动作选择
                    new_actions = self._select_actions_batch_ps(first_station_states)
                    for k, a in new_actions.items():
                        action_dict[k] = a

                # 非第一次到达某个站点的情况
                second_station_states = {k: obs[k] for k in obs if len(obs[k]) == 2}
                
                # 收集状态转换的智能体
                transitions_list = []
                for k, states in second_station_states.items():
                    if states[0][1] != states[1][1]:  # 站点变化
                        transitions_list.append({
                            "agent_id": k,
                            "old_state": states[0][2:],
                            "new_state": states[1][2:],
                            "action": action_dict[k],
                            "reward": agent_reward[k]
                        })
                        self.episode_rewards += agent_reward[k]
                        transitions_added += 1
                        agent_steps[k] += 1

                    # 更新观察
                    obs[k] = [states[1]]

                if second_station_states:
                    new_actions = self._select_actions_batch_ps(second_station_states)
                    for k, a in new_actions.items():
                        action_dict[k] = a

                timing_stats["select_actions"] += time.time() - t0

                ### 2. 批量添加经验 ###
                t0 = time.time()
                if transitions_list:
                    # 并行添加经验
                    future_adds = []
                    for trans in transitions_list:
                        agent_id = trans["agent_id"]
                        active_agents.add(agent_id)
                        local_obs = {agent_id: [trans["old_state"], trans["new_state"]]}
                        local_action = {agent_id: trans["action"]}
                        local_reward = {agent_id: trans["reward"]}

                        # 提交添加任务
                        future_adds.append(
                            executor.submit(
                                self.agent.add,
                                local_obs,
                                local_action,
                                local_reward,
                                None,
                                self.done
                            )
                        )

                    # 等待所有添加完成
                    for future in future_adds:
                        future.result()

                timing_stats["add_experiences"] += time.time() - t0

                ### 3. 环境步进 ###
                t0 = time.time()
                action_zero = {k: np.zeros(self.env.action_space.shape[0]) for k in action_dict.keys()}
                obs, agent_reward, self.done = self.env.step(action_dict, render=render)
                timing_stats["step_env"] += time.time() - t0

                ### 4. 训练共享网络 ###
                t0 = time.time()
                if transitions_added > self.args.random_steps:
                    # 收集所有有足够数据的智能体进行训练
                    valid_agents = []
                    for agent_id in active_agents:
                        buffer_size = len(self.agent.buffers.get(agent_id, []))
                        if buffer_size >= self.args.batch_size:
                            valid_agents.append(agent_id)

                    # Parameter sharing: 每个有效智能体都可以用来训练共享网络
                    if valid_agents and trained_steps < sum(agent_steps.values()):
                        # 随机选择一个智能体的数据来训练共享网络
                        training_agent = np.random.choice(valid_agents)
                        q_value = self.agent.learn(
                            self.args.batch_size,
                            self.args.gamma,
                            training_agent
                        )
                        
                        if q_value is not None:
                            q_values.append(q_value)
                            training_steps += 1
                            trained_steps += 1

                    # 更新目标网络（parameter sharing只需要更新一次）
                    if trained_steps % self.args.training_freq == 0 and trained_steps > 0:
                        self.agent.update_target(self.args.tau)

                timing_stats["train_agents"] += time.time() - t0

            # Episode结束，记录数据
            rewards.append(self.episode_rewards)
            if training_steps > 0:
                q_values_episode.append(np.mean(q_values[-training_steps:]))
            elif q_values:
                q_values_episode.append(q_values[-1])

            active_agents.clear()

            # 计算总时间
            episode_time = time.time() - ep_start_time
            timing_stats["total"] += episode_time

            # 绘图和保存模型
            if episode % self.args.plot_interval == 0:
                plot(rewards, q_values_episode, model_path)
                np.save("rewards.npy", rewards)
                np.save("q_values.npy", q_values_episode)
                self.agent.save_model()

                timing_stats = {k: 0 for k in timing_stats}  # 重置计时器

            # 打印统计信息
            print(
                f"Episode: {episode} | Reward: {self.episode_rewards:.2f} | "
                f"Time: {episode_time:.2f}s | Transitions: {transitions_added} | "
                f"Training Steps: {trained_steps} | "
                f"CPU: {psutil.Process().memory_info().rss / 1024 ** 2:.1f}MB | "
                f"GPU: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB | "
                f"Total Steps: {step_counter}"
            )

        # 关闭线程池
        executor.shutdown()

    def _select_actions_batch_ps(self, obs_dict):
        """使用parameter sharing批量选择动作"""
        if not obs_dict:
            return {}

        # 收集需要动作的智能体
        agent_ids = []
        observations = []

        for agent_id, obs in obs_dict.items():
            if len(obs) > 0:
                agent_ids.append(agent_id)
                # 提取状态特征（排除前两个元素：bus_id和station_id）
                state_features = obs[-1][2:] if isinstance(obs, list) else obs[2:]
                # 添加agent_id信息
                obs_with_id = self.agent._add_agent_id(state_features, agent_id)
                observations.append(obs_with_id)

        if not agent_ids:
            return {}

        # 批量转换为张量
        try:
            batch_tensor = torch.FloatTensor(np.array(observations)).to(self.device)

            # 使用共享网络批量计算动作
            with torch.no_grad():
                batch_actions, _ = self.agent.shared_agent.action(batch_tensor)

            actions = {}
            for i, agent_id in enumerate(agent_ids):
                actions[agent_id] = batch_actions[i].squeeze(0).cpu().numpy()

            return actions
        except Exception as e:
            print(f"Error in parameter sharing batch action selection: {e}")
            return {}

    def get_running_reward(self, arr):
        if len(arr) == 0:
            arr = self.all_reward_record

        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        window = self.args.size_win
        running_reward = np.zeros_like(arr)

        for i in range(len(arr)):
            start_idx = max(0, i - window + 1)
            running_reward[i] = np.mean(arr[start_idx:i + 1])
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

    def save_rewards_to_csv(self, adversary_rewards, sum_rewards, filename=None):
        """保存奖励数据到CSV文件"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        if filename is None:
            filename = f"data_rewards_{timestamp}.csv"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(current_dir, '..', 'plot', 'data')
        os.makedirs(plot_dir, exist_ok=True)

        full_filename = os.path.join(plot_dir, filename)

        header = ['Episode', 'Adversary Average Reward', 'Sum Reward of All Agents']
        data = list(zip(range(1, len(adversary_rewards) + 1), adversary_rewards, sum_rewards))
        
        with open(full_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data)

        print(f"Rewards data saved to {full_filename}")

    def evaluate(self):
        """评估模式"""
        self.reward_sum_record = []
        self.episode_rewards = {agent_id: np.zeros(self.args.episode_num) for agent_id in self.env_agents}
        
        for episode in range(self.args.episode_num):
            step = 0
            print(f"评估第 {episode + 1} 回合")
            
            obs, _ = self.env.reset()
            self.done = {agent_id: False for agent_id in self.env_agents}
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            
            while self.env.agents:
                step += 1
                # 使用parameter sharing选择动作
                action = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                obs = next_obs

                if step % 10 == 0:
                    print(f"Step {step}, obs: {obs}, action: {action}, reward: {reward}, done: {self.done}")

            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)