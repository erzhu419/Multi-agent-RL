import os

import numpy as np
import torch
import torch.nn.functional as F
from agents.maddpg.DDPG_agent_bus import DDPG
from agents.maddpg.buffer import BUFFER

class MADDPG_PS():
    """Parameter Sharing版本的MADDPG"""
    
    def __init__(self, args, dim_info, capacity, batch_size, actor_lr, critic_lr, action_bound, _chkpt_dir, _device = 'cpu', _model_timestamp = None):
        # 确保模型保存路径存在
        if _chkpt_dir is not None:
            os.makedirs(_chkpt_dir, exist_ok=True)
        self.args = args
        self.device = _device
        self.model_timestamp = _model_timestamp
        
        # 获取观测和动作维度（假设所有智能体的维度相同）
        first_agent_id = list(dim_info.keys())[0]
        obs_dim, act_dim = dim_info[first_agent_id]
        
        # 添加agent_id作为输入特征（categorical feature）
        # 观测维度+1用于agent_id的one-hot编码
        self.num_agents = len(dim_info)
        self.obs_dim_with_id = obs_dim + self.num_agents
        
        # 全局观测和动作维度
        global_obs_act_dim = 32
        
        # 创建单一的共享DDPG智能体
        self.shared_agent = DDPG(
            obs_dim=self.obs_dim_with_id, 
            act_dim=act_dim, 
            global_obs_dim=global_obs_act_dim,
            actor_lr=actor_lr, 
            critic_lr=critic_lr, 
            device=self.device, 
            action_bound=action_bound[first_agent_id],  # 假设所有智能体动作边界相同
            chkpt_dir=_chkpt_dir, 
            chkpt_name='shared_'
        )
        
        # 为每个智能体创建独立的buffer
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.buffers[agent_id] = BUFFER(capacity, self.obs_dim_with_id, act_dim, self.device)
        
        self.dim_info = dim_info
        self.batch_size = batch_size
        
        # 记录更新次数
        self.actor_update_count = 0
        self.critic_update_count = 0

    def _add_agent_id(self, obs, agent_id):
        """为观测添加agent_id的one-hot编码"""
        # 创建one-hot编码
        agent_onehot = np.zeros(self.num_agents)
        agent_onehot[agent_id] = 1.0
        
        # 将one-hot编码添加到观测中
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs, agent_onehot])
        else:
            obs_array = np.array(obs)
            return np.concatenate([obs_array, agent_onehot])

    def update_target_for_agent(self, agent_id, tau):
        """更新目标网络"""
        def soft_update(from_network, to_network):
            """Copy parameters with proportion tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        # 所有智能体共享同一个网络
        soft_update(self.shared_agent.actor, self.shared_agent.target_actor)
        soft_update(self.shared_agent.critic, self.shared_agent.target_critic)

    def add(self, obs, action, reward, next_obs, done):
        """添加经验到buffer"""
        keys_with_two_obs = [key for key, val in obs.items() if isinstance(val, list) and len(val) == 2]
        
        for agent_id in keys_with_two_obs:
            o = obs[agent_id][0]
            next_o = obs[agent_id][1]
            a = action[agent_id]
            if isinstance(a, int):
                a = np.eye(self.dim_info[agent_id][1])[a]
            r = reward[agent_id]
            d = done
            
            # 添加agent_id信息
            o_with_id = self._add_agent_id(o, agent_id)
            next_o_with_id = self._add_agent_id(next_o, agent_id)
            
            # 添加到对应智能体的buffer
            self.buffers[agent_id].add(o_with_id, a, r, next_o_with_id, d)

    def sample(self, batch_size, agent_id):
        """从指定智能体的buffer中采样"""
        if agent_id not in self.buffers:
            print(f"Buffer for agent {agent_id} not found")
            return None

        buffer = self.buffers[agent_id]
        total_num = len(buffer)
        if total_num < batch_size:
            print(f"Buffer for agent {agent_id} has only {total_num} samples, need {batch_size}")
            return None

        # 随机采样索引
        indices = np.random.choice(total_num, size=batch_size, replace=True)
        o, a, r, n_o, d = buffer.sample(indices)

        # 使用共享网络计算next_action
        next_a, _ = self.shared_agent.target_action(n_o)

        return o, a, r, n_o, d, next_a

    def select_action(self, obs, action, noise_std=0.2):
        """使用共享网络为所有智能体选择动作"""
        with torch.no_grad():
            keys_to_process = [k for k in obs if len(obs[k]) > 0]
            if not keys_to_process:
                return action
                
            # 批处理观察，添加agent_id信息
            batch_obs_list = []
            for k in keys_to_process:
                obs_with_id = self._add_agent_id(obs[k][0], k)
                batch_obs_list.append(torch.FloatTensor(obs_with_id))
            
            batch_obs = torch.stack(batch_obs_list)
            batch_obs = batch_obs.to(self.device)
            
            # 使用共享网络获取动作
            batch_actions, _ = self.shared_agent.action(batch_obs)
            
            # 为每个智能体分配动作
            for i, k in enumerate(keys_to_process):
                a = batch_actions[i].cpu().numpy()
                # 加高斯噪声
                a += np.random.normal(0, noise_std, size=a.shape)
                # clip到动作空间
                action_bound = list(self.dim_info.values())[0]  # 假设所有智能体动作边界相同
                # 这里需要根据实际的action_bound结构调整
                if hasattr(self.shared_agent.actor, 'action_bound'):
                    a = np.clip(a, self.shared_agent.actor.action_bound[0], self.shared_agent.actor.action_bound[1])
                action[k] = a
                
            return action

    def learn(self, batch_size, gamma, agent_id):
        """训练共享网络"""
        # 采样数据
        sample_data = self.sample(batch_size, agent_id)
        if sample_data is None:
            return None

        o, a, r, n_o, d, next_a = sample_data

        # 标准化奖励
        reward_scale = 10.0
        r = reward_scale * (r - r.mean(dim=0)) / (r.std(dim=0) + 1e-6)

        # 更新Critic网络
        critic_value = self.shared_agent.critic_value([o], [a])
        next_target_critic_value = self.shared_agent.target_critic_value([n_o], [next_a])
        target_value = r + gamma * next_target_critic_value * (1 - d)
        mask = (d < 1.0).float()
        critic_loss = (F.mse_loss(critic_value, target_value.detach(), reduction='none') * mask).sum() / mask.sum()
        self.shared_agent.update_critic(critic_loss)
        self.critic_update_count += 1

        # 更新Actor网络
        if self.critic_update_count % self.args.critic_actor_ratio == 0:
            action, logits = self.shared_agent.action(o, model_out=True)
            actor_loss = -self.shared_agent.critic_value([o], [action]).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()  # Regularization term
            self.shared_agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)

        return critic_value.mean().item()

    def update_target(self, tau):
        """更新目标网络"""
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau """
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        # 只需要更新一次共享网络
        soft_update(self.shared_agent.actor, self.shared_agent.target_actor)
        soft_update(self.shared_agent.critic, self.shared_agent.target_critic)

    def save_model(self):
        """保存共享模型"""
        self.shared_agent.actor.save_checkpoint(is_target=False, timestamp=True)
        self.shared_agent.target_actor.save_checkpoint(is_target=True, timestamp=True)
        self.shared_agent.critic.save_checkpoint(is_target=False, timestamp=True)
        self.shared_agent.target_critic.save_checkpoint(is_target=True, timestamp=True)

    def load_model(self):
        """加载共享模型"""
        self.shared_agent.actor.load_checkpoint(device=self.device, is_target=False, timestamp=self.model_timestamp)
        self.shared_agent.target_actor.load_checkpoint(device=self.device, is_target=True, timestamp=self.model_timestamp)
        self.shared_agent.critic.load_checkpoint(device=self.device, is_target=False, timestamp=self.model_timestamp)
        self.shared_agent.target_critic.load_checkpoint(device=self.device, is_target=True, timestamp=self.model_timestamp)

    @classmethod
    def load(cls, dim_info, file):
        """ init maddpg using the model saved in `file` """
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file, map_location=instance.device)
        instance.shared_agent.actor.load_state_dict(data['shared_actor'])
        return instance