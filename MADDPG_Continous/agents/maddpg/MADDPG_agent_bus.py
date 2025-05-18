import os

import numpy as np
import torch
import torch.nn.functional as F
from agents.maddpg.DDPG_agent_bus import DDPG
from agents.maddpg.buffer import BUFFER

class MADDPG():
    # device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, action_bound, _chkpt_dir, _device = 'cpu', _model_timestamp = None):
        # 确保模型保存路径存在
        if _chkpt_dir is not None:
            os.makedirs(_chkpt_dir, exist_ok=True)

        self.device = _device
        self.model_timestamp = _model_timestamp
        # 状态（全局观测）与所有智能体动作维度的和 即critic网络的输入维度  dim_info =  [obs_dim, act_dim]
        global_obs_act_dim = 32
        # 创建智能体与buffer，每个智能体有自己的buffer, actor, critic
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            # print("dim_info -> agent_id:",agent_id)
            # 每一个智能体都是一个DDPG智能体
            
            self.agents[agent_id] = DDPG(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device, action_bound[agent_id], chkpt_name = (str(agent_id) + '_'), chkpt_dir = _chkpt_dir)
            # buffer均只是存储自己的观测与动作
            self.buffers[agent_id] = BUFFER(capacity, obs_dim, act_dim, self.device)
        self.dim_info = dim_info
        self.batch_size = batch_size

    def add(self, obs, action, reward, next_obs, done):
        keys_with_two_obs = [key for key, val in obs.items() if isinstance(val, list) and len(val) == 2]
        
        for agent_id in keys_with_two_obs:
            o = obs[agent_id][0]
            next_o = obs[agent_id][1]
            a = action[agent_id]
            if isinstance(a, int):
                a = np.eye(self.dim_info[agent_id][1])[a]
            r = reward[agent_id]
            d = done
            
            # 只添加一次
            self.buffers[agent_id].add(o, a, r, next_o, d)
    
    def sample(self, batch_size, agent_id):
        """Sample experience from the buffer of a specific agent."""
        # Get the buffer for the specified agent
        buffer = self.buffers[agent_id]
        
        # Get the total number of transitions in the buffer
        total_num = len(buffer)
        if total_num == 0:
            raise ValueError(f"Buffer for agent {agent_id} is empty. Cannot sample.")

        # Randomly sample indices
        indices = np.random.choice(total_num, size=batch_size, replace=True)

        # Sample data for the specified agent
        o, a, r, n_o, d = buffer.sample(indices)

        # Calculate next_action using the target network and next_state
        next_a, _ = self.agents[agent_id].target_action(n_o)

        return o, a, r, n_o, d, next_a
    
    # 在select_action中实现批处理
    def select_action(self, obs, action):
        with torch.no_grad():  # 避免不必要的梯度计算
            keys_to_process = [k for k in obs if len(obs[k]) > 0]
            if not keys_to_process:
                return action
                
            # 批处理观察
            batch_obs = torch.stack([torch.FloatTensor(obs[k][0]) for k in keys_to_process])
            batch_obs = batch_obs.to(self.device)
            
            # 为每个待处理智能体获取动作
            for i, k in enumerate(keys_to_process):
                a, _ = self.agents[k].action(batch_obs[i:i+1])
                action[k] = a.cpu().numpy().squeeze(0)
                
            return action
            
        return action
    # 更多解释-飞书链接：https://m6tsmtxj3r.feishu.cn/docx/Kb1vdqvBholiIUxcvYxcIcBcnEg?from=from_copylink   密码：6u2257#8
    def learn(self, batch_size, gamma, agent_id):
        """Train the specified agent."""
        agent = self.agents[agent_id]
        o, a, r, n_o, d, next_a = self.sample(batch_size, agent_id)
        # --- reward 标准化和缩放（与 SAC 一致） ---
        reward_scale = 10.0
        r = reward_scale * (r - r.mean(dim=0)) / (r.std(dim=0) + 1e-6)
        # ----------------------------------------
        # Update Critic Network
        critic_value = agent.critic_value([o], [a])
        next_target_critic_value = agent.target_critic_value([n_o], [next_a])
        target_value = r + gamma * next_target_critic_value * (1 - d)
        critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction="mean")
        agent.update_critic(critic_loss)

        # Update Actor Network
        action, logits = agent.action(o, model_out=True)
        actor_loss = -agent.critic_value([o], [action]).mean()
        actor_loss_pse = torch.pow(logits, 2).mean()  # Regularization term
        agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)

        return critic_value.mean().item()
    def update_target(self, tau): #  嵌套函数定义
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau """
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)  #体现使用嵌套函数的作用！ 易于维护和使用
            soft_update(agent.critic, agent.target_critic)

    @classmethod
    def load( cls, dim_info, file):
        """ init maddpg using the model saved in `file` """
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file, map_location=instance.device)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
    
    def save_model(self):
        for agent_id in self.dim_info.keys():
            self.agents[agent_id].actor.save_checkpoint(is_target = False, timestamp = True)
            self.agents[agent_id].target_actor.save_checkpoint(is_target = True, timestamp = True)
            self.agents[agent_id].critic.save_checkpoint(is_target = False, timestamp = True)
            self.agents[agent_id].target_critic.save_checkpoint(is_target = True, timestamp = True)

        agent_id = list(self.dim_info.keys())[0]  # 获取第一个代理的 ID
        agent = self.agents[agent_id]
        for name, param in agent.actor.state_dict().items():
        # 仅打印前几个值（例如前5个）
            print(f"Layer: {name}, Shape: {param.shape}, Values: {param.flatten()[:5]}")  # flatten() 展开参数为一维数组


    def load_model(self):
        for agent_id in self.dim_info.keys():
            self.agents[agent_id].actor.load_checkpoint(device = self.device, is_target = False, timestamp = self.model_timestamp)
            self.agents[agent_id].target_actor.load_checkpoint(device = self.device, is_target = True, timestamp = self.model_timestamp)
            self.agents[agent_id].critic.load_checkpoint(device = self.device, is_target = False, timestamp = self.model_timestamp)
            self.agents[agent_id].target_critic.load_checkpoint(device = self.device, is_target = True, timestamp = self.model_timestamp)

        agent_id = list(self.dim_info.keys())[0]  # 获取第一个代理的 ID
        agent = self.agents[agent_id]
        for name, param in agent.actor.state_dict().items():
        # 仅打印前几个值（例如前5个）
            print(f"Layer: {name}, Shape: {param.shape}, Values: {param.flatten()[:5]}")  # flatten() 展开参数为一维数组
  
