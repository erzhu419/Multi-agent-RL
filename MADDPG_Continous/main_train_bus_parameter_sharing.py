from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from envs import simple_tag_env, custom_agents_dynamics
from envs.sim import env_bus

from main_parameters import main_parameters
from utils.runner_bus_parameter_sharing import RUNNER_PS  # 修改import

from agents.maddpg.MADDPG_parameter_sharing import MADDPG_PS  # 修改import
import torch
import os
import time
from datetime import timedelta
from utils.logger import TrainingLogger

def get_env(render):
    """create environment and get observation and action dimension of each agent in this environment"""
    path = os.getcwd() + '/MADDPG_Continous/envs'
    debug = True
    new_env = env_bus(path, debug=debug, render=render)
    new_env.reset()
    _dim_info = {}
    action_bound = {}
    for agent_id in range(new_env.max_agent_num):
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = []  # [low action, high action]
        _dim_info[agent_id].append(new_env.observation_space.shape[0])
        _dim_info[agent_id].append(new_env.action_space.shape[0])
        action_bound[agent_id].append(new_env.action_space.low)
        action_bound[agent_id].append(new_env.action_space.high)
    
    return new_env, _dim_info, action_bound

if __name__ == '__main__':
    device_idx = 0
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Using Parameter Sharing MADDPG")
    start_time = time.time()
    render = False
    
    # 模型保存路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models/maddpg_ps_models/')
    
    # 定义参数
    args = main_parameters()
    
    # 创建环境
    print("Using Env's name", args.env_name)
    env, dim_info, action_bound = get_env(render=render)
    
    # 创建Parameter Sharing MA-DDPG智能体
    agent = MADDPG_PS(
        args, 
        dim_info, 
        args.buffer_capacity, 
        args.batch_size, 
        args.actor_lr, 
        args.critic_lr, 
        action_bound, 
        _chkpt_dir=chkpt_dir, 
        _device=device
    )
    
    # 创建运行对象
    runner = RUNNER_PS(agent, env, args, device, mode='train')
    
    # 开始训练
    print("Starting Parameter Sharing Training...")
    runner.train(render)
    print("agent", agent)

    # 计算训练时间
    end_time = time.time()
    training_time = end_time - start_time
    training_duration = str(timedelta(seconds=int(training_time)))
    print(f"\n===========训练完成!===========")
    print(f"训练设备: {device}")
    print(f"训练用时: {training_duration}")
    print(f"使用Parameter Sharing: 是")

    # 使用logger保存训练日志
    logger = TrainingLogger()
    current_time = logger.save_training_log(args, device, training_duration, runner)
    print(f"完成时间: {current_time}")

    print("--- saving trained Parameter Sharing models ---")
    agent.save_model()
    print("--- trained Parameter Sharing models saved ---")