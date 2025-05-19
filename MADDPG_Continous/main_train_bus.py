from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from envs import simple_tag_env, custom_agents_dynamics
from envs.sim import env_bus

from main_parameters import main_parameters
from utils.runner_bus import RUNNER

from agents.maddpg.MADDPG_agent_bus import MADDPG
import torch
import os
import time
from datetime import timedelta
from utils.logger import TrainingLogger  # 添加导入

def get_env(render):
    """create environment and get observation and action dimension of each agent in this environment"""
    # new_env = None
    # if env_name == 'simple_adversary_v3':
    #     new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    # if env_name == 'simple_spread_v3':
    #     new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array")
    # if env_name == 'simple_tag_v3':
    #     new_env = simple_tag_v3.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    # if env_name == 'simple_tag_env':
    #     new_env = simple_tag_env.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    path = os.getcwd() + '/MADDPG_Continous/envs'
    debug = True
    new_env = env_bus(path, debug=debug, render=render)
    new_env.reset()
    _dim_info = {}
    action_bound = {}
    for agent_id in range(new_env.max_agent_num):
        # print("agent_id:",agent_id)
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = [] #[low action,  hign action]
        _dim_info[agent_id].append(new_env.observation_space.shape[0])
        _dim_info[agent_id].append(new_env.action_space.shape[0])
        action_bound[agent_id].append(new_env.action_space.low)
        action_bound[agent_id].append(new_env.action_space.high)
    # print("_dim_info:",_dim_info)
    # print("action_bound:",action_bound)
    return new_env, _dim_info, action_bound



if __name__ == '__main__':
    device_idx = 0
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    # device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
    #                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    print("Using device:",device)
    start_time = time.time() # 记录开始时间
    render = False
    # 模型保存路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models/maddpg_models/')
    # 定义参数
    args = main_parameters()
    # 创建环境
    print("Using Env's name",args.env_name)
    env, dim_info, action_bound = get_env(render=render)
    # print(env, dim_info, action_bound)
    # 创建MA-DDPG智能体 dim_info: 字典，键为智能体名字 内容为二维数组 分别表示观测维度和动作维度 是观测不是状态 需要注意。
    agent = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, _chkpt_dir = chkpt_dir, _device = device)
    # 创建运行对象
    runner = RUNNER(agent, env, args, device, mode = 'train')
    # 开始训练
    runner.train(render)
    print("agent",agent)

    # 计算训练时间
    end_time = time.time()
    training_time = end_time - start_time
    # 转换为时分秒格式
    training_duration = str(timedelta(seconds=int(training_time)))
    print(f"\n===========训练完成!===========")
    print(f"训练设备: {device}")
    print(f"训练用时: {training_duration}")

    # 使用logger保存训练日志
       # 使用logger保存训练日志
    logger = TrainingLogger()
    current_time = logger.save_training_log(args, device, training_duration, runner)
    print(f"完成时间: {current_time}")

    print("--- saving trained models ---")
    agent.save_model()
    print("--- trained models saved ---")
    


