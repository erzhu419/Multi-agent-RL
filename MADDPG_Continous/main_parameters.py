import argparse

def main_parameters():
    parser = argparse.ArgumentParser()
    ############################################ 选择环境 ############################################
    parser.add_argument("--env_name", type =str, default = "simple_tag_v3", help = "name of the env",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_tag_env']) 
    parser.add_argument("--render_mode", type=str, default = "None", help = "None | human | rgb_array")
    parser.add_argument("--episode_num", type = int, default = 500) # 5000
    parser.add_argument("--episode_length", type = int, default = 500) #50
    parser.add_argument("--max_workers", type = int, default = 8, help = "number of parallel workers")
    parser.add_argument('--training_freq', type=int, default=5,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=50000, help='random steps before the agent start to learn') #  2e3
    parser.add_argument('--tau', type=float, default=0.001, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e5), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=128, help='batch-size of replay buffer')  
    parser.add_argument('--actor_lr', type=float, default=0.0002, help='learning rate of actor') # .00002
    parser.add_argument('--critic_lr', type=float, default=0.002, help='learning rate of critic') # .002
    parser.add_argument('--plot_interval', type=int, default=1, help='plot interval') # 100
    # The parameters for the communication network
    # TODO
    parser.add_argument('--visdom', type=bool, default=False, help="Open the visdom")
    parser.add_argument('--size_win', type=int, default=200, help="Open the visdom") # 1000


    args = parser.parse_args()
    return args