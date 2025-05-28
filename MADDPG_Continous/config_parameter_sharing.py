"""
Parameter Sharing MADDPG 专用配置文件
"""

class ParameterSharingConfig:
    """Parameter Sharing MADDPG的配置类"""
    
    def __init__(self):
        # 基础训练参数
        self.episode_num = 1000
        self.batch_size = 64
        self.buffer_capacity = 10000
        
        # 学习率设置
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        
        # 网络更新参数
        self.gamma = 0.95
        self.tau = 0.01
        self.critic_actor_ratio = 2  # 每训练2次critic训练1次actor
        
        # Parameter Sharing 特定参数
        self.shared_network_hidden_dim = 128  # 共享网络的隐藏层维度
        self.agent_id_embedding_dim = None  # None表示使用one-hot，否则使用embedding
        self.use_agent_id_in_critic = True  # 是否在critic中也使用agent_id信息
        
        # 训练控制参数
        self.random_steps = 1000  # 随机探索步数
        self.training_freq = 10  # 目标网络更新频率
        self.plot_interval = 50  # 绘图间隔
        
        # 探索参数
        self.noise_std = 0.1  # 动作噪声标准差
        self.noise_decay = 0.995  # 噪声衰减率
        self.min_noise_std = 0.01  # 最小噪声标准差
        
        # 其他参数
        self.max_workers = 4  # 并行处理线程数
        self.visdom = False  # 是否使用visdom可视化
        self.env_name = "bus_holding"

def get_hyperparameter_suggestions():
    """获取不同场景下的超参数建议"""
    
    suggestions = {
        "fast_convergence": {
            "description": "快速收敛配置",
            "actor_lr": 2e-4,
            "critic_lr": 2e-3,
            "batch_size": 128,
            "tau": 0.02,
            "noise_std": 0.2,
            "shared_network_hidden_dim": 256
        },
        
        "stable_training": {
            "description": "稳定训练配置",
            "actor_lr": 5e-5,
            "critic_lr": 1e-3,
            "batch_size": 64,
            "tau": 0.005,
            "noise_std": 0.1,
            "shared_network_hidden_dim": 128
        },
        
        "large_scale": {
            "description": "大规模多智能体配置",
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "batch_size": 256,
            "tau": 0.01,
            "noise_std": 0.05,
            "shared_network_hidden_dim": 512,
            "buffer_capacity": 50000
        },
        
        "exploration_heavy": {
            "description": "重探索配置",
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "batch_size": 64,
            "tau": 0.01,
            "noise_std": 0.3,
            "noise_decay": 0.99,
            "min_noise_std": 0.05,
            "random_steps": 2000
        }
    }
    
    return suggestions

def create_config_from_suggestion(suggestion_name):
    """根据建议创建配置"""
    config = ParameterSharingConfig()
    suggestions = get_hyperparameter_suggestions()
    
    if suggestion_name in suggestions:
        suggestion = suggestions[suggestion_name]
        print(f"Using configuration: {suggestion['description']}")
        
        # 更新配置
        for key, value in suggestion.items():
            if key != "description" and hasattr(config, key):
                setattr(config, key, value)
                print(f"  {key}: {value}")
    else:
        print(f"Unknown suggestion: {suggestion_name}")
        print(f"Available suggestions: {list(suggestions.keys())}")
    
    return config

def print_config_comparison():
    """打印配置对比"""
    print("="*60)
    print("PARAMETER SHARING CONFIGURATION COMPARISON")
    print("="*60)
    
    suggestions = get_hyperparameter_suggestions()
    
    # 打印表头
    params = ["actor_lr", "critic_lr", "batch_size", "tau", "noise_std", "hidden_dim"]
    print(f"{'Config':<20}", end="")
    for param in params:
        print(f"{param:<12}", end="")
    print()
    print("-" * 80)
    
    # 打印每个配置
    for name, config in suggestions.items():
        print(f"{name:<20}", end="")
        for param in params:
            value = config.get(param.replace("hidden_dim", "shared_network_hidden_dim"), "N/A")
            print(f"{str(value):<12}", end="")
        print()

def validate_config(config):
    """验证配置的合理性"""
    warnings = []
    
    # 检查学习率
    if config.actor_lr > config.critic_lr:
        warnings.append("Warning: Actor learning rate is higher than critic learning rate")
    
    # 检查批量大小
    if config.batch_size > config.buffer_capacity // 10:
        warnings.append("Warning: Batch size might be too large compared to buffer capacity")
    
    # 检查噪声参数
    if config.noise_std < config.min_noise_std:
        warnings.append("Warning: Initial noise std is less than minimum noise std")
    
    # 检查更新频率
    if config.critic_actor_ratio < 1:
        warnings.append("Warning: Critic-Actor ratio should be >= 1")
    
    if warnings:
        print("Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Configuration validation passed!")
    
    return len(warnings) == 0

# 使用示例
if __name__ == "__main__":
    print("Parameter Sharing MADDPG Configuration Tools")
    print("="*50)
    
    # 显示所有配置选项
    print_config_comparison()
    
    print("\nCreating different configurations...")
    
    # 创建不同的配置
    configs = {}
    for suggestion_name in ["fast_convergence", "stable_training", "large_scale"]:
        print(f"\n--- {suggestion_name.upper()} CONFIGURATION ---")
        config = create_config_from_suggestion(suggestion_name)
        validate_config(config)
        configs[suggestion_name] = config
    
    print("\nConfiguration creation completed!")
    print("You can import these configurations in your training script:")
    print("from config_parameter_sharing import create_config_from_suggestion")
    print("config = create_config_from_suggestion('stable_training')")