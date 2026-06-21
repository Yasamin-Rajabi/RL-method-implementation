import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN Variants - Lunar Lander")

    # -------------------------------------------------------------------------
    # Model selection
    # -------------------------------------------------------------------------
    parser.add_argument("--model_name", type=str, default="DQN",
                        choices=["DQN", "DDQN", "DuelingQN", "PERDQN", "CombinedDQN"],
                        help="Which agent to train")

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    parser.add_argument("--env_name", type=str, default="LunarLander-v3",
                        help="Gymnasium environment ID")
    parser.add_argument("--num_episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # -------------------------------------------------------------------------
    # Core hyperparameters (shared across all agents)
    # -------------------------------------------------------------------------
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument("--epsilon_min", type=float, default=0.001,
                        help="Minimum exploration rate (DQN / DDQN / DuelingQN)")
    parser.add_argument("--epsilon_decay", type=float, default=0.999,
                        help="Epsilon decay factor (DQN / DDQN / DuelingQN)")
    parser.add_argument("--batch_size", type=int, default=30,
                        help="Replay buffer batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam optimizer learning rate")
    parser.add_argument("--target_update_freq", type=int, default=500,
                        help="Steps between target network syncs (DQN / DDQN / DuelingQN)")
    parser.add_argument("--buffer_capacity", type=int, default=10000,
                        help="Replay buffer capacity (DQN / DDQN / DuelingQN)")

    # -------------------------------------------------------------------------
    # PER-specific hyperparameters (PERDQN / CombinedDQN)
    # -------------------------------------------------------------------------
    parser.add_argument("--epsilon_min_per", type=float, default=0.01,
                        help="Minimum exploration rate for PER-based agents")
    parser.add_argument("--epsilon_decay_per", type=float, default=0.9999,
                        help="Epsilon decay factor for PER-based agents")
    parser.add_argument("--target_update_freq_per", type=int, default=250,
                        help="Steps between target network syncs for PER-based agents")
    parser.add_argument("--per_buffer_capacity", type=int, default=8192,
                        help="PrioritizedReplayBuffer capacity (must be power of 2 for SumTree)")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="PER priority exponent (0 = uniform, 1 = full prioritization)")
    parser.add_argument("--beta", type=float, default=0.4,
                        help="PER importance sampling exponent (annealed towards 1.0)")
    parser.add_argument("--beta_increment", type=float, default=1e-5,
                        help="Per-step increment for beta annealing")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)