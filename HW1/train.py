import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from utils import save_agent_video, show_video
from arguments import get_args
from DQN import DQNAgent
from DoubleDQN import DDQNAgent
from DuelingQN import DuelingDQNAgent
from PERDQN import PERDQNAgent
from CombinedDQN import CombinedDQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_args() 

SEED = args.seed  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def train(env, agent, num_episodes, max_steps, seed):
    scores = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        score = 0
        episode_losses = []
        
        for step in range(max_steps):
            action = agent.select_action(state) 
            
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated 
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()         
            if loss is not None:
                episode_losses.append(loss)  
            
            score += reward
            state = next_state
            
            if done:
                break
        
        scores.append(score)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        
        if (episode + 1) % 50 == 0:
            avg = np.mean(scores[-50:])
            print(f"Episode {episode+1}/{num_episodes} | Score: {score:.1f} | Avg: {avg:.1f} | ε: {agent.epsilon:.3f}")
    
    return scores, losses


if __name__ == "__main__":

    print(f"Using device: {device}")
    model_name = args.model_name

    print("="*60)
    print(f"TRAINING {model_name} ON LUNAR LANDER")
    print("="*60)

    env = gym.make('LunarLander-v3')
    video_env = gym.make('LunarLander-v3', render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dim: {state_dim}, Action dim: {action_dim}\n")

    # Agent
    agent = None

    match model_name:
        case "DQN":
            agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

        case "DDQN":
            agent = DDQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

        case "DuelingQN":  
            agent = DuelingDQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

        case "PERDQN":                  
            agent = PERDQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

        case "CombinedDQN":
            agent = CombinedDQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

        case _:
            agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)


    # Train
    scores, losses = train(env, agent, args.num_episodes, args.max_steps, seed=SEED)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Video
    save_agent_video(video_env, agent, filename=f"{model_name}.mp4")
    env.close()
    video_env.close()

    axes[0].plot(scores)
    axes[0].plot(np.convolve(scores, np.ones(50)/50, mode='valid'), 'r-', linewidth=2, label='Moving Avg')
    axes[0].axhline(y=200, color='g', linestyle='--', label='Target')
    axes[0].set_title(f'{model_name} - Scores')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses)
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Episode')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{model_name} Final Avg Score (last 50): {np.mean(scores[-50:]):.1f}")

    show_video(f"{model_name}.mp4", width=600)