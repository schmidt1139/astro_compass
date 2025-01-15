
import gymnasium as gym;

#gym.pprint_registry()

env = gym.make("LunarLander-v3", render_mode="human")

print(env.action_space);

for _ in range(100):

    observation, info = env.reset()
    
    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
    
        episode_over = terminated or truncated
    #for _ in range(100):

env.close()



