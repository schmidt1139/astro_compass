import numpy as np


def extract_rollouts(buffer):
    size = buffer.size()
    obs = buffer.observations
    actions = buffer.actions
    rewards = buffer.rewards
    next_obs = buffer.next_observations
    dones = buffer.dones
    infos = buffer.infos

    episodes = []
    episode = {"obs": [], "actions": [], "rewards": [], "next_obs": [], "infos": []}

    for i in range(size):
        episode["obs"].append(obs[i][0])
        episode["actions"].append(actions[i][0])
        episode["rewards"].append(rewards[i][0])
        episode["next_obs"].append(next_obs[i][0])
        episode["infos"].append(infos[i])
        if np.asarray(dones[i][0]).flatten()[0]:
            episodes.append(episode)
            episode = {
                "obs": [],
                "actions": [],
                "rewards": [],
                "next_obs": [],
                "infos": [],
            }

    # Add last episode if not ended with done
    if episode["obs"]:
        episodes.append(episode)

    return episodes
