from stable_baselines3.common.callbacks import BaseCallback
import os

class ReplayBufferCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path  # This should be a directory, not a file

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Ensure the directory exists
            os.makedirs(self.save_path, exist_ok=True)
            # Create a unique filename for each checkpoint
            filename = f"replay_buffer_step_{self.n_calls}.pkl"
            full_path = os.path.join(self.save_path, filename)
            try:
                self.model.save_replay_buffer(full_path)  # type: ignore
                print(f"Replay buffer saved to {full_path}")
            except Exception as e:
                print(f"Error saving replay buffer: {e}")
        return True