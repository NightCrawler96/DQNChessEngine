import keras
import numpy as np


class DQNTrainer:
    def __init__(self, model: keras.Model, memory, action, training):
        self._active_model = model
        self._target_model = keras.models.clone_model(model)
        self._memory = memory
        self.action = action
        self.training = training

    """Train your model
    
    # Arguments
        batch_size: number of states used in training
        gamma: power of reinforcement (q = Q(s) + gamma * Q(s'))
        theta: coefficient by which target network should be updated
            If continuous updating is not wanted, then leave it at zero and target network won't be changed
    
    """
    def train(self, batch_size: int = 32, gamma: float = 0.99, theta: float = 0.):
        self.training(self._active_model, self._target_model, self._memory, batch_size, gamma)
        if theta > 0:
            self._update_target(theta)

    def _update_target(self, theta: float):
        t_weights = np.array(self._target_model.get_weights())
        a_weights = np.array(self._active_model.get_weights())
        new_t_weights = a_weights * theta + (1 - theta) * t_weights
        self._target_model.set_weights(new_t_weights)

    def copy_weights_to_target(self):
        self._target_model.set_weights(self._active_model.get_weights())

    def take_action(self, environment, epsilon=0.):
        self.action(self._active_model, self._memory, environment, epsilon)

    def save(self, path: str):
        self._target_model.save(path)
