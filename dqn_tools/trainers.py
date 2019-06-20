import keras


class DQNTrainer:
    def __init__(self, model: keras.Model, memory, action, training):
        self._model = model
        self._memory = memory
        self.action = action
        self.training = training

    def train(self, batch_size: int, gamma: float):
        self.training(self._model, self._memory, batch_size, gamma)

    def take_action(self, environment, epsilon=0.):
        self.action(self._model, self._memory, environment, epsilon)

    def save(self, path: str):
        self._model.save(path)
