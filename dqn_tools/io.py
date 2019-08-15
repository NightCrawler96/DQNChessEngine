import os
import pickle
import keras
from keras.engine.saving import load_model


def save(directory: str, name: str, active_model: keras.models.Model, target_model: keras.models.Model, memory=None):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    active_model.save("{}/{}_active.h5f".format(directory, name))
    target_model.save("{}/{}_target.h5f".format(directory, name))
    if memory is not None:
        with open("{}/{}_memory.obj".format(directory, name), 'wb') as handler:
            pickle.dump(memory, handler, pickle.HIGHEST_PROTOCOL)


def load(directory: str, name: str, has_memory: bool=False):
    active_model = load_model("{}/{}_active.h5f".format(directory, name))
    target_model = load_model("{}/{}_target.h5f".format(directory, name))
    memory = None
    if has_memory:
        with open("{}/{}_memory.obj".format(directory, name), 'rb') as handler:
            memory = pickle.load(handler)
    return active_model, target_model, memory
