"""AlphaZero 训练模块"""
from importlib import import_module
from .config import AlphaZeroConfig
from .replay import ReplayBuffer


def __getattr__(name):
    """Load GPU-dependent modules only when the training runtime needs them."""
    if name == 'Trainer':
        from .trainer import Trainer
        return Trainer
    if name == 'Arena':
        from .arena import Arena
        return Arena
    if name in ('InferenceServer', 'BatchRequest', 'BatchResponse'):
        return getattr(import_module('.inference_server', __name__), name)
    if name in ('BatchRemoteEvaluator', 'self_play_worker_fn'):
        return getattr(import_module('.self_play_worker', __name__), name)
    if name in ('TrainingMonitor', 'monitor'):
        return getattr(import_module('.monitor', __name__), name)
    raise AttributeError(name)

