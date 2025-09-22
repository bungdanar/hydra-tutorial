from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING, DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class Experiment:
    model: str = MISSING
    nrof_epochs: int = 30
    learning_rate: float = 5e-3


@dataclass
class Resnet18Experiment(Experiment):
    model: str = 'resnet18'
    batch_size: int = 256


@dataclass
class Resnet50Experiment(Experiment):
    model: str = 'resnet50'
    lr_scheduler: str = 'MultiStepLR'


DEFAULT = [
    {'experiment': 'resnet50'},
    '_self_'
]


@dataclass
class MainConfig:
    defaults: List[Any] = field(default_factory=lambda: DEFAULT)


cs = ConfigStore.instance()
cs.store(name='config', node=MainConfig)
cs.store(group='experiment', name='resnet18', node=Resnet18Experiment)
cs.store(group='experiment', name='resnet50', node=Resnet50Experiment)


@hydra.main(version_base=None, config_path=None, config_name='config')
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == '__main__':
    main()
