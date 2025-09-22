# from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING, DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from pydantic import validator, field_validator


@dataclass
class ExperimentSchema:
    model: str = MISSING
    nrof_epochs: int = 30
    batch_size: int = 512

    @field_validator('batch_size')
    def batch_size_multiple_of_32(cls, batch_size: int) -> int:
        if batch_size % 32 != 0:
            raise ValueError('batch_size should be a multiple of 32')
        return batch_size


@dataclass
class Resnet18ExperimentSchema(ExperimentSchema):
    model: str = 'resnet18'


@dataclass
class Resnet50ExperimentSchema(ExperimentSchema):
    model: str = 'resnet50'


@dataclass
class ConfigSchema:
    experiment: ExperimentSchema


cs = ConfigStore.instance()
cs.store(name='config_schema', node=ConfigSchema)
cs.store(group='experiment', name='resnet18_schema',
         node=Resnet18ExperimentSchema)
cs.store(group='experiment', name='resnet50_schema',
         node=Resnet50ExperimentSchema)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig) -> None:
    OmegaConf.to_object(config)
    print(OmegaConf.to_yaml(config))


if __name__ == '__main__':
    main()
