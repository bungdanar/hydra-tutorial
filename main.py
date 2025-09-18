import logging

from omegaconf import DictConfig, OmegaConf
import hydra

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    logger.info('Default info message')
    logger.debug('Default debug message')


if __name__ == "__main__":
    main()
