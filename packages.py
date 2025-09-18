import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))


if __name__ == '__main__':
    main()
