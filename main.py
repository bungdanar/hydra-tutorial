from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path='.', config_name='config')
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
