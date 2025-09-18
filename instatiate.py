import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch


class Hello:
    def __init__(self, name: str) -> None:
        self.name = name

    def say_hello(self):
        print(f'Hello {self.name}')


@hydra.main(version_base=None, config_path='.', config_name='instatiate-config')
def main(config: DictConfig):
    hello = Hello('John Doe')
    hello.say_hello()

    hello2 = instantiate(config.hello_class)
    hello2.say_hello()

    params = torch.nn.Parameter(torch.randn(10, 10))
    partial_optimizer = instantiate(config.optimizer)

    optimizer = partial_optimizer([params])
    print(optimizer)


if __name__ == '__main__':
    main()
