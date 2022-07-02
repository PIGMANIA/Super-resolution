import hydra
from trainer import Trainer


@hydra.main(config_path="./archs/SCUNet/configs", config_name="train.yaml")
def main(cfg):
    Trainer(cfg, 0)


if __name__ == "__main__":
    main()
