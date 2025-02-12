import logging
import os
import yaml
import argparse
import torch
import random

# import hydra
import pytorch_lightning as pl
import torchvision.transforms as T
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

# from models.vae import VAE
from models.vae import VAE
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    return config

#@hydra.main(config_path="configs")
def train(config_path):
    # import config yaml
    config = load_config(config_path)

###############################################################################################################
# SELECT THE DATASET
dataset_name = "celeba"       # mnist, fmnist, cifar10, celeba, cubicc is supported
###############################################################################################################


def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    return config

def reset_random_seeds(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # No determinism as nn.Upsample has no deterministic implementation
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

#@hydra.main(config_path="configs")
def train():
    # Get config and setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=f'{dataset_name}', type=str,
                        choices=['mnist', 'fmnist', 'news20', 'omniglot', 'cifar10', 'cifar100', 'celeba', 'cubicc'],
                        help='the override file name for config.yml')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    args = parser.parse_args()

    if args.config_name == 'celeba':
        config_path = f"configs/dataset/celeba64/train.yaml"
    else:
        config_path = f"configs/dataset/{args.config_name}/train.yaml"

    # import config yaml
    config = load_config(config_path)

    # Get config and setup
    config = config.vae
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    # seed_everything(config.training.seed, workers=True)
    reset_random_seeds(args.seed)


    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(d_type, root, image_size, norm=False, flip=config.data.hflip)
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    vae = VAE(
        input_res=image_size,
        enc_block_str=config.model.enc_block_config,
        dec_block_str=config.model.dec_block_config,
        enc_channel_str=config.model.enc_channel_config,
        dec_channel_str=config.model.dec_channel_config,
        lr=config.training.lr,
        alpha=config.training.alpha,
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path is not None:
        # Restore checkpoint
        pass
        # train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vae-{config.training.chkpt_prefix}"
        + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    device = config.training.device
    loader_kws = {}
    # if device.startswith("gpu"):
    #     _, devs = configure_device(device)
    #     train_kwargs["gpus"] = devs
    #
    #     # Disable find_unused_parameters when using DDP training for performance reasons
    #     # from pytorch_lightning.plugins import DDPPlugin
    #     #
    #     # train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
    #     loader_kws["persistent_workers"] = True
    # elif device == "tpu":
    #     train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(vae, train_dataloaders=loader)


if __name__ == "__main__":
    config_path = "configs/dataset/cifar10/train.yaml"
    train(config_path)