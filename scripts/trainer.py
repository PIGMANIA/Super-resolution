import os
import hydra
import logging
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import torchvision.utils as vutils

from dataset import Dataset

logging.basicConfig(level=logging.INFO)
logging.getLogger("Trainer").setLevel(logging.INFO)
log = logging.getLogger("Trainer")


class Trainer:
    def __init__(self, cfg, gpu) -> None:
        ### GPU Device setting
        self.gpu = gpu
        cudnn.benchmark = True
        cudnn.deterministic = True

        ### Common setting
        torch.manual_seed(cfg.train.common.seed)
        self.save_path = cfg.train.common.ckpt_dir
        os.makedirs(self.save_path, exist_ok=True)

        self.gan_train = cfg.train.common.GAN
        self.scale = cfg.train.dataset.scale
        self.start_iters = 0
        self.end_iters = cfg.train.common.iteration
        self.seed = cfg.train.common.seed
        self.use_wandb = cfg.train.common.use_wandb

        ### Model setting
        self.generator = None
        self.discriminator = None

        ### Optimizer setting
        self.g_optim = None
        self.d_optim = None

        ### Scheduler setting
        self.scheduler = None

        ### Loss setting
        self.l1loss = None
        self.perceptual_loss = None
        self.gan_loss = None

        ### Dataset setting
        self.dataloader = None

        ### Initializer
        self._init_model(cfg.models)
        self._init_optim(cfg.train.optim)
        self._init_loss(cfg.train.loss)
        self._load_state_dict(cfg.models)
        self._init_scheduler(cfg.train.scheduler)
        self._init_dataset(cfg.train.dataset)

        ### Train
        self.train()

    def _init_model(self, model):
        if model.name.lower() == "scunet":
            from archs.SCUNet.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)

            if self.gan_train:
                from archs.SCUNet.models import Discriminator

                self.discriminator = Discriminator(model.discriminator).to(self.gpu)

        elif model.name.lower() == "realesrgan":
            from archs.RealESRGAN.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)

            if self.gan_train:
                from archs.RealESRGAN.models import Discriminator

                self.discriminator = Discriminator(model.discriminator).to(self.gpu)

        elif model.name.lower() == "bsrgan":
            from archs.BSRGAN.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)

            if self.gan_train:
                from archs.BSRGAN.models import Discriminator

                self.discriminator = Discriminator(model.discriminator).to(self.gpu)

        elif model.name.lower() == "edsr":
            from archs.EDSR.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)

        log.info("Initialized the model")

    def _load_state_dict(self, model):
        if model.generator.path:
            log.info("Train the generator with checkpoint")
            log.info(f"Loading the checkpoint from : {model.generator.path}")
            ckpt = torch.load(
                model.generator.path, map_location=lambda storage, loc: storage
            )
            if len(ckpt) == 3:
                self.num_iteration = ckpt["iteration"]
                self.generator.load_state_dict(ckpt["g"])
                self.g_optim.load_state_dict(ckpt["g_optim"])
                self.start_iters = ckpt["iteration"] + 1
            else:
                self.generator.load_state_dict(ckpt)
        else:
            log.info("Train the generator without checkpoint")

        if self.gan_train:
            if model.discriminator.path:
                log.info("Train the discriminator with checkpoint")
                log.info(f"Loading the checkpoint from : {model.discriminator.path}")
                ckpt = torch.load(
                    model.discriminator.path, map_location=lambda storage, loc: storage
                )
                if len(ckpt) == 3:
                    self.num_iteration = ckpt["iteration"]
                    self.discriminator.load_state_dict(ckpt["d"])
                    self.d_optim.load_state_dict(ckpt["d_optim"])
                else:
                    self.discriminator.load_state_dict(ckpt)
            else:
                log.info("Train the discriminator without checkpoint")

        log.info("Initialized the checkpoints")

    def _init_loss(self, losses):
        if not self.gan_train:
            self.l1loss = nn.L1Loss().to(self.gpu)
        else:
            from loss import GANLoss, PerceptualLoss

            self.l1loss = nn.L1Loss().to(self.gpu)
            self.gan_loss = GANLoss(losses.GANLoss).to(self.gpu)
            self.perceptual_loss = PerceptualLoss(losses.PerceptualLoss).to(self.gpu)

        log.info("Initialized the losses functions")

    def _init_optim(self, optims):
        if optims.type == "Adam":
            self.g_optim = torch.optim.Adam(
                self.generator.parameters(),
                lr=optims.Adam.lr,
                betas=(optims.Adam.betas[0], optims.Adam.betas[1]),
                eps=optims.Adam.eps,
                weight_decay=optims.Adam.weight_decay,
                amsgrad=optims.Adam.amsgrad,
            )

            if self.discriminator:
                self.d_optim = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=optims.Adam.lr,
                    betas=(optims.Adam.betas[0], optims.Adam.betas[1]),
                    eps=optims.Adam.eps,
                    weight_decay=optims.Adam.weight_decay,
                    amsgrad=optims.Adam.amsgrad,
                )
        log.info("Initialized the optimizers")

    def _init_scheduler(self, scheduler):
        if self.start_iters > 0:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.g_optim,
                scheduler.milestones,
                scheduler.gamma,
                last_epoch=self.start_iters,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.g_optim, scheduler.milestones, scheduler.gamma
            )
        log.info("Initialized the schedulers")

    def _init_dataset(self, dataset):
        def sample_data(loader):
            while True:
                for batch in loader:
                    yield batch

        dataloader = DataLoader(
            dataset=Dataset(dataset),
            batch_size=dataset.batch_size,
            shuffle=True,  # TODO None for now, this will be changed when ddp is applied
            num_workers=dataset.num_workers,
            pin_memory=True,
            sampler=None,  # TODO None for now, this will be changed when ddp is applied
            drop_last=True,
        )
        self.dataloader = sample_data(dataloader)

        log.info("Initialized the datasets")

    def train_psnr(self):
        self.generator.train()

        for i in range(self.start_iters, self.end_iters):
            lr, hr = next(self.dataloader)
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            preds = self.generator(lr)

            loss = self.l1loss(preds, hr)

            self.generator.zero_grad()
            loss.backward()
            self.g_optim.step()
            self.scheduler.step()

            results = torch.cat(
                (
                    hr.detach(),
                    F.interpolate(lr, scale_factor=self.scale, mode="nearest").detach(),
                    preds.detach(),
                ),
                2,
            )

            vutils.save_image(results, os.path.join(self.save_path, f"preds.jpg"))

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": self.generator.state_dict(),
                        "g_optim": self.g_optim.state_dict(),
                        "iteration": i,
                    },
                    os.path.join(self.save_path, f"{str(i).zfill(6)}.pth"),
                )

    def train_gan(self):
        self.generator.train()
        self.discriminator.train()

        def requires_grad(model, flag=True):
            for p in model.parameters():
                p.requires_grad = flag

        for i in range(self.start_iters, self.end_iters):
            lr, hr = next(self.dataloader)
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            d_loss = 0.0

            preds = self.generator(lr)
            real_pred = self.discriminator(hr)
            d_loss_real = self.gan_loss(real_pred, True)

            fake_pred = self.discriminator(preds)
            d_loss_fake = self.gan_loss(fake_pred, False)

            d_loss = (d_loss_real + d_loss_fake) / 2

            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            preds = self.generator(lr)
            fake_pred = self.discriminator(preds)

            g_loss = 0.0
            g_loss += self.l1loss(preds, hr)
            g_loss += self.perceptual_loss(preds, hr)
            g_loss += self.gan_loss(fake_pred, True)

            self.generator.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            self.scheduler.step()

            results = torch.cat(
                (
                    hr.detach(),
                    F.interpolate(lr, scale_factor=self.scale, mode="nearest").detach(),
                    preds.detach(),
                ),
                2,
            )

            vutils.save_image(results, os.path.join(self.save_path, f"preds.jpg"))

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": self.generator.state_dict(),
                        "g_optim": self.g_optim.state_dict(),
                        "iteration": i,
                    },
                    os.path.join(self.save_path, f"g_{str(i).zfill(6)}.pth"),
                )

                torch.save(
                    {
                        "d": self.discriminator.state_dict(),
                        "d_optim": self.d_optim.state_dict(),
                        "iteration": i,
                    },
                    os.path.join(self.save_path, f"d_{str(i).zfill(6)}.pth"),
                )

    def train(self):
        if not self.gan_train:
            self.train_psnr()
        else:
            self.train_gan()


@hydra.main(config_path="../configs/", config_name="train.yaml")
def main(cfg):
    Trainer(cfg, 0)


if __name__ == "__main__":
    main()
