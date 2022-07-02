import os
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
        self.scale = cfg.train.dataset.scale
        torch.manual_seed(cfg.train.common.seed)
        self.save_path = cfg.train.common.ckpt_dir

        os.makedirs(self.save_path, exist_ok=True)

        ### GPU Device setting
        self.gpu = gpu
        cudnn.benchmark = True
        cudnn.deterministic = True

        ### Common setting
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
        self.loss = []

        ### Dataset setting
        self.dataloader = None

        ### Initializer
        self._init_model(cfg.models)
        self._init_optim(cfg.train.optim)
        self._init_scheduler(cfg.train.scheduler)
        self._init_loss(cfg.train.loss.lists)
        self._load_state_dict(cfg.models)
        self._init_dataset(cfg.train.dataset)

        ### Train
        self.train()

    def _init_model(self, model):
        if model.name.lower() == "scunet":
            from archs.SCUNet.models import Generator

            self.generator = Generator(model.generator).to(self.gpu)
        elif model.name.lower() == "real-esrgan":
            pass

        log.info("Generator is Loaded")

        if (k == "discriminator" for k in list(model.keys())):
            from archs.SCUNet.models import Discriminator

            self.discriminator = Discriminator(model.discriminator).to(self.gpu)
        else:
            log.info("The model does not have discriminator")

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
            else:
                self.generator.load_state_dict(ckpt)
        else:
            log.info("Train the generator without checkpoint")

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
        for loss in losses:
            if loss == "MAE":
                self.loss.append(nn.L1Loss().to(self.gpu))
            elif loss == "MSE":
                self.loss.append(nn.MSELoss().to(self.gpu))
            elif loss == "PerceptualLoss":
                from loss import PerceptualLoss

                self.loss.append(PerceptualLoss(losses.PerceptualLoss).to(self.gpu))
            elif loss == "GANLoss":
                from loss import GANLoss

                self.loss.append(GANLoss(losses.GANLoss).to(self.gpu))
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
            shuffle=None,  # TODO None for now, this will be changed when ddp is applied
            num_workers=dataset.num_workers,
            pin_memory=True,
            sampler=None,  # TODO None for now, this will be changed when ddp is applied
            drop_last=True,
        )
        self.dataloader = sample_data(dataloader)

        log.info("Initialized the datasets")

    def train(self):
        for i in range(self.start_iters, self.end_iters):
            lr, hr = next(self.dataloader)
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            preds = self.generator(lr)

            loss = 0.0
            for l in self.loss:
                loss += l(preds, hr)

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
