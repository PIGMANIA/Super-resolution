import os
import hydra
import builtins
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import torchvision.utils as vutils

import torch.multiprocessing as mp
import torch.distributed as dist

from data.dataset import Dataset
from models import define_model


class Trainer:
    def __init__(self, gpu, cfg) -> None:
        ### GPU Device setting
        self.gpu = gpu
        self.ngpus_per_node = torch.cuda.device_count()

        if cfg.train.ddp.distributed and self.gpu != 0:

            def print_pass(*args):
                pass

            builtins.print = print_pass

        ### Common setting
        self.save_path = cfg.train.common.ckpt_dir
        os.makedirs(self.save_path, exist_ok=True)

        self.gan_train = cfg.train.common.GAN
        self.scale = cfg.models.generator.scale
        self.start_iters = 0
        self.end_iters = cfg.train.common.iteration
        self.seed = cfg.train.common.seed
        self.use_wandb = cfg.train.common.use_wandb
        self.save_img_every = cfg.train.common.save_img_every
        self.save_model_every = cfg.train.common.save_model_every

        ### Model setting
        self.generator = None
        self.discriminator = None

        ### Optimizer setting
        self.g_optim = None
        self.d_optim = None

        ### Scheduler setting
        self.g_scheduler = None
        self.d_scheduler = None

        ### Loss setting
        self.l1loss = None
        self.perceptual_loss = None
        self.gan_loss = None

        ### Dataset setting
        self.distributed = cfg.train.ddp.distributed
        self.dataloader = None

        ### Initializer
        self._init_model(cfg.models)
        if cfg.train.ddp.distributed:
            self._init_distributed_data_parallel(cfg)
        self._init_optim(cfg.train.optim)
        self._init_loss(cfg.train.loss)
        self._load_state_dict(cfg.models)
        self._init_scheduler(cfg)
        self._init_dataset(cfg)

        ### Train
        self.train()

    def _init_model(self, model):
        self.generator, self.discriminator = define_model(
            model, self.gpu, self.gan_train
        )
        print("Initialized the model")

    def _load_state_dict(self, model):
        if model.generator.path:
            print("Train the generator with checkpoint")
            print(f"Loading the checkpoint from : {model.generator.path}")
            ckpt = torch.load(
                model.generator.path, map_location=lambda storage, loc: storage
            )
            if len(ckpt) == 3:
                self.num_iteration = ckpt["iteration"]
                if isinstance(model, nn.DataParallel):
                    self.generator.module.load_state_dict(ckpt["g"])
                else:
                    self.generator.load_state_dict(ckpt["g"])
                self.g_optim.load_state_dict(ckpt["g_optim"])
                self.start_iters = ckpt["iteration"] + 1
            else:
                self.generator.load_state_dict(ckpt)
        else:
            print("Train the generator without checkpoint")

        if self.gan_train:
            if model.discriminator.path:
                print("Train the discriminator with checkpoint")
                print(
                    f"Loading the checkpoint from : {model.discriminator.path}"
                )
                ckpt = torch.load(
                    model.discriminator.path,
                    map_location="cuda:{}".format(self.gpu),
                )
                if len(ckpt) == 3:
                    self.num_iteration = ckpt["iteration"]
                    if isinstance(model, nn.DataParallel):
                        self.discriminator.module.load_state_dict(ckpt["d"])
                    else:
                        self.discriminator.load_state_dict(ckpt["d"])
                    self.d_optim.load_state_dict(ckpt["d_optim"])
                else:
                    self.discriminator.load_state_dict(ckpt)
            else:
                print("Train the discriminator without checkpoint")

        print("Initialized the checkpoints")

    def _init_loss(self, losses):
        if losses.lists[0] == "MAE":
            self.l1loss = nn.L1Loss().to(self.gpu)
        elif losses.lists[0] == "Charbonnier":
            from loss import L1_Charbonnier_loss

            self.l1loss = L1_Charbonnier_loss().to(self.gpu)

        if self.gan_train:
            from loss import GANLoss, PerceptualLoss

            self.gan_loss = GANLoss(losses.GANLoss).to(self.gpu)
            self.perceptual_loss = PerceptualLoss(losses.PerceptualLoss).to(
                self.gpu
            )

        print("Initialized the losses functions")

    def _init_optim(self, optims):
        if optims.type == "Adam":
            self.g_optim = torch.optim.Adam(
                self.generator.parameters(),
                lr=optims.Adam.lr,
                betas=(optims.Adam.betas[0], optims.Adam.betas[1]),
                weight_decay=optims.Adam.weight_decay,
            )

            if self.discriminator:
                self.d_optim = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=optims.Adam.lr,
                    betas=(optims.Adam.betas[0], optims.Adam.betas[1]),
                    weight_decay=optims.Adam.weight_decay,
                )
        print("Initialized the optimizers")

    def _init_scheduler(self, cfg):
        if not cfg.models.generator.path:
            self.g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.g_optim,
                cfg.train.scheduler.g_milestones,
                cfg.train.g_gamma,
                last_epoch=self.start_iters,
            )
        else:
            self.g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.g_optim,
                cfg.train.scheduler.g_milestones,
                cfg.train.scheduler.g_gamma,
            )

        if self.gan_train:
            if len(cfg.models.discriminator.path) > 0:
                self.d_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.d_optim,
                    cfg.train.scheduler.d_milestones,
                    cfg.train.scheduler.d_gamma,
                    last_epoch=self.start_iters,
                )
            else:
                self.d_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.d_optim,
                    cfg.train.scheduler.d_milestones,
                    cfg.train.scheduler.d_gamma,
                )
        print("Initialized the schedulers")

    def _init_dataset(self, cfg):
        def sample_data(loader):
            while True:
                for batch in loader:
                    yield batch

        cfg.train.dataset.num_workers = 4 * torch.cuda.device_count()
        train_dataset = Dataset(cfg)

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=cfg.train.ddp.world_size,
                rank=cfg.train.ddp.rank,
            )
        else:
            train_sampler = None

        dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.train.dataset.batch_size,
            shuffle=(train_sampler is None),
            num_workers=cfg.train.dataset.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
        self.dataloader = sample_data(dataloader)

        print("Initialized the datasets")

    def _init_distributed_data_parallel(self, cfg):
        cfg.train.ddp.rank = cfg.train.ddp.nr * cfg.train.ddp.gpus + self.gpu

        dist.init_process_group(
            backend=cfg.train.ddp.dist_backend,
            init_method=cfg.train.ddp.dist_url,
            world_size=cfg.train.ddp.world_size,
            rank=cfg.train.ddp.rank,
        )

        torch.cuda.set_device(self.gpu)
        self.generator.to(self.gpu)

        cfg.train.dataset.batch_size = int(
            cfg.train.dataset.batch_size / self.ngpus_per_node
        )

        self.generator = torch.nn.parallel.DistributedDataParallel(
            self.generator, device_ids=[self.gpu]
        )

        if self.gan_train:
            self.discriminator.to(self.gpu)
            self.discriminator = torch.nn.parallel.DistributedDataParallel(
                self.discriminator, device_ids=[self.gpu]
            )

        print("Initialized the Distributed Data Parallel")

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
            self.g_scheduler.step()

            results = torch.cat(
                (
                    hr.detach(),
                    F.interpolate(
                        lr, scale_factor=self.scale, mode="nearest"
                    ).detach(),
                    preds.detach(),
                ),
                2,
            )

            if self.gpu == 0:
                if i % self.save_img_every == 0:
                    vutils.save_image(
                        results, os.path.join(self.save_path, f"preds.jpg")
                    )

                if i % self.save_model_every == 0:
                    if isinstance(self.generator, nn.DataParallel):
                        g_state_dict = self.generator.module.state_dict()
                    else:
                        g_state_dict = self.generator.state_dict()

                    torch.save(
                        {
                            "g": g_state_dict,
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

            self.g_scheduler.step()
            self.d_scheduler.step()

            results = torch.cat(
                (
                    hr.detach(),
                    F.interpolate(
                        lr, scale_factor=self.scale, mode="nearest"
                    ).detach(),
                    preds.detach(),
                ),
                2,
            )

            if self.gpu == 0:
                if i % self.save_img_every == 0:
                    vutils.save_image(
                        results, os.path.join(self.save_path, f"preds.jpg")
                    )
                if i % self.save_model_every == 0:
                    if isinstance(self.generator, nn.DataParallel):
                        g_state_dict = self.generator.module.state_dict()
                        d_state_dict = self.discriminator.module.state_dict()
                    else:
                        g_state_dict = self.generator.state_dict()
                        d_state_dict = self.discriminator.state_dict()

                    torch.save(
                        {
                            "g": g_state_dict,
                            "g_optim": self.g_optim.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(
                            self.save_path, f"g_{str(i).zfill(6)}.pth"
                        ),
                    )

                    torch.save(
                        {
                            "d": d_state_dict,
                            "d_optim": self.d_optim.state_dict(),
                            "iteration": i,
                        },
                        os.path.join(
                            self.save_path, f"d_{str(i).zfill(6)}.pth"
                        ),
                    )

    def train(self):
        if not self.gan_train:
            self.train_psnr()
        else:
            self.train_gan()


@hydra.main(config_path="../configs/", config_name="train.yaml")
def main(cfg):
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(cfg.train.common.seed)

    if torch.cuda.device_count() > 1:
        print("Train with multiple GPUs")
        cfg.train.ddp.distributed = True
        gpus = torch.cuda.device_count()
        cfg.train.ddp.world_size = gpus * cfg.train.ddp.nodes
        mp.spawn(
            Trainer,
            nprocs=gpus,
            args=(cfg,),
        )
        dist.destroy_process_group()
    else:
        print("Train with single GPUs")
        Trainer(0, cfg)


if __name__ == "__main__":
    main()
