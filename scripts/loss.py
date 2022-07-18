import os
import torch
import torch.nn.functional
import torchvision
import torch.nn as nn


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class GANLoss(nn.Module):
    def __init__(self, gan):
        super(GANLoss, self).__init__()
        # target_real_label=1.0, target_fake_label=0.0
        self.loss_weight = gan.loss_weight
        self.register_buffer("real_label", torch.tensor(gan.real_label_val))
        self.register_buffer("fake_label", torch.tensor(gan.fake_label_val))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss_weight * self.loss(input, target_tensor)


class PerceptualLoss(torch.nn.Module):
    r"""Where VGG19 represents the feature map of 7/8/35/36th layer in pretrained VGG19 model.
    `"Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" <https://arxiv.org/pdf/1609.04802.pdf>`_
    `"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" <https://arxiv.org/pdf/1809.00219.pdf>`_
    `"Perceptual Extreme Super Resolution Network with Receptive Field Block" <https://arxiv.org/pdf/2005.12597.pdf>`_
    A loss defined on feature maps of higher level features from deeper network layers
    with more potential to focus on the content of the images. We refer to this network
    as SRGAN in the following.
    """

    def __init__(self, cfg) -> None:
        """Constructing characteristic loss function of VGG network. For VGG19 35th layer.
        Args:
            feature_layer (int): How many layers in VGG19. (Default: 35).
        Notes:
            features(
              (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU(inplace=True)
              (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): ReLU(inplace=True)
              (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (6): ReLU(inplace=True)
              (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (8): ReLU(inplace=True)
              (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (11): ReLU(inplace=True)
              (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (13): ReLU(inplace=True)
              (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (15): ReLU(inplace=True)
              (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (17): ReLU(inplace=True)
              (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (20): ReLU(inplace=True)
              (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (22): ReLU(inplace=True)
              (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (24): ReLU(inplace=True)
              (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (26): ReLU(inplace=True)
              (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (29): ReLU(inplace=True)
              (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (31): ReLU(inplace=True)
              (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (33): ReLU(inplace=True)
              (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (35): ReLU(inplace=True)
              (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
        """
        super(PerceptualLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(
            *list(model.features.children())[: cfg.feature_layer]
        ).eval()
        # Freeze parameters. Don't train.
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        self.loss_weight = cfg.loss_weight

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        vgg_loss = torch.nn.functional.l1_loss(
            self.features(source), self.features(target)
        )

        return self.loss_weight * vgg_loss
