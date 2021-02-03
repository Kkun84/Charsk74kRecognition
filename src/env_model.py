from logging import getLogger

from typing import Any, Dict, List, Iterable, Union
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data


logger = getLogger(__name__)


class EnvModel(pl.LightningModule):
    def __init__(
        self,
        output_n: int,
        pool_mode: str,
        input_n: int = 3,
        # optimizer: str = None,
        # lr: float = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_n = input_n
        self.output_n = output_n

        self.pool_mode = pool_mode

        self.accuracy = pl.metrics.Accuracy()

        self.f_1 = nn.Sequential(
            nn.Conv2d(input_n, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
        )

        self.f_2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_n),
        )

    @auto_move_data
    def encode(
        self, patch_set: Union[Iterable[Tensor], Tensor]
    ) -> Union[List[Tensor], Tensor]:
        if isinstance(patch_set, Tensor):
            x = patch_set.reshape(
                patch_set.shape[0] * patch_set.shape[1], *patch_set.shape[2:]
            )
            x = self.f_1(x)
            x = x.reshape(*patch_set.shape[:2], *x.shape[1:])
        elif isinstance(patch_set, list):
            x = patch_set
            x = [self.f_1(x) for x in x]
        else:
            assert False
        return x

    @auto_move_data
    def pool(
        self, feature_set: Union[Iterable[Tensor], Tensor], keepdim: bool = False
    ) -> Tensor:
        if self.pool_mode == 'max':
            pool = lambda *args, **kwargs,: torch.max(*args, **kwargs)[0]
        else:
            pool = getattr(torch, self.pool_mode)

        if isinstance(feature_set, Tensor):
            x = pool(feature_set, 1, keepdim=keepdim)
        elif isinstance(feature_set, list):
            x = [pool(i, 0, keepdim=keepdim) for i in feature_set]
            x = torch.stack(x)
        else:
            assert False

        return x

    @auto_move_data
    def decode(self, feature: Tensor) -> Tensor:
        x = self.f_2(feature)
        return x

    @auto_move_data
    def forward(self, patch_set: Union[Iterable[Tensor], Tensor]) -> Tensor:
        feature_set = self.encode(patch_set)
        features = self.pool(feature_set)
        output = self.decode(features)
        return output

    # def configure_optimizers(self) -> optim.Optimizer:
    #     optimizer = getattr(optim, self.hparams.optimizer)(
    #         self.parameters(), lr=self.hparams.lr
    #     )
    #     return optimizer

    def _step(self, batch: List[Tensor]) -> Dict[str, Any]:
        x, y = batch
        batch_size = len(y)
        y_hat = self.forward(x)
        if y.device != y_hat.device:
            y = y.to(y_hat.device)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return {'batch_size': batch_size, 'loss': loss, 'accuracy': accuracy}

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        items = self._step(batch)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('train_loss', loss, on_step=True)
        self.log('train_accuracy', accuracy, on_step=True, prog_bar=True)
        return loss, accuracy

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str, Any]:
        items = self._step(batch)
        batch_size = items['batch_size']
        loss = items['loss'] * batch_size
        accuracy = items['accuracy'] * batch_size
        return {
            'batch_size': batch_size,
            'loss': loss,
            'accuracy': accuracy,
        }

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Dict[str, Any]:
        items = self._step(batch)
        batch_size = items['batch_size']
        loss = items['loss'] * batch_size
        accuracy = items['accuracy'] * batch_size
        return {
            'batch_size': batch_size,
            'loss': loss,
            'accuracy': accuracy,
        }

    def _epoch_end(self, step_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        data_size = sum([i['batch_size'] for i in step_outputs])
        loss = sum([i['loss'] for i in step_outputs]) / data_size
        accuracy = sum([i['accuracy'] for i in step_outputs]) / data_size
        return {'loss': loss, 'accuracy': accuracy}

    def validation_epoch_end(self, step_outputs: List[Dict[str, Any]]):
        items = self._epoch_end(step_outputs)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_accuracy', accuracy, prog_bar=True)

    def test_epoch_end(self, step_outputs: List[Dict[str, Any]]):
        items = self._epoch_end(step_outputs)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)


# if __name__ == '__main__':
#     from torchsummary import summary

#     image_size = 100
#     patch_size = 25
#     output_n = 2
#     pool_mode = 'max'
#     patch_num_min = 1
#     patch_num_max = 5

#     model = EnvModel(
#         output_n,
#         pool_mode,
#         patch_num_min,
#         patch_num_max,
#     )
#     summary(model)

#     batch_size = 10
#     image = (
#         torch.arange(image_size ** 2)
#         .float()
#         .reshape(1, 1, *[image_size] * 2)
#         .expand(batch_size, -1, -1, -1)
#     )

#     x = sm.cutout_patch2d(
#         image,
#         torch.randint(patch_num_max, patch_num_max + 1, [batch_size]),
#         patch_size,
#     )
#     x = torch.stack(x)
#     y = model(x)

#     print('image', image.shape)
#     print('x', x.shape)
#     print('y', y.shape)
#     print()

#     x = sm.cutout_patch2d(
#         image,
#         torch.randint(patch_num_min, patch_num_max + 1, [batch_size]),
#         patch_size,
#     )
#     y = model(x)

#     print('image', image.shape)
#     print('x', *[i.shape for i in x], '', sep='\n')
#     print('y', y.shape)
