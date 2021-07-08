from typing import Any, Callable, Sequence, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from src.model import Model
from torch import Tensor


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: bool,
        num_classes: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.accuracy = torchmetrics.Accuracy()

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[Callable]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], []

    def forward(self, batch: Union[Tensor, tuple[Tensor, dict[str, Any]]]):
        if isinstance(batch, Tensor):
            x = batch
            return self.model(x)
        elif isinstance(batch, tuple):
            x, y = batch
            assert isinstance(x, Tensor)
            assert isinstance(y, dict)
            return self.model(x), y
        assert False, type(batch)

    def _step(self, batch: Sequence[Tensor]) -> dict[str, Any]:
        x = batch[0]
        y = batch[1]['label_id']
        batch_size = len(y)
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy: float = self.accuracy(y_hat.softmax(1), y)
        return {'batch_size': batch_size, 'loss': loss, 'accuracy': accuracy}

    def training_step(self, batch: Sequence[Tensor], batch_idx: int) -> Tensor:
        items = self._step(batch)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('metrics/train_loss', loss, on_step=True)
        self.log('metrics/train_accuracy', accuracy, on_step=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Sequence[Tensor], batch_idx: int
    ) -> dict[str, Any]:
        items = self._step(batch)
        batch_size = items['batch_size']
        loss = items['loss'] * batch_size
        accuracy = items['accuracy'] * batch_size
        return {
            'batch_size': batch_size,
            'loss': loss,
            'accuracy': accuracy,
        }

    def test_step(self, batch: Sequence[Tensor], batch_idx: int) -> dict[str, Any]:
        items = self._step(batch)
        batch_size = items['batch_size']
        loss = items['loss'] * batch_size
        accuracy = items['accuracy'] * batch_size
        return {
            'batch_size': batch_size,
            'loss': loss,
            'accuracy': accuracy,
        }

    def _epoch_end(self, step_outputs: list[dict[str, Any]]) -> dict[str, Any]:
        data_size = sum([i['batch_size'] for i in step_outputs])
        loss = sum([i['loss'] for i in step_outputs]) / data_size
        accuracy = sum([i['accuracy'] for i in step_outputs]) / data_size
        return {'loss': loss, 'accuracy': accuracy}

    def validation_epoch_end(self, step_outputs: list[dict[str, Any]]):
        items = self._epoch_end(step_outputs)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('metrics/valid_loss', loss, prog_bar=True)
        self.log('metrics/valid_accuracy', accuracy, prog_bar=True)

    def test_epoch_end(self, step_outputs: list[dict[str, Any]]):
        items = self._epoch_end(step_outputs)
        loss = items['loss']
        accuracy = items['accuracy']
        self.log('metrics/test_loss', loss)
        self.log('metrics/test_accuracy', accuracy)


if __name__ == '__main__':
    from torchinfo import summary

    lightning_module = LightningModule(
        model_name='resnet18',
        pretrained=False,
        num_classes=10,
        lr=0.001,
    )

    summary(lightning_module, [2, 3, 100, 100])
