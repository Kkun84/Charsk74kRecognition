from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def import_chars74k():
    import sys

    sys.path.append(str(Path('/', 'dataset', 'Chars74k')))
    import chars74k

    del sys.path[-1]
    return chars74k


chars74k = import_chars74k()


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        path: Union[str, Path],
        image_size: int,
        k: int,
        number: bool,
        upper: bool,
        lower: bool,
        good: bool,
        bad: bool,
        language: str = 'English',
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.path = path if isinstance(path, Path) else Path(path)

        assert 0 <= k < chars74k.N_SPLIT - 1
        self.k = k

        self.language = language
        self.number = number
        self.upper = upper
        self.lower = lower
        self.good = good
        self.bad = bad

        self.dims = (3, 100, 100)
        self.num_classes = (
            (10 if self.number else 0)
            + (26 if self.upper else 0)
            + (26 if self.lower else 0)
        )
        assert self.num_classes > 0

        self.transform = transforms.Compose(
            [transforms.Resize([image_size, image_size]), transforms.ToTensor()]
        )

    def prepare_data(self) -> None:
        # http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        kwargs = dict(
            root=self.path,
            transform=self.transform,
            target_transform=None,
            number=self.number,
            upper=self.upper,
            lower=self.lower,
            good=self.good,
            bad=self.bad,
            language=self.language,
        )
        if stage == 'fit' or stage is None:
            self.train_dataset = chars74k.Chars74kImageDataset(
                data_type=chars74k.data_type_list[0 : self.k]
                + chars74k.data_type_list[self.k + 1 : -1],
                **kwargs,
            )
            self.val_dataset = chars74k.Chars74kImageDataset(
                data_type=chars74k.data_type_list[self.k], **kwargs
            )
        if stage == 'test' or stage is None:
            self.test_dataset = chars74k.Chars74kImageDataset(
                data_type=chars74k.data_type_list[-1], **kwargs
            )

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.test_dataset)

    def make_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_dataset(self, sample_type: str) -> Dataset:
        return {
            'train': self.train_dataset,
            'valid': self.val_dataset,
            'validation': self.val_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset,
        }[sample_type]

    def get_dataloader(self, sample_type: str) -> DataLoader:
        return {
            'train': self.train_dataloader,
            'valid': self.val_dataloader,
            'validation': self.val_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader,
        }[sample_type]()


if __name__ == "__main__":
    print(chars74k.data_type_list)
    print(chars74k.N_SPLIT)
    print(chars74k.Chars74kImageDataset)

    datamodule = DataModule(
        batch_size=64,
        num_workers=4,
        path=Path('/', 'dataset', 'Chars74k', 'splitted'),
        image_size=100,
        k=0,
        number=True,
        upper=True,
        lower=True,
        good=True,
        bad=True,
    )
    print(datamodule)
    datamodule.prepare_data()
    datamodule.setup()
