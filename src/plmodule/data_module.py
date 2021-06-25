from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


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
