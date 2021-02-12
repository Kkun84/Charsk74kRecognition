from logging import getLogger
from torch.functional import Tensor
from tqdm import tqdm
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from typing import Dict, Iterable, Union, Optional, Tuple, Any, Callable, List
import yaml


logger = getLogger(__name__)


class Chars74kImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        data_type: Union[str, Iterable[str]],
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[Dict[str, Union[str, int]]], Any]] = (
            lambda y: y['label_id']
        ),
        language: str = 'English',
        number: bool = True,
        upper: bool = True,
        lower: bool = True,
        good: bool = True,
        bad: bool = True,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform

        root: Path = Path(root, language, 'Image')
        self.data_type = data_type

        if isinstance(data_type, str):
            self.data_property = pd.read_csv(
                root / f'{self.data_type}.csv', index_col=0
            )
        else:
            self.data_property = pd.concat(
                [pd.read_csv(root / f'{i}.csv', index_col=0) for i in self.data_type]
            )

        with open(root / 'list.yaml') as file:
            self.uniques = yaml.safe_load(file)

        assert any([i is True for i in [number, upper, lower]])
        f = self.uniques['label'].index
        if number is False:
            self.data_property = self.data_property[
                (self.data_property['label'].apply(f) < f('0'))
                | (f('9') < self.data_property['label'].apply(f))
            ]
        if upper is False:
            self.data_property = self.data_property[
                (self.data_property['label'].apply(f) < f('A'))
                | (f('Z') < self.data_property['label'].apply(f))
            ]
        if lower is False:
            self.data_property = self.data_property[
                (self.data_property['label'].apply(f) < f('a'))
                | (f('z') < self.data_property['label'].apply(f))
            ]

        assert any([i is True for i in [good, bad]])
        if good is False:
            self.data_property = self.data_property[
                ~(self.data_property['quality'] == 'good')
            ]
        if bad is False:
            self.data_property = self.data_property[
                ~(self.data_property['quality'] == 'bad')
            ]

        self.data_property = self.data_property.sort_index()

        self.has_uniques = {}
        for column in self.uniques.keys():
            column_id = column + '_id'
            self.has_uniques[column_id] = sorted(
                self.data_property[column_id].unique().tolist()
            )
            self.has_uniques[column] = sorted(
                self.data_property[column].unique().tolist()
            )

        self.data: List[Tuple[Image.Image, Dict[str, Union[str, int]]]] = []
        self.data_property['image'] = None
        for index, item in tqdm(list(self.data_property.iterrows())):
            data_path = root / item['split'] / (item['name'] + '.png')
            image = Image.open(data_path)
            self.data.append((image, dict(item)))
            self.data_property.loc[index, 'image'] = image
        return

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Any]:
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


if __name__ == "__main__":
    root = 'splitted'

    dataset = Chars74kImageDataset(
        root=root,
        data_type=['test_data'],
        number=False,
        upper=False,
        # lower=False,
        good=False,
        # bad=False,
    )
    print(
        len(dataset),
        len(dataset.has_uniques),
    )
    print(dataset[0])
    print(dataset[-1])
    print()
    assert (
        (ord('a') <= dataset.data_property['label'].apply(ord))
        & (dataset.data_property['label'].apply(ord) <= ord('z'))
    ).all()
    assert (dataset.data_property['quality'] == 'bad').all()

    dataset = Chars74kImageDataset(
        root=root,
        data_type=['test_data'],
        number=False,
        upper=False,
        # lower=False,
        good=False,
        # bad=False,
    )
    print(
        len(dataset),
        len(dataset.has_uniques),
    )
    print(dataset[0])
    print(dataset[-1])
    print()
    assert (
        (ord('a') <= dataset.data_property['label'].apply(ord))
        & (dataset.data_property['label'].apply(ord) <= ord('z'))
    ).all()
    assert (dataset.data_property['quality'] == 'bad').all()

    dataset = Chars74kImageDataset(
        root=root,
        data_type=['train_data_0', 'train_data_1', 'train_data_2', 'train_data_3'],
        number=False,
        # upper=False,
        lower=False,
        # good=False,
        # bad=False,
    )
    print(
        len(dataset),
        len(dataset.has_uniques),
    )
    print(dataset[0])
    print(dataset[-1])
    print()
    assert (
        (ord('A') <= dataset.data_property['label'].apply(ord))
        & (dataset.data_property['label'].apply(ord) <= ord('Z'))
    ).all()

    dataset = Chars74kImageDataset(
        root=root,
        data_type=['train_data_4'],
        # number=False,
        upper=False,
        lower=False,
        # good=False,
        bad=False,
    )
    print(
        len(dataset),
        len(dataset.has_uniques),
    )
    print(dataset[0])
    print(dataset[-1])
    print()
    assert (
        (ord('0') <= dataset.data_property['label'].apply(ord))
        & (dataset.data_property['label'].apply(ord) <= ord('9'))
    ).all()
    assert (dataset.data_property['quality'] == 'good').all()

    dataset = Chars74kImageDataset(
        root=root,
        data_type=[
            'train_data_0',
            'train_data_1',
            'train_data_2',
            'train_data_3',
            'train_data_4',
            'test_data',
        ],
    )
    print(
        len(dataset),
        len(dataset.has_uniques),
    )
    print(dataset[0])
    print(dataset[-1])
    print()
