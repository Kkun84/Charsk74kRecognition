from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class AdobeFontDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        data_type: Union[str, Iterable[str]],
        transform: Optional[Callable[[Image.Image], Any]] = transforms.ToTensor(),
        target_transform: Optional[Callable[[Dict[str, Union[str, int]]], Any]] = (
            lambda y: y['alphabet_id']
        ),
        upper: bool = True,
        lower: bool = True,
    ):
        self.transform = transform
        self.target_transform = target_transform

        root: Path = Path(root)
        self.data_type = data_type

        if isinstance(data_type, str):
            self.data_property = pd.read_csv(
                root / f'{self.data_type}.csv', index_col=0
            )
        else:
            self.data_property = pd.concat(
                [pd.read_csv(root / f'{i}.csv', index_col=0) for i in self.data_type]
            )

        if upper == False:
            self.data_property = self.data_property[
                ~self.data_property['alphabet'].str.startswith('cap')
            ]
        if lower == False:
            self.data_property = self.data_property[
                ~self.data_property['alphabet'].str.startswith('small')
            ]

        with open(root / 'list.yaml') as file:
            self.uniques = yaml.safe_load(file)
        self.has_uniques = {}
        for label in self.uniques.keys():
            self.has_uniques[label] = sorted(
                self.data_property[label].unique().tolist()
            )

        self.data: List[Tuple[Image.Image, Dict[str, Union[str, int]]]] = []
        self.data_property['image'] = None
        for index, item in tqdm(list(self.data_property.iterrows())):
            data_path = (
                root / item['split'] / (item['alphabet'] + '_' + item['font'] + '.png')
            )
            image = Image.open(data_path)
            self.data.append((image, dict(item)))
            self.data_property.loc[index, 'image'] = image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Any]:
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


if __name__ == "__main__":
    root = '/dataset/AdobeFontCharImages/splitted/'
    dataset = AdobeFontDataset(root=root, data_type='test_data', upper=True, lower=True)
    print(
        len(dataset),
        len(dataset.has_uniques['alphabet']),
        len(dataset.has_uniques['font']),
        dataset.has_uniques['alphabet'],
    )
    print(dataset[0])
    print(dataset[-1])
    print()
    assert len(dataset.has_uniques['alphabet']) == 52

    dataset = AdobeFontDataset(
        root=root, data_type='test_data', upper=False, lower=True
    )
    print(
        len(dataset),
        len(dataset.has_uniques['alphabet']),
        len(dataset.has_uniques['font']),
        dataset.has_uniques['alphabet'],
    )
    print()
    assert len(dataset.has_uniques['alphabet']) == 26

    dataset = AdobeFontDataset(
        root=root, data_type='train_data_0', upper=True, lower=True
    )
    print(
        len(dataset),
        len(dataset.has_uniques['alphabet']),
        len(dataset.has_uniques['font']),
        dataset.has_uniques['alphabet'],
    )
    print()
    assert len(dataset.has_uniques['alphabet']) == 52

    dataset = AdobeFontDataset(
        root=root,
        data_type=[
            'train_data_0',
            'train_data_1',
            'train_data_2',
            'train_data_3',
            'train_data_4',
        ],
        upper=True,
        lower=False,
    )
    print(
        len(dataset),
        len(dataset.has_uniques['alphabet']),
        len(dataset.has_uniques['font']),
        dataset.has_uniques['alphabet'],
    )
    print()
    assert len(dataset.has_uniques['alphabet']) == 26

    dataset = AdobeFontDataset(
        root=root,
        data_type=[
            'train_data_0',
            'train_data_1',
            'train_data_2',
            'train_data_3',
            'train_data_4',
            'test_data',
        ],
        upper=True,
        lower=True,
    )
    print(
        len(dataset),
        len(dataset.has_uniques['alphabet']),
        len(dataset.has_uniques['font']),
        dataset.has_uniques['alphabet'],
    )
    print()
    assert len(dataset.has_uniques['alphabet']) == 52

    dataset = AdobeFontDataset(
        root=root,
        data_type=[
            'train_data_0',
            'train_data_1',
            'train_data_2',
            'train_data_3',
            'train_data_4',
            'test_data',
        ],
        upper=False,
        lower=False,
    )
    print(
        len(dataset),
        len(dataset.has_uniques['alphabet']),
        len(dataset.has_uniques['font']),
        dataset.has_uniques['alphabet'],
    )
    print()
    assert len(dataset.font_list) == 0
    assert len(dataset.has_uniques['alphabet']) == 0
