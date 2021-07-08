import os
from pathlib import Path
from typing import Callable, Sequence, Union

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
import torch
import torchmetrics
import yaml
from torch import Tensor
from torch.distributions import Categorical


def get_config(model_root_path: Path) -> dict:
    yaml_path = model_root_path / '.hydra' / 'config.yaml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def display_config(st, config: dict) -> None:
    st.beta_expander('Display "config.yaml"').json(config)
    return


def input_model_root_path(st) -> Path:
    model_root_path_filter = Path(
        st.text_input(
            'Filter of models',
            value='outputs/evaluate/Chars74kImageDataset/2021-07-08/10-10-47-multiple/0',
        ),
        'checkpoints',
    )
    filtered_model_root_path_list = sorted(
        Path().glob(str(model_root_path_filter)), key=os.path.getmtime
    )
    filtered_model_root_path_list = [i.parent for i in filtered_model_root_path_list]
    model_root_path = st.selectbox(
        f'Select model from "{model_root_path_filter}"',
        filtered_model_root_path_list,
        index=len(filtered_model_root_path_list) - 1,
    )
    st.write(f'`{model_root_path}`')
    return model_root_path


def input_model_path(model_root_path: Path) -> Path:
    model_path_list = list(
        (model_root_path / 'checkpoints').glob('epoch=*-step=*.ckpt')
    )
    assert len(model_path_list) == 1
    model_path = model_path_list[0]
    return model_path


def input_use_gpus(st) -> int:
    st_col_1, st_col_2 = st.beta_columns(2)
    with st_col_1:
        use_gpus = st.number_input(
            'use_gpus',
            min_value=0,
            max_value=torch.cuda.torch.cuda.device_count(),
            value=1,
        )
    with st_col_2:
        st.write('Selected:')
        st.write(f'`{use_gpus}`')
    return use_gpus


def input_sample_type(st) -> str:
    st_col_1, st_col_2 = st.beta_columns(2)
    with st_col_1:
        sample_type = st.selectbox('sample_type', ['train', 'valid', 'test'], index=2)
    with st_col_2:
        st.write('Selected:')
        st.write(f'`{sample_type}`')
    return sample_type


@st.cache(suppress_st_warning=True)
def get_data_module(class_: Callable, config_data_module):
    data_module = class_(**(dict(config_data_module) | {'shuffle': False}))
    data_module.prepare_data()
    data_module.setup()
    return data_module


def reshape_predicted(predicted) -> tuple[Tensor, dict[str, Union[list[str], Tensor]]]:
    batch_x = [i[0] for i in predicted]
    x = torch.cat(batch_x, 0)

    batch_y = [i[1] for i in predicted]
    y: dict[str, Union[list[str], Tensor]] = {
        key: sum([list(i[key]) for i in batch_y], []) for key in batch_y[0].keys()
    }
    for key, value in y.items():
        if isinstance(value[0], Tensor):
            y[key] = torch.stack(value)

    for key, value in y.items():
        assert len(x) == len(value)
    return x, y


def calc_metrics(pred: Tensor, target: Tensor, num_classes: int) -> dict[str, float]:
    assert pred.shape[1:] == torch.Size([num_classes])
    assert pred.shape[:1] == target.shape

    accuracy = torchmetrics.functional.accuracy(pred.softmax(1), target)
    recall_macro = torchmetrics.functional.recall(
        pred.softmax(1),
        target,
        average='macro',
        num_classes=num_classes,
    )
    precision_macro = torchmetrics.functional.precision(
        pred.softmax(1),
        target,
        average='macro',
        num_classes=num_classes,
    )
    f1_macro = torchmetrics.functional.f1(
        pred.softmax(1),
        target,
        average='macro',
        num_classes=num_classes,
    )
    return {
        'accuracy': accuracy.item(),
        'recall_macro': recall_macro.item(),
        'precision_macro': precision_macro.item(),
        'f1_macro': f1_macro.item(),
    }


def display_metrics(st, pred: Tensor, target: Tensor, num_classes: int):
    metrics_dict = calc_metrics(pred, target, num_classes)
    for key, value in metrics_dict.items():
        st.write(f'{key}:', value)


def get_confusion_matrix(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    cm = torchmetrics.functional.confusion_matrix(
        pred, target, num_classes=num_classes
    ).cpu()
    return cm


def display_confusion_matrix(st, label_list: Sequence, cm, is_normalize: bool):
    col1, col2 = st.beta_columns(2)
    with col1:
        fig = px.imshow(
            cm / cm.sum(1, keepdims=True) if is_normalize else cm,
            labels={'x': 'Estimated', 'y': 'True', 'color': 'Count'},
            x=label_list,
            y=label_list,
            title='Confusion matrix',
        )
        fig.update(
            data=[
                {
                    'customdata': cm / cm.sum(1, keepdims=True) * 100,
                    'hovertemplate': 'True: %{y}<br>Estimated: %{x}<br>Count: %{z}<br>Percentage: %{customdata:.2f}<extra></extra>',
                }
            ]
        )
        fig.update_xaxes(side='top')
        st.plotly_chart(fig)
    with col2:
        plt.figure(figsize=[13, 10])
        sns.heatmap(
            cm / cm.sum(1, keepdims=True) if is_normalize else cm,
            # annot=True,
            # fmt='d',
            square=True,
            xticklabels=[i[-1] for i in label_list],
            yticklabels=[i[-1] for i in label_list],
        )
        plt.yticks(rotation=0)
        st.pyplot(plt, True)


def display_datalabel_bar(st, dataset):
    df = dataset.data_property['label'].value_counts().sort_index()
    fig = px.bar(df, x=df.index, y='label')
    st.plotly_chart(fig)


def input_data_index(st, dataset):
    st_col_1, st_col_2 = st.beta_columns(2)
    with st_col_1:
        data_label = st.selectbox('data_label', dataset.has_uniques['label'])
    with st_col_2:
        st.write('Selected:')
        st.write(f'`{data_label}`')
    st_col_1, st_col_2 = st.beta_columns(2)
    with st_col_1:
        data_num = (dataset.data_property['label'] == data_label).sum().item()
        data_index = st.number_input(f'data_index (0~{data_num - 1})', 0, data_num - 1)
    with st_col_2:
        st.write('Selected:')
        st.write(f'`{data_index}`')

    data_index = (dataset.data_property['label'] == data_label).index[data_index]
    return data_index


def get_model_output(lightning_module, x: Tensor) -> tuple[Tensor, Categorical]:
    assert x.dim() == 3
    y_hat = lightning_module.model(x[None])[0]
    m = Categorical(logits=lightning_module.model(x[None])[0])
    return y_hat, m
