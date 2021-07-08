from pathlib import Path

import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import streamlit as st
import torch
from omegaconf import DictConfig
from src.plmodule import LightningModule
from stqdm import stqdm
from torch.functional import Tensor
from torchvision.transforms.functional import to_pil_image

import utils


def additional_import(model_root_path: Path):
    # path = str(model_root_path / 'snapshot')
    # import sys

    # sys.path.insert(0, path)

    global DataModule
    import src.plmodule.data_module
    from src.plmodule import DataModule

    src.plmodule.data_module.chars74k.tqdm = stqdm

    # del sys.path[0]

    return


def main():
    pl.seed_everything(0)

    st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.write('# ' + __file__)

    model_root_path = utils.input_model_root_path(st)
    model_path = utils.input_model_path(model_root_path)

    additional_import(model_root_path)

    with st.sidebar:
        sample_type = utils.input_sample_type(st)
        st.markdown('---')

    config = utils.get_config(model_root_path)
    utils.display_config(st, config)
    config = DictConfig(config)
    st.markdown('---')

    lightning_module = LightningModule.load_from_checkpoint(str(model_path))
    lightning_module.eval()

    data_module = utils.get_data_module(DataModule, config.data_module)

    dataloader = data_module.get_dataloader(sample_type)
    dataset = dataloader.dataset

    with st.beta_expander('datalabel_bar'):
        utils.display_datalabel_bar(st, dataset)

    with st.sidebar:
        data_index = utils.input_data_index(st, dataset)

    x, y = dataset[data_index]

    with st.sidebar:
        pil_image = to_pil_image(x)
        st.image(pil_image)
        st.markdown('---')

    st.write('# model outputs')

    y_hat, m = utils.get_model_output(lightning_module, x)

    st.write(
        f"target: `{y['label']}`, pred: `{dataset.uniques['label'][y_hat.argmax()]}`"
    )
    st.write(f'entropy: `{m.entropy().item()}`')

    df = pd.DataFrame(
        {
            'raw': y_hat,
            'prob': m.probs,
            'logit': m.logits,
        },
        dataset.uniques['label'],
    )

    st.dataframe(df)

    display_columns = ['prob', 'raw']
    for st_col, key in zip(st.beta_columns(len(display_columns)), display_columns):
        with st_col:
            st.write(f'## {key}')
            fig = px.bar(df, y=key)
            st.plotly_chart(fig)

    return


if __name__ == "__main__":
    with torch.no_grad():
        main()
