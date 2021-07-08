from pathlib import Path

import pytorch_lightning as pl
import streamlit as st
import torch
from omegaconf import DictConfig
from src.plmodule import LightningModule
from stqdm import stqdm

import utils


def additional_import(model_root_path: Path):
    # path = str(model_root_path / 'snapshot')
    # import sys

    # sys.path.insert(0, path)

    global DataModule
    from src.plmodule import DataModule

    import src.plmodule.data_module

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
        use_gpus = utils.input_use_gpus(st)
        st.markdown('---')

    with st.sidebar:
        sample_type = utils.input_sample_type(st)
        st.markdown('---')

    config = utils.get_config(model_root_path)
    utils.display_config(st, config)
    config = DictConfig(config)

    trainer = pl.Trainer(
        **(dict(config.trainer) | {'gpus': use_gpus}), callbacks=None, logger=None
    )

    lightning_module = LightningModule.load_from_checkpoint(str(model_path))
    lightning_module.eval()

    data_module = utils.get_data_module(DataModule, config.data_module)

    predicted = trainer.predict(
        model=lightning_module, dataloaders=data_module.get_dataloader(sample_type)
    )
    x, y = utils.reshape_predicted(predicted)

    label_list = data_module.test_dataset.uniques['label']

    st.write('# metrics')
    utils.display_metrics(st, x, y['label_id'], config.lightning_module.num_classes)

    st.write('# Confusion matrix')
    with st.beta_expander('Confusion matrix'):
        cm = utils.get_confusion_matrix(
            x.softmax(1), y['label_id'], config.lightning_module.num_classes
        )

        st.write(f'正規化済みなら行の総和が1')
        for is_normalize in [False, True]:
            st.write(f'## 正規化: `{is_normalize}`')
            utils.display_confusion_matrix(st, label_list, cm, is_normalize)
            st.markdown('---')

    return


if __name__ == "__main__":
    with torch.no_grad():
        main()
