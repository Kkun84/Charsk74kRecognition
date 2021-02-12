import os
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import more_itertools
import pandas as pd
import pfrl
import pytorch_lightning as pl
import streamlit as st
import torch
import torch.nn.functional as F
import yaml
from hydra.utils import DictConfig
from torch import Tensor
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from src.agent_model import AgentModel
from src.dataset import AdobeFontDataset
from src.env import PatchSetsClassificationEnv
from src.env_model import EnvModel

logger = getLogger(__name__)


@st.cache()
def make_dataset(config_dataset: DictConfig, data_type: List[str]) -> AdobeFontDataset:
    config_dataset_test = config_dataset.test.copy()
    config_dataset_test['data_type'] = data_type
    dataset = AdobeFontDataset(transform=transforms.ToTensor(), **config_dataset_test)
    return dataset


@st.cache(allow_output_mutation=True)
def make_env_model(
    config_env_model: DictConfig, env_model_path: str, dataset: AdobeFontDataset, device
) -> EnvModel:
    env_model = EnvModel(**config_env_model)
    env_model.load_state_dict(torch.load(env_model_path, map_location='cpu'))
    env_model: EnvModel = env_model.to(device)
    env_model.eval()
    summary(env_model)
    return env_model


@st.cache(allow_output_mutation=True)
def make_agent_model(
    config_agent_model: DictConfig, agent_model_path: str, device
) -> pfrl.agent.Agent:
    if 'input_n' in config_agent_model:
        agent_model = AgentModel(**config_agent_model)
    else:
        agent_model = AgentModel(**config_agent_model, input_n=2)
    agent_model.load_state_dict(torch.load(agent_model_path, map_location='cpu'))
    agent_model = agent_model.to(device)
    agent_model.eval()
    summary(agent_model)
    return agent_model


# @st.cache(hash_funcs={Tensor: lambda x: x.cpu().detach().numpy()})
def make_all_patch(image: Tensor, patch_size: Dict[str, int]) -> Tensor:
    height, width = image.shape[1:]
    y, x = torch.meshgrid(
        torch.arange(height - patch_size['y'] + 1),
        torch.arange(width - patch_size['x'] + 1),
    )
    x = x.flatten()
    y = y.flatten()
    all_patch = [
        PatchSetsClassificationEnv.make_patch(image, u, v, patch_size)
        for u, v in zip(x, y)
    ]
    all_patch = torch.stack(all_patch)
    return all_patch


# @st.cache(hash_funcs={Tensor: lambda x: x.cpu().detach().numpy()})
def predict_all_patch(
    env_model: EnvModel,
    all_patch: Tensor,
    additional_patch_set: Tensor = None,
):
    assert all_patch.dim() == 4, all_patch.shape
    assert len(all_patch) > 1, all_patch.shape
    all_patch = all_patch[:, None]
    if additional_patch_set is not None:
        all_patch = torch.cat(
            [
                all_patch,
                additional_patch_set.expand(all_patch.shape[0], -1, -1, -1, -1),
            ],
            1,
        )
    assert all_patch.dim() == 5, all_patch.shape
    predicted_all = env_model(all_patch)
    assert predicted_all.dim() == 2, predicted_all.shape
    assert len(predicted_all) == len(all_patch), predicted_all.shape
    return predicted_all


# @st.cache(hash_funcs={Tensor: lambda x: x.cpu().detach().numpy()})
def make_loss_map(
    predicted_all: Tensor, target: int, image: Tensor, patch_size: Dict[str, int]
) -> Tensor:
    loss_map = F.cross_entropy(
        predicted_all,
        torch.full(
            [len(predicted_all)],
            target,
            dtype=torch.long,
            device=predicted_all.device,
        ),
        reduction='none',
    ).reshape(
        1,
        image.shape[1] - patch_size['y'] + 1,
        image.shape[2] - patch_size['x'] + 1,
    )
    return loss_map


# @st.cache(hash_funcs={Tensor: lambda x: x.cpu().detach().numpy()})
def make_action_df(loss_map: Tensor, predicted_all: Tensor, label_list) -> pd.DataFrame:
    loss_sorted, loss_ranking = loss_map.reshape(-1).sort()
    loss_ranking_x = loss_ranking % loss_map.shape[2]
    loss_ranking_y = loss_ranking // loss_map.shape[2]

    df = pd.DataFrame(
        dict(
            action=loss_ranking.cpu().detach(),
            x=loss_ranking_x.cpu().detach(),
            y=loss_ranking_y.cpu().detach(),
            loss=loss_sorted.cpu().detach(),
            **{
                alphabet: data
                for alphabet, data in zip(
                    label_list,
                    predicted_all.softmax(1).T.cpu().detach(),
                )
            },
        ),
    )
    return df


def one_step(
    env: PatchSetsClassificationEnv,
    env_model: EnvModel,
    agent_model: AgentModel,
    patch_size: Dict[str, int],
    image: Tensor,
    target: int,
    step: int,
    default_action_mode: str,
):
    (
        col_loss_map,
        col_state,
        col_actoin_map,
        col_action_ranking,
        col_patch_select,
    ) = st.beta_columns([0.01, 2, 4, 4, 2])

    # with col_loss_map:
    #     all_patch = make_all_patch(image, patch_size)
    #     if step == 0:
    #         predicted_all = predict_all_patch(env_model, all_patch)
    #     else:
    #         predicted_all = predict_all_patch(
    #             env_model,
    #             all_patch,
    #             additional_patch_set=torch.stack(env.trajectory['patch']),
    #         )
    #     loss_map = make_loss_map(predicted_all, target, image, patch_size)
    #     st.write('Loss map')
    #     fig = plt.figure()
    #     plt.imshow(loss_map.cpu().detach().numpy()[0])
    #     plt.colorbar()
    #     st.pyplot(fig, True)
    #     plt.close()

    with col_state:
        obs = env.trajectory['observation'][-1]

        st.write('State')
        fig = plt.figure()
        plt.imshow(obs[0].cpu().detach().numpy())
        plt.colorbar()
        st.pyplot(fig, True)
        plt.close()

        fig = plt.figure()
        plt.imshow(obs[1].cpu().detach().numpy(), cmap='tab10', vmin=0, vmax=10)
        plt.colorbar()
        st.pyplot(fig, True)
        plt.close()

    with col_actoin_map:
        obs = env.trajectory['observation'][-1]
        action, policy = agent_model(obs[None])
        select_probs = action.probs[0].clone()

        select_probs_map = select_probs.reshape(
            image.shape[1] - patch_size['y'] + 1,
            image.shape[2] - patch_size['x'] + 1,
        )
        st.write('Select prob. map')
        fig = plt.figure()
        plt.imshow(select_probs_map.cpu().detach().numpy())
        plt.colorbar()
        st.pyplot(fig, True)
        plt.close()

    with col_action_ranking:
        st.write('entropy:', action.entropy().item())

        select_ranking = torch.zeros_like(select_probs, dtype=torch.float)
        select_ranking[select_probs.argsort(descending=True)] = torch.arange(
            len(select_probs), dtype=torch.float
        )
        select_ranking_map = select_ranking.reshape(select_probs_map.shape)
        select_ranking_map[select_probs_map == 0] = float('nan')
        fig = plt.figure()
        plt.imshow(select_ranking_map.cpu().detach().numpy(), cmap='plasma_r')
        plt.colorbar()
        st.pyplot(fig, True)
        plt.close()

    if default_action_mode == 'RL':
        best_x = (select_probs.argmax() % (image.shape[2] - patch_size['x'] + 1)).item()
        best_y = (
            select_probs.argmax() // (image.shape[2] - patch_size['y'] + 1)
        ).item()
    elif default_action_mode == 'Minimum loss':
        best_x = int(df['x'][0])
        best_y = int(df['y'][0])
    else:
        assert False

    with col_patch_select:
        action_x = st.number_input(
            f'action_x_{step}',
            0,
            image.shape[2] - patch_size['x'],
            value=best_x,
        )
        action_y = st.number_input(
            f'action_y_{step}',
            0,
            image.shape[1] - patch_size['y'],
            value=best_y,
        )
        action = action_x + action_y * (image.shape[2] - patch_size['x'] + 1)

        _, _, done, _ = env.step(action)
        patch = env.trajectory['patch'][-1]
        patch = patch[0:1]
        assert patch.dim() == 3
        assert patch.shape[0] == 1
        st.image(
            to_pil_image(patch.cpu()),
            use_column_width=True,
            output_format='png',
        )
        done = st.checkbox(f'Done on step {step}', value=((step + 1) % 16 == 0) or done)
    return done


def main():
    pl.seed_everything(0)

    st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.write('# RL test app')

    folder = Path(st.text_input('Select folder', value='outputs/**/*/'), 'train.log')
    path_list = sorted(Path().glob(str(folder)), key=os.path.getmtime)
    path_list = [i.parent for i in path_list]
    folder_path = st.selectbox(
        f'Select folder from "{folder}"', path_list, index=len(path_list) - 1
    )
    st.write(f'`{folder_path}`')

    device = st.selectbox(
        'device',
        [None, 'cpu', *[f'cuda:{i}' for i in range(torch.cuda.device_count())]],
    )
    if device is None:
        return ()
    device = torch.device(device)
    st.write(device)

    with st.beta_expander('Setting'):
        st.write('## Select agent model')
        agent_model_path_pattern = st.text_input(
            'Agent model path pattern', value='**/*_finish/model.pt'
        )
        path_list = sorted(
            Path(folder_path).glob(agent_model_path_pattern), key=os.path.getmtime
        )
        agent_model_path = st.selectbox(
            f'Select agent model path from "{Path(folder_path, agent_model_path_pattern)}"',
            path_list,
            index=len(path_list) - 1,
        )
        st.write(f'`{agent_model_path}`')

        st.write('## Select env model')
        env_model_path_pattern = st.text_input(
            'Env model path pattern', value='**/env_model/env_model_finish.pt'
        )
        path_list = sorted(
            Path(folder_path).glob(env_model_path_pattern), key=os.path.getmtime
        )
        env_model_path = st.selectbox(
            f'Select agent model path from "{Path(folder_path, env_model_path_pattern)}"',
            path_list,
            index=len(path_list) - 1,
        )
        st.write(f'`{env_model_path}`')

        st.write('## Select config file')
        yaml_path_pattern = st.text_input(
            'Yaml config path pattern', value='**/*/config.yaml'
        )
        path_list = sorted(
            Path(folder_path).glob(yaml_path_pattern), key=os.path.getmtime
        )
        yaml_path = st.selectbox(
            f'Select agent model path from "{Path(folder_path, yaml_path_pattern)}"',
            path_list,
            index=len(path_list) - 1,
        )
        st.write(f'`{yaml_path}`')
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            st.write(config)
            config = DictConfig(config)

    st.sidebar.write('# Select data')

    data_type = []
    for i in config.dataset.values():
        if isinstance(i.data_type, str):
            data_type.append(i.data_type)
        else:
            data_type.extend(i.data_type)
    data_type = st.sidebar.multiselect('Select data type', data_type, 'test_data')

    dataset = make_dataset(config.dataset, data_type)

    default_action = st.radio('Mode to select default action', ['RL', 'Minimum loss'])

    mode_to_select = st.sidebar.radio(
        'Mode to select', ['Number input', 'Select box'], 0
    )
    if mode_to_select == 'Number input':
        font_index = st.sidebar.number_input(
            f'Font index (0~{len(dataset.has_uniques["font"]) - 1})',
            0,
            len(dataset.has_uniques['font']) - 1,
            value=1,
            step=1,
        )
        alphabet_index = st.sidebar.number_input(
            f'Alphabet index (0~{len(dataset.has_uniques["alphabet"]) - 1})',
            0,
            len(dataset.has_uniques['alphabet']) - 1,
            value=0,
            step=1,
        )
        font = dataset.has_uniques['font'][font_index]
        alphabet = dataset.has_uniques['alphabet'][alphabet_index]
    elif mode_to_select == 'Select box':
        font = st.sidebar.selectbox('Font', dataset.has_uniques['font'], index=1)
        alphabet = st.sidebar.selectbox('Alphabet', dataset.has_uniques['alphabet'])
    else:
        assert False
    df_property = dataset.data_property.reset_index()
    df_property = df_property[df_property['font'] == font]
    df_property = df_property[df_property['alphabet'] == alphabet]
    data_index = df_property.index.item()
    st.sidebar.write(f'Data index:', data_index, f'(id={df_property["index"].item()})')
    st.sidebar.write(f'Font: `{font}` (id={df_property["font_id"].item()})')
    st.sidebar.write(f'Alphabet: `{alphabet}` (id={df_property["alphabet_id"].item()})')
    st.sidebar.write(
        f'Category: `{df_property["category"].item()}` (id={df_property["category_id"].item()})'
    )
    st.sidebar.write(
        f'Sub category: `{df_property["sub_category"].item()}` (id={df_property["sub_category_id"].item()})'
    )

    env_model = make_env_model(config.env_model, env_model_path, dataset, device)
    env = PatchSetsClassificationEnv(dataset=dataset, env_model=env_model, **config.env)
    agent_model = make_agent_model(config.agent_model, agent_model_path, device)

    _ = env.reset(data_index=data_index)
    image = env.data[0]
    target = env.data[1]

    st.sidebar.write('# Original data')
    st.sidebar.image(
        to_pil_image(image.cpu()), use_column_width=True, output_format='png'
    )

    step = 0
    done = False
    while not done:
        st.write(f'## step {step}')
        done = one_step(
            env,
            env_model,
            agent_model,
            env.patch_size,
            image,
            target,
            step,
            default_action,
        )
        step += 1

    st.sidebar.write('# History')

    df = pd.DataFrame(
        dict(
            action=env.trajectory['action'],
            x=[i % image.shape[2] for i in env.trajectory['action']],
            y=[i // image.shape[2] for i in env.trajectory['action']],
            loss=env.trajectory['loss'],
            reward=env.trajectory['reward'],
            **{
                alphabet: data
                for alphabet, data in zip(
                    dataset.has_uniques['alphabet'],
                    torch.cat(env.trajectory['output'], 0).softmax(1).T.detach().cpu(),
                )
            },
        ),
    )

    st.sidebar.write('state[1]')
    obs = env.trajectory['observation'][-1]
    fig = plt.figure()
    plt.imshow(obs[1].cpu().detach().numpy(), cmap='tab10', vmin=0, vmax=10)
    plt.colorbar()
    st.sidebar.pyplot(fig, True)
    plt.close()

    st.sidebar.write('patch')
    for i in more_itertools.chunked(env.trajectory['patch'], 4):
        for col, patch in zip(st.sidebar.beta_columns(4), list(i) + [None] * 3):
            if patch is not None:
                patch = patch[0:1]
                assert patch.dim() == 3
                assert patch.shape[0] == 1
                col.image(
                    to_pil_image(patch.cpu()),
                    use_column_width=True,
                    output_format='png',
                )

    st.sidebar.write('output')
    output_df = df[dataset.has_uniques['alphabet']]
    fig = plt.figure()
    plt.plot(
        range(1, step + 1),
        output_df[dataset.has_uniques['alphabet'][target]],
        color='black',
        marker='o',
        label=dataset.has_uniques['alphabet'][target],
    )
    top_acc_label = []
    for _, row in output_df.iterrows():
        top_acc_label.extend(row.sort_values(ascending=False)[:2].index)
    display_label = sorted(
        set(top_acc_label) - {dataset.has_uniques['alphabet'][target]}
    )
    for label in display_label:
        plt.plot(
            range(1, step + 1),
            output_df[label],
            marker='.',
            label=label,
        )
    plt.ylim([0, 1])
    plt.legend()
    st.sidebar.pyplot(fig, True)

    st.sidebar.write('loss')
    fig = plt.figure()
    plt.plot(range(1, step + 1), env.trajectory['loss'], marker='o', label='loss')
    plt.hlines(
        [
            F.cross_entropy(
                torch.zeros([1, 26]),
                torch.tensor([target], dtype=torch.long),
            ).item()
        ],
        1,
        step,
        color='gray',
        linestyles='--',
    )
    plt.ylim([0, max(plt.ylim())])
    st.sidebar.pyplot(fig, True)

    st.sidebar.write('reward')
    fig = plt.figure()
    plt.plot(range(1, step + 1), env.trajectory['reward'], marker='o', label='reward')
    plt.hlines(
        [0],
        1,
        step,
        color='gray',
        linestyles='--',
    )
    st.sidebar.pyplot(fig, True)

    st.sidebar.write('data')
    st.sidebar.dataframe(df)
    return


if __name__ == "__main__":
    with torch.no_grad():
        main()
