from typing import Dict
from logging import getLogger
from pathlib import Path

import hydra
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
from torch import Tensor, nn
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from src.agent_model import AgentModel
from src.dataset import AdobeFontDataset
from src.env import PatchSetsClassificationEnv
from src.env_model import EnvModel

logger = getLogger(__name__)


@st.cache()
def make_dataset(config_dataset: DictConfig) -> AdobeFontDataset:
    if 'test' in config_dataset:
        dataset = AdobeFontDataset(
            transform=transforms.ToTensor(), **config_dataset.test
        )
    else:
        dataset = AdobeFontDataset(
            transform=transforms.ToTensor(), **config_dataset.valid
        )
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
    agent = pfrl.agents.PPO(model=agent_model, optimizer=None)
    return agent


# @st.cache(hash_funcs={Tensor: lambda x: x.cpu().detach().numpy()})
def predict_all_patch(
    env_model: EnvModel,
    image: Tensor,
    patch_size: Dict[str, int],
    additional_patch_set: Tensor = None,
):
    patch_all = (
        image.unfold(1, patch_size['y'], 1)
        .unfold(2, patch_size['x'], 1)
        .reshape(-1, 1, 1, patch_size['y'], patch_size['x'])
    )
    if additional_patch_set is not None:
        patch_all = torch.cat(
            [
                patch_all,
                additional_patch_set.expand(patch_all.shape[0], -1, -1, -1, -1),
            ],
            1,
        )
    predicted_all = env_model(patch_all)
    return predicted_all


@st.cache(hash_funcs={Tensor: lambda x: x.cpu().detach().numpy()})
def make_loss_map(
    predicted_all: Tensor, target: int, image: Tensor, patch_size: int
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
        col_rl_map,
        col_action_dataframe,
        col_patch_select,
    ) = st.beta_columns([4, 2, 4, 3, 2])

    with col_loss_map:
        if step == 0:
            predicted_all = predict_all_patch(env_model, image, patch_size)
        else:
            predicted_all = predict_all_patch(
                env_model,
                image,
                patch_size,
                additional_patch_set=torch.stack(env.trajectory['patch']),
            )
        loss_map = make_loss_map(predicted_all, target, image, patch_size)
        st.write('Loss map')
        fig = plt.figure()
        plt.imshow(loss_map.cpu().detach().numpy()[0])
        plt.colorbar()
        st.pyplot(fig, True)
        plt.close()

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

    with col_rl_map:
        obs = env.trajectory['observation'][-1]
        select_probs = agent_model(obs[None])[0].probs[0]
        select_probs_map = select_probs.reshape(
            image.shape[1] - patch_size['y'] + 1,
            image.shape[2] - patch_size['x'] + 1,
        )

        st.write('Slect prob. map')
        fig = plt.figure()
        plt.imshow(select_probs_map.cpu().detach().numpy())
        plt.colorbar()
        st.pyplot(fig, True)
        plt.close()

    with col_action_dataframe:
        st.write('Action dataframe')
        df = make_action_df(
            loss_map, predicted_all, env.dataset.has_uniques['alphabet']
        )
        st.dataframe(df.head())

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
        action_x = st.slider(
            f'action_x_{step}',
            0,
            image.shape[2] - patch_size['x'],
            value=best_x,
        )
        action_y = st.slider(
            f'action_y_{step}',
            0,
            image.shape[1] - patch_size['y'],
            value=best_y,
        )
        action = action_x + action_y * (image.shape[2] - patch_size['x'] + 1)

        _, _, done, _ = env.step(action)
        st.image(
            to_pil_image(env.trajectory['patch'][-1].cpu()),
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

    with st.beta_expander('Setting'):
        device = st.selectbox('device', ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])

        st.write('## Select agent model')
        agent_model_path_pattern = st.text_input(
            'Agent model path pattern', value='outputs/**//**/*h/model'
        )
        agent_model_path_pattern = f'{agent_model_path_pattern}.pt'
        path_list = sorted(
            Path(hydra.utils.get_original_cwd()).glob(agent_model_path_pattern)
        )
        agent_model_path = st.selectbox(
            f'Select agent model path from "{agent_model_path_pattern}"',
            path_list,
            index=len(path_list) - 1,
        )
        st.write(f'`{agent_model_path}`')

        st.write('## Select env model')
        env_model_path_pattern = st.text_input(
            'Env model path pattern', value='outputs/**//**/env_model_finish'
        )
        env_model_path_pattern = f'{env_model_path_pattern}.pth'
        path_list = sorted(
            Path(hydra.utils.get_original_cwd()).glob(env_model_path_pattern)
        )
        env_model_path = st.selectbox(
            f'Select agent model path from "{env_model_path_pattern}"',
            path_list,
            index=len(path_list) - 1,
        )
        st.write(f'`{env_model_path}`')

        st.write('## Select config file')
        yaml_path_pattern = st.text_input(
            'Yaml file path pattern', value='outputs/**//**/config'
        )
        yaml_path_pattern = f'{yaml_path_pattern}.yaml'
        path_list = sorted(Path(hydra.utils.get_original_cwd()).glob(yaml_path_pattern))
        yaml_path = st.selectbox(
            f'Select agent model path from "{yaml_path_pattern}"',
            path_list,
            index=len(path_list) - 1,
        )
        st.write(f'`{yaml_path}`')
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            st.write(config)
            config = DictConfig(config)

    dataset = make_dataset(config.dataset)
    env_model = make_env_model(config.env_model, env_model_path, dataset, device)
    env = PatchSetsClassificationEnv(dataset=dataset, env_model=env_model, **config.env)

    agent_model = make_agent_model(config.agent_model, agent_model_path, device)

    default_action = st.radio('Mode to select default action', ['RL', 'Minimum loss'])

    st.sidebar.write('# Select data')

    mode_to_select = st.sidebar.radio(
        'Mode to select', ['index', 'font index & alphabet index', 'font & alphabet'], 1
    )
    if mode_to_select == 'index':
        data_index = st.sidebar.number_input(
            f'Data index (0~{len(dataset) - 1})',
            0,
            len(dataset) - 1,
            value=1300,
            step=1,
        )
        font = dataset.data_property.iloc[data_index]['font']
        alphabet = dataset.data_property.iloc[data_index]['alphabet']
    else:
        if mode_to_select == 'font index & alphabet index':
            font_index = st.sidebar.number_input(
                f'Font index (0~{len(dataset.has_uniques["font"]) - 1})',
                0,
                len(dataset.has_uniques['font']) - 1,
                value=4,
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
        elif mode_to_select == 'font & alphabet':
            font = st.sidebar.selectbox('Font', dataset.has_uniques['font'], index=3)
            alphabet = st.sidebar.selectbox('Alphabet', dataset.has_uniques['alphabet'])
        else:
            assert False
        df_property = dataset.data_property
        df_property = df_property[df_property['font'] == font]
        df_property = df_property[df_property['alphabet'] == alphabet]
        data_index = df_property.index.item()
    st.sidebar.write(f'Data index:', data_index)
    st.sidebar.write(f'Font: `{font}`')
    st.sidebar.write(f'Alphabet: `{alphabet}`')
    st.sidebar.write(f'Category: `{df_property["category"].item()}`')
    st.sidebar.write(f'Sub category: `{df_property["sub_category"].item()}`')

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
        for col, image in zip(st.sidebar.beta_columns(4), list(i) + [None] * 3):
            if image is not None:
                col.image(
                    to_pil_image(image.cpu()),
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
