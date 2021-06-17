import itertools
import more_itertools
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import seaborn as sns

import numpy as np
import pandas as pd
import pfrl
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl
import src.dataset
import streamlit as st
import torch
import yaml
from hydra.utils import DictConfig
from sklearn.metrics import f1_score, confusion_matrix
from src.agent_model import AgentModel
from src.env import PatchSetsClassificationEnv
from src.env_model import EnvModel
from stqdm import stqdm
from torch import Tensor
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import matplotlib.pyplot as plt


@st.cache()
def make_dataset(
    dataset_class: Callable, config_dataset: DictConfig, data_type: List[str]
) -> torch.utils.data.Dataset:
    config_dataset_test = config_dataset.test.copy()
    config_dataset_test['data_type'] = data_type
    dataset = dataset_class(**config_dataset_test)
    return dataset


@st.cache(allow_output_mutation=True)
def make_env_model(
    config_env_model: DictConfig,
    env_model_path: str,
    dataset: torch.utils.data.Dataset,
    device,
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
) -> AgentModel:
    if 'input_n' in config_agent_model:
        agent_model = AgentModel(**config_agent_model)
    else:
        agent_model = AgentModel(**config_agent_model, input_n=2)
    agent_model.load_state_dict(torch.load(agent_model_path, map_location='cpu'))
    agent_model = agent_model.to(device)
    agent_model.eval()
    summary(agent_model)
    return agent_model


@st.cache(allow_output_mutation=True)
def make_agent(
    config_agent: DictConfig, agent_model: AgentModel, gpu: int
) -> pfrl.agents.PPO:
    config_agent = {
        key: config_agent[key] for key in set(config_agent.keys()) - {'gpu'}
    }
    config_agent['gpu'] = gpu
    agent = pfrl.agents.PPO(model=agent_model, optimizer=None, **config_agent)
    return agent


def step_index(
    env: PatchSetsClassificationEnv,
    agent: pfrl.agents.PPO,
    data_index: int,
    max_episode_length: int,
    act_deterministically: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    act_deterministically, agent.act_deterministically = (
        agent.act_deterministically,
        act_deterministically,
    )
    with agent.eval_mode():
        obs = env.reset(data_index)

        total_reward = 0
        episode_length = 0
        reset = False
        action_list = []
        action_x_list = []
        action_y_list = []
        while not reset:
            action = agent.act(obs)
            x, y = env.convert_action(
                *obs.shape[1:], patch_size=env.patch_size, action=action
            )

            action_list.append(action)
            action_x_list.append(x)
            action_y_list.append(y)

            obs, r, done, info = env.step(action)
            total_reward += r
            episode_length += 1
            reset = (
                done
                or episode_length == max_episode_length
                or info.get("needs_reset", False)
            )
            if not done and episode_length == max_episode_length:
                episode_length += 1
            agent.observe(obs, r, done, reset)
    estimated = env.trajectory['output'][-1][0].argmax().item()
    return {
        'episode_length': episode_length,
        'total_reward': total_reward,
        'estimated': estimated,
    }, {
        'action_list': action_list,
        'action_x_list': action_x_list,
        'action_y_list': action_y_list,
    }


def eval_dataset(
    dataset: torch.utils.data.Dataset,
    env: PatchSetsClassificationEnv,
    agent: pfrl.agents.PPO,
    max_episode_length: int,
    target_name: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_evaled = dataset.data_property.copy()
    # df_evaled = dataset.data_property.copy()[:10]
    evaled_dict = {}
    action_dict = {'data_index': [], 'step': []}
    for data_index in stqdm(range(len(dataset))):
        evaled, action = step_index(
            env, agent, data_index=data_index, max_episode_length=max_episode_length
        )
        for key, value in evaled.items():
            if key not in evaled_dict:
                evaled_dict[key] = []
            evaled_dict[key].append(value)
        for key, value in action.items():
            if key not in action_dict:
                action_dict[key] = []
            action_dict[key].extend(value)
        action_dict['data_index'].extend(
            [df_evaled.index[data_index]] * len(action['action_list'])
        )
        action_dict['step'].extend(range(len(action['action_list'])))
    for key, value in evaled_dict.items():
        df_evaled[key] = value
    df_evaled['success'] = df_evaled[target_name + '_id'] == df_evaled['estimated']
    df_action = pd.DataFrame(action_dict)
    return df_evaled, df_action


def main():
    pl.seed_everything(0)

    st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.write('# RL test app')

    # folder = Path(st.text_input('Select folder', value='outputs/**/*/'), 'train.log')
    folder = Path(
        st.text_input(
            'Select folder',
            value='outputs/AdobeFontDataset/baseline/gamma00/2021-02-06/02-18-34',
        ),
        'train.log',
    )
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
    data_type = sorted(
        st.sidebar.multiselect('Select data type', data_type, 'test_data')
    )

    if 'dataset_name' not in config:
        config.dataset_name = 'AdobeFontDataset'
    if config.dataset_name == 'AdobeFontDataset':
        target_name = 'alphabet'
    elif config.dataset_name == 'Chars74kImageDataset':
        target_name = 'label'
    dataset_class = getattr(src.dataset, config.dataset_name)
    dataset = make_dataset(dataset_class, config.dataset, data_type)
    dataset_uniques = dataset.uniques
    dataset_has_uniques = dataset.has_uniques

    max_episode_length = st.sidebar.number_input('Max episode length', 1, value=8)

    self_file_path = Path(__file__).relative_to(Path('.').absolute())
    save_file_path_base = Path(
        '.',
        self_file_path.with_suffix(''),
        str(Path(*folder_path.parts[1:])),
        '-'.join([*data_type, f'{max_episode_length}']),
    )
    if save_file_path_base.is_dir():
        df_evaled = pd.read_csv(save_file_path_base / 'df_evaled.csv', index_col=0)
        df_action = pd.read_csv(save_file_path_base / 'df_action.csv', index_col=0)
    else:
        env_model = make_env_model(config.env_model, env_model_path, dataset, device)
        env = PatchSetsClassificationEnv(
            dataset=dataset, env_model=env_model, **config.env
        )
        agent_model = make_agent_model(config.agent_model, agent_model_path, device)

        agent = make_agent(config.agent, agent_model, gpu=device.index)

        df_evaled, df_action = eval_dataset(
            dataset, env, agent, max_episode_length, target_name=target_name
        )
        save_file_path_base.mkdir(exist_ok=True, parents=True)
        df_evaled = df_evaled.drop('image', axis=1)
        df_evaled.to_csv(save_file_path_base / 'df_evaled.csv')
        df_action.to_csv(save_file_path_base / 'df_action.csv')

    df_action = pd.merge(df_evaled, df_action, left_index=True, right_on='data_index')

    if config.dataset_name == 'AdobeFontDataset':
        hover_data = ['font', 'alphabet', 'category', 'sub_category']
    elif config.dataset_name == 'Chars74kImageDataset':
        hover_data = [
            'name',
            'label',
            'quality',
            'episode_length',
            'total_reward',
            'estimated',
            'success',
        ]
    else:
        assert False

    st.write('# Dataframe')

    with st.beta_expander('Dataframe'):
        st.write('df_evaled')
        st.dataframe(df_evaled)

        st.write('df_action')
        st.dataframe(df_action)

    with st.sidebar:
        st.write('# Setting of analysis')
        target_column = st.selectbox('Target column', df_evaled.columns, index=1)
        is_normalize = st.checkbox('Normalize', True)

    st.write('# Statistics')

    st.write(f'Recognition rate: `{df_evaled["success"].mean()}`')
    st.write(
        f'F1-measure (micro): `{f1_score(df_evaled[target_name + "_id"], df_evaled["estimated"], average="micro")}`'
    )
    st.write(
        f'F1-measure (macro): `{f1_score(df_evaled[target_name + "_id"], df_evaled["estimated"], average="macro")}`'
    )
    st.write(
        f'F1-measure (weighted): `{f1_score(df_evaled[target_name + "_id"], df_evaled["estimated"], average="weighted")}`'
    )

    cm = confusion_matrix(
        df_evaled[target_name + '_id'],
        df_evaled['estimated'],
    )
    with st.beta_expander('Confusion matrix'):
        col1, col2 = st.beta_columns(2)
        with col1:
            fig = px.imshow(
                cm,
                labels={'x': 'Estimated', 'y': 'True', 'color': 'Count'},
                x=dataset_has_uniques[target_name],
                y=dataset_has_uniques[target_name],
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
                annot=True,
                # fmt='d',
                square=True,
                xticklabels=[i[-1] for i in dataset_has_uniques[target_name]],
                yticklabels=[i[-1] for i in dataset_has_uniques[target_name]],
            )
            plt.yticks(rotation=0)
            save_path = save_file_path_base / 'confusion_matrix'
            save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / 'confusion_matrix.png')
            st.pyplot(plt, True)

    if config.dataset_name == 'AdobeFontDataset':
        with st.beta_expander('Episode length & success map of all fonts'):
            col1, col2 = st.beta_columns(2)
            with col1:
                image = (
                    np.array(df_evaled['episode_length'])
                    .reshape(-1, len(df_evaled['alphabet'].unique()))
                    .T.astype(float)
                )
                fig = px.imshow(
                    image,
                    x=df_evaled['font'].unique(),
                    y=df_evaled['alphabet'].unique(),
                    title=f'Episode length map',
                    range_color=[1, max_episode_length],
                    width=2000,
                )
                fig.update(
                    data=[
                        {
                            'customdata': np.array(df_evaled['category'])
                            .reshape(-1, len(df_evaled['alphabet'].unique()))
                            .T,
                            'hovertemplate': 'Font: %{x}<br>Alphabet: %{y}<br>Category: %{customdata}<br>Success: %{z}<extra></extra>',
                        }
                    ]
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                image = (
                    np.array(df_evaled['success'])
                    .reshape(-1, len(df_evaled['alphabet'].unique()))
                    .T.astype(float)
                )
                fig = px.imshow(
                    image,
                    x=df_evaled['font'].unique(),
                    y=df_evaled['alphabet'].unique(),
                    title='Success map',
                    range_color=[0, 1],
                    width=2000,
                )
                fig.update(
                    data=[
                        {
                            'customdata': np.array(df_evaled['category'])
                            .reshape(-1, len(df_evaled['alphabet'].unique()))
                            .T,
                            'hovertemplate': 'Font: %{x}<br>Alphabet: %{y}<br>Category: %{customdata}<br>Success: %{z}<extra></extra>',
                        }
                    ]
                )
                st.plotly_chart(fig, use_container_width=True)

        with st.beta_expander('Episode length map only succeed'):
            col, _ = st.beta_columns([10, 1])
            with col:
                image = (
                    np.array(df_evaled['episode_length'])
                    .reshape(-1, len(df_evaled['alphabet'].unique()))
                    .T.astype(float)
                )
                image[image > max_episode_length] = float('nan')
                image[
                    np.array(df_evaled['success'])
                    .reshape(-1, len(df_evaled['alphabet'].unique()))
                    .T
                    == False
                ] = float('nan')

                fig = px.imshow(
                    image,
                    x=[
                        f'{value:.4f}  {name}'
                        for name, value in zip(
                            df_evaled['font'].unique(), image.mean(0)
                        )
                    ],
                    y=[
                        f'{name}  {value:.4f}'
                        for name, value in zip(
                            df_evaled['alphabet'].unique(), image.mean(1)
                        )
                    ],
                    title=f'Episode length map',
                    # width=2000,
                )
                fig.update(
                    data=[
                        {
                            'customdata': np.array(df_evaled['category'])
                            .reshape(-1, len(df_evaled['alphabet'].unique()))
                            .T,
                            'hovertemplate': 'Font: %{x}<br>Alphabet: %{y}<br>Category: %{customdata}<br>Success: %{z}<extra></extra>',
                        }
                    ],
                )
                fig.update_layout({'plot_bgcolor': 'green'})
                fig.update_layout(xaxis={'showgrid': False}, yaxis={'showgrid': False})
                st.plotly_chart(fig, use_container_width=True)

                fig = px.imshow(
                    image,
                    x=[
                        f'{value:.4f}  {name}'
                        for name, value in zip(
                            df_evaled['font'].unique(), np.nanmean(image, 0)
                        )
                    ],
                    y=[
                        f'{name}  {value:.4f}'
                        for name, value in zip(
                            df_evaled['alphabet'].unique(), np.nanmean(image, 1)
                        )
                    ],
                    title=f'Episode length map',
                    # width=2000,
                )
                fig.update(
                    data=[
                        {
                            'customdata': np.array(df_evaled['category'])
                            .reshape(-1, len(df_evaled['alphabet'].unique()))
                            .T,
                            'hovertemplate': 'Font: %{x}<br>Alphabet: %{y}<br>Category: %{customdata}<br>Success: %{z}<extra></extra>',
                        }
                    ],
                )
                fig.update_layout({'plot_bgcolor': 'green'})
                fig.update_layout(xaxis={'showgrid': False}, yaxis={'showgrid': False})
                st.plotly_chart(fig, use_container_width=True)

        with st.beta_expander('Episode length & success map of all categories'):
            col1, col2 = st.beta_columns(2)
            result = dict(category=[], alphabet=[], success=[], episode_length=[])
            for category, alphabet in itertools.product(
                df_evaled['category'].unique(), df_evaled['alphabet'].unique()
            ):
                df = df_evaled
                df = df[df['category'] == category]
                df = df[df['alphabet'] == alphabet]
                result['category'].append(category)
                result['alphabet'].append(alphabet)
                result['success'].append(df['success'].mean())
                result['episode_length'].append(df['episode_length'].mean())
            df = pd.DataFrame(result)
            with col1:
                image = (
                    np.array(df['success'])
                    .reshape(-1, len(df['alphabet'].unique()))
                    .T.astype(float)
                )
                fig = px.imshow(
                    image,
                    x=df['category'].unique(),
                    y=df['alphabet'].unique(),
                    title='Success map',
                    range_color=[0, 1],
                )
                st.plotly_chart(fig)
            with col2:
                image = (
                    np.array(df['episode_length'])
                    .reshape(-1, len(df['alphabet'].unique()))
                    .T.astype(float)
                )
                fig = px.imshow(
                    image,
                    x=df['category'].unique(),
                    y=df['alphabet'].unique(),
                    title='Episode length map',
                    range_color=[1, max_episode_length],
                )
                st.plotly_chart(fig)

        with st.beta_expander('Episode length & success map by categories'):
            for category in df_evaled['category'].unique():
                st.write(category)

                df = df_evaled[df_evaled['category'] == category]

                col1, col2 = st.beta_columns(2)
                with col1:
                    image = (
                        np.array(df['episode_length'])
                        .reshape(-1, len(df['alphabet'].unique()))
                        .T.astype(float)
                    )
                    fig = px.imshow(
                        image,
                        # x=[f'{value:.2f}  {i:02}' for i, value in enumerate(image.mean(0))],
                        # y=[f'{i:02}  {value:.2f}' for i, value in enumerate(image.mean(1))],
                        x=[
                            f'{value:.2f}  {name}'
                            for name, value in zip(df['font'].unique(), image.mean(0))
                        ],
                        y=[
                            f'{name}  {value:.2f}'
                            for name, value in zip(
                                df['alphabet'].unique(), image.mean(1)
                            )
                        ],
                        title=f'Episode length map ({category})',
                        range_color=[1, max_episode_length],
                    )
                    fig.update(
                        data=[
                            {
                                'customdata': np.array(df['category'])
                                .reshape(-1, len(df['alphabet'].unique()))
                                .T,
                                'hovertemplate': 'Font: %{x}<br>Alphabet: %{y}<br>Category: %{customdata}<br>Success: %{z}<extra></extra>',
                            }
                        ]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    image = (
                        np.array(df['success'])
                        .reshape(-1, len(df['alphabet'].unique()))
                        .T.astype(float)
                    )
                    fig = px.imshow(
                        image,
                        # x=[f'{value:.2f}  {i:02}' for i, value in enumerate(image.mean(0))],
                        # y=[f'{i:02}  {value:.2f}' for i, value in enumerate(image.mean(1))],
                        x=[
                            f'{value:.2f}  {name}'
                            for name, value in zip(df['font'].unique(), image.mean(0))
                        ],
                        y=[
                            f'{name}  {value:.2f}'
                            for name, value in zip(
                                df['alphabet'].unique(), image.mean(1)
                            )
                        ],
                        title=f'Success map ({category})',
                        range_color=[0, 1],
                    )
                    fig.update(
                        data=[
                            {
                                'customdata': np.array(df['category'])
                                .reshape(-1, len(df['alphabet'].unique()))
                                .T,
                                'hovertemplate': 'Font: %{x}<br>Alphabet: %{y}<br>Category: %{customdata}<br>Success: %{z}<extra></extra>',
                            }
                        ]
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.write('# Analysis of each column')

    with st.beta_expander('Recognition rate'):
        df = {target_column: [], 'success': [], 'count': [], 'percent': []}
        for target_value, success in itertools.product(
            df_evaled[target_column].unique(), df_evaled['success'].unique()
        ):
            df[target_column].append(target_value)
            df['success'].append(success)
            df['count'].append(
                (
                    (df_evaled[target_column] == target_value)
                    & (df_evaled['success'] == success)
                ).sum()
            )
            df['percent'].append(
                df['count'][-1] / (df_evaled[target_column] == target_value).sum()
            )
        df = pd.DataFrame(df)

        fig = px.bar(
            df,
            x=target_column,
            y='count',
            color='success',
            title='Recognition rate',
        )
        fig.update_layout({'plot_bgcolor': 'white'})
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            df,
            x=target_column,
            y='percent',
            color='success',
            title='Recognition rate',
        )
        fig.update_layout({'plot_bgcolor': 'white'})
        st.plotly_chart(fig, use_container_width=True)

        # fig = px.histogram(
        #     df_evaled,
        #     x=target_column,
        #     color='success',
        #     hover_data=hover_data,
        #     title='Recognition rate',
        # )
        # fig.update_xaxes(categoryorder='category ascending')
        # st.plotly_chart(fig, use_container_width=True)

    with st.beta_expander('Episode length histogram'):
        col1, col2 = st.beta_columns(2)
        with col1:
            df_episode_length = pd.DataFrame()
            for value in df_evaled[target_column].unique():
                df = df_evaled
                df = df[df[target_column] == value]
                df_episode_length[value] = df['episode_length'].value_counts()
            df_episode_length = df_episode_length.fillna(0)
            for i in range(1, df_episode_length.index.max()):
                if i not in df_episode_length.index:
                    df_episode_length.loc[i] = [0] * len(df_episode_length.columns)
            df_episode_length = df_episode_length.sort_index(0).sort_index(1)

            if not is_normalize:
                labels = {'index': 'Episode length', 'value': 'Count'}
            else:
                df_episode_length = df_episode_length / df_episode_length.sum()
                labels = {'index': 'Episode length', 'value': 'Percentage'}

            fig = px.area(
                df_episode_length,
                labels=labels,
                title='Episode length histogram',
            )
            st.plotly_chart(fig, use_container_width=True)
            fig = px.line(
                df_episode_length,
                labels=labels,
                title='Episode length histogram',
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if not is_normalize:
                histnorm = None
            else:
                histnorm = 'percent'

            fig = px.histogram(
                df_evaled.sort_values(target_column),
                x='episode_length',
                color=target_column,
                marginal='rug',
                hover_data=hover_data,
                labels={'episode_length': 'Episode length'},
                title='Episode length histogram',
                histnorm=histnorm,
            )
            fig.update_layout(barmode='stack')
            st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(
                df_evaled.sort_values(target_column),
                x='episode_length',
                color=target_column,
                marginal='rug',
                hover_data=hover_data,
                labels={'episode_length': 'Episode length'},
                title='Episode length histogram',
                histnorm=histnorm,
            )
            fig.update_layout(barmode='overlay')
            fig.update_traces(opacity=0.75)
            st.plotly_chart(fig, use_container_width=True)

        # fig = px.histogram(
        #     df_evaled,
        #     x='total_reward',
        #     color='success',
        #     marginal='rug',
        #     hover_data=hover_data,
        #     labels={'total_reward': 'Total reward'},
        #     title='Total reward',
        # )
        # fig.update_layout(barmode='overlay')
        # fig.update_traces(opacity=0.75)
        # st.plotly_chart(fig)

        # fig = px.scatter(
        #     df_evaled,
        #     x='episode_length',
        #     y='total_reward',
        #     color='alphabet',
        #     hover_data=hover_data,
        #     labels={'episode_length': 'Episode length', 'total_reward': 'total reward'},
        #     title='Episode length histogram',
        # )
        # fig.update_layout(barmode='overlay')
        # fig.update_traces(opacity=0.75)
        # st.plotly_chart(fig)

    with st.beta_expander('Action heatmap'):
        top_n_step = st.number_input(
            'Top n step', 1, max_episode_length, value=max_episode_length
        )
        save_path = (
            save_file_path_base / f'action_heatmap-top_{top_n_step}' / target_column
        )
        save_path.mkdir(exist_ok=True, parents=True)
        if not is_normalize:
            histnorm = None
        else:
            histnorm = 'percent'
        for target_value_list in more_itertools.chunked(
            df_evaled[target_column].unique(), 4
        ):
            for col, target_value in zip(st.beta_columns(4), target_value_list):
                with col:
                    # df = df_action[df_action[target_column] == target_value]
                    # df = df[df['step'] < top_n_step]
                    # st.write(target_value)
                    # fig = px.density_heatmap(df, x='action_x_list', y='action_y_list', nbinsx=76, nbinsy=76, color_continuous_scale='magma', labels={'action_x_list': 'x', 'action_y_list': 'y'}, histnorm=histnorm, width=270, height=200, title=target_value)
                    # fig.update_xaxes(showticklabels=False, title_text='')
                    # fig.update_yaxes(showticklabels=False, title_text='')
                    # fig.update_yaxes(autorange='reversed')
                    # fig.update_xaxes(matches=None)
                    # fig.update_layout(title_x=0.5, margin=dict(l=30, r=30, t=30, b=30))
                    # fig.update_layout(font=dict(size=24))
                    # st.plotly_chart(fig)

                    df = df_action[df_action[target_column] == target_value]
                    df = df[df['step'] < top_n_step]
                    fig, ax = plt.subplots(1, 1, figsize=[2.5, 2])
                    h, xedges, yedges, pc = ax.hist2d(
                        df['action_x_list'],
                        df['action_y_list'],
                        bins=76,
                        range=[[0, 75]] * 2,
                    )
                    clb = fig.colorbar(pc, aspect=20, format='%.0f')
                    ax.set_title(target_value)
                    ax.set_ylim(ax.get_ylim()[::-1])
                    plt.savefig(save_path / (str(target_value) + '.png'))
                    st.pyplot(fig, True)

    return


if __name__ == "__main__":
    with torch.no_grad():
        main()
