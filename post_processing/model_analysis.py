import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
from sklearn.metrics import confusion_matrix
from src.agent_model import AgentModel
from src.env import PatchSetsClassificationEnv
from src.env_model import EnvModel
from stqdm import stqdm
from torch import Tensor
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


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
) -> Dict[str, Any]:
    act_deterministically, agent.act_deterministically = (
        agent.act_deterministically,
        act_deterministically,
    )

    obs = env.reset(data_index)

    total_reward = 0
    episode_length = 0
    reset = False
    while not reset:
        action = agent.act(obs)
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
    }


def eval_dataset(
    dataset: torch.utils.data.Dataset,
    env: PatchSetsClassificationEnv,
    agent: pfrl.agents.PPO,
    max_episode_length: int,
    target_name: str = None,
):
    with agent.eval_mode():
        df = dataset.data_property.copy()
        result_list = {}
        for data_index in stqdm(range(len(dataset))):
            result = step_index(
                env, agent, data_index=data_index, max_episode_length=max_episode_length
            )
            for key, value in result.items():
                if key not in result_list:
                    result_list[key] = []
                result_list[key].append(value)
        for key, value in result_list.items():
            df[key] = value
    df['success'] = df[target_name + '_id'] == df['estimated']
    return df


def main():
    pl.seed_everything(0)

    st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.write('# RL test app')

    path_list = sorted(Path().glob('outputs/**/*/.hydra/'), key=os.path.getmtime)
    path_list = [i.parent for i in path_list]
    folder_path = st.selectbox(
        f'Select folder from "outputs/**/*/.hydra/"',
        path_list,
        index=len(path_list) - 1,
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

    max_episode_length = st.sidebar.number_input('Max episode length', 1, value=16)

    env_model = make_env_model(config.env_model, env_model_path, dataset, device)
    env = PatchSetsClassificationEnv(dataset=dataset, env_model=env_model, **config.env)
    agent_model = make_agent_model(config.agent_model, agent_model_path, device)
    agent = make_agent(config.agent, agent_model, gpu=device.index)

    self_file_path = Path(__file__).relative_to(Path('.').absolute())
    save_file_path = Path(
        '.',
        self_file_path.with_suffix(''),
        str(folder_path).replace('/', '-'),
        '-'.join(data_type),
    )
    file_name = save_file_path / f'{max_episode_length}.csv'
    if file_name.exists():
        df_evaled = pd.read_csv(file_name, index_col=0)
    else:
        df_evaled = eval_dataset(
            dataset, env, agent, max_episode_length, target_name=target_name
        )
        file_name.parent.mkdir(parents=True, exist_ok=True)
        df_evaled.to_csv(file_name)

    hover_data = ['font', 'alphabet', 'category', 'sub_category']

    st.write('# Dataframe')
    st.dataframe(df_evaled)

    st.write('# Statistics')

    st.write(f'Recognition rate: `{df_evaled["success"].mean()}`')

    cm = confusion_matrix(
        df_evaled[target_name + '_id'],
        df_evaled['estimated'],
    )
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

    col1, col2 = st.beta_columns(2)
    with col1:
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
        st.plotly_chart(fig)
    with col2:
        image = (
            np.array(df_evaled['episode_length'])
            .reshape(-1, len(df_evaled['alphabet'].unique()))
            .T.astype(float)
        )
        fig = px.imshow(
            image,
            x=df_evaled['font'].unique(),
            y=df_evaled['alphabet'].unique(),
            title='Episode length map',
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
        st.plotly_chart(fig)

    st.write('# Analysis of each column')

    target_column = st.selectbox('Target column', df_evaled.columns, index=1)

    fig = px.histogram(
        df_evaled,
        x=target_column,
        color='success',
        hover_data=hover_data,
        title='Recognition rate',
    )
    fig.update_xaxes(categoryorder='category ascending')
    st.plotly_chart(fig, use_container_width=True)

    normalize_histogram = st.checkbox('Normalize histogram', True)
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

        if not normalize_histogram:
            labels = {'index': 'Episode length', 'value': 'Count'}
        else:
            df_episode_length = df_episode_length / df_episode_length.sum()
            labels = {'index': 'Episode length', 'value': 'Percentage'}

        fig = px.area(
            df_episode_length,
            labels=labels,
            title='Episode length',
        )
        st.plotly_chart(fig, use_container_width=True)
        fig = px.line(
            df_episode_length,
            labels=labels,
            title='Episode length',
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not normalize_histogram:
            histnorm = None
        else:
            histnorm = 'probability'

        fig = px.histogram(
            df_evaled.sort_values(target_column),
            x='episode_length',
            color=target_column,
            marginal='rug',
            hover_data=hover_data,
            labels={'episode_length': 'Episode length', 'count': 'Count'},
            title='Episode length',
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
            labels={'episode_length': 'Episode length', 'count': 'Count'},
            title='Episode length',
            histnorm=histnorm,
        )
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        df_evaled,
        x='total_reward',
        color='success',
        marginal='rug',
        hover_data=hover_data,
        labels={'total_reward': 'Total reward', 'count': 'Count'},
        title='Total reward',
    )
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig)

    fig = px.scatter(
        df_evaled,
        x='episode_length',
        y='total_reward',
        color='alphabet',
        hover_data=hover_data,
        labels={'episode_length': 'Episode length', 'total_reward': 'total reward'},
        title='Episode length',
    )
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig)

    return


if __name__ == "__main__":
    with torch.no_grad():
        main()
