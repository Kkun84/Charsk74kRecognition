import streamlit as st
import torch
import more_itertools
from torch import Tensor, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from torchsummary import summary
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import pandas as pd
import hydra
from logging import getLogger
from pathlib import Path
import yaml

from src.dataset import AdobeFontDataset
from src.env import PatchSetsClassificationEnv
import src.env_model
import src.agent_model


logger = getLogger(__name__)


@st.cache(allow_output_mutation=True)
def make_dataset() -> torch.utils.data.Dataset:
    dataset = AdobeFontDataset(
        path='/dataset/AdobeFontCharImages',
        transform=transforms.ToTensor(),
        target_transform=lambda x: x['alphabet'],
        upper=True,
        lower=False,
    )
    return dataset


@st.cache(allow_output_mutation=True)
def make_env(env_model_path: str, done_prob: float, device):
    model = src.env_model.EnvModel.load_from_checkpoint(
        checkpoint_path='/workspace/src/epoch=1062.ckpt'
    ).to(device)
    model.load_state_dict(state_dict=torch.load(env_model_path, map_location='cpu'))
    hparams = model.hparams
    model.eval()
    summary(model)

    dataset = make_dataset()

    patch_size = hparams.patch_size

    env = PatchSetsClassificationEnv(dataset, model, patch_size, done_prob)
    return env


@st.cache(allow_output_mutation=True)
def make_agent(agent_model_path: str, obs_size: int, patch_size: int, device):
    model = src.agent_model.Model(obs_size, patch_size).to(device)
    model.load_state_dict(torch.load(agent_model_path, map_location='cpu'))
    model.eval()
    summary(model)
    return model


# @st.cache(hash_funcs={Tensor: lambda x: x.cpu().detach().numpy()})
def predict_all_patch(
    model: nn.Module,
    image: Tensor,
    patch_size: int,
    additional_patch_set: Tensor = None,
):
    patch_all = (
        image.unfold(1, patch_size, 1)
        .unfold(2, patch_size, 1)
        .reshape(-1, 1, patch_size, patch_size)
    )
    if additional_patch_set is not None:
        patch_all = torch.cat(
            [
                patch_all,
                additional_patch_set.expand(patch_all.shape[0], -1, -1, -1),
            ],
            1,
        )
    predicted_all = model(patch_all)
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
        image.shape[1] - patch_size + 1,
        image.shape[2] - patch_size + 1,
    )
    return loss_map


@st.cache(hash_funcs={Tensor: lambda x: x.cpu().detach().numpy()})
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
    env,
    agent_model: nn.Module,
    patch_size: int,
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
            predicted_all = predict_all_patch(env.model, image, patch_size)
        else:
            predicted_all = predict_all_patch(
                env.model,
                image,
                patch_size,
                torch.stack(env.trajectory['patch'], 1),
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
            image.shape[1] - patch_size + 1,
            image.shape[2] - patch_size + 1,
        )

        st.write('Slect prob. map')
        fig = plt.figure()
        plt.imshow(select_probs_map.cpu().detach().numpy())
        plt.colorbar()
        st.pyplot(fig, True)
        plt.close()

    with col_action_dataframe:
        st.write('Action dataframe')
        df = make_action_df(loss_map, predicted_all, env.dataset.unique_alphabet)
        st.dataframe(df.head())

    if default_action_mode == 'RL':
        best_x = (select_probs.argmax() % (image.shape[2] - patch_size + 1)).item()
        best_y = (select_probs.argmax() // (image.shape[2] - patch_size + 1)).item()
    elif default_action_mode == 'Minimum loss':
        best_x = int(df['x'][0])
        best_y = int(df['y'][0])
    else:
        assert False

    with col_patch_select:
        action_x = st.slider(
            f'action_x_{step}',
            0,
            image.shape[2] - patch_size,
            value=best_x,
        )
        action_y = st.slider(
            f'action_y_{step}',
            0,
            image.shape[1] - patch_size,
            value=best_y,
        )
        action = action_x + action_y * (image.shape[2] - patch_size + 1)

        _, _, done, _ = env.step(action)
        st.image(
            to_pil_image(env.trajectory['patch'][-1].cpu()),
            use_column_width=True,
            output_format='png',
        )
        done = st.checkbox(f'Done on step {step}', value=((step + 1) % 16 == 0) or done)
    return done


@hydra.main(config_path='../config', config_name='app.yaml')
def main(config):
    logger.info('\n' + hydra.utils.OmegaConf.to_yaml(config))

    st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.write('# RL test app')

    pl.seed_everything(config.seed)

    with st.beta_expander('Setting'):
        device = st.selectbox('device', ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])

        st.write('## Select agent model')
        agent_model_path_pattern = st.text_input(
            'Agent model path pattern', value='**//**/*h/model'
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
            'Env model path pattern', value='**//**/env_model*'
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

    with st.beta_expander('Display yaml file'):
        yaml_path_pattern = st.text_input(
            'Yaml file path pattern', value='**//**/config'
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
            st.write(yaml.safe_load(f))

    env = make_env(env_model_path, 0.99, device)
    dataset = env.dataset

    patch_size = env.model.hparams.patch_size
    obs_size = 2

    agent_model = make_agent(agent_model_path, obs_size, patch_size, device)

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
    elif mode_to_select == 'font index & alphabet index':
        font_index = st.sidebar.number_input(
            f'Font index (0~{len(dataset.unique_font) - 1})',
            0,
            len(dataset.unique_font) - 1,
            value=3,
            step=1,
        )
        alphabet_index = st.sidebar.number_input(
            f'Alphabet index (0~{len(dataset.unique_alphabet) - 1})',
            0,
            len(dataset.unique_alphabet) - 1,
            value=0,
            step=1,
        )
        data_index = dataset.font_alphabet_to_index(font_index, alphabet_index)
    elif mode_to_select == 'font & alphabet':
        font = st.sidebar.selectbox('Font', dataset.unique_font, index=3)
        alphabet = st.sidebar.selectbox('Alphabet', dataset.unique_alphabet)
        font_index = dataset.unique_font.index(font)
        alphabet_index = dataset.unique_alphabet.index(alphabet)
        data_index = dataset.font_alphabet_to_index(font_index, alphabet_index)
    else:
        assert False
    st.sidebar.write(f'Data index:', data_index)
    st.sidebar.write(
        f'Font: `{dataset.index_to_font(data_index)}`, `{dataset.unique_font[dataset.index_to_font(data_index)]}`'
    )
    st.sidebar.write(
        f'Alphabet: `{dataset.index_to_alphabet(data_index)}`, `{dataset.unique_alphabet[dataset.index_to_alphabet(data_index)]}`'
    )

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
            env, agent_model, env.patch_size, image, target, step, default_action
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
                    dataset.unique_alphabet,
                    torch.cat(env.trajectory['likelihood'], 0)
                    .softmax(1)
                    .T.detach()
                    .cpu(),
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

    st.sidebar.write('likelihood')
    likelihood_df = df[dataset.unique_alphabet]
    fig = plt.figure()
    plt.plot(
        range(1, step + 1),
        likelihood_df[dataset.unique_alphabet[target]],
        color='black',
        marker='o',
        label=dataset.unique_alphabet[target],
    )
    top_acc_label = []
    for _, row in likelihood_df.iterrows():
        top_acc_label.extend(row.sort_values(ascending=False)[:2].index)
    display_label = sorted(set(top_acc_label) - {dataset.unique_alphabet[target]})
    for label in display_label:
        plt.plot(
            range(1, step + 1),
            likelihood_df[label],
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
    main()
