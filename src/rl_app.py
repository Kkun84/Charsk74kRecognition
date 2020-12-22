from matplotlib import figure
import streamlit as st
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from datetime import datetime
from dataset import AdobeFontDataset
from torchvision import transforms
from torchsummary import summary
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import pandas as pd

from src.env import PatchSetsClassificationEnv
import src.env_model
import src.agent_model


device = 'cuda'


@st.cache(allow_output_mutation=True)
def make_env():
    model = src.env_model.Model.load_from_checkpoint(
        checkpoint_path='/workspace/src/env_model/epoch=711.ckpt'
    ).to(device)
    hparams = model.hparams
    model.eval()
    summary(model)

    image_size = 100
    patch_size = hparams.patch_size
    feature_n = hparams.feature_n
    obs_size = feature_n + 1
    n_actions = (image_size - patch_size) ** 2
    done_loss = 0.1

    # dataset = [
    #     (
    #         torch.rand(1, image_size, image_size),
    #         torch.randint(26, [1]).item(),
    #     )
    #     for i in range(100)
    # ]
    dataset = AdobeFontDataset(
        path='/dataset/AdobeFontCharImages',
        transform=transforms.ToTensor(),
        target_transform=lambda x: x['alphabet'],
        upper=True,
        lower=False,
    )

    env = PatchSetsClassificationEnv(dataset, model, patch_size, feature_n, done_loss)
    return env


@st.cache(allow_output_mutation=True)
def make_agent(obs_size, n_actions, hidden_n, patch_size):
    model = src.agent_model.Model(obs_size, n_actions, hidden_n, patch_size).to(device)
    model.load_state_dict(
        torch.load('/workspace/outputs/Default/2020-12-22/12-32-24/best/model.pt')
    )
    model.eval()
    summary(model)
    return model


def predict_all_pach(
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


def one_step(
    env, agent_model: nn.Module, patch_size: int, image: Tensor, target: int, step: int
):
    col_loss_map, col_rl_map, col_loss_dataflame, col_patch_select = st.beta_columns(
        [2, 2, 1, 1]
    )

    with col_loss_map:
        if step == 0:
            predicted_all = predict_all_pach(env.model, image, patch_size)
        else:
            predicted_all = predict_all_pach(
                env.model,
                image,
                patch_size,
                torch.stack(env.trajectory['patch'], 1),
            )
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
        st.write('Loss map')
        fig = plt.figure()
        plt.imshow(loss_map.detach().cpu().numpy()[0])
        plt.colorbar()
        st.pyplot(fig, True)

    with col_rl_map:
        obs = env.trajectory['observation'][-1]
        select_probs = agent_model(obs[None])[0].probs[0]
        select_probs_map = select_probs.reshape(
            image.shape[1] - patch_size + 1,
            image.shape[2] - patch_size + 1,
        )

        st.write('Slect prob. map')
        fig = plt.figure()
        plt.imshow(select_probs_map.detach().cpu().numpy())
        plt.colorbar()
        st.pyplot(fig, True)

    with col_loss_dataflame:
        st.write('Low loss actions')

        loss_sorted, loss_ranking = loss_map.reshape(-1).sort()
        loss_ranking_x = loss_ranking % (image.shape[2] - patch_size + 1)
        loss_ranking_y = loss_ranking // (image.shape[2] - patch_size + 1)

        df = pd.DataFrame(
            dict(
                action=loss_ranking.detach().cpu(),
                x=loss_ranking_x.detach().cpu(),
                y=loss_ranking_y.detach().cpu(),
                loss=loss_sorted.detach().cpu(),
            ),
        )
        st.dataframe(df)

    best_x = (select_probs.argmax() % (image.shape[2] - patch_size + 1)).item()
    best_y = (select_probs.argmax() // (image.shape[2] - patch_size + 1)).item()

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
        st.image(to_pil_image(env.trajectory['patch'][-1].cpu()).resize((200, 200)))
        if not done:
            done = st.checkbox(f'Done on step {step}', value=True)
        else:
            st.write('Done')
    return done


def main():
    st.set_page_config(
        layout='wide',
        initial_sidebar_state='expanded',
    )

    pl.seed_everything(0)

    env = make_env()
    dataset = env.dataset

    image_size = 100
    patch_size = env.model.hparams.patch_size
    feature_n = env.model.hparams.feature_n

    obs_size = feature_n + 1
    n_actions = (image_size - patch_size) ** 2

    agent_model = make_agent(obs_size, n_actions, 64, patch_size)

    st.write('# RL test app')

    if st.sidebar.radio('Mode to select', ['index', 'font & alphabet']) == 'index':
        data_index = st.sidebar.number_input(
            f'Data index (0~{len(dataset)})', 0, len(dataset), value=0, step=1
        )
        st.sidebar.write(
            f'Font: `{dataset.unique_font[dataset.index_to_font(data_index)]}`',
            '\n',
            f'Alphabet: `{dataset.unique_alphabet[dataset.index_to_alphabet(data_index)]}`',
        )
    else:
        font = st.sidebar.selectbox('Font', dataset.unique_font)
        alphabet = st.sidebar.selectbox('Alphabet', dataset.unique_alphabet)
        font_index = dataset.unique_font.index(font)
        alphabet_index = dataset.unique_alphabet.index(alphabet)
        data_index = dataset.font_alphabet_to_index(font_index, alphabet_index)
        st.sidebar.write('Data index:', data_index)

    _ = env.reset(data_index=data_index)
    image = env.data[0]
    target = env.data[1]

    st.sidebar.write('# Original data')
    st.sidebar.image(to_pil_image(image.cpu()), use_column_width=True)

    step = 0
    done = False
    while not done:
        st.write(f'# step {step}')
        done = one_step(env, agent_model, env.patch_size, image, target, step)
        step += 1

    st.sidebar.write('# History')

    df = pd.DataFrame(
        dict(
            action=env.trajectory['action'],
            x=[i % image.shape[2] for i in env.trajectory['action']],
            y=[i // image.shape[2] for i in env.trajectory['action']],
            loss=env.trajectory['loss'],
            reward=env.trajectory['reward'],
        ),
    )
    st.sidebar.dataframe(df)

    st.sidebar.write('loss')
    fig = plt.figure()
    plt.plot(env.trajectory['loss'], marker='o', label='loss')
    st.sidebar.pyplot(fig, True)

    st.sidebar.write('reward')
    fig = plt.figure()
    plt.plot(env.trajectory['reward'], marker='o', label='reward')
    st.sidebar.pyplot(fig, True)

    st.sidebar.write('patch')
    for image in env.trajectory['patch']:
        st.sidebar.image(to_pil_image(image.cpu()))

    return


if __name__ == "__main__":
    main()
