import os
import time
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from agent.base import SPACE_SIZE, transpose
from agent.path import ActionType

EPISODES_DIR = "dataset/episodes"
AGENT_EPISODES_DIR = "dataset/agent_episodes"
MODEL_NAME = "unit_unet"

N_CHANNELS = 17
N_GLOBAL = 15
N_CLASSES = 6


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def select_episodes(submission_ids, min_opp_score, val_ratio=0.1):
    seed_everything(42)

    submissions_df = pd.read_csv("dataset/submissions.csv")
    sid_to_score = dict(zip(submissions_df["submission_id"], submissions_df["score"]))

    games_df = pd.read_csv("dataset/games.csv")
    games_df["opp_score"] = [sid_to_score[x] for x in games_df["OppSubmissionId"]]
    games_df = games_df[
        games_df["SubmissionId"].isin(submission_ids)
        & (games_df["opp_score"] >= min_opp_score)
    ]

    episodes = set()
    for sid, episode_id in zip(games_df["SubmissionId"], games_df["EpisodeId"]):
        path = f"{AGENT_EPISODES_DIR}/{sid}_{episode_id}.npz"
        if os.path.exists(path):
            train_data = np.load(path)
            num_steps = len(train_data["steps"])
            episodes.add((sid, episode_id, num_steps))

    episodes = sorted(episodes, key=lambda x: x[1])
    print(f"total number of episodes: {len(episodes)}")

    random.shuffle(episodes)
    num_train = int(len(episodes) * (1 - val_ratio))

    train_episode_steps, val_episode_steps = [], []
    for i, (sid, episode_id, num_steps) in enumerate(episodes):
        episode_steps = [(sid, episode_id, s) for s in range(num_steps)]

        if i <= num_train:
            train_episode_steps += episode_steps
        else:
            val_episode_steps += episode_steps

    print(f"train size: {len(train_episode_steps)}")
    print(f"val size: {len(val_episode_steps)}")

    return train_episode_steps, val_episode_steps


# ===================#
#      Dataset       #
# ===================#


class LuxDataset(Dataset):

    def __init__(self, episode_steps, aug=True):
        self.episode_steps = episode_steps
        self.aug = aug

    def __len__(self):
        return len(self.episode_steps)

    def __getitem__(self, idx):
        sid, episode_id, step = self.episode_steps[idx]

        path = f"{AGENT_EPISODES_DIR}/{sid}_{episode_id}.npz"
        data = np.load(path)

        state = data["states"][step]
        gf_values = data["gfs"][step]
        actions = data["actions"][step]

        if self.aug:
            aug = random.random() > 0.5
        else:
            aug = False

        gf = np.zeros((len(gf_values), 3, 3), dtype=np.float32)
        for i, x in enumerate(gf_values):
            gf[i] = x

        label = np.zeros((6, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        for x, y, action_type in actions:
            if x == -1 or y == -1:
                break
            if aug:
                action_type = ActionType(action_type).transpose()
            label[action_type, y, x] = 1

        if aug:
            state = transpose(state)
            label = transpose(label)
            gf[0] *= -1  # nebula_tile_drift_direction

        return state, gf, label


# ===================#
#       Model        #
# ===================#


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AddGlobal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gf):
        return torch.cat([x, gf], dim=1)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        n_global=N_GLOBAL,
        bilinear=True,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.add_global = AddGlobal()
        self.up1 = Up(512 + n_global, 256, bilinear)
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, gf):
        x1 = self.inc(x)  # 24
        x2 = self.down1(x1)  # 12
        x3 = self.down2(x2)  # 6
        x4 = self.down3(x3)  # 3
        x5 = self.add_global(x4, gf)
        x = self.up1(x5, x3)  # 6
        x = self.up2(x, x2)  # 12
        x = self.up3(x, x1)  # 24
        logits = self.outc(x)
        return logits


# ===================#
#       Train        #
# ===================#


criterion = nn.BCEWithLogitsLoss()


def get_loss(policy, label):
    index = []
    x0 = []
    y0 = []
    ans = []
    for i, lb in enumerate(label):
        with_action = lb.any(axis=0)
        for x, y in zip(*torch.where(with_action)):
            index.append(i)
            x0.append(x)
            y0.append(y)

    def to_cuda(x):
        return torch.from_numpy(np.array(x)).cuda().long()

    index = to_cuda(index)
    x0 = torch.tensor(x0)
    y0 = torch.tensor(y0)

    preds = policy[index, :, x0, y0]
    ans = label[index, :, x0, y0]

    loss = criterion(preds, ans)

    return loss


def get_acc(policy, label, label_to_acc):
    correct = 0
    total = 0
    for p, lb in zip(policy, label):
        with_action = lb.any(axis=0)
        for x, y in zip(*torch.where(with_action)):
            _p = p[:, x, y]
            _lb = lb[:, x, y]
            _p = torch.argmax(_p)
            _lb = torch.argmax(_lb)

            correct += _p == _lb
            total += 1

            label_to_acc[int(_lb)][0] += int(_p == _lb)
            label_to_acc[int(_lb)][1] += 1

    return correct, total


def train_model(
    model, dataloaders_dict, optimizer, scheduler, num_epochs, model_name="model"
):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0
            coorect = 0
            total = 0

            label_to_acc = {x: [0, 0] for x in ActionType}

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                gf = item[1].cuda().float()
                label = item[2].cuda().float()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    policy = model(states, gf)
                    loss = get_loss(policy, label)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    else:
                        _coorect, _total = get_acc(policy, label, label_to_acc)
                        coorect += _coorect
                        total += _total

                    epoch_loss += loss.item() * len(policy)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            if phase != "train":
                epoch_acc = coorect.double() / total
                print(
                    "label to auc",
                    {
                        ActionType(l): c / t if t else None
                        for l, (c, t) in label_to_acc.items()
                    },
                )
                if scheduler is not None:
                    scheduler.step(epoch_loss)

            time.sleep(10)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.5f} | Acc: {epoch_acc:.4f}"
            )

        if epoch_acc > best_acc:
            traced = torch.jit.trace(
                model.cpu(),
                example_inputs=(
                    torch.rand(1, N_CHANNELS, 24, 24),
                    torch.rand(1, N_GLOBAL, 3, 3),
                ),
            )
            model_path = f"{model_name}.pth"
            print(f"Saving model to `{model_path}`.")
            traced.save(model_path)
            best_acc = epoch_acc


def train(
    train_data,
    val_data,
    model_name="model",
    num_epochs=30,
    batch_size=64,
):
    seed_everything(42)

    model = UNet()  # torch.jit.load(f'{model_name}.pth')

    train_loader = DataLoader(
        LuxDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
    )
    val_loader = DataLoader(
        LuxDataset(val_data), batch_size=batch_size, shuffle=False, num_workers=4
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=0, min_lr=1e-6, verbose=True
    )
    train_model(
        model,
        dataloaders_dict,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        model_name=model_name,
    )


def main(submission_ids, min_opp_score):
    train_episode_steps, val_episode_steps = select_episodes(
        submission_ids, min_opp_score
    )
    train(
        train_episode_steps,
        val_episode_steps,
        model_name=MODEL_NAME,
        num_epochs=15,
        batch_size=256,
    )
