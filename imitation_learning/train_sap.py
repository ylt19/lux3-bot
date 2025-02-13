import os
import glob
import json
import time
import torch
import random
import pickle
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from enum import IntEnum
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imshow

from agent.base import (
    Global,
    SPACE_SIZE,
    transpose,
    get_opposite,
    get_nebula_tile_drift_speed,
    manhattan_distance,
    nearby_positions,
    clip_int,
)
from agent.path import Action, ActionType
from agent.state import State


EPISODES_DIR = "dataset/episodes"
AGENT_EPISODES_DIR = "dataset/agent_episodes"
MODEL_NAME = "sap_unet"

N_CHANNELS = 20
N_GLOBAL = 16
N_CLASSES = 1

GF_INFO = [
    {"name": "nebula_tile_drift_direction", "m": 1},
    {"name": "nebula_tile_energy_reduction", "m": Global.MAX_UNIT_ENERGY},
    {"name": "unit_move_cost", "m": Global.MAX_UNIT_ENERGY},
    {"name": "unit_sap_cost", "m": Global.MAX_UNIT_ENERGY},
    {"name": "unit_sap_range", "m": Global.SPACE_SIZE},
    {"name": "unit_sap_dropoff_factor", "m": 1},
    {"name": "unit_energy_void_factor", "m": 1},
    {"name": "match_step", "m": Global.MAX_STEPS_IN_MATCH},
    {"name": "match_number", "m": Global.NUM_MATCHES_IN_GAME},
    {"name": "num_steps_before_obstacle_movement", "m": Global.MAX_STEPS_IN_MATCH},
    {"name": "my_points", "m": 1000},
    {"name": "opp_points", "m": 1000},
    {"name": "my_reward", "m": 1000},
    {"name": "opp_reward", "m": 1000},
    {"name": "num_relics_found", "m": 3},
    {"name": "unit_energy", "m": Global.MAX_UNIT_ENERGY},
]


def create_dataset_from_pickle(submission_ids, num_episodes=None):
    paths = []
    for submission_id in submission_ids:
        paths += list(glob.glob(f"{AGENT_EPISODES_DIR}/{submission_id}_*.pkl"))

    if num_episodes is not None:
        paths = paths[:num_episodes]

    obses = []
    for path in tqdm(paths):
        obses += pars_agent_episode(pickle.load(open(path, "rb")))

    return obses


def pars_agent_episode(agent_episode):
    episode_id = agent_episode["episode_id"]
    team_id = agent_episode["team_id"]
    wins = agent_episode["wins"]
    if not any(wins):
        return []

    # print(f"start parsing episode {episode_id} team {team_id}")

    Global.clear()
    Global.VERBOSITY = 1

    data = []

    game_params = agent_episode["params"]
    Global.MAX_UNITS = game_params["max_units"]
    Global.UNIT_MOVE_COST = game_params["unit_move_cost"]
    Global.UNIT_SAP_COST = game_params["unit_sap_cost"]
    Global.UNIT_SAP_RANGE = game_params["unit_sap_range"]
    Global.UNIT_SENSOR_RANGE = game_params["unit_sensor_range"]

    r = Global.UNIT_SAP_RANGE * 2 + 1
    sap_kernel = np.ones((r, r), dtype=np.int32)

    exploration_flags = agent_episode["exploration_flags"]

    state = State(team_id)
    previous_step_unit_array = np.zeros((4, SPACE_SIZE, SPACE_SIZE), dtype=np.float16)
    previous_step_sap_array = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float16)
    for team_observation, team_actions in zip(
        agent_episode["observations"], agent_episode["actions"]
    ):
        state.update(team_observation)

        step = state.global_step
        match_step = state.match_step
        match_number = state.match_number
        # print(f"start step {step}, episode_id {episode_id}, team {team_id}")

        if match_step == 0:
            previous_step_unit_array[:] = 0
            previous_step_sap_array[:] = 0
            continue

        if any(
            num_wins > Global.NUM_MATCHES_IN_GAME / 2
            for num_wins in team_observation["team_wins"]
        ):
            break

        is_win = wins[match_number]

        if not is_win:
            continue

        nebula_tile_energy_reduction_ = (
            game_params["nebula_tile_energy_reduction"]
            if step >= exploration_flags["nebula_energy_reduction"]
            else 0
        )
        obs_array, saps = pars_obs(state, team_actions, nebula_tile_energy_reduction_)

        obs_array[6:10] = previous_step_unit_array
        previous_step_unit_array = obs_array[:4].copy()

        obs_array[10] = previous_step_sap_array
        unit_sap_dropoff_factor = (
            game_params["unit_sap_dropoff_factor"]
            if step >= exploration_flags["unit_sap_dropoff_factor"]
            else 0.5
        )
        fill_sap_array(
            state, team_actions, previous_step_sap_array, unit_sap_dropoff_factor
        )

        if saps:

            if team_id == 1:
                obs_array = transpose(obs_array, reflective=True)
                flipped_saps = []
                for sap in saps:
                    flipped_saps.append(
                        {
                            "unit_position": get_opposite(*sap["unit_position"]),
                            "unit_energy": sap["unit_energy"],
                            "sap_position": get_opposite(*sap["sap_position"]),
                        }
                    )
                saps = flipped_saps

        for sap in saps:

            obs_array_coppy = np.array(obs_array)

            unit_x, unit_y = sap["unit_position"]
            ship_arr = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)
            ship_arr[unit_y, unit_x] = 1
            ship_arr = convolve2d(
                ship_arr,
                sap_kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            obs_array_coppy[19] = ship_arr

            if Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
                nebula_tile_drift_direction = (
                    1 if get_nebula_tile_drift_speed() > 0 else -1
                )
                num_steps_before_obstacle_movement = (
                    state.num_steps_before_obstacle_movement()
                )
            else:
                nebula_tile_drift_direction = 0
                num_steps_before_obstacle_movement = -Global.MAX_STEPS_IN_MATCH

            gf = (
                nebula_tile_drift_direction,
                (
                    game_params["nebula_tile_energy_reduction"] / Global.MAX_UNIT_ENERGY
                    if step >= exploration_flags["nebula_energy_reduction"]
                    else -1
                ),
                Global.UNIT_MOVE_COST / Global.MAX_UNIT_ENERGY,
                Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY,
                Global.UNIT_SAP_RANGE / Global.SPACE_SIZE,
                (
                    game_params["unit_sap_dropoff_factor"]
                    if step >= exploration_flags["unit_sap_dropoff_factor"]
                    else -1
                ),
                (
                    game_params["unit_energy_void_factor"]
                    if step >= exploration_flags["unit_energy_void_factor"]
                    else -1
                ),
                match_step / Global.MAX_STEPS_IN_MATCH,
                match_number / Global.NUM_MATCHES_IN_GAME,
                num_steps_before_obstacle_movement / Global.MAX_STEPS_IN_MATCH,
                state.fleet.points / 1000,
                state.opp_fleet.points / 1000,
                state.fleet.reward / 1000,
                state.opp_fleet.reward / 1000,
                sum(Global.RELIC_RESULTS) / 3,
                sap["unit_energy"] / Global.MAX_UNIT_ENERGY,
            )

            d = {
                "state": obs_array_coppy,
                "sap_position": sap["sap_position"],
                "step": step,
                "episode_id": episode_id,
                "team_id": team_id,
                "gf": gf,
            }

            data.append(d)

    return data


def fill_sap_array(
    state, team_actions, previous_step_sap_array, unit_sap_dropoff_factor
):
    previous_step_sap_array[:] = 0
    for ship, (action_type, dx, dy) in zip(state.fleet.ships, team_actions):
        if action_type == ActionType.sap and ship.node is not None and ship.can_sap():
            sap_x = ship.node.x + dx
            sap_y = ship.node.y + dy
            for x, y in nearby_positions(sap_x, sap_y, 1):
                if x == sap_x and y == sap_y:
                    previous_step_sap_array[y, x] += (
                        Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY
                    )
                else:
                    previous_step_sap_array[y, x] += (
                        Global.UNIT_SAP_COST
                        / Global.MAX_UNIT_ENERGY
                        * unit_sap_dropoff_factor
                    )


def pars_obs(state, team_actions, nebula_tile_energy_reduction):
    d = np.zeros((20, SPACE_SIZE, SPACE_SIZE), dtype=np.float16)
    saps = []

    energy_field = state.field.energy
    nebulae_field = state.field.nebulae

    # 0 - unit positions
    # 1 - unit energy
    # 2 - unit next positions
    # 3 - unit next energy
    for ship, action in zip(state.fleet.ships, team_actions):
        if (
            ship.node is not None
            and ship.energy >= 0
            and ship.steps_since_last_seen == 0
        ):
            x, y = ship.coordinates
            d[0, y, x] += 1
            d[1, y, x] += ship.energy

            action_type, sap_dx, sap_dy = action
            action_type = ActionType(action_type)
            if action_type == ActionType.sap:
                saps.append(
                    {
                        "unit_position": (x, y),
                        "unit_energy": ship.energy,
                        "sap_position": (x + sap_dx, y + sap_dy),
                    }
                )

            dx, dy = action_type.to_direction()

            next_x = clip_int(x + dx)
            next_y = clip_int(y + dy)

            # if state.global_step == 75:
            #     print(ship, action_type, next_x, next_y)

            next_energy = ship.energy + energy_field[next_y, next_x]
            if action_type == ActionType.sap:
                next_energy -= Global.UNIT_SAP_COST
            elif action_type != ActionType.center:
                next_energy -= Global.UNIT_MOVE_COST

            if nebulae_field[next_y, next_x]:
                next_energy -= nebula_tile_energy_reduction

            d[2, next_y, next_x] += 1
            d[3, next_y, next_x] += next_energy

    # 4 - opp unit position
    # 5 - opp unit energy
    for unit in state.opp_fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[4, y, x] += 1
            d[5, y, x] += unit.energy

    d[0] /= 10
    d[1] /= Global.MAX_UNIT_ENERGY
    d[2] /= 10
    d[3] /= Global.MAX_UNIT_ENERGY
    d[4] /= 10
    d[5] /= Global.MAX_UNIT_ENERGY

    # 6 - previous step unit positions
    # 7 - previous step unit energy
    # 8 - previous step opp unit positions
    # 9 - previous step opp unit energy
    # 10 - previous step sap array

    f = state.field
    d[11] = f.vision
    d[12] = f.energy / Global.MAX_UNIT_ENERGY
    d[13] = f.asteroid
    d[14] = f.nebulae
    d[15] = f.relic
    d[16] = f.reward
    d[17] = (state.global_step - f.last_relic_check) / Global.MAX_STEPS_IN_MATCH
    d[18] = (state.global_step - f.last_step_in_vision) / Global.MAX_STEPS_IN_MATCH

    return d, saps


# ===================#
#      Dataset       #
# ===================#


class LuxDataset(Dataset):

    def __init__(self, obses):
        self.obses = obses

    def __len__(self):
        return len(self.obses)

    def __getitem__(self, idx):
        obs = self.obses[idx]

        state = obs["state"]

        aug = random.random() > 0.5

        label = np.zeros((1, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        sap_x, sap_y = obs["sap_position"]
        label[0, sap_y, sap_x] = 1
        if aug:
            state = transpose(state)
            label = transpose(label)

        gf = np.zeros((len(GF_INFO), 3, 3), dtype=np.float32)

        for i, (val, gf_info) in enumerate(zip(obs["gf"], GF_INFO)):
            gf[i] = val
            if aug and gf_info["name"] == "nebula_tile_drift_speed":
                gf[i] = -val

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


def get_loss(policy, label):
    loss = criterion(policy, label)
    return loss


def get_acc(policy, label):
    correct = 0
    total = 0
    for p, a in zip(policy, label):
        p = p.squeeze(0)
        p = (p == p.max()).float()

        a = a.squeeze(0)

        is_correct = (p == a).sum() == SPACE_SIZE * SPACE_SIZE

        correct += is_correct
        total += 1

    return correct, total


def train_model(
    model, dataloaders_dict, optimizer, scheduler, num_epochs, model_name="model"
):
    best_loss = 10**9

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0
            correct = 0
            total = 0

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
                        _correct, _total = get_acc(policy, label)
                        correct += _correct
                        total += _total

                    epoch_loss += loss.item() * len(policy)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            if phase != "train":
                epoch_acc = correct.double() / total

                if scheduler is not None:
                    scheduler.step(epoch_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.5f} | Acc: {epoch_acc:.4f}"
            )
            time.sleep(10)

        if epoch_loss < best_loss:
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
            best_loss = epoch_loss


def train(data, model_name="model", num_epochs=5, batch_size=64):
    seed_everything(42)

    model = UNet()  # torch.jit.load(f'{model_name}.pth')

    train, val = train_test_split(data, test_size=0.1, random_state=42)
    train_loader = DataLoader(
        LuxDataset(train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        LuxDataset(val), batch_size=batch_size, shuffle=False, num_workers=0
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


def main():
    data = create_dataset_from_pickle([42340565], num_episodes=None)
    train(data, model_name=MODEL_NAME, num_epochs=15, batch_size=128)
