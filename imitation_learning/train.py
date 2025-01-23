import os
import glob
import time
import torch
import random
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from agent.base import (
    Global,
    SPACE_SIZE,
    transpose,
    get_opposite,
    get_nebula_tile_drift_speed,
    nearby_positions,
)
from agent.path import ActionType
from agent.state import State

EPISODES_DIR = "dataset/episodes"
AGENT_EPISODES_DIR = "dataset/agent_episodes"
MODEL_NAME = "unit_unet"

N_CHANNELS = 15
N_GLOBAL = 14
N_CLASSES = 6

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

        obs_array, actions = pars_obs(state, team_actions)

        obs_array[4:8] = previous_step_unit_array
        previous_step_unit_array = obs_array[:4]

        obs_array[8] = previous_step_sap_array
        unit_sap_dropoff_factor = (
            game_params["unit_sap_dropoff_factor"]
            if step >= exploration_flags["unit_sap_dropoff_factor"]
            else 0.5
        )
        fill_sap_array(
            state, team_actions, previous_step_sap_array, unit_sap_dropoff_factor
        )

        add_to_dataset = True
        if not actions:
            add_to_dataset = False
        if (
            all(x == ActionType.center for x in actions.values())
            and random.random() > 0.1
        ):
            add_to_dataset = False

        if add_to_dataset:

            if team_id == 1:
                obs_array = transpose(obs_array, reflective=True)
                flipped_actions = {}
                for (x, y), action_id in actions.items():
                    x, y = get_opposite(x, y)
                    action_id = ActionType(action_id).transpose(reflective=True).value
                    flipped_actions[(x, y)] = action_id
                actions = flipped_actions

            if step >= exploration_flags["nebula_tile_drift_speed"]:
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
            )

            d = {
                "state": obs_array,
                "actions": actions,
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
                    previous_step_sap_array[y, x] += 1
                else:
                    previous_step_sap_array[y, x] += unit_sap_dropoff_factor
    previous_step_sap_array *= Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY


def pars_obs(state, team_actions):
    d = np.zeros((15, SPACE_SIZE, SPACE_SIZE), dtype=np.float16)

    # 0 - unit positions
    # 1 - unit energy
    for unit in state.fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[0, y, x] += 1 / 10
            d[1, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    # 2 - opp unit position
    # 3 - opp unit energy
    for unit in state.opp_fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[2, y, x] += 1 / 10
            d[3, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    # 4 - previous step unit positions
    # 5 - previous step unit energy
    # 6 - previous step opp unit positions
    # 7 - previous step opp unit energy

    # 8 - previous step sap positions

    d[9] = state.field.vision
    d[10] = state.field.energy / Global.MAX_UNIT_ENERGY
    d[11] = state.field.asteroid
    d[12] = state.field.nebulae
    d[13] = state.field.relic
    d[14] = state.field.reward
    # d[15] = state.field.unexplored_for_reward

    actions = {}
    for ship, action in zip(state.fleet.ships, team_actions):
        if ship.node is not None and ship.energy >= 0:
            action_type, dx, dy = action
            position = ship.coordinates
            if position not in actions:
                actions[position] = action_type
            else:
                if action_type != ActionType.center:
                    actions[position] = action_type

    return d, actions


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

        label = np.zeros((6, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        for (x, y), action_type in obs["actions"].items():
            if aug:
                action_type = ActionType(action_type).transpose()
            label[action_type, y, x] = 1

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


def train(data, model_name="model", num_epochs=30, batch_size=64):
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
    train(data, model_name=MODEL_NAME, num_epochs=15, batch_size=256)
