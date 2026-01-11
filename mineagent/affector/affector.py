import torch
import torch.nn as nn
from gymnasium.spaces import MultiDiscrete

from ..utils import add_forward_hooks


class LinearAffector(nn.Module):
    """
    Feed-forward affector (action) module.

    This module produces distributions over actions for the environment given some input using linear layers.
    """

    def __init__(self, embed_dim: int, action_space: MultiDiscrete):
        """
        Parameters
        ----------
        embed_dim : int
            Dimension of the input embeddings
        action_space : MultiDiscrete
            The action space for Minecraft is a length 8 numpy array:
            0: longitudinal movement (i.e. moving forward and back)
            1: lateral movement (i.e. moving left and right)
            2: vertical movement (i.e. jumping)
            3: pitch movement (vertical rotation, i.e. looking up and down)
            4: yaw movement (hortizontal rotation, i.e. looking left and right)
            5: functional (0: noop, 1: use, 2: drop, 3: attack, 4: craft, 5: equip, 6: place, 7: destroy)
            6: item index to craft
            7: inventory index
        """
        super().__init__()

        # Movement
        self.longitudinal_action = nn.Linear(embed_dim, int(action_space.nvec[0]))
        self.lateral_action = nn.Linear(embed_dim, int(action_space.nvec[1]))
        self.vertical_action = nn.Linear(embed_dim, int(action_space.nvec[2]))
        self.pitch_action = nn.Linear(embed_dim, int(action_space.nvec[3]))
        self.yaw_action = nn.Linear(embed_dim, int(action_space.nvec[4]))

        # Manipulation
        self.functional_action = nn.Linear(embed_dim, int(action_space.nvec[5]))
        self.craft_action = nn.Linear(embed_dim, int(action_space.nvec[6]))
        self.inventory_action = nn.Linear(embed_dim, int(action_space.nvec[7]))

        # Internal
        ## distribution for which we can sample regions of interest
        self.focus_means = nn.Linear(embed_dim, 2)
        self.focus_stds = nn.Linear(embed_dim, 2)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.action_space = action_space

        # Monitoring
        self.start_monitoring()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        long_dist = self.softmax(self.longitudinal_action(x))
        lat_dist = self.softmax(self.lateral_action(x))
        vert_dist = self.softmax(self.vertical_action(x))
        pitch_dist = self.softmax(self.pitch_action(x))
        yaw_dist = self.softmax(self.yaw_action(x))
        func_dist = self.softmax(self.functional_action(x))
        craft_dist = self.softmax(self.craft_action(x))
        inventory_dist = self.softmax(self.inventory_action(x))
        roi_means = self.focus_means(x)
        roi_stds = self.softplus(self.focus_stds(x))

        return (
            long_dist,
            lat_dist,
            vert_dist,
            pitch_dist,
            yaw_dist,
            func_dist,
            craft_dist,
            inventory_dist,
            roi_means,
            roi_stds,
        )

    def stop_monitoring(self):
        for hook in self.hooks:
            hook.remove()

    def start_monitoring(self):
        self.hooks = add_forward_hooks(self, "Affector")
