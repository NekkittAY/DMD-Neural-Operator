import numpy as np
import torch
import torch.nn as nn
from pydmd import DMD


class DMDProcessor:
    def __init__(self, data: torch.Tensor, rank: int):
        """Process input data using Dynamic Mode Decomposition.

        Args:
            data: Input tensor of shape (batch_size, ny, nx)
            rank: Rank for SVD approximation
        """
        self.data = data
        self.rank = rank

    def _validate_input(self):
        if self.rank <= 0:
            raise ValueError("Rank must be positive integer")

    def _compute_dmd(self):
        """Perform DMD and return reconstructed data."""
        try:
            snapshots = self.data.reshape(self.data.shape[0], -1).T
            dmd = DMD(svd_rank=self.rank)
            dmd.fit(snapshots)

            if dmd.reconstructed_data is None:
                raise RuntimeError("DMD reconstruction failed")

            return dmd

        except Exception as e:
            raise RuntimeError(f"DMD processing failed: {str(e)}")

    def _calc_energy(self):
        dmd = self._compute_dmd()
        energy = np.cumsum(np.abs(dmd.amplitudes)) / np.sum(np.abs(dmd.amplitudes))
        n_modes = np.argmax(energy > 0.95) + 1
        return n_modes

    def method(self):
        dmd = self._compute_dmd()

        modes = [dmd.modes.real[:, i] for i in range(len(dmd.amplitudes))]
        dynamics = [dmd.dynamics.real[i] for i in range(len(dmd.amplitudes))]
        return [modes, dynamics]


  class DMDNeuralOperator(nn.Module):
  def __init__(self, branch1_dim, branch_dmd_dim_modes, branch_dmd_dim_dynamics, trunk_dim):
      """Neural operator with DMD preprocessing.

      Args:
          branch1_dim: Layer dimensions for primary branch
          branch_dmd_dim_modes: Layer dimensions for DMD modes branch
          branch_dmd_dim_dynamics: Layer dimensions for DMD dynamics branch
          trunk_dims: Layer dimensions for trunk network
      """
      super(DMDNeuralOperator, self).__init__()

      modules = []
      for i, h_dim in enumerate(branch1_dim):
        if i == 0:
          in_channels = h_dim
        else:
          modules.append(nn.Sequential(
              nn.Linear(in_channels, h_dim),
              nn.Tanh()
              )
          )
          in_channels = h_dim

      self._branch_1 = nn.Sequential(*modules)

      modules = []
      for i, h_dim in enumerate(branch_dmd_dim_modes):
        if i == 0:
          in_channels = h_dim
        else:
          modules.append(nn.Sequential(
              nn.Linear(in_channels, h_dim),
              nn.Tanh()
              )
          )
          in_channels = h_dim
      self._branch_dmd_modes = nn.Sequential(*modules)

      modules = []
      for i, h_dim in enumerate(branch_dmd_dim_dynamics):
        if i == 0:
          in_channels = h_dim
        else:
          modules.append(nn.Sequential(
              nn.Linear(in_channels, h_dim),
              nn.Tanh()
              )
          )
          in_channels = h_dim
      self._branch_dmd_dynamics = nn.Sequential(*modules)

      modules = []
      for i, h_dim in enumerate(trunk_dim):
        if i == 0:
          in_channels = h_dim
        else:
          modules.append(nn.Sequential(
              nn.Linear(in_channels, h_dim),
              nn.Tanh()
              )
          )
          in_channels = h_dim

      self._trunk = nn.Sequential(*modules)

      self.final_linear = nn.Linear(trunk_dim[-1], 10)

  def forward(self, f: torch.Tensor, f_dmd_modes: torch.Tensor, f_dmd_dynamics: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
        f: Input function (batch_size, *spatial_dims)
        x: Evaluation points (num_points, coord_dim)

    Returns:
        Output tensor (batch_size, num_points)
    """
    modes, dynamics = f_dmd_modes, f_dmd_dynamics

    branch_dmd_modes = self._branch_dmd_modes(modes)
    branch_dmd_dynamics = self._branch_dmd_dynamics(dynamics)
    y_branch_dmd = branch_dmd_modes * branch_dmd_dynamics

    y_branch1 = self._branch_1(f)
    y_br = y_branch1 * y_branch_dmd

    y_tr = self._trunk(x)

    y_out = y_br @ y_tr

    linear_out = nn.Linear(y_out.shape[-1], 10)
    tanh_out = nn.Tanh()

    y_out = self.final_linear(y_out)

    return y_out

  def loss(self, f, f_dmd_modes, f_dmd_dynamics, x, y):
    y_out = self.forward(f, f_dmd_modes, f_dmd_dynamics, x)
    loss = ((y_out - y) ** 2).mean()
    return loss
