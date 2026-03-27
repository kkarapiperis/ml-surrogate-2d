import os
import sys
import torch
import numpy as np

class MLP(torch.nn.Module):
    """ 
    Simple Multi Layer Perceptron
    """
    def __init__(self, input_dim, hidden_dim, out_dim, n_layers, activation):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc2.append(torch.nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim, bias=True)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.fc3(x)
        return x

class ConstitutiveModel:
    """
    Loads a trained energy model and provides energy() and stress() methods.

    Usage:
      cm = ConstitutiveModel(model_path='output/model-W.pt', device='cpu',
                             eps_dim=3, hidden_dim=32, layers=4)
      W = cm.compute_energy(eps_np)         # returns torch.Tensor (N,1) on device
      sig = cm.compute_stress(eps_np)       # returns torch.Tensor (N,eps_dim) on device
      tangent = cm.compute_stiffness(eps_np)  # returns torch.Tensor (N,eps_dim,eps_dim) on device

    Notes:
      - eps input can be numpy array shape (N,eps_dim) or torch tensor (N,eps_dim)
      - If a single sample provided (eps_dim,), it will be treated as batch size 1.
    """
    def __init__(self, model_path, eps_dim=3, hidden_dim=32, out_dim=1,
                 layers=4, activation=torch.nn.Tanh(), device='cpu'):
        self.device = torch.device(device)
        self.model = MLP(eps_dim, hidden_dim, out_dim, layers, activation).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.eval()

    def _prepare_input(self, eps):
        """
        Convert eps to torch tensor on device with batch dim.
        """
        if isinstance(eps, np.ndarray):
            eps_t = torch.from_numpy(eps).to(dtype=torch.float32, device=self.device)
        elif torch.is_tensor(eps):
            eps_t = eps.to(device=self.device, dtype=torch.float32)
        else:
            eps_t = torch.tensor(eps, dtype=torch.float32, device=self.device)
        return eps_t

    def _batch_jacobian(self, func, x, create_graph=False):
        """
        Compute d func(x) / d x for batch inputs.
        """
        def _func_sum(x):
            return func(x).sum(dim=0)
        J = torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph, vectorize=True)
        return J

    def compute_energy(self, eps):
        """
        Compute energy W(eps).
        """
        eps_t = self._prepare_input(eps).to(self.device)
        eps_t = eps_t.requires_grad_(False)
        W_pred = self.model(eps_t)
        return W_pred

    def compute_stress(self, eps):
        """
        Compute stress = dW/deps (by differentiation).
        """
        eps_t = self._prepare_input(eps).to(self.device)
        eps_t = eps_t.requires_grad_(True)
        J = self._batch_jacobian(self.model, eps_t, create_graph=False)  
        return J

    def compute_stiffness(self, eps):
        """
        Compute stiffness C = d^2 W / d eps^2.
        """
        eps_t = self._prepare_input(eps).to(self.device)
        eps_t = eps_t.requires_grad_(True)
        C = torch.autograd.functional.hessian(self.model, eps_t, create_graph=False)
        return C
    
    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self