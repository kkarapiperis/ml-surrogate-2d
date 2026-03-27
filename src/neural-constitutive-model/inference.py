import sys
import h5py
import copy
import time
import torch
import argparse
import numpy as np
from itertools import repeat

"""
Neural network architecture
"""
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

"""
Model inference
"""
if __name__ == "__main__":

	# Parser options
	parser = argparse.ArgumentParser()
	parser.add_argument('--hidden_dim', type=int, default=32)
	parser.add_argument('--layers', type=int, default=4)
	parser.add_argument('--device', type=str, default='cpu') 
	parser.add_argument('--model_name', type=str, default='model')
	args = parser.parse_args()

	# Model parameters
	hidden_dim = args.hidden_dim
	layers = args.layers
	device = args.device	
	model_name = args.model_name
	print(model_name)

	# Read test path
	file = h5py.File('./data.h5', 'r')
	eps = torch.tensor(file['eps'][:], dtype=torch.float32)
	n_data = len(eps)
	file.close()

	# Model 
	out_dim = 1
	eps_dim = 3

	# Initialize model
	energy_model = MLP(eps_dim, hidden_dim, out_dim, layers, torch.nn.Tanh())

	# Load trained model state
	energy_model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))

	# Move to device
	energy_model.to(device)
	energy_model.eval()

	def dW_deps(x):
		return torch.autograd.functional.jacobian(energy_model, x.unsqueeze(0), 
			create_graph=False, vectorize=True).squeeze(0,1,2)

	def d2W_deps(x, create_graph=False):
		return torch.autograd.functional.hessian(energy_model, x.unsqueeze(0), 
			create_graph=create_graph, 
			vectorize=True).diagonal(dim1=0,dim2=2).permute(2,1,0).flatten(1)
	
	# Initialize output of model for all data
	sig = torch.zeros(n_data, eps_dim)
	C = torch.zeros(n_data, eps_dim*eps_dim)

	# Timestepping
	start = time.time()
	for i,eps_i in enumerate(eps):
		
		# Append to output
		sig[i] = dW_deps(eps_i)
		C[i] = d2W_deps(eps_i)

	header_path='eps11, eps22, eps12, sig11, sig22, sig12, C11, C12, C13, C21, C22, C23, C31, C32, C33'

  	# Save the predicted path quantities to file
	np.savetxt(model_name + '-prediction.csv', 
		np.c_[eps, sig.detach().cpu().numpy(), C.detach().cpu().numpy()],
		delimiter=",", header=header_path)

	# Evaluate also the potentials on a grid and save the data
	eps_ii_grid = np.linspace(-0.1, 0.1, 2)
	eps_ij_grid = [0] # evaluate with zero shear stresses
	eps_grid = np.meshgrid(*tuple(repeat(eps_ii_grid,2)), 
		                   *tuple(repeat(eps_ij_grid,1)))
	eps_grid = np.vstack([grid.ravel() for grid in eps_grid]).T
	eps_grid = torch.tensor(eps_grid, dtype=torch.float32)
	W_grid = np.zeros(len(eps_grid))

	for i, eps_i in enumerate(zip(eps_grid)):
		W_grid[i] = energy_model(eps_i[0]).squeeze(0)

	header_free_energy='eps11, eps22, eps12, W'

	np.savetxt(model_name + '-free-energy-landscape.csv', 
		np.c_[eps_grid, W_grid], delimiter=",", header=header_free_energy)
