import sys
import h5py
import copy
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

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
Data reader 
"""
def read_data(test_size, batch_size, seed=0):
	"""
	Reads training data from a file
	"""
	file = h5py.File('./data.h5', 'r')
	eps = file['eps'][:]
	W = file['W'][:]
	sig  = file['sig'][:]
	C = file['C'][:]
	file.close()

	# Split to train/test set
	np.random.seed(seed)
	torch.manual_seed(seed)
	shuffle_idx = np.arange(len(eps))
	np.random.shuffle(shuffle_idx)
	test_split = int(np.floor(test_size*len(eps))) 

	# Convert to torch tensors
	train_idx = shuffle_idx[test_split:]
	eps_train = torch.from_numpy(eps[train_idx]).to(torch.float32)
	W_train = torch.from_numpy(W[train_idx]).to(torch.float32)
	sig_train = torch.from_numpy(sig[train_idx]).to(torch.float32)
	C_train = torch.from_numpy(C[train_idx]).to(torch.float32)
	test_idx = shuffle_idx[:test_split]
	eps_test = torch.from_numpy(eps[test_idx]).to(torch.float32)
	W_test = torch.from_numpy(W[test_idx]).to(torch.float32)
	sig_test = torch.from_numpy(sig[test_idx]).to(torch.float32)
	C_test = torch.from_numpy(C[test_idx]).to(torch.float32)

	return DataLoader(list(zip(eps_train,W_train,sig_train,C_train)), 
					  batch_size=batch_size), \
		   DataLoader(list(zip(eps_test,W_test,sig_test,C_test)), 
					  batch_size=batch_size)

"""
Model training
"""
if __name__ == "__main__":

	# Parser options
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--test_size', type=float, default=0.1)
	parser.add_argument('--hidden_dim', type=int, default=32)
	parser.add_argument('--layers', type=int, default=4)
	parser.add_argument('--learning_rate', type=float, default=5e-5)
	parser.add_argument('--weight_decay', type=float, default=1e-6)
	parser.add_argument('--gamma1', type=float, default=1.0)
	parser.add_argument('--gamma2', type=float, default=2.0)
	parser.add_argument('--gamma3', type=float, default=4.0)
	parser.add_argument('--device', type=str, default='cpu') 
	parser.add_argument('--optimizer', type=str, default='Adam')   
	parser.add_argument('--gradient_clip', type=int, default=0)
	parser.add_argument('--model_name', type=str, default='model')
	args = parser.parse_args()

	# Training/model parameters
	epochs = args.epochs 
	batch_size = args.batch_size
	test_size = args.test_size
	hidden_dim = args.hidden_dim
	layers = args.layers
	learning_rate = args.learning_rate 
	device = args.device
	weight_decay = args.weight_decay
	gamma1 = args.gamma1
	gamma2 = args.gamma2
	gamma3 = args.gamma3
	gradient_clip = args.gradient_clip
	model_name = args.model_name
	print(model_name)

	# Read data
	train_loader, test_loader = read_data(test_size, batch_size) 

	# Directory to save trained model 
	save_dir = './'

	# Model dims
	out_dim = 1
	eps_dim = 3
	
    # Initialize model
	energy_model = MLP(eps_dim, hidden_dim, out_dim, layers, torch.nn.Tanh())
	state = copy.deepcopy(energy_model.state_dict())

    # Move to device
	energy_model.to(device)
	
    # Define optimizer and scheduler
	optimizer = torch.optim.Adam(list(energy_model.parameters()),lr=learning_rate, weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
	
	# Loss function
	loss_func = torch.nn.MSELoss()  

	n_data_train = len(train_loader.dataset)
	n_data_test = len(test_loader.dataset)
	print('Number of training data pts:', n_data_train)
	print('Number of testing data pts:', n_data_test)

    # Initialize losses
	min_loss = 1e16
	loss_W = np.zeros(epochs)
	loss_dW = np.zeros(epochs)
	loss_d2W = np.zeros(epochs)
	loss_total = np.zeros(epochs)
	test_loss_W = np.zeros(epochs)
	test_loss_dW = np.zeros(epochs)
	test_loss_d2W = np.zeros(epochs)
	test_loss_total = np.zeros(epochs)
	
	def batch_jacobian(func, x, create_graph=True):
		# x in shape (Batch, Length)
		def _func_sum(x):
			return func(x).sum(dim=0)
		return torch.autograd.functional.jacobian(_func_sum, x, 
			create_graph=create_graph, vectorize=True).squeeze(0)

	def batch_hessian(func, x, create_graph=False):
		# x in shape (Batch, Length)
		def _func_sum(x):
			return func(x).sum(dim=0)
		return torch.autograd.functional.hessian(_func_sum, x, 
			create_graph=create_graph, 
			vectorize=True).diagonal(dim1=0,dim2=2).permute(2,1,0).flatten(1)

	# Define testing
	def test(model_W, test_loader):

		# Switch to evaluation mode
		energy_model.eval()

		test_loss_W_epoch = 0.0
		test_loss_dW_epoch = 0.0
		test_loss_d2W_epoch = 0.0

		for batch_idx, (eps, W, dW, d2W) in enumerate(test_loader): 

			# Move to device
			eps = eps.to(device)
			W = W.to(device)
			dW = dW.to(device)
			d2W = d2W.to(device)

			W_pred = model_W(eps)
			dW_deps_pred = batch_jacobian(model_W, eps, create_graph=True)
			d2W_deps_pred = batch_hessian(model_W, eps, create_graph=True) 

			test_loss_W_batch = loss_func(W_pred, W)
			test_loss_dW_batch = loss_func(dW_deps_pred, dW)
			test_loss_d2W_batch = loss_func(d2W_deps_pred, d2W) 
			test_loss_W_epoch += test_loss_W_batch.item()
			test_loss_dW_epoch += test_loss_dW_batch.item()
			test_loss_d2W_epoch += test_loss_d2W_batch.item()

		test_loss_W_epoch /= n_data_test
		test_loss_dW_epoch /= n_data_test
		test_loss_d2W_epoch /= n_data_test  
		test_loss_epoch = gamma1*test_loss_W_epoch + gamma2*test_loss_dW_epoch + gamma3*test_loss_d2W_epoch

		return test_loss_W_epoch, test_loss_dW_epoch, test_loss_d2W_epoch, test_loss_epoch

	# Train
	for epoch in range(epochs):
		start = time.time()

		# Switch to training mode
		energy_model.train()

		loss_W_epoch = 0.0
		loss_dW_epoch = 0.0
		loss_d2W_epoch = 0.0

		for batch_idx, (eps, W, dW, d2W) in enumerate(train_loader): 

			# Move to device
			eps = eps.to(device)
			W = W.to(device)
			dW = dW.to(device)
			d2W = d2W.to(device)

			W_pred = energy_model(eps)
			dW_deps_pred = batch_jacobian(energy_model, eps, create_graph=True)
			d2W_deps_pred = batch_hessian(energy_model, eps, create_graph=True)

			loss_W_batch = loss_func(W_pred, W)
			loss_dW_batch = loss_func(dW_deps_pred, dW)
			loss_d2W_batch = loss_func(d2W_deps_pred, d2W)
			loss_W_epoch += loss_W_batch.item()
			loss_dW_epoch += loss_dW_batch.item()
			loss_d2W_epoch += loss_d2W_batch.item() 
			loss = gamma1*loss_W_batch + gamma2*loss_dW_batch + gamma3*loss_d2W_batch

			optimizer.zero_grad() 
			loss.backward() 

			if gradient_clip:
				torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=0.1, norm_type=2)

			optimizer.step()          

        # Train loss
		loss_W_epoch /= n_data_train
		loss_dW_epoch /= n_data_train
		loss_d2W_epoch /= n_data_train
		loss_W[epoch] = loss_W_epoch
		loss_dW[epoch] = loss_dW_epoch
		loss_d2W[epoch] = loss_d2W_epoch
		loss_total[epoch] = gamma1*loss_W_epoch + gamma2*loss_dW_epoch + gamma3*loss_d2W_epoch

		# Test loss
		test_loss_epoch = test(energy_model, test_loader)
		test_loss_W[epoch] = test_loss_epoch[0]
		test_loss_dW[epoch] = test_loss_epoch[1]
		test_loss_d2W[epoch] = test_loss_epoch[2]
		test_loss_total[epoch] = test_loss_epoch[3]

		scheduler.step()
		
		if epoch % 1 == 0:
			print('Epoch:', epoch)
			print('(Training: loss_W, loss_dW, loss_d2W, loss_total):', 
				loss_W[epoch], loss_dW[epoch], loss_d2W[epoch], loss_total[epoch])
			print('(Test: loss_W, loss_dW, loss_d2W, loss_total):', 
				test_loss_W[epoch], test_loss_dW[epoch], test_loss_d2W[epoch], test_loss_total[epoch])
			print('Time: ', time.time() - start)

		# Save model if loss keeps improving
		if loss_total[epoch] < min_loss:
			min_loss = loss_total[epoch]
			state = copy.deepcopy(energy_model.state_dict())
		# if test_loss_total[epoch] < min_loss:
		# 	min_loss = test_loss_total[epoch]
		# 	state = copy.deepcopy(energy_model.state_dict())

	# Save the best state
	torch.save(state, save_dir + model_name + '.pt')

	# Save loss
	np.savetxt(save_dir + model_name + '-loss.csv', 
		np.c_[loss_W, loss_dW, loss_d2W, loss_total,
		test_loss_W, test_loss_dW, test_loss_d2W, test_loss_total,], delimiter=",", 
		header="Train loss W, Train loss dW, Train loss d2W, Train loss, \
				Test loss W, Test loss dW, Test loss d2W, Test loss")

	# Evaluate best model
	energy_model.load_state_dict(state)
	energy_model.eval()

	save_test_predictions = True
	if save_test_predictions:
		data = []
		pred = []
		for batch_idx, (eps, W, dW, d2W) in enumerate(test_loader): 

			# Move to device
			eps = eps.to(device)
			W = W.to(device)
			dW = dW.to(device)
			d2W = d2W.to(device)

			W_pred = energy_model(eps)
			dW_deps_pred = batch_jacobian(energy_model, eps)
			d2W_deps_pred = batch_hessian(energy_model, eps)

			# Save to file
			for i in range(W_pred.shape[0]):
				data.append(np.concatenate((eps[i].detach().cpu().numpy(),
											W[i].detach().cpu().numpy(),
											dW[i].detach().cpu().numpy(),
											d2W[i].detach().cpu().numpy().ravel())))
				pred.append(np.concatenate((W_pred[i].detach().cpu().numpy(),
											dW_deps_pred[i].detach().cpu().numpy(),
											d2W_deps_pred[i].detach().cpu().numpy().ravel())))

		np.savetxt(save_dir + model_name + '-prediction.csv', 
			np.c_[np.array(data),np.array(pred)],
			delimiter=",", header='e11, e22, e12, ' \
            'W_data, dW11_data, dW22_data, dW12_data, ' \
            'd2W11_data, d2W12_data, d2W13_data, d2W21_data, d2W22_data, d2W23_data, d2W31_data, d2W32_data, d2W33_data, ' \
            'W_pred, dW11_pred, dW22_pred, dW12_pred, ' \
            'd2W11_pred, d2W12_pred, d2W13_pred, d2W21_pred, d2W22_pred, d2W23_pred, d2W31_pred, d2W32_pred, d2W33_pred')