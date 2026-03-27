import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from sklearn.metrics import r2_score

if __name__ == "__main__":

	data_dir = './'
	fig_dir = './figures/'
	if not os.path.exists(fig_dir):
		os.makedirs(fig_dir)

	# ========================================================================
	# Loss figures
	# ========================================================================
	loss_figures = [plt.figure(figsize=(3, 2.7)) for _ in range(4)]
	loss_axes = [fig.add_subplot(111) for fig in loss_figures]

	# Define loss column indices
	loss_columns = {
		'W': 4,
		'dW': 5,
		'd2W': 6,
		'total': 7,
	}

	model_name = 'model'

	# Read data
	data = pd.read_csv(data_dir + model_name + '-loss.csv').to_numpy()
	epochs = np.arange(len(data[:, loss_columns['W']]))
	
	# Append to plot_data for plotting
	plot_data = []
	plot_data.append({
		'epochs': epochs,
		'W': data[:, loss_columns['W']],
		'dW': data[:, loss_columns['dW']],
		'd2W': data[:, loss_columns['d2W']],
		'total': data[:, loss_columns['total']],
	})

	# Plot data
	for i, loss_type in enumerate(['W', 'dW', 'd2W', 'total']):
		for data in plot_data:
			loss_axes[i].plot(
				data['epochs'], data[loss_type],
				c='r', ls='-', label=model_name
			)
		loss_axes[i].set_yscale("log")
		loss_axes[i].set_xlabel('Epochs', fontsize=14)
		loss_axes[i].set_ylabel(f'{loss_type} loss', fontsize=14)
		loss_axes[i].tick_params(axis='both', which='major', labelsize=14)
		loss_axes[i].legend()
		loss_figures[i].tight_layout()
		loss_figures[i].savefig(fig_dir + f'{model_name}-{loss_type}-loss.png', format='png')

	#========================================================================
	# R2 Figures
	#========================================================================
	r2_figures = [plt.figure(figsize=(3, 2.7)) for _ in range(12)]
	r2_axes = [fig.add_subplot(111) for fig in r2_figures]
	r2_scores = []

	# Read the predicted values
	data = pd.read_csv(data_dir + model_name + '-prediction.csv')
	sig11_pred = data[' sig11']
	sig22_pred = data[' sig22']
	sig12_pred = data[' sig12']
	C11_pred = data[' C11']
	C12_pred = data[' C12']
	C13_pred = data[' C13']
	C21_pred = data[' C21']
	C22_pred = data[' C22']
	C23_pred = data[' C23']
	C31_pred = data[' C31']
	C32_pred = data[' C32']
	C33_pred = data[' C33']

	# Read the data
	file = h5py.File(data_dir + '/data.h5', 'r')
	eps = file['eps'][:]
	sig11_data = file['sig'][:,0]
	sig22_data = file['sig'][:,1]
	sig12_data = file['sig'][:,2]
	C11_data = file['C'][:,0]
	C12_data = file['C'][:,1]
	C13_data = file['C'][:,2]
	C21_data = file['C'][:,3]
	C22_data = file['C'][:,4]
	C23_data = file['C'][:,5]
	C31_data = file['C'][:,6]
	C32_data = file['C'][:,7]
	C33_data = file['C'][:,8]

	# Calculate r2 scores
	sig11_r2 = r2_score(sig11_data, sig11_pred)	
	sig22_r2 = r2_score(sig22_data, sig22_pred)
	sig12_r2 = r2_score(sig12_data, sig12_pred)
	C11_r2 = r2_score(C11_data, C11_pred)
	C12_r2 = r2_score(C12_data, C12_pred)
	C13_r2 = r2_score(C13_data, C13_pred)
	C21_r2 = r2_score(C21_data, C21_pred)
	C22_r2 = r2_score(C22_data, C22_pred)
	C23_r2 = r2_score(C23_data, C23_pred)
	C31_r2 = r2_score(C31_data, C31_pred)
	C32_r2 = r2_score(C32_data, C32_pred)
	C33_r2 = r2_score(C33_data, C33_pred)

	# Plot
	r2_axes[0].scatter(sig11_data, sig11_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {sig11_r2:.3f}')
	r2_axes[0].plot([sig11_data.min(), sig11_data.max()],
					[sig11_data.min(), sig11_data.max()], '--', c='r')

	r2_axes[1].scatter(sig22_data, sig22_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {sig22_r2:.3f}')
	r2_axes[1].plot([sig22_data.min(), sig22_data.max()],
					[sig22_data.min(), sig22_data.max()], '--', c='r')

	r2_axes[2].scatter(sig12_data, sig12_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {sig12_r2:.3f}')
	r2_axes[2].plot([sig12_data.min(), sig12_data.max()],
					[sig12_data.min(), sig12_data.max()], '--', c='r')
	
	r2_axes[3].scatter(C11_data, C11_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C11_r2:.3f}')
	r2_axes[3].plot([C11_data.min(), C11_data.max()],
					[C11_data.min(), C11_data.max()], '--', c='r')	
	
	r2_axes[4].scatter(C12_data, C12_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C12_r2:.3f}')
	r2_axes[4].plot([C12_data.min(), C12_data.max()],
					[C12_data.min(), C12_data.max()], '--', c='r')	\
					
	r2_axes[5].scatter(C13_data, C13_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C13_r2:.3f}')
	r2_axes[5].plot([C13_data.min(), C13_data.max()],
					[C13_data.min(), C13_data.max()], '--', c='r')	
	
	r2_axes[6].scatter(C21_data, C21_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C21_r2:.3f}')
	r2_axes[6].plot([C21_data.min(), C21_data.max()],
					[C21_data.min(), C21_data.max()], '--', c='r')	

	r2_axes[7].scatter(C22_data, C22_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C22_r2:.3f}')
	r2_axes[7].plot([C22_data.min(), C22_data.max()],
					[C22_data.min(), C22_data.max()], '--', c='r')	
	
	r2_axes[8].scatter(C23_data, C23_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C23_r2:.3f}')
	r2_axes[8].plot([C23_data.min(), C23_data.max()],
					[C23_data.min(), C23_data.max()], '--', c='r')
	
	r2_axes[9].scatter(C31_data, C31_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C31_r2:.3f}')
	r2_axes[9].plot([C31_data.min(), C31_data.max()],
					[C31_data.min(), C31_data.max()], '--', c='r')	
	
	r2_axes[10].scatter(C32_data, C32_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C32_r2:.3f}')
	r2_axes[10].plot([C32_data.min(), C32_data.max()],
					[C32_data.min(), C32_data.max()], '--', c='r')	
	
	r2_axes[11].scatter(C33_data, C33_pred, marker='o',
		c='r', alpha=0.5, s=10, label=f'R2 Score: {C33_r2:.3f}')
	r2_axes[11].plot([C33_data.min(), C33_data.max()],
					[C33_data.min(), C33_data.max()], '--', c='r')

	# Configure plots
	r2_axes[0].set_xlabel(r'Actual $\sigma_{11}$', fontsize=14)
	r2_axes[0].set_ylabel(r'Predicted $\sigma_{11}$', fontsize=14)
	r2_axes[0].set_ylim(r2_axes[0].get_xlim())
	r2_axes[0].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[0].legend(fontsize=8)
	r2_figures[0].tight_layout()
	r2_figures[0].savefig(fig_dir + f'r2-score-sig11.png', format='png')

	r2_axes[1].set_xlabel(r'Actual $\sigma_{22}$', fontsize=14)			
	r2_axes[1].set_ylabel(r'Predicted $\sigma_{22}$', fontsize=14)
	r2_axes[1].set_ylim(r2_axes[1].get_xlim())
	r2_axes[1].tick_params(axis='both', which='major', labelsize=	14)
	r2_axes[1].legend(fontsize=8)
	r2_figures[1].tight_layout()
	r2_figures[1].savefig(fig_dir + f'r2-score-sig22.png', format='png')

	r2_axes[2].set_xlabel(r'Actual $\sigma_{12}$', fontsize=14)			
	r2_axes[2].set_ylabel(r'Predicted $\sigma_{12}$', fontsize=14)
	r2_axes[2].set_ylim(r2_axes[2].get_xlim())	
	r2_axes[2].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[2].legend(fontsize=8)
	r2_figures[2].tight_layout()
	r2_figures[2].savefig(fig_dir + f'r2-score-sig12.png', format='png')

	# ... similarly for C components ...
	r2_axes[3].set_xlabel(r'Actual $C_{11}$', fontsize=14)			
	r2_axes[3].set_ylabel(r'Predicted $C_{11}$', fontsize=14)
	r2_axes[3].set_ylim(r2_axes[3].get_xlim())	
	r2_axes[3].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[3].legend(fontsize=8)
	r2_figures[3].tight_layout()
	r2_figures[3].savefig(fig_dir + f'r2-score-C11.png', format='png')

	r2_axes[4].set_xlabel(r'Actual $C_{12}$', fontsize=14)			
	r2_axes[4].set_ylabel(r'Predicted $C_{12}$', fontsize=14)
	r2_axes[4].set_ylim(r2_axes[4].get_xlim())	
	r2_axes[4].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[4].legend(fontsize=8)
	r2_figures[4].tight_layout()
	r2_figures[4].savefig(fig_dir + f'r2-score-C12.png', format='png')

	r2_axes[5].set_xlabel(r'Actual $C_{13}$', fontsize=14)			
	r2_axes[5].set_ylabel(r'Predicted $C_{13}$', fontsize=14)
	r2_axes[5].set_ylim(r2_axes[5].get_xlim())	
	r2_axes[5].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[5].legend(fontsize=8)
	r2_figures[5].tight_layout()
	r2_figures[5].savefig(fig_dir + f'r2-score-C13.png', format='png')

	r2_axes[6].set_xlabel(r'Actual $C_{21}$', fontsize=14)			
	r2_axes[6].set_ylabel(r'Predicted $C_{21}$', fontsize=14)
	r2_axes[6].set_ylim(r2_axes[6].get_xlim())	
	r2_axes[6].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[6].legend(fontsize=8)
	r2_figures[6].tight_layout()
	r2_figures[6].savefig(fig_dir + f'r2-score-C21.png', format='png')

	r2_axes[7].set_xlabel(r'Actual $C_{22}$', fontsize=14)			
	r2_axes[7].set_ylabel(r'Predicted $C_{22}$', fontsize=14)
	r2_axes[7].set_ylim(r2_axes[7].get_xlim())	
	r2_axes[7].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[7].legend(fontsize=8)
	r2_figures[7].tight_layout()
	r2_figures[7].savefig(fig_dir + f'r2-score-C22.png', format='png')

	r2_axes[8].set_xlabel(r'Actual $C_{23}$', fontsize=14)			
	r2_axes[8].set_ylabel(r'Predicted $C_{23}$', fontsize=14)
	r2_axes[8].set_ylim(r2_axes[8].get_xlim())	
	r2_axes[8].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[8].legend(fontsize=8)
	r2_figures[8].tight_layout()
	r2_figures[8].savefig(fig_dir + f'r2-score-C23.png', format='png')

	r2_axes[9].set_xlabel(r'Actual $C_{31}$', fontsize=14)			
	r2_axes[9].set_ylabel(r'Predicted $C_{31}$', fontsize=14)
	r2_axes[9].set_ylim(r2_axes[9].get_xlim())	
	r2_axes[9].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[9].legend(fontsize=8)
	r2_figures[9].tight_layout()
	r2_figures[9].savefig(fig_dir + f'r2-score-C31.png', format='png')

	r2_axes[10].set_xlabel(r'Actual $C_{32}$', fontsize=14)			
	r2_axes[10].set_ylabel(r'Predicted $C_{32}$', fontsize=14)
	r2_axes[10].set_ylim(r2_axes[10].get_xlim())	
	r2_axes[10].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[10].legend(fontsize=8)
	r2_figures[10].tight_layout()
	r2_figures[10].savefig(fig_dir + f'r2-score-C32.png', format='png')	

	r2_axes[11].set_xlabel(r'Actual $C_{33}$', fontsize=14)			
	r2_axes[11].set_ylabel(r'Predicted $C_{33}$', fontsize=14)
	r2_axes[11].set_ylim(r2_axes[11].get_xlim())	
	r2_axes[11].tick_params(axis='both', which='major', labelsize=14)
	r2_axes[11].legend(fontsize=8)
	r2_figures[11].tight_layout()
	r2_figures[11].savefig(fig_dir + f'r2-score-C33.png', format='png')
