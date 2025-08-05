import matplotlib
import matplotlib.colors as clrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom
import numpy as np
import pandas as pd
import warnings
import os
import statistics

from loguru import logger

from time import asctime
from matplotlib.lines import Line2D


warnings.filterwarnings("ignore")

def listify(value, length):
	return [value] * length


def divide(dividend, divisor):
	return dividend / divisor


def color_fader(c1, c2, mix=0):
	clr1 = np.array(clrs.to_rgb(c1))
	clr2 = np.array(clrs.to_rgb(c2))
	return clrs.to_hex((1 - mix) * clr1 + mix * clr2)

def plot_results(results, folder):
	"""
	Function to plot the relevant results from the optimization problem

	Parameters:
	- opt: Optimization object from optimization.
	- structure: Structure of the community.
	- objective: Objective of the optimization problem.
	- common_fname: Common file name.

	Returns:
	- plt: Plots of the results of the optimization problem. Stored in png format.
	"""
	
	members = results['e_slc_pool'].keys()
	horizon = len(results['e_slc_pool'][list(members)[0]])
	
	ec_total = 0
	ec_geral = [0 for _ in range(horizon)]
	eg_total = 0
	eg_geral = [0 for _ in range(horizon)]
	e_sur_total = 0
	e_sur_geral = [0 for _ in range(horizon)]
	e_sup_total = 0
	e_sup_geral = [0 for _ in range(horizon)]


	for p in members:

		logger.info(f'Plotting results for member {p}...')

		# ******************************************************************************************************************
		#        INITIALIZE VARIABLES
		# ******************************************************************************************************************
		aux_bar_length = 0.5
		time_series = range(horizon)

		eCharge = sum(results['e_bc'][p])
		eDischarge = sum(results['e_bd'][p])

		logo_img = mpimg.imread('graphics/INESCTEC_logo_secnd_COLOR.png')

		# Original size of the logo
		original_height, original_width, _ = logo_img.shape
		scaling_factor = 0.08

		# Resize the logo image
		logo_img_resized = zoom(logo_img, (scaling_factor, scaling_factor, 1))

		esup = sum(results['e_sup'][p])
		e_sup_total += esup
		esur = sum(results['e_sur'][p])
		e_sur_total += esur
		epur = sum(results['e_pur_pool'][p])
		esale = sum(results['e_sale_pool'][p])


		# -- GIVEN DATA PLOT -----------------------------------------------------------------------------------------------
		E_Generation = results['e_g'][p]  # kWH
		if len(E_Generation) == len(eg_geral):
			eg_geral = [x + y for (x,y) in zip(E_Generation, eg_geral)]
		eg = sum(results['e_g'][p])
		eg_total += eg

		E_Consumed = results['e_c'][p]
		if len(E_Consumed) == len(eg_geral):
			ec_geral = [x + y for (x,y) in zip(E_Consumed, ec_geral)]
		ec = sum(results['e_c'][p])
		ec_total += ec

		# **************************************************************************************************************
		#        PLOTS
		# **************************************************************************************************************
		matplotlib.rcParams.update({'font.size': 14})
		nrows = 4
		fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(30, 3 * nrows), sharex=True)

		fig.suptitle(f'Renewable Energy Communities | {results["milp_status"]}: {round(results["obj_value"], 2)} | Prosumer {p}', fontsize=30)

		# --- Title definition -----------------------------------------------------------------------------------------
		title1 = f'Energy generated: {eg:.2f} kWh // Energy consumed: {ec:.2f} kWh'
		title2 = f'Energy charged: {eCharge:.2f} kWh // Energy discharged: {eDischarge:.2f} kWh'
		title3 = f'Energy supply: {esup: .2f} kWh // Energy surplus: {esur: .2f} kWh'
		title4 = f'Energy purchased: {epur: .2f} kWh // Energy sold: {esale: .2f} kWh'

		# **************************************************************************************************************
		#        PLOT 1 - Energy demand and PV production
		# **************************************************************************************************************
		plt.tight_layout(pad=3.0)

		vertical = 0  # Vertical relative position of the plot
		ax = axes[vertical]

		pd.DataFrame(E_Generation).plot(title=title1, kind='bar', width=aux_bar_length, align='edge', edgecolor='darkturquoise',
						   color='darkturquoise', alpha=0.7, ax=ax)
		pd.DataFrame(E_Consumed).plot(title=title1, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='goldenrod',
						color='goldenrod', alpha=0.7, ax=ax, bottom=0)

		# Create handles from scratch
		handles = list()
		handles.append(mpatches.Patch(color='darkturquoise', edgecolor='darkturquoise', alpha=0.7, label='PV production'))
		handles.append(mpatches.Patch(color='goldenrod', edgecolor='goldenrod', alpha=0.7, label='Energy consumed'))

		ax.grid(which='major', axis='x', linestyle='--')

		ax2 = ax.twinx()

		pd.DataFrame(results['l_buy'][p]).plot(kind='line', color='orangered', alpha=0.7, ax=ax2, linewidth=2.0)

		handles.append(mpatches.Patch(color='orangered', edgecolor='red', alpha=0.7, label='Price of energy'))

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		ax.xaxis.set_tick_params(labelbottom=True)
		for label in ax.xaxis.get_ticklabels()[::2]:
			label.set_visible(False)


		# Tweak plot parameters
		ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fancybox=True, shadow=True)
		ax.set_ylabel('kWh')
		ax.set_xlabel(f'Hours')

		ax2.set_ylabel(f'€/kWh')

		vertical += 1

		# **************************************************************************************************************
		#        PLOT 2 - Energy charged and discharged in the battery and SOC
		# **************************************************************************************************************
		handles = []
		ax = axes[vertical]
		ax.grid(which='major', axis='x', linestyle='--')


		E_C = pd.DataFrame(results['e_bc'][p])
		E_C.plot(title=title2, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='limegreen',
					color='limegreen', alpha=0.7, ax=ax)
		
		e_d = pd.DataFrame(results['e_bd'][p])
		e_d.plot(kind='bar', width=aux_bar_length, position=-1.0, edgecolor='firebrick',
					color='firebrick', alpha=0.7, ax=ax)

		handles = list()
		handles.append(
			mpatches.Patch(color='limegreen', edgecolor='limegreen', alpha=0.7,
							label='Energy charged into the battery'))
		handles.append(
			mpatches.Patch(color='firebrick', edgecolor='firebrick', alpha=0.7,
							label='Energy discharged from the battery'))
		
		ax2 = ax.twinx()

		soc = pd.DataFrame(results['soc'][p])
		soc.plot(kind='line', color='orangered', alpha=0.7, ax=ax2, linewidth=2.0)

		for i in time_series:
			ax2.hlines(y=20, xmin=i, xmax=i + 1, linewidth=2.0, color='red', linestyle='--')
			ax2.hlines(y=90, xmin=i, xmax=i + 1, linewidth=2.0, color='red', linestyle='--')

		handles.append(mpatches.Patch(color='orangered', edgecolor='red', alpha=0.7, label='SOC'))
		handles.append(Line2D([0], [90], color='red', linestyle='dashed', lw=2.0, label=f'Maximum SOC'))
		handles.append(Line2D([0], [20], color='red', linestyle='dashed', lw=2.0, label=f'Minimum SOC'))

		ax.set_ylabel(f'kWh')
		ax.set_xlabel(f'Hours')
		ax2.set_ylabel('%')
		ax.grid(which='major', axis='x', linestyle='--')

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		ax.xaxis.set_tick_params(labelbottom=False)
		for label in ax.xaxis.get_ticklabels()[::2]:
			label.set_visible(False)

		# Tweak plot parameters
		ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fancybox=True, shadow=True)
		ax.set_ylabel('kWh')

		vertical += 1

		# **************************************************************************************************************
		#        PLOT 3 - Energy supply and surplus
		# **************************************************************************************************************
		handles = []
		ax = axes[vertical]
		ax.grid(which='major', axis='x', linestyle='--')

		eSUP = results[f'e_sup'][p]
		if len(eSUP) == len(e_sup_geral):
			e_sup_geral = [x + y for (x,y) in zip(eSUP, e_sup_geral)]
		eSUR = results[f'e_sur'][p]
		if len(eSUR) == len(e_sur_geral):
			e_sur_geral = [x + y for (x,y) in zip(eSUR, e_sur_geral)]

		pd.DataFrame(eSUP).plot(title=title3, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='firebrick',
				  color='firebrick', alpha=0.7, ax=ax, bottom=0)
		pd.DataFrame(eSUR).plot(title=title3, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='limegreen',
				  color='limegreen', alpha=0.7, ax=ax, bottom=0)

		handles = list()
		handles.append(mpatches.Patch(color='firebrick', edgecolor='firebrick', alpha=0.7,
									  label='Energy supplied'))
		handles.append(mpatches.Patch(color='limegreen', edgecolor='limegreen', alpha=0.7,
									  label='Energy surplus'))

		ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fancybox=True, shadow=True)

		ax.set_ylabel(f'E demand [kWh]')
		ax.set_xlabel(f'Hours')
		ax.grid(which='major', axis='x', linestyle='--')

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		ax.xaxis.set_tick_params(labelbottom=True)
		for label in ax.xaxis.get_ticklabels()[::2]:
			label.set_visible(False)

		vertical += 1

		# **************************************************************************************************************
		#        PLOT 5 - Energy sale and purchased in the community
		# **************************************************************************************************************
		handles = []
		ax = axes[vertical]
		ax.grid(which='major', axis='x', linestyle='--')

		eSALE = results['e_sale_pool'][p]
		ePUR = results['e_pur_pool'][p]

		pd.DataFrame(ePUR).plot(title=title4, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='firebrick',
				  color='firebrick', alpha=0.7, ax=ax, bottom=0)
		pd.DataFrame(eSALE).plot(title=title4, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='limegreen',
				  color='limegreen', alpha=0.7, ax=ax, bottom=0)

		handles = list()
		handles.append(mpatches.Patch(color='firebrick', edgecolor='firebrick', alpha=0.7,
									  label='Energy purchased'))
		handles.append(mpatches.Patch(color='limegreen', edgecolor='limegreen', alpha=0.7,
									  label='Energy sold'))

		ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fancybox=True, shadow=True)

		ax.set_ylabel(f'kWh')
		ax.set_xlabel(f'Hours')
		ax.grid(which='major', axis='x', linestyle='--')

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		ax.xaxis.set_tick_params(labelbottom=True)
		for label in ax.xaxis.get_ticklabels()[::2]:
			label.set_visible(False)

		plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=True, rotation=0)
		plt.xlim(0, horizon)
		for ax in fig.axes:
			plt.setp(ax.get_xticklabels(), visible=True)
			plt.setp(ax.get_xticklabels()[::4], visible=True)

		vertical += 1

		plt.figimage(logo_img_resized, xo=50, yo=1100)

		plt.savefig(rf'./{folder}/Prosumer{p}.png')

	logger.info('Plotting REC general view...')

	matplotlib.rcParams.update({'font.size': 14})
	nrows = 3
	fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(30, 8 * nrows), sharex=True)

	fig.suptitle(
		f'Renewable Energy Communities | {results["milp_status"]}: {round(results["obj_value"], 2)} | General View',
		fontsize=30)

	# --- Title definition -----------------------------------------------------------------------------------------
	title1 = f'Energy generated: {eg_total:.2f} kWh // Energy consumed: {ec_total:.2f} kWh'
	title2 = f'Energy supply: {e_sup_total: .2f} kWh // Energy surplus: {e_sur_total: .2f} kWh'
	title3 = f'Resilience: {statistics.mean(results["resilience"]):.2f} hours, in each hour'

	# **************************************************************************************************************
	#        PLOT 1 - Energy demand and PV production
	# **************************************************************************************************************
	plt.tight_layout(pad=3.0)

	vertical = 0  # Vertical relative position of the plot
	ax = axes[vertical]

	pd.DataFrame(eg_geral).plot(title=title1, kind='bar', width=aux_bar_length, align='edge', edgecolor='darkturquoise',
					  color='darkturquoise', alpha=0.7, ax=ax)
	pd.DataFrame(ec_geral).plot(title=title1, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='goldenrod',
					color='goldenrod', alpha=0.7, ax=ax, bottom=0)

	# Create handles from scratch
	handles = list()
	handles.append(mpatches.Patch(color='darkturquoise', edgecolor='darkturquoise', alpha=0.7, label='PV production'))
	handles.append(mpatches.Patch(color='goldenrod', edgecolor='goldenrod', alpha=0.7, label='Energy consumed'))

	ax.grid(which='major', axis='x', linestyle='--')

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.xaxis.set_tick_params(labelbottom=True)
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_visible(False)

	# Tweak plot parameters
	ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fancybox=True, shadow=True)
	ax.set_ylabel('kWh')
	ax.set_xlabel(f'Hours')

	vertical += 1

	# **************************************************************************************************************
	#        PLOT 2 - Energy charged and discharged in the battery and SOC
	# **************************************************************************************************************
	handles = []
	ax = axes[vertical]
	ax.grid(which='major', axis='x', linestyle='--')

	pd.DataFrame(e_sup_geral).plot(title=title2, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='firebrick',
			  color='firebrick', alpha=0.7, ax=ax, bottom=0)
	pd.DataFrame(e_sur_geral).plot(title=title2, kind='bar', width=aux_bar_length, position=-1.0, edgecolor='limegreen',
			   color='limegreen', alpha=0.7, ax=ax, bottom=0)

	handles = list()
	handles.append(mpatches.Patch(color='firebrick', edgecolor='firebrick', alpha=0.7,
								  label='Energy supply'))
	handles.append(mpatches.Patch(color='limegreen', edgecolor='limegreen', alpha=0.7,
								  label='Energy surplus'))

	ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fancybox=True, shadow=True)

	ax.set_ylabel(f'kWh')
	ax.set_xlabel(f'Hours')
	ax.grid(which='major', axis='x', linestyle='--')

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.xaxis.set_tick_params(labelbottom=True)
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_visible(False)

	vertical += 1

	# **************************************************************************************************************
	#        PLOT 3 - Resilience view
	# **************************************************************************************************************
	handles = []
	ax = axes[vertical]
	ax.grid(which='major', axis='x', linestyle='--')

	res_values = pd.Series(results['resilience'])
	avg_value = res_values.mean()
	diff_values = res_values - avg_value

	# Plot the average line
	ax.axhline(y=avg_value, color='blue', linestyle='--', label='Average')

	# Plot bars centered on the average line
	ax.bar(x=range(len(diff_values)), height=diff_values, bottom=avg_value, 
		   width=aux_bar_length, color='darkorange', alpha=0.7, edgecolor='darkorange')

	handles = list()
	handles.append(mpatches.Patch(color='darkorange', edgecolor='darkorange', alpha=0.7, label='Deviation from HACP'))
	handles.append(mpatches.Patch(color='blue', label='HACP'))

	ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fancybox=True, shadow=True)

	ax.set_ylabel(f'HACP (hours)')
	ax.set_xlabel(f'Hours')
	ax.grid(which='major', axis='x', linestyle='--')

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.xaxis.set_tick_params(labelbottom=True)
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_visible(False)



	plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=True, rotation=0)
	plt.xlim(0, horizon)
	for ax in fig.axes:
		plt.setp(ax.get_xticklabels(), visible=True)
		plt.setp(ax.get_xticklabels()[::4], visible=True)

	plt.figimage(logo_img_resized, xo=50, yo=2300)

	plt.savefig(rf'./{folder}/GeneralView.png')
