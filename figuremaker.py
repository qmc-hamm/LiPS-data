#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 15:31:58 2025

@author: tawfiqurrakib
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import pylustrator

#pylustrator.start()

csv_filename = 'LPS_barrier_integrated.csv'
df = pd.read_csv(csv_filename)
energy_qmc_hop1 = df[(df['barrier']=='barrier') & (df['method']=='DMC')]['energy_ev_per_cell']
energy_qmc_hop2 = df[(df['barrier']=='barrier2') & (df['method']=='DMC')]['energy_ev_per_cell']
energy_qmc_hop3 = df[(df['barrier']=='barrier3') & (df['method']=='DMC')]['energy_ev_per_cell']
error_qmc_hop1 = df[(df['barrier']=='barrier') & (df['method']=='DMC')]['error_ev']
error_qmc_hop2 = df[(df['barrier']=='barrier2') & (df['method']=='DMC')]['error_ev']
error_qmc_hop3 = df[(df['barrier']=='barrier3') & (df['method']=='DMC')]['error_ev']

datasize = ['100', '1000', '24000']


fig = plt.figure(figsize=(10.5, 3))
plt.rcParams.update({'font.size': 12, 'font.family': 'Helvetica'})
color = [ 'b', 'r', 'g']
# Define custom GridSpec layout
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

csv_filename = 'barrier_energy_hop2.csv'
df = pd.read_csv(csv_filename)
energy_DFT = df['DFT']
avg_DFT2 = np.mean(energy_DFT)
shift_DFT = np.min(energy_DFT)
x = np.arange(len(energy_DFT))
ax1 = fig.add_subplot(gs[0, 1])
shift_strata = np.zeros((3,3))
for i in range (len(datasize)):
    for j in range (3):
        energy = df['datasize'+datasize[i]+'_MLIP'+str(j+1)]
        shift_strata[i,j] = np.min(energy)
        if j ==0:
            ax1.plot(x, energy-np.min(energy), color=color[i], marker='o', linestyle='-', linewidth=2, markersize = 5, label=f'Stratified {datasize[i]}')
        else:
            ax1.plot(x, energy-np.min(energy), color=color[i], marker='o', linestyle='-', linewidth=2, markersize = 5)
        print(datasize[i], np.max(energy)-energy[0])
        print(datasize[i], np.max(energy)-energy[8])
shift_heating = np.zeros(3)        
for j in range(3):
    energy = df['datasize_heating'+str(j+1)]
    shift_heating[j] = np.min(energy)
    if j ==0:
        ax1.plot(x, energy-np.min(energy), color='m', marker='o', linestyle='-', linewidth=2, markersize = 5, label='Heating 2000')
    else:
        ax1.plot(x, energy-np.min(energy), color='m', marker='o', linestyle='-', linewidth=2, markersize = 5)
    print(np.max(energy)-energy[0])
    print(np.max(energy)-energy[8])
shift_qmc = np.min(energy_qmc_hop2)
ax1.errorbar(x, energy_qmc_hop2-shift_qmc , yerr=error_qmc_hop2, fmt='o', color='c', ecolor='c', elinewidth=2, markersize=5, capsize=3, label='DMC')
ax1.plot(x, energy_DFT - np.min(energy_DFT), color='k', marker='o', linestyle='-', linewidth=2, markersize=5, label='DFT')
print(np.max(energy_DFT)-energy_DFT[0])
print(np.max(energy_DFT)-energy_DFT[8])
ax1.set_ylim([-0.05, 0.39])
ax1.set_xlabel('Reaction coordinates')
ax1.set_ylabel('')
ax1.tick_params(axis='y', which='both', labelleft=False)
#ax1.legend()
plt.tight_layout()


csv_filename = 'barrier_energy_hop1.csv'
df = pd.read_csv(csv_filename)
energy_DFT = df['DFT']
avg_DFT1 = np.mean(energy_DFT)
shift = avg_DFT2-avg_DFT1
x = np.arange(len(energy_DFT))
ax2 = fig.add_subplot(gs[0, 0])
for i in range (len(datasize)):
    for j in range (3):
        energy = df['datasize'+datasize[i]+'_MLIP'+str(j+1)]
        if j ==0:
            ax2.plot(x, energy-np.min(energy)+shift, color=color[i], marker='o', linestyle='-', linewidth=2, markersize = 5, label=f'Stratified {datasize[i]}')
        else:
            ax2.plot(x, energy-np.min(energy)+shift, color=color[i], marker='o', linestyle='-', linewidth=2, markersize = 5)
        print(datasize[i], np.max(energy)-energy[8]) 
        print(datasize[i], np.max(energy)-energy[0]) 
for j in range(3):
    energy = df['datasize_heating'+str(j+1)]
    if j ==0:
        ax2.plot(x, energy-np.min(energy)+shift, color='m', marker='o', linestyle='-', linewidth=2, markersize = 5, label='Heating 2000')
    else:
        ax2.plot(x, energy-np.min(energy)+shift, color='m', marker='o', linestyle='-', linewidth=2, markersize = 5)
    print(np.max(energy)-energy[8])
    print(np.max(energy)-energy[0])
ax2.plot(x, energy_DFT - np.min(energy_DFT)+shift, color='k', marker='o', linestyle='-', linewidth=2, markersize=5, label='DFT')
ax2.errorbar(x, energy_qmc_hop1-np.min(energy_qmc_hop1)+shift , yerr=error_qmc_hop1, fmt='o', color='c', ecolor='c', elinewidth=2, markersize=5, capsize=3, label='DMC')
print(np.max(energy_DFT)-energy_DFT[8])
print(np.max(energy_DFT)-energy_DFT[0])
ax2.set_ylim([-0.02, 0.39])
ax2.set_xlabel('Reaction coordinates')
ax2.set_ylabel('Energy (eV)')
#ax2.legend()
plt.tight_layout()

csv_filename = 'barrier_energy_hop3.csv'
df = pd.read_csv(csv_filename)
energy_DFT = df['DFT']
avg_DFT3 = np.mean(energy_DFT)
shift = avg_DFT2-avg_DFT3
x = np.arange(len(energy_DFT))
ax3 = fig.add_subplot(gs[0, 2])
for i in range (len(datasize)):
    for j in range (3):
        energy = df['datasize'+datasize[i]+'_MLIP'+str(j+1)]
        if j ==0:
            ax3.plot(x, energy-np.min(energy)+shift, color=color[i], marker='o', linestyle='-', linewidth=2, markersize = 5, label=f'Stratified {datasize[i]}')
        else:
            ax3.plot(x, energy-np.min(energy)+shift, color=color[i], marker='o', linestyle='-', linewidth=2, markersize = 5)
        print(datasize[i], np.max(energy)-energy[0])
        print(datasize[i], np.max(energy)-energy[8])
for j in range(3):
    energy = df['datasize_heating'+str(j+1)]
    if j ==0:
        ax3.plot(x, energy-np.min(energy)+shift, color='m', marker='o', linestyle='-', linewidth=2, markersize = 5, label='Heating 2000')
    else:
        ax3.plot(x, energy-np.min(energy)+shift, color='m', marker='o', linestyle='-', linewidth=2, markersize = 5)
    print(np.max(energy)-energy[0])
    print(np.max(energy)-energy[8])
ax3.plot(x, energy_DFT - np.min(energy_DFT)+shift, color='k', marker='o', linestyle='-', linewidth=2, markersize=5, label='DFT')
ax3.errorbar(x, energy_qmc_hop3- np.min(energy_qmc_hop3)+shift , yerr=error_qmc_hop3, fmt='o', color='c', ecolor='c', elinewidth=2, markersize=5, capsize=3, label='DMC')
print(np.max(energy_DFT)-energy_DFT[0])
print(np.max(energy_DFT)-energy_DFT[8])
ax3.set_ylim([-0.02, 0.39])
ax3.set_xlabel('Reaction coordinates')
ax3.set_ylabel('')
ax3.tick_params(axis='y', which='both', labelleft=False)
ax3.legend()
plt.tight_layout()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(26.670000/2.54, 7.780000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.386457, 0.208571, 0.287314, 0.687004])
plt.figure(1).axes[0].text(0.0054, 1.0524, '(b) Hop 2', transform=plt.figure(1).axes[0].transAxes, )  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[1].set(position=[0.07737, 0.2086, 0.2873, 0.687])
plt.figure(1).axes[1].text(-0.0068, 1.0524, '(a) Hop 1', transform=plt.figure(1).axes[1].transAxes, )  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[2].legend(loc=(0.4235, 0.3653), borderpad=0.25, labelspacing=0.3, handlelength=1.8)
plt.figure(1).axes[2].set(position=[0.6955, 0.2086, 0.2873, 0.687])
plt.figure(1).axes[2].text(-0.0061, 1.0524, '(c) Hop 3', transform=plt.figure(1).axes[2].transAxes, )  # id=plt.figure(1).axes[2].texts[0].new
#% end: automatic generated code from pylustrator

#% end: automatic generated code from pylustrator
figure_name = 'fig_with_qmc.pdf'
plt.savefig(figure_name, dpi=600)
plt.show()

