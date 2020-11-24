from BroadcastAndWeight.BroadcastAndWeightScalibility import BroadcastAndWeight
from BroadcastWafferMap import *


import utils.readconfig as rd
import numpy as np
import sys
import traceback
import logging
import csv

DIE_NO = "die_no"
CHANNEL_SPACING_LIMIT = 0.2

config_path = 'input/inputconfigbroadcast.json'
result_filename = 'results/dievariation200by200.csv'

config_dict = rd.readJsonConfig(config_path)
arc_obj = BroadcastAndWeight(config_dict)

ER_dies_config = arc_obj.getParameterDieFiles(arc_obj.ER_data_path)
LAMDA_R_dies_config = arc_obj.getParameterDieFiles(arc_obj.LAMDA_R_data_path)
Q_dies_config = arc_obj.getParameterDieFiles(arc_obj.Q_data_path)
FINESSE_dies_config = arc_obj.getParameterDieFiles(arc_obj.FINESSE_data_path)



die_resolution_nlamda_rslts = []
supported_nlamda = []
res_median = []
die_no = 0
no_of_dies = 80
verbose = 0
arc_obj.verbose = verbose
n_lamda_range = np.arange(64,arc_obj.nlamda)
try:
    channel_spacing = (arc_obj.FSR / (1 + arc_obj.nlamda))
    if channel_spacing < CHANNEL_SPACING_LIMIT:
        raise Exception("Channel Spacing is not feasible")
except Exception as inst:
    print(inst)
    print("The Limit of Channel Spacing is ", CHANNEL_SPACING_LIMIT)
    print("As the channel spacing for given Nlamda and FSR exiting the program")
    print("Max supported N Lamda is :", (arc_obj.FSR/CHANNEL_SPACING_LIMIT)-1)
    sys.exit()

for die in range(no_of_dies):
    print("Die Numeber", die)
    die_rslts = {}
    resoution_of_weight_blk = []
    die_rslts[DIE_NO] = die
    ER_config = np.loadtxt(ER_dies_config[die], delimiter=',')
    Q_config = np.genfromtxt(Q_dies_config[die], delimiter=',')
    LAMDA_R_config = np.genfromtxt(LAMDA_R_dies_config[die], delimiter=',')
    FINESSE_config = np.genfromtxt(FINESSE_dies_config[die], delimiter=',')

    ER_config = np.transpose(ER_config)
    Q_config = np.transpose(Q_config)
    LAMDA_R_config = np.transpose(LAMDA_R_config)
    FINESSE_config = np.transpose(FINESSE_config)

    no_of_blocks_x = ER_config.shape[1]
    no_of_blocks_y = ER_config.shape[0]
    # print(no_of_blocks_x)
    # print(no_of_blocks_y)
    no_of_blocks_x_for_preweigh = int(arc_obj.preweighingblock / arc_obj.pitch)
    arch_start_block_indx = no_of_blocks_x_for_preweigh + 1
    for nlamda in n_lamda_range:
        channel_spacing = (arc_obj.FSR / (1 + nlamda))
        per_wavelength_power = (arc_obj.p_waveguide_maop / nlamda)  # mw
        per_wavelength_power_dbm = 10*np.log10(per_wavelength_power) #dBm
        wavelengths_in_wvg = nlamda
        per_wavelength_power = (arc_obj.p_waveguide_maop/nlamda)
        if verbose != 0:
                print("Nlamda :", nlamda)
                print("spacing :", channel_spacing)
                print("Total power available mW :",arc_obj.p_waveguide_maop)
                print("Per Wavelength Power mW :",per_wavelength_power )
                print("Per Wavelength Power dBm :", per_wavelength_power_dbm)
        lamdar_per_channel = np.arange(arc_obj.RESONANCE_WAVELENGTH, (arc_obj.RESONANCE_WAVELENGTH + channel_spacing * nlamda),
                                               channel_spacing)

        arch= getArchConfigParamsFromDie(arc_obj.no_of_modules,nlamda,arch_start_block_indx,ER_config,Q_config,LAMDA_R_config,FINESSE_config)
        supported = True
        # Each Module
        for module_name in arch.keys():
            total_per_wavelength_error = 0
            array_idx = 0  # array or lamda number
            module_result = {}
            per_wavlength_power = []

            # print("Check Array ", array_idx)
            if supported == False:
                break
            lamda_power_floor = []
            lamda_power_ceiling = []
            module = arch[module_name]
            # Each Array is a one lamda
            for array_name in module.keys():
                array_result = {}
                if supported == False:
                    break

                array = module[array_name]
                # weighing block mr selection
                for mr_no in list(array.keys()):
                    mr = array[mr_no]
                    # calculate total loss including preweighing,weighing block,
                    total_insertion_loss_dB = arc_obj.getTotalLossOfArray(module_name, array_idx, arch, lamdar_per_channel, nlamda)
                    print("Total Insertion Lopss ", total_insertion_loss_dB)
