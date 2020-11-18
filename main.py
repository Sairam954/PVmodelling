from HitlessArchitecture.HitLessScalability import HitLessArchitecture
from HitlessArchWafferMap import *

import utils.readconfig as rd
import numpy as np
import sys
import traceback
import logging

DIE_NO = "die_no"
CHANNEL_SPACING_LIMIT = 0.2

config_path = 'input/inputconfig.json'
config_dict = rd.readJsonConfig(config_path)
arc_obj = HitLessArchitecture(config_dict)

ER_dies_config = arc_obj.getParameterDieFiles(arc_obj.ER_data_path)
LAMDA_R_dies_config = arc_obj.getParameterDieFiles(arc_obj.LAMDA_R_data_path)
Q_dies_config = arc_obj.getParameterDieFiles(arc_obj.Q_data_path)
FINESSE_dies_config = arc_obj.getParameterDieFiles(arc_obj.FINESSE_data_path)

die_resolution_nlamda_rslts = []
supported_nlamda = []
total_power_for_lamda = []
die_no = 0
no_of_dies = 80
verbose = 1
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
    sys.exit()

for die in range(no_of_dies):
    die_rslts = {}
    resoution_of_weight_blk = []
    die_rslts[DIE_NO] = die
    ER_config = np.loadtxt(ER_dies_config[die], delimiter=',')
    Q_config = np.genfromtxt(Q_dies_config[die], delimiter=',')
    LAMDA_R_config = np.genfromtxt(LAMDA_R_dies_config[die], delimiter=',')
    FINESSE_config = np.genfromtxt(FINESSE_dies_config[die], delimiter=',')
    no_of_blocks_x = ER_config.shape[1]
    no_of_blocks_y = ER_config.shape[0]
    no_of_blocks_x_for_preweigh = int(arc_obj.preweighingblock / arc_obj.pitch)
    no_of_blocks_x_for_vector_imprint = int(arc_obj.vectorimprintblock / arc_obj.pitch)
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

        arch, aggre_blk, pre_filter = getArchConfigParamsFromDie(arc_obj.no_of_modules, nlamda, arc_obj.no_of_rings_weighing_blk,
                                                             arch_start_block_indx, no_of_blocks_x_for_vector_imprint, ER_config,
                                                             Q_config, LAMDA_R_config, FINESSE_config)
        supported = True
        # Each Module

        for module_name in arch.keys():
            total_per_wavelength_error = 0
            array_idx = 0 # array or lamda number
            module_result = {}
            per_wavlength_power = []

            # print("Check Array ", array_idx)
            if supported == False:
                break
            lamda_power = []
            module = arch[module_name]
            # Each Array is a one lamda
            for array_name in module.keys():
                array_result = {}
                if supported == False:
                    break

                array = module[array_name]
                #weighing block mr selection
                mr_0 = array[list(array.keys())[0]]
                #calculate total loss including prefilter,weighing block,imprintblock and aggregation block
                total_insertion_loss_dB = arc_obj.getTotalLossOfArray(module_name,array_idx,pre_filter,aggre_blk,lamdar_per_channel,nlamda)
                #calculate power of a wavelength
                mr_wgt_ER = mr_0['ER']
                mr_wgt_ER_dB = 10*np.log10(mr_wgt_ER)
                pow_wavelength_at_pd_dbm = arc_obj.getPowerOfWavelength(mr_wgt_ER, total_insertion_loss_dB, per_wavelength_power_dbm)
                lamda_power.append(pow_wavelength_at_pd_dbm)

                # Resolution calculation:
                resolution = arc_obj.getResolution(mr_0)
                resoution_of_weight_blk.append(resolution)
                if verbose!=0:
                    print("Resolution", resolution)
                    print("Power of Wavelength at Pd :", pow_wavelength_at_pd_dbm)

                array_idx += 1

        pow_wavelength_at_pd_mW = []
        for wavelength_pw in lamda_power:
            pow_wavelength_at_pd_mW.append(np.power(10,(wavelength_pw/10)))
        # print("pow_wavelength_at_pd_mW ",pow_wavelength_at_pd_mW)

        total_power_for_lamda.append(sum(pow_wavelength_at_pd_mW) )
        # print("Total power at PD dBm", 10*np.log10(sum(pow_wavelength_at_pd_mW)))
        total_pw_at_pd_mW = sum(pow_wavelength_at_pd_mW)
        print("Total power at PD mW :", total_pw_at_pd_mW)
        detector_sensitivity_mW = np.power(10, (arc_obj.detector_sensitivity/10))
        # print("Detector Sensitivity in mW :", dectector_sensitivity_mW)
        # print("Detector Sensitivity in dBm", 10*np.log10(dectector_sensitivity_mW))

        if(total_pw_at_pd_mW>detector_sensitivity_mW):
            print(f'Nlamda Supported {nlamda}')
            supported_nlamda.append(nlamda)
        else:
            print(f'Nlamda Not Supported {nlamda}')
    if verbose>=0:
        print("Max supported Nlamda", max(supported_nlamda))
        print("Min Resolution ", min(resoution_of_weight_blk))
        print("Resolution ", resoution_of_weight_blk)
    die_rslts['max_supported_nlamda'] = max(supported_nlamda)
    die_rslts['resolution'] = min(resoution_of_weight_blk)
    die_resolution_nlamda_rslts.append(die_rslts)
print("Die Results:", die_resolution_nlamda_rslts)
keys = die_resolution_nlamda_rslts[0].keys()
try:
    with open('results/dievariation100by100.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(die_resolution_nlamda_rslts)
except PermissionError as inst:
    print("File is already open please close and re run")
