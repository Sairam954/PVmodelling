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
result_filename = 'results/optical/dievariation100by100.csv'

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
no_of_dies = arc_obj.no_of_dies
verbose = 1
arc_obj.verbose = verbose
n_lamda_range = np.arange(4,arc_obj.nlamda)
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
    print(no_of_blocks_x)
    print(no_of_blocks_y)
    no_of_blocks_x_for_mux = int(arc_obj.premuxblock / arc_obj.pitch)
    no_of_blocks_x_for_imprint = int(arc_obj.vectorimprintblock/arc_obj.pitch)
    no_of_blocks_x_for_preweigh = int(arc_obj.preweighingblock/arc_obj.pitch)
    arch_start_block_indx = no_of_blocks_x_for_preweigh + 1

    for nlamda in n_lamda_range:
        resoution_of_weight_blk = []

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

        arch, demul_mr, mul_mr = getArchConfigParamsFromDie(arc_obj.no_of_modules,nlamda,arch_start_block_indx,no_of_blocks_x_for_imprint,no_of_blocks_x_for_preweigh,ER_config,Q_config,LAMDA_R_config,FINESSE_config)


        # Each Module
        for module_name in arch.keys():
            total_per_wavelength_error = 0
            array_idx = 0  # array or lamda number
            module_result = {}
            per_wavlength_power = []

            lamda_power_floor = []
            lamda_power_ceiling = []
            module = arch[module_name]



            # Each Array is a one lamda
            for array_name in module.keys():
                array_result = {}
                array = module[array_name]
                # weighing block mr selection
                wavelength_idx = 0
                # find the optimal channel gap between mr to avoid the cross talk
                # find the worst case Q
                parameter = 'Q'
                minQ, maxQ = arc_obj.findMinAndMaxConfig(array,'Q')
                # finding the dynamic range
                #considering the middle mr for analysis
                array_keys  = list(array.keys())
                middle_mr_idx = int(len(array_keys)/2)
                middle_mr_key = array_keys[middle_mr_idx]
                middle_mr = array[middle_mr_key]
                # operating wavelength
                carrier_wavelength1 = lamdar_per_channel[middle_mr_idx]
                carrier_wavelength2 = lamdar_per_channel[middle_mr_idx-1]
                operating_wavelength = carrier_wavelength1
                FWHM = operating_wavelength / minQ
                period = FWHM / 4
                FWHM_array = np.arange(0, channel_spacing, period)
                mr_trans1 = []
                mr_trans2 = []
                diff_trans = []
                delta_lamda = {}
                for shift in FWHM_array:
                    operating_wavelength -= shift
                    delta1 = operating_wavelength-carrier_wavelength1
                    delta2 = operating_wavelength-carrier_wavelength2
                    t1 = arc_obj.getTransmission(middle_mr, operating_wavelength, delta1)
                    t2 = arc_obj.getTransmission(middle_mr, operating_wavelength, delta2)
                    delta_lamda[delta1] = t1-t2
                    mr_trans1.append(t1)
                    mr_trans2.append(t2)
                    diff_trans.append(t1-t2)
                delta_trns_values = np.array(list(delta_lamda.values()))
                # print(delta_trns_values)
                max_negitive_idx = np.where(delta_trns_values < 0, delta_trns_values, -np.inf).argmax()
                min_positive_idx = np.where(delta_trns_values > 0, delta_trns_values, np.inf).argmin()
                # print("Max Negative", delta_trns_values[max_negitive_idx])
                # print("Min Positive", delta_trns_values[min_positive_idx])
                optimal_delta_csp = 0

                for key in list(delta_lamda.keys()):

                    if delta_lamda[key] == delta_trns_values[max_negitive_idx]:
                        optimal_delta_csp = key
                        print("Optimal Delta", optimal_delta_csp)

                        break
                # print("Optimal Delta CSP", optimal_delta_csp)
                mr_transmission_at_optimal = arc_obj.getTransmission(middle_mr, carrier_wavelength1, optimal_delta_csp)
                mr_transmission_at_zero = arc_obj.getTransmission(middle_mr, carrier_wavelength1, 0)
                mr_transmission_at_optimal = -10 * np.log10(abs(mr_transmission_at_optimal))
                mr_transmission_at_zero = -10 * np.log10(abs(mr_transmission_at_zero))

                ER = abs(mr_transmission_at_optimal - mr_transmission_at_zero)
                # print("ER", ER)
                for mr_no in list(array.keys()):

                    mr = array[mr_no]

                    # calculate total loss including preweighing,weighing block,
                    total_insertion_loss_dB = arc_obj.getTotalLossOfArray(module_name, wavelength_idx, array,demul_mr, mul_mr, lamdar_per_channel, nlamda)
                    # calculate power of a wavelength
                    mr_wgt_ER = mr['ER']
                    mr_wgt_ER_dB = 10 * np.log10(mr_wgt_ER)
                    pow_wavelength_at_pd_dbm_floor, pow_wavelength_at_pd_dbm_ceiling = arc_obj.getPowerOfWavelengthFloorandCeiling(
                        mr_wgt_ER, total_insertion_loss_dB, per_wavelength_power_dbm)
                    lamda_power_floor.append(pow_wavelength_at_pd_dbm_floor)
                    lamda_power_ceiling.append(pow_wavelength_at_pd_dbm_ceiling)
                    # print("Lamda Power Floor", lamda_power_floor)
                    # print("Lamda Power Ceiling", lamda_power_ceiling)
                    # print("Total Insertion Loss ", total_insertion_loss_dB)
                    # Resolution calculation:
                    r_sources = []
                    thermal_crosstalk_radius = 1000
                    max_no_of_sources_each_direction = int(thermal_crosstalk_radius/arc_obj.pitch)
                    sources_on_left = max_no_of_sources_each_direction if wavelength_idx>=max_no_of_sources_each_direction else wavelength_idx
                    sources_on_right = max_no_of_sources_each_direction if (nlamda-wavelength_idx)>max_no_of_sources_each_direction else (nlamda-wavelength_idx-1)
                    for r in range(1,sources_on_left+1):
                        r_sources.append(arc_obj.pitch*r*1e-6)
                    for r in range(1, sources_on_right + 1):
                        r_sources.append(arc_obj.pitch * r*1e-6)
                    print("Nlamda ", nlamda)
                    print("Wavelength Idx", wavelength_idx)
                    print("sources_on_left ", sources_on_left)
                    print("sources_on_right", sources_on_right)
                    print("R sources",r_sources)
                    mr['ER'] = ER
                    resolution = arc_obj.getResolution(mr,r_sources)
                    resoution_of_weight_blk.append(resolution)
                    wavelength_idx += 1

        # print(resoution_of_weight_blk)
        min_resolution = min(resoution_of_weight_blk)

        pow_floor_wavelength_at_pd_mW = []
        pow_ceiling_wavelength_at_pd_mW = []
        for wavelength_pw in lamda_power_floor:
            pow_floor_wavelength_at_pd_mW.append(np.power(10,(wavelength_pw/10)))
        for wavelength_pw in lamda_power_ceiling:
            pow_ceiling_wavelength_at_pd_mW.append(np.power(10, (wavelength_pw / 10)))

        total_pw_at_pd_floor_mW = sum(pow_floor_wavelength_at_pd_mW)
        print("Total power at PD floor mW :", total_pw_at_pd_floor_mW)
        total_pw_at_pd_ceiling_mW = sum(pow_ceiling_wavelength_at_pd_mW)
        print("Total power at PD ceiling mW :", total_pw_at_pd_ceiling_mW)
        detector_sensitivity_mW = np.power(10, (arc_obj.detector_sensitivity / 10))
        print("Detector Sensitivity ",detector_sensitivity_mW)
        print("Minimum Resolution", min_resolution)
        max_bits_after_weight = 2 * min_resolution

        ceiling_bits = int(np.floor(np.log2(nlamda * (2 ** max_bits_after_weight))) + 1)
        ceiling_limit = (ceiling_bits ** 2) * arc_obj.adc_pw_mw + detector_sensitivity_mW
        print("Ceiling Bits ", ceiling_bits)
        print("Ceiling Limit ", ceiling_limit)

        if (total_pw_at_pd_floor_mW > detector_sensitivity_mW):
            print("Flooring power supported")
            if (total_pw_at_pd_ceiling_mW > ceiling_limit):
                print("Ceiling power supported")
                if(min_resolution>0):
                    supported_nlamda.append(nlamda)
                    print(f'Nlamda Supported {nlamda}')
                else:
                    print("Resolution is below 1")
                    print(f'Nlamda not Supported {nlamda}')

            else:
                print("Ceiling Power Not Supported")
                print(f'Nlamda not Supported {nlamda}')

        else:
            print(f'Nlamda Not Supported {nlamda}')
    if verbose>=0:
        print("Max supported Nlamda", max(supported_nlamda))
        print("Min Resolution ", min(resoution_of_weight_blk))
        res_median.append(np.median(resoution_of_weight_blk))
        print("Median Resolution",np.median(resoution_of_weight_blk))
        print("Resolution ", resoution_of_weight_blk)
    die_rslts['max_supported_nlamda'] = max(supported_nlamda)
    die_rslts['resolution'] = min_resolution
    die_resolution_nlamda_rslts.append(die_rslts)
print("Die Results:", die_resolution_nlamda_rslts)
print("Median Resolution across Dies", np.median(res_median))
print("Res Median", res_median)
keys = die_resolution_nlamda_rslts[0].keys()
try:
    with open(result_filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(die_resolution_nlamda_rslts)
except PermissionError as inst:
    print("File is already open please close and re run")