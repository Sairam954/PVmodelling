import numpy as np
import pandas as pd
from HitlessArchWafferMap import *
import pprint
import csv
import os
import re

class HitLessArchitecture():

    def __init__(self,config_dict):
        # set architecture configuration all of these are in um
        arch_config = config_dict["arch_config"]
        self.preweighingblock = arch_config["preweighingblock"]
        self.vectorimprintblock = arch_config["vectorimprintblock"]
        self.pre_filter_wvg_length = arch_config["pre_filter_wvg_length"]
        self.post_filter_wvg_length = arch_config["post_filter_wvg_length"]
        self.ring_radius = arch_config["ring_radius"]
        self.pitch = arch_config["pitch"]
        self.no_of_modules = arch_config["no_of_modules"]
        self.no_of_rings_weighing_blk = arch_config["no_of_rings_weighing_blk"]
        self.nlamda = arch_config["nlamda"]
        # die configuration in mm
        die_config = config_dict["die_config"]
        self.die_x = die_config["die_x"]
        self.die_y = die_config["die_y"]
        #resolution config in mm
        resolution_config = config_dict["resolution_config"]
        self.HEATER_THICKNESS = resolution_config["HEATER_THICKNESS"]
        self.CHIP_BOUNDARY_RADIUS = resolution_config["CHIP_BOUNDARY_RADIUS"]
        self.DESIGN_ER = resolution_config["DESIGN_ER"]
        #design config mw dBm and nm
        design_config = config_dict["design_config"]
        self.p_waveguide_maop = design_config["p_waveguide_maop"]
        self.detector_sensitivity = design_config["detector_sensitivity"]
        self.RESONANCE_WAVELENGTH = design_config["RESONANCE_WAVELENGTH"]
        self.FSR = design_config["FSR"]
        self.insertion_loss_percm = design_config["insertion_loss_percm"]
        self.mzi_insertion_loss = design_config["mzi_insertion_loss"]
        # pv dies path containing csvs with design parameters like Q,FINESSE,ER and LAMDA_R
        die_data_path = config_dict["die_data_path"]
        self.ER_data_path = die_data_path["ER_data_path"]
        self.LAMDA_R_data_path = die_data_path["LAMDA_R_data_path"]
        self.Q_data_path = die_data_path["Q_data_path"]
        self.FINESSE_data_path = die_data_path["FINESSE_data_path"]

    def getInsertionLoss(self, no_of_rings):
        length_of_blk_waveguide = (self.pitch + self.ring_radius * 2) * no_of_rings   # um
        perumloss = self.insertion_loss_percm * 1e-4
        return length_of_blk_waveguide * perumloss


    def getInsertionLossImprintBlk(self):
        perumloss = self.insertion_loss_percm * 1e-4
        imprint_waveguide_loss = (self.vectorimprintblock + 2 * self.pitch) * perumloss
        return imprint_waveguide_loss+self.mzi_insertion_loss


    def getInsertionLossSummationBlk(self,no_of_rings):
        perumloss = self.insertion_loss_percm * 1e-4
        length_of_summation_blk_waveguide = (self.pitch + self.ring_radius * 2) * no_of_rings + self.pitch
        return length_of_summation_blk_waveguide * perumloss



    def getCouplingLoss(self):
        couplinglosspercoupler = 1.5  # dB
        return couplinglosspercoupler


    def getThroughLossOfRing(self,Q, ER, LamdaR, delta_lamda):
        A = 1 - (1/ER)
        through_loss = 1 - (A/(1+np.square((2*Q*delta_lamda)/LamdaR)))
        return -10*np.log10(through_loss)


    def getDropLossOfRing(self,Q, ER, LamdaR, delta_lamda):
        A = 1 - (1/ER)
        dropport_loss = (A/(1+np.square((2*Q*delta_lamda)/LamdaR)))
        return -10*np.log10(dropport_loss)

    def getPreWeighBlkLoss(self,module_name,pre_filter,wavelength_idx,lamdar_per_channel):
        # calculate the pre weighing blk waveguide propagation loss and filter loss

        filter_drop_loss = self.getDropLossOfRing(pre_filter[module_name]['mr_w' + str(wavelength_idx)]['Q'],
                                             pre_filter[module_name]['mr_w' + str(wavelength_idx)]['ER'],
                                             lamdar_per_channel[wavelength_idx], 0)
        filter_wvg_loss = self.getInsertionLoss(wavelength_idx + 1)
        pre_weigh_blk_wvg_loss = (self.insertion_loss_percm * 1e-4) * (
                    self.pre_filter_wvg_length + self.post_filter_wvg_length) + filter_wvg_loss
        pre_weigh_loss = pre_weigh_blk_wvg_loss + filter_drop_loss
        # print("Pre Waveguide Filter Drop Loss ", filter_drop_loss)
        return pre_weigh_loss

    def getAggregateBlkLoss(self,module_name,aggre_blk,wavelength_idx,lamdar_per_channel,nlamda):
        # calculate summation Mr ring drop loss
        aggr_mr_drop_loss = self.getDropLossOfRing(aggre_blk[module_name]['mr_w' + str(wavelength_idx)]['Q'],
                                              aggre_blk[module_name]['mr_w' + str(wavelength_idx)]['ER'],
                                              lamdar_per_channel[wavelength_idx], 0)

        # pd is at the top
        # calculate the summation waveguide propagation loss depending on the length a lamda travels
        aggr_wvg_loss = self.getInsertionLoss(wavelength_idx + 1)



        # calculate through loss

        lamda_idx = wavelength_idx
        resonant_wavelength = lamdar_per_channel[lamda_idx]
        aggr_through_loss = 0
        aggr_mr_Q = aggre_blk[module_name]['mr_w' + str(wavelength_idx)]['Q']
        aggr_mr_ER = aggre_blk[module_name]['mr_w' + str(wavelength_idx)]['ER']

        for lamda in range(lamda_idx + 1, nlamda):
            operating_wavelength = lamdar_per_channel[lamda]

            through_loss = 0
            delta_lamda = abs(operating_wavelength - resonant_wavelength)
            # print("Delta Lamda", delta_lamda)
            through_loss = self.getThroughLossOfRing(aggr_mr_Q, aggr_mr_ER, resonant_wavelength, delta_lamda)
            # print("Through Loss on ring :" + str(wavelength_idx + 1) + " due to ring :" + str(lamda + 1) + " is :" + str(
            #     through_loss))
            aggr_through_loss += through_loss
        # print("Through Loss of Ring ", (wavelength_idx + 1))
        # print("Aggre Through Loss dB", aggr_through_loss)

        aggre_blk_loss = aggr_mr_drop_loss + aggr_wvg_loss + aggr_through_loss
        return aggre_blk_loss

    def getPowerOfWavelength(self,mr_wgt_ER,total_insertion_loss_dB,per_wavelength_power_dbm):
        weigh_blk_mr_ER = mr_wgt_ER
        weigh_blk_mr_ER_dB = 10 * np.log10(weigh_blk_mr_ER)
        # print("ER of Weight block in dB", weigh_blk_mr_ER_dB)
        pow_wavelength_at_pd_dbm = per_wavelength_power_dbm - weigh_blk_mr_ER_dB - total_insertion_loss_dB
        return pow_wavelength_at_pd_dbm

    def getMaxThermalCrossTalkError(self,r_source,finesse):

        # for source in r_source:
        #     heat_diffusion = heat_diffusion+(np.log(r_chip_boundary/source)/np.log(r_chip_boundary/ring_radius))

        heat_diffusion = (np.log(self.CHIP_BOUNDARY_RADIUS/ r_source) / np.log(self.CHIP_BOUNDARY_RADIUS / (self.ring_radius*1e-6)))
        # we consider in an array if MR adjacent two MRs incur thermal losses so summation of the sources
        # considetring both the rings are at a similar or equal distance
        # heat_diffusion = 2*heat_diffusion
        max_error = 0.65*finesse*(self.HEATER_THICKNESS/(2*self.ring_radius*1e-6))*heat_diffusion
        return max_error

    def getResolution(self,mr_0):
        thermal_crosstalk_error = self.getMaxThermalCrossTalkError(r_source=(self.pitch * 1e-6), finesse=mr_0['FINESSE_R'])
        resolution = 1 / (thermal_crosstalk_error)
        resolution = resolution.astype(int)
        mr_pv_ER = mr_0['ER']
        resolution = np.ceil(np.log2(resolution * (mr_pv_ER / self.DESIGN_ER))).astype(int)
         # resolution = int(resolution * (mr_pv_ER / DESIGN_ER))
        return resolution
    def getParameterDieFiles(self,path):
        files = [file for file in os.listdir(path) if (file.lower().endswith('.csv'))]
        sort_files = []
        for file in files:
            if file.endswith(".csv"):
                sort_files.append(os.path.join(path, file))
                # print(os.path.join(path, file))
        sort_files.sort(key=lambda x: os.path.getmtime(x))
        return sort_files


# preweighingblock = 1  # mm
# vectorimprintblock = 1  # mm
# ring_radius = 7  # um
# pitch = 100  # um
# no_of_modules = 1  # this equivalent to M dimension in MXN matrix
# no_of_rings_weighing_blk = 2
# DIE_DIM = '30mm X 15mm'
# DIE_Y = 30  # mm
# DIE_X = 15  # mm
#
#
#
#
#
#
# p_waveguide_maop = 100  # mW
# dectector_sensitivity = -20  # dBm
#
# n_lamda_range = np.arange(64,300)
#
#
# # Resolution calculation global params
#
# HEATER_THICKNESS = 100*1e-9
# CHIP_BOUNDARY_RADIUS = 1*1e-3
# DESIGN_ER = 10
#
# configuration = {}
# supported_nlamda = []
# total_power_for_lamda = []
#
#
# RESONANCE_WAVELENGTH = 1550  # nm
# FSR = 20  # nm
#
#
# ER_data_path = "interpvfolder/ER/Block1by1/"
# LAMDA_R_data_path = "interpvfolder/LAMDA_R/Block1by1/"
# Q_data_path = "interpvfolder/Q/Block1by1/"
# FINESSE_data_path = "interpvfolder/FINESSE/Block1by1/"
#
# ER_dies_config = getParameterDieFiles(ER_data_path)
# LAMDA_R_dies_config = getParameterDieFiles(LAMDA_R_data_path)
# Q_dies_config = getParameterDieFiles(Q_data_path)
# FINESSE_dies_config = getParameterDieFiles(FINESSE_data_path)
#
# die_resolution_nlamda_rslts = []
# die_no = 0
# no_of_dies = 80
# verbose = 0
# for die in range(no_of_dies):
#     die_rslts = {}
#     die_rslts['die_no'] = die
#
#     ER_config = np.loadtxt(ER_dies_config[die], delimiter=',')
#     Q_config = np.genfromtxt(Q_dies_config[die], delimiter=',')
#     LAMDA_R_config = np.genfromtxt(LAMDA_R_dies_config[die], delimiter=',')
#     FINESSE_config = np.genfromtxt(FINESSE_dies_config[die], delimiter=',')
#     # Procession Variation Dimension are swapped so in mean time just transposing the matrixxes
#     # ER_config = np.transpose(ER_config)
#     # Q_config = np.transpose(Q_config)
#     # LAMDA_R_config = np.transpose(LAMDA_R_config)
#     # FINESSE_config = np.transpose(FINESSE_config)
#
#     each_block_dim = '100umX100um'
#     no_of_blocks_x = ER_config.shape[1]
#     no_of_blocks_y = ER_config.shape[0]
#     # print("Y blocks :", no_of_blocks_y)
#     # print("X blocks :",no_of_blocks_x)
#
#     no_of_blocks_x_for_preweigh = int(preweighingblock / 200)
#     no_of_blocks_x_for_vector_imprint = int(vectorimprintblock / 200)
#     arch_start_block_indx = no_of_blocks_x_for_preweigh + 1
#     resoution_of_weight_blk = []
#     for nlamda in n_lamda_range:
#
#         channel_spacing = (FSR / (1 + nlamda))
#         per_wavelength_power = (p_waveguide_maop/nlamda) #mw
#         per_wavelength_power_dbm = 10*np.log10(per_wavelength_power) #dBm
#         wavelengths_in_wvg = nlamda
#         if verbose!=0:
#             print("Nlamda :", nlamda)
#             print("spacing :", channel_spacing)
#             print("Total power available mW :",p_waveguide_maop)
#             print("Per Wavelength Power mW :",per_wavelength_power )
#             print("Per Wavelength Power dBm :", per_wavelength_power_dbm)
#
#
#         lamdar_per_channel = np.arange(RESONANCE_WAVELENGTH, (RESONANCE_WAVELENGTH + channel_spacing * nlamda),
#                                        channel_spacing)
#
#
#         configuration['N'] = nlamda
#         configuration['M'] = no_of_modules
#
#         arch, aggre_blk, pre_filter = getArchConfigParamsFromDie(no_of_modules, nlamda, no_of_rings_weighing_blk,
#                                                      arch_start_block_indx, no_of_blocks_x_for_vector_imprint, ER_config,
#                                                      Q_config, LAMDA_R_config, FINESSE_config)
#
#         supported = True
#         # Each Module
#
#         for module_name in arch.keys():
#             total_per_wavelength_error = 0
#             array_idx = 0 # array or lamda number
#             module_result = {}
#             per_wavlength_power = []
#
#             # print("Check Array ", array_idx)
#             if supported == False:
#                 break
#             lamda_power = []
#             # Each Array is a one lamda
#             for array_name in arch[module_name].keys():
#                 array_result = {}
#                 if supported == False:
#                     break
#
#                 array = arch[module_name][array_name]
#                 #weighing block mr selection
#                 mr_0 = array[list(array.keys())[0]]
#                 insertion_loss_percm = 3.7  # dB
#                 pre_filter_wvg_length = 300  # um
#                 post_filter_wvg_length = 600  # um
#                 # print("Calculate power for Wavelength ",(array_idx+1))
#                 # pre weighing blk
#                 pre_weigh_loss = getPreWeighBlkLoss(pre_filter,wavelength_idx=array_idx,lamdar_per_channel=lamdar_per_channel)
#                 #Weight and Imprint blk
#                 #calculate the weighing block wvg propogation loss and vector imprint block wvg loss plus MZI insertion loss
#                 weigh_blk_loss = getInsertionLoss(insertion_loss_percm, pitch, ring_radius,no_of_rings_weighing_blk)
#                 imprint_blk_loss = getInsertionLossImprintBlk(insertion_loss_percm, pitch, vectorimprintblock*1e-3)
#
#                 # print("Weigh blk wvg IL ",weigh_blk_loss)
#                 # print("Imprint blk wvg IL ",imprint_blk_loss)
#
#                 #aggregation block
#                 # calculate summation Mr ring drop loss
#                 aggr_mr_drop_loss = getDropLossOfRing(aggre_blk[module_name]['mr_w' + str(array_idx)]['Q'],
#                                      aggre_blk[module_name]['mr_w' + str(array_idx)]['ER'],
#                                      lamdar_per_channel[array_idx], 0)
#                 # print("Aggr blk Mr loss ",aggr_mr_drop_loss)
#
#
#                 # pd is at the top
#
#                 aggre_blk_loss = getAggregateBlkLoss(aggre_blk,wavelength_idx=array_idx,lamdar_per_channel=lamdar_per_channel)
#
#                 # print("Total Aggregate block loss :",aggre_blk_loss)
#                 total_insertion_loss_dB = pre_weigh_loss+weigh_blk_loss+imprint_blk_loss+aggre_blk_loss
#                 if verbose!=0:
#                     print("Pre Weigh Block : ", pre_weigh_loss)
#                     print("Weigh Blk Loss : ", weigh_blk_loss)
#                     print("Imprint Blk Loss : ", imprint_blk_loss)
#                     print("Aggre_blk_Loss : ", aggre_blk_loss)
#                     print("Total Insertion losses dB: ", total_insertion_loss_dB)
#                 #calculate power of a wavelength
#                 mr_wgt_ER = mr_0['ER']
#                 mr_wgt_ER_dB = 10*np.log10(mr_wgt_ER)
#                 pow_wavelength_at_pd_dbm = getPowerOfWavelength(mr_wgt_ER, total_insertion_loss_dB, per_wavelength_power_dbm)
#                 lamda_power.append(pow_wavelength_at_pd_dbm)
#
#                 # Resolution calculation:
#                 resolution = getResolution(mr_0)
#                 resoution_of_weight_blk.append(resolution)
#                 if verbose!=0:
#                     print("Resolution", resolution)
#                     print("Power of Wavelength at Pd :", pow_wavelength_at_pd_dbm)
#
#                 array_idx += 1
#
#         pow_wavelength_at_pd_mW = []
#         for wavelength_pw in lamda_power:
#             pow_wavelength_at_pd_mW.append(np.power(10,(wavelength_pw/10)))
#         # print("pow_wavelength_at_pd_mW ",pow_wavelength_at_pd_mW)
#
#         total_power_for_lamda.append(sum(pow_wavelength_at_pd_mW) )
#         # print("Total power at PD dBm", 10*np.log10(sum(pow_wavelength_at_pd_mW)))
#         total_pw_at_pd_mW = sum(pow_wavelength_at_pd_mW)
#         print("Total power at PD mW :", total_pw_at_pd_mW)
#         dectector_sensitivity_mW = np.power(10,(dectector_sensitivity/10))
#         # print("Detector Sensitivity in mW :", dectector_sensitivity_mW)
#         # print("Detector Sensitivity in dBm", 10*np.log10(dectector_sensitivity_mW))
#
#         if(total_pw_at_pd_mW>dectector_sensitivity_mW):
#             print(f'Nlamda Supported {nlamda}')
#             supported_nlamda.append(nlamda)
#         else:
#             print(f'Nlamda Not Supported {nlamda}')
#     if verbose>=0:
#         print("Max supported Nlamda", max(supported_nlamda))
#         print("Min Resolution ", min(resoution_of_weight_blk))
#         print("Resolution ", resoution_of_weight_blk)
#     die_rslts['max_supported_nlamda'] = max(supported_nlamda)
#     die_rslts['resolution'] = min(resoution_of_weight_blk)
#     die_resolution_nlamda_rslts.append(die_rslts)
# print("Die Results:", die_resolution_nlamda_rslts)
# keys = die_resolution_nlamda_rslts[0].keys()
# with open('results/dievariation100by100.csv', 'w', newline='')  as output_file:
#     dict_writer = csv.DictWriter(output_file, keys)
#     dict_writer.writeheader()
#     dict_writer.writerows(die_resolution_nlamda_rslts)
#
#
#
