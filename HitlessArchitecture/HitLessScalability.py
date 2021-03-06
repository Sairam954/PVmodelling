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
        self.mzi_ER_dB = design_config["mzi_ER_dB"]
        self.adc_pw_mw = design_config["adc_pw_mw"]
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
    def getCnnBlkLoss(self,cnn_blk,module_name,wavelength_idx,lamdar_per_channel):

        cnn_blk_loss = self.getDropLossOfRing(cnn_blk[module_name]['mr_w' + str(wavelength_idx)]['Q'],
                                              cnn_blk[module_name]['mr_w' + str(wavelength_idx)]['ER'],
                                              lamdar_per_channel[wavelength_idx], 0)
        no_of_cnn_blk = 4
        cnn_blk_loss += (self.insertion_loss_percm * 1e-4)*(no_of_cnn_blk*self.pitch+1)

        return cnn_blk_loss

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

    def getPowerOfWavelengthFloorandCeiling(self,mr_wgt_ER,total_insertion_loss_dB,per_wavelength_power_dbm):
        weigh_blk_mr_ER = mr_wgt_ER
        weigh_blk_mr_ER_dB = 10 * np.log10(weigh_blk_mr_ER)
        imprint_blk_mzi_ER_dB = self.mzi_ER_dB
        # print("ER of Weight block in dB", weigh_blk_mr_ER_dB)
        pow_wavelength_at_pd_dbm_floor = per_wavelength_power_dbm - weigh_blk_mr_ER_dB -imprint_blk_mzi_ER_dB- total_insertion_loss_dB
        pow_wavelength_at_pd_dbm_ceiling = per_wavelength_power_dbm-total_insertion_loss_dB
        return pow_wavelength_at_pd_dbm_floor,pow_wavelength_at_pd_dbm_ceiling

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
        mr_pv_ER = mr_0['ER']
        resolution = 1 / (thermal_crosstalk_error)
        resolution = resolution*((mr_pv_ER / self.DESIGN_ER))
        # resolution = resolution.astype(int)
        # print("Resoultion",np.log2(resolution))
        resolution = np.ceil(np.log2(resolution)).astype(int)
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

    def getTotalLossOfArray(self,module_name,array_idx,pre_filter,aggre_blk,cnn_blk,lamdar_per_channel,nlamda):


        # pre weighing blk
        pre_weigh_loss = self.getPreWeighBlkLoss(module_name, pre_filter, wavelength_idx=array_idx,
                                                    lamdar_per_channel=lamdar_per_channel)
        # Weight and Imprint blk
        # calculate the weighing block wvg propogation loss and vector imprint block wvg loss plus MZI insertion loss
        weigh_blk_loss = self.getInsertionLoss(self.no_of_rings_weighing_blk)
        imprint_blk_loss = self.getInsertionLossImprintBlk()

        # print("Weigh blk wvg IL ",weigh_blk_loss)
        # print("Imprint blk wvg IL ",imprint_blk_loss)
        cnn_blk_loss = 0
        #Cnn block loss
        if array_idx<16:
            cnn_blk_loss = self.getCnnBlkLoss(cnn_blk, module_name, array_idx,lamdar_per_channel)
            # print("Cnn Blk Loss",cnn_blk_loss)


        # aggregation block
        # calculate summation Mr ring drop loss
        aggr_mr_drop_loss = self.getDropLossOfRing(aggre_blk[module_name]['mr_w' + str(array_idx)]['Q'],
                                                      aggre_blk[module_name]['mr_w' + str(array_idx)]['ER'],
                                                      lamdar_per_channel[array_idx], 0)
        # print("Aggr blk Mr loss ",aggr_mr_drop_loss)

        # pd is at the top

        aggre_blk_loss = self.getAggregateBlkLoss(module_name, aggre_blk, wavelength_idx=array_idx,
                                                     lamdar_per_channel=lamdar_per_channel, nlamda=nlamda)

        aggre_blk_loss += aggr_mr_drop_loss
        # print("Total Aggregate block loss :",aggre_blk_loss)
        total_insertion_loss_dB = pre_weigh_loss + weigh_blk_loss + imprint_blk_loss + aggre_blk_loss +cnn_blk_loss
        if self.verbose != 0:
            print("Pre Weigh Block : ", pre_weigh_loss)
            print("Weigh Blk Loss : ", weigh_blk_loss)
            print("Imprint Blk Loss : ", imprint_blk_loss)
            print("Aggre_blk_Loss : ", aggre_blk_loss)
            print("Total Insertion losses dB: ", total_insertion_loss_dB)
        return total_insertion_loss_dB