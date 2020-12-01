from BroadcastWafferMap import *

import utils.readconfig as rd
import numpy as np
import sys
import traceback
import logging
import csv
import os

class BroadcastAndWeight():

    def __init__(self, config_dict):
        arch_config = config_dict["arch_config"]
        self.preweighingblock = arch_config["preweighingblock"]
        self.premuxblock = arch_config["premuxblock"]
        self.vectorimprintblock = arch_config["vectorimprintblock"]
        self.ring_radius = arch_config["ring_radius"]
        self.pitch = arch_config["pitch"]
        self.no_of_modules = arch_config["no_of_modules"]
        self.nlamda = arch_config["nlamda"]
        # die configuration in mm
        die_config = config_dict["die_config"]
        self.die_x = die_config["die_x"]
        self.die_y = die_config["die_y"]
        self.no_of_dies = die_config["no_of_dies"]
        # resolution config in mm
        resolution_config = config_dict["resolution_config"]
        self.HEATER_THICKNESS = resolution_config["HEATER_THICKNESS"]
        self.CHIP_BOUNDARY_RADIUS = resolution_config["CHIP_BOUNDARY_RADIUS"]
        self.DESIGN_ER = resolution_config["DESIGN_ER"]
        # design config mw dBm and nm
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


    def getPreWeighBlkLoss(self):
        # calculate the pre weighing blk waveguide propagation loss and filter loss


        pre_weigh_blk_wvg_loss = (self.insertion_loss_percm * 1e-4) * (
                self.preweighingblock + self.pitch+ 4*self.ring_radius +self.pitch)
        print("Pre Waveguide loss ",pre_weigh_blk_wvg_loss)
        return pre_weigh_blk_wvg_loss


    def getInsertionLoss(self, no_of_rings):
        length_of_blk_waveguide = (self.pitch + self.ring_radius * 2) * no_of_rings  # um
        perumloss = self.insertion_loss_percm * 1e-4
        return length_of_blk_waveguide * perumloss

    def getThroughLossOfRing(self, Q, ER, LamdaR, delta_lamda):
        A = 1 - (1/ER)
        through_loss = 1 - (A/(1+np.square((2*Q*delta_lamda)/LamdaR)))
        return -10*np.log10(through_loss)

    def getInsertionLossImprintBlk(self):
        perumloss = self.insertion_loss_percm * 1e-4
        imprint_waveguide_loss = (self.vectorimprintblock + 2 * self.pitch) * perumloss
        return imprint_waveguide_loss + self.mzi_insertion_loss

    def getDropLossOfRing(self, Q, ER, LamdaR, delta_lamda):
        A = 1 - (1/ER)
        dropport_loss = (A/(1+np.square((2*Q*delta_lamda)/LamdaR)))
        return -10*np.log10(dropport_loss)

    def getMaxThermalCrossTalkError(self, r_source, finesse):

        # for source in r_source:
        #     heat_diffusion = heat_diffusion+(np.log(r_chip_boundary/source)/np.log(r_chip_boundary/ring_radius))

        heat_diffusion = (np.log(self.CHIP_BOUNDARY_RADIUS / r_source) / np.log(
            self.CHIP_BOUNDARY_RADIUS / (self.ring_radius * 1e-6)))
        # we consider in an array if MR adjacent two MRs incur thermal losses so summation of the sources
        # considetring both the rings are at a similar or equal distance
        # heat_diffusion = 2*heat_diffusion

        max_error = 0.65 * finesse * (self.HEATER_THICKNESS / (2 * self.ring_radius * 1e-6)) * heat_diffusion
        return max_error

    def getResolution(self, mr_0):
        thermal_crosstalk_error = self.getMaxThermalCrossTalkError(r_source=(self.pitch * 1e-6),
                                                                   finesse=mr_0['FINESSE_R'])
        mr_pv_ER = mr_0['ER']
        resolution = 1 / (thermal_crosstalk_error)
        resolution = resolution * ((mr_pv_ER / self.DESIGN_ER))
        # resolution = resolution.astype(int)
        # print("Resoultion",np.log2(resolution))
        resolution = np.ceil(np.log2(resolution)).astype(int)
        # resolution = int(resolution * (mr_pv_ER / DESIGN_ER))
        return resolution

    def getPowerOfWavelengthFloorandCeiling(self, mr_wgt_ER, total_insertion_loss_dB, per_wavelength_power_dbm):
        weigh_blk_mr_ER = mr_wgt_ER
        weigh_blk_mr_ER_dB = 10 * np.log10(weigh_blk_mr_ER)
        imprint_blk_mzi_ER_dB = self.mzi_ER_dB
        # print("ER of Weight block in dB", weigh_blk_mr_ER_dB)
        pow_wavelength_at_pd_dbm_floor = per_wavelength_power_dbm - weigh_blk_mr_ER_dB - imprint_blk_mzi_ER_dB - total_insertion_loss_dB
        print("Per Wavelength ",per_wavelength_power_dbm)
        print("total_insertion_loss_dB",total_insertion_loss_dB)
        pow_wavelength_at_pd_dbm_ceiling = per_wavelength_power_dbm - total_insertion_loss_dB
        return pow_wavelength_at_pd_dbm_floor, pow_wavelength_at_pd_dbm_ceiling

    def getParameterDieFiles(self, path):
        files = [file for file in os.listdir(path) if (file.lower().endswith('.csv'))]
        sort_files = []
        for file in files:
            if file.endswith(".csv"):
                sort_files.append(os.path.join(path, file))
                # print(os.path.join(path, file))
        sort_files.sort(key=lambda x: os.path.getmtime(x))
        return sort_files


    def getTotalLossOfArray(self,module_name, wavelength_idx, array, demul_mr, mul_mr,lamdar_per_channel,nlamda):

        # pre mux and demux blk
        pre_mux_loss = (self.insertion_loss_percm*1e-4)*(self.premuxblock+self.pitch+wavelength_idx*self.pitch)
        # print("Pre Mux Loss",pre_mux_loss)
        # demux loss
        # calculate through loss

        lamda_idx = wavelength_idx
        resonant_wavelength = lamdar_per_channel[lamda_idx]
        demux_through_loss = 0
        demux_mr_Q = demul_mr['mr_w' + str(wavelength_idx)]['Q']
        demux_mr_ER = demul_mr['mr_w' + str(wavelength_idx)]['ER']

        for lamda in range(lamda_idx + 1, nlamda):
            operating_wavelength = lamdar_per_channel[lamda]

            through_loss = 0
            delta_lamda = abs(operating_wavelength - resonant_wavelength)
            # print("Delta Lamda", delta_lamda)
            through_loss = self.getThroughLossOfRing(demux_mr_Q, demux_mr_ER, resonant_wavelength, delta_lamda)
            # print("Through Loss on ring :" + str(wavelength_idx + 1) + " due to ring :" + str(lamda + 1) + " is :" + str(
            #     through_loss))
            demux_through_loss += through_loss
        # print("Through Loss of Demux Ring ", (wavelength_idx + 1))
        # print("Demux Through Loss dB", demux_through_loss)
        #imprint loss
        imprint_blk_loss = self.getInsertionLossImprintBlk()
        # print("Insertion Loss Imprint block",imprint_blk_loss)
        #mux through loss
        mux_mr_Q = mul_mr['mr_w' + str(wavelength_idx)]['Q']
        mux_mr_ER = mul_mr['mr_w' + str(wavelength_idx)]['ER']
        mux_drop_loss = self.getDropLossOfRing(mux_mr_Q,mux_mr_ER,resonant_wavelength,0)
        # print("Mul Mr drop loss", mux_drop_loss)

        #mux demux wvg propagation loss
        demux_blk_wvg_loss = (self.insertion_loss_percm*1e-4)*(self.ring_radius*4+wavelength_idx*self.pitch+self.pitch)
        # print("Demux Blk Wvg Loss",demux_blk_wvg_loss)

        # pre weigh block loss
        pre_weigh_loss = (self.insertion_loss_percm*1e-4)*(self.preweighingblock+self.pitch)
        # print("Pre Weight loss", pre_weigh_loss)
        #
        # Weighing block loss
        weigh_mr = array['mr_w' + str(wavelength_idx)]
        mr_through_loss = 0
        weigh_mr_Q = weigh_mr['Q']
        weigh_mr_ER = weigh_mr['ER']
        weigh_through_loss =0
        for lamda in range(0, nlamda):
            operating_wavelength = lamdar_per_channel[lamda]

            through_loss = 0
            delta_lamda = abs(operating_wavelength - resonant_wavelength)
            # print("Delta Lamda", delta_lamda)
            if delta_lamda==0:
                through_loss = 0
            else:
                through_loss = self.getThroughLossOfRing(weigh_mr_Q, weigh_mr_ER, resonant_wavelength, delta_lamda)
            # print("Through Loss on ring :" + str(wavelength_idx + 1) + " due to ring :" + str(lamda + 1) + " is :" + str(through_loss))
            weigh_through_loss += through_loss
        total_insertion_loss = pre_mux_loss+demux_through_loss+imprint_blk_loss+mux_drop_loss+demux_blk_wvg_loss+pre_weigh_loss+weigh_through_loss
        print("Total Inserstion Loss ", total_insertion_loss)
        return total_insertion_loss




