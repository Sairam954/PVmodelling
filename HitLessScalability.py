import numpy as np
import pandas as pd
from HitlessArchWafferMap import *
import pprint
import os
from Weight import Weight


preweighingblock = 1  # mm
vectorimprintblock = 1  # mm
ring_radius = 7  # um
pitch = 200  # um
no_of_modules = 1  # this equivalent to M dimension in MXN matrix
DIE_DIM = '30mm X 15mm'
DIE_Y = 30  # mm
DIE_X = 15  # mm

no_of_rings_weighing_blk = 2
length_of_weighing_blk = ring_radius * 2 * no_of_rings_weighing_blk + pitch
length_of_imprinting_blk = 1  # mm
length_of_summation_blk = 1  # mm

# read configuration parameters dies
# die_path = "C://Users//saira//Google Drive//ResearchPapers//Fall2020//Code//interpvfolder//"
# folder_name = "//Block2by2//"
# # ER_data_path = "C://Users//saira//Google Drive//ResearchPapers//Fall2020//Code//interpvfolder//ER//Block2by2//"
# # LAMDA_R_data_path = "C://Users//saira//Google Drive//ResearchPapers//Fall2020//Code//interpvfolder//LAMDA_R//Block2by2//"
# # Q_data_path = "C://Users//saira//Google Drive//ResearchPapers//Fall2020//Code//interpvfolder//Q//Block2by2//"
# # FINESSE_data_path = "C://Users//saira//Google Drive//ResearchPapers//Fall2020//Code//interpvfolder//FINESSE//Block2by2//"
#
# # import glob
# # print(glob.glob("/pvfigures/Q/*.png"))
# #
#
#
#
# search_dir = die_path+"ER"+folder_name
# os.chdir(search_dir)
# files = filter(os.path.isfile, os.listdir(search_dir))
# files = [os.path.join(search_dir, f) for f in files]  # add path to each file
# print(files)
#
# files.sort(key=lambda x: os.path.getmtime(x))
# #
ER_config = np.loadtxt("InterDie_ER_0.csv", delimiter=',')
Q_config = np.genfromtxt('InterDie_Q_0.csv', delimiter=',')
LAMDA_R_config = np.genfromtxt('InterDie_LAMDA_R_0.csv', delimiter=',')
FINESSE_config = np.genfromtxt('InterDie_FINESSE_0.csv', delimiter=',')
each_block_dim = '200umX200um'
no_of_blocks_x = ER_config.shape[1]
no_of_blocks_y = ER_config.shape[0]
# print(no_of_blocks_y)
# #
# print(getMaxNlamdaPerModule(no_of_blocks_y,no_of_modules,200,pitch))


# no_of_rings_aggrtion_blk = 2  # this is equivalent to the N in the matrix
# total_rings_per_module = no_of_rings_weighing_blk * no_of_arrays + no_of_rings_aggrtion_blk  # N waveguides with 2 MRs each and N rings at the aggregation block
# total_rings_per_arch = total_rings_per_module * no_of_modules

no_of_blocks_x_for_preweigh = int(preweighingblock * 1e3 / 200)
no_of_blocks_x_for_vector_imprint = int(vectorimprintblock * 1e3 / 200)
arch_start_block_indx = no_of_blocks_x_for_preweigh + 1

p_waveguide_maop = 100  # mW
dectector_sensitivity = -20  # dBm

n_lamda_range = np.arange(32,150)


# Resolution calculation global params
weightObj = Weight()
HEATER_THICKNESS = 100*1e-9
CHIP_BOUNDARY_RADIUS = 1*1e-3
DESIGN_ER = 10

configuration = {}
supported_nlamda = []
total_power_for_lamda = []
resoution_of_weight_blk = []

RESONANCE_WAVELENGTH = 1550  # nm
FSR = 20  # nm



def getInsertionLoss(percmloss, pitch, ring_radius, no_of_rings):
    length_of_blk_waveguide = (pitch + ring_radius * 2) * no_of_rings   # um
    perumloss = percmloss * 1e-4
    return length_of_blk_waveguide * perumloss


def getInsertionLossImprintBlk(percmloss, pitch, imprint_blk_length):
    perumloss = percmloss * 1e-4
    imprint_waveguide_loss = (imprint_blk_length * 1000 + 2 * pitch) * perumloss
    MZI_INSERTION_LOSS = 3.30 #dB
    return imprint_waveguide_loss+MZI_INSERTION_LOSS


def getInsertionLossSummationBlk(percmloss, pitch, no_of_rings, ring_radius):
    perumloss = percmloss * 1e-4
    length_of_summation_blk_waveguide = (pitch + ring_radius * 2) * no_of_rings + pitch
    return length_of_summation_blk_waveguide * perumloss



def getCouplingLoss():
    couplinglosspercoupler = 1.5  # dB
    return couplinglosspercoupler


def getThroughLossOfRing(Q, ER, LamdaR, delta_lamda):
    A = 1 - (1/ER)
    through_loss = 1 - (A/(1+np.square((2*Q*delta_lamda)/LamdaR)))
    return -10*np.log10(through_loss)


def getDropLossOfRing(Q, ER, LamdaR, delta_lamda):
    A = 1 - (1/ER)
    dropport_loss = (A/(1+np.square((2*Q*delta_lamda)/LamdaR)))
    return -10*np.log10(dropport_loss)

def getPreWeighBlkLoss(pre_filter,wavelength_idx,lamdar_per_channel):
    # calculate the pre weighing blk waveguide propagation loss and filter loss

    filter_drop_loss = getDropLossOfRing(pre_filter[module_name]['mr_w' + str(wavelength_idx)]['Q'],
                                         pre_filter[module_name]['mr_w' + str(wavelength_idx)]['ER'],
                                         lamdar_per_channel[wavelength_idx], 0)
    filter_wvg_loss = getInsertionLoss(insertion_loss_percm, pitch, ring_radius, wavelength_idx + 1)
    pre_weigh_blk_wvg_loss = (insertion_loss_percm * 1e-4) * (
                pre_filter_wvg_length + post_filter_wvg_length) + filter_wvg_loss
    pre_weigh_loss = pre_weigh_blk_wvg_loss + filter_drop_loss
    print("Pre Waveguide Filter Drop Loss ", filter_drop_loss)
    return pre_weigh_loss

def getAggregateBlkLoss(aggre_blk,wavelength_idx,lamdar_per_channel):
    # calculate summation Mr ring drop loss
    aggr_mr_drop_loss = getDropLossOfRing(aggre_blk[module_name]['mr_w' + str(wavelength_idx)]['Q'],
                                          aggre_blk[module_name]['mr_w' + str(wavelength_idx)]['ER'],
                                          lamdar_per_channel[wavelength_idx], 0)
    # print("Aggr blk Mr loss ",aggr_mr_drop_loss)

    # pd is at the top
    # calculate the summation waveguide propagation loss depending on the length a lamda travels
    aggr_wvg_loss = getInsertionLoss(insertion_loss_percm, pitch, ring_radius, wavelength_idx + 1)

    # print("Aggr Wvg propagation loss", aggr_wvg_loss)

    # calculate through loss

    lamda_idx = wavelength_idx
    resonant_wavelength = lamdar_per_channel[lamda_idx]
    aggr_through_loss = 0
    aggr_mr_Q = aggre_blk[module_name]['mr_w' + str(wavelength_idx)]['Q']
    aggr_mr_ER = aggre_blk[module_name]['mr_w' + str(wavelength_idx)]['ER']

    for lamda in range(lamda_idx + 1, nlamda):
        operating_wavelength = lamdar_per_channel[lamda]
        # if resonant_wavelength == operating_wavelength:
        #     # print("Matched")
        #     break
        # else:
        through_loss = 0
        delta_lamda = abs(operating_wavelength - resonant_wavelength)
        print("Delta Lamda", delta_lamda)
        through_loss = getThroughLossOfRing(aggr_mr_Q, aggr_mr_ER, resonant_wavelength, delta_lamda)
        print("Through Loss on ring :" + str(wavelength_idx + 1) + " due to ring :" + str(lamda + 1) + " is :" + str(
            through_loss))
        aggr_through_loss += through_loss
    print("Through Loss of Ring ", (wavelength_idx + 1))
    print("Aggre Through Loss dB", aggr_through_loss)

    aggre_blk_loss = aggr_mr_drop_loss + aggr_wvg_loss + aggr_through_loss
    return aggre_blk_loss

def getPowerOfWavelength(mr_wgt_ER,total_insertion_loss_dB,per_wavelength_power_dbm):
    weigh_blk_mr_ER = mr_wgt_ER
    weigh_blk_mr_ER_dB = 10 * np.log10(weigh_blk_mr_ER)
    print("ER of Weight block in dB", weigh_blk_mr_ER_dB)
    pow_wavelength_at_pd_dbm = per_wavelength_power_dbm - weigh_blk_mr_ER_dB - total_insertion_loss_dB
    return pow_wavelength_at_pd_dbm

def getMaxThermalCrossTalkError(heater_thickness,ring_radius,r_chip_boundary,r_source,finesse):

    # for source in r_source:
    #     heat_diffusion = heat_diffusion+(np.log(r_chip_boundary/source)/np.log(r_chip_boundary/ring_radius))

    heat_diffusion = (np.log(r_chip_boundary / r_source) / np.log(r_chip_boundary / ring_radius))
    # we consider in an array if MR adjacent two MRs incur thermal losses so summation of the sources
    # considetring both the rings are at a similar or equal distance
    # heat_diffusion = 2*heat_diffusion
    max_error = 0.65*finesse*(heater_thickness/(2*ring_radius))*heat_diffusion
    return max_error

def getResolution(mr_0):
    thermal_crosstalk_error = getMaxThermalCrossTalkError(HEATER_THICKNESS, ring_radius * 1e-6,
                                                          CHIP_BOUNDARY_RADIUS,
                                                          r_source=(pitch * 1e-6), finesse=mr_0['FINESSE_R'])
    resolution = 1 / (thermal_crosstalk_error)
    resolution = resolution.astype(int)
    resolution = np.ceil(np.log2(resolution)).astype(int)
    mr_pv_ER = mr_0['ER']

    resolution = int(resolution * (mr_pv_ER / DESIGN_ER))
    return resolution


for nlamda in n_lamda_range:

    channel_spacing = (FSR / (1 + nlamda))
    per_wavelength_power = (p_waveguide_maop/nlamda) #mw
    per_wavelength_power_dbm = 10*np.log10(per_wavelength_power) #dBm
    wavelengths_in_wvg = nlamda
    print("Nlamda :", nlamda)
    print("spacing :", channel_spacing)
    print("Total power available mW :",p_waveguide_maop)
    print("Per Wavelength Power mW :",per_wavelength_power )
    print("Per Wavelength Power dBm :", per_wavelength_power_dbm)


    lamdar_per_channel = np.arange(RESONANCE_WAVELENGTH, (RESONANCE_WAVELENGTH + channel_spacing * nlamda),
                                   channel_spacing)


    configuration['N'] = nlamda
    configuration['M'] = no_of_modules

    arch, aggre_blk, pre_filter = getArchConfigParamsFromDie(no_of_modules, nlamda, no_of_rings_weighing_blk,
                                                 arch_start_block_indx, no_of_blocks_x_for_vector_imprint, ER_config,
                                                 Q_config, LAMDA_R_config, FINESSE_config)
    # # print(arch.keys())
    # print(aggre_blk)
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
        # Each Array is a one lamda
        for array_name in arch[module_name].keys():
            array_result = {}
            if supported == False:
                break

            array = arch[module_name][array_name]
            #weighing block mr selection
            mr_0 = array[list(array.keys())[0]]
            insertion_loss_percm = 3.7  # dB
            pre_filter_wvg_length = 300  # um
            post_filter_wvg_length = 600  # um
            print("Calculate power for Wavelength ",(array_idx+1))
            # pre weighing blk
            pre_weigh_loss = getPreWeighBlkLoss(pre_filter,wavelength_idx=array_idx,lamdar_per_channel=lamdar_per_channel)
            #Weight and Imprint blk
            #calculate the weighing block wvg propogation loss and vector imprint block wvg loss plus MZI insertion loss
            weigh_blk_loss = getInsertionLoss(insertion_loss_percm, pitch, ring_radius,no_of_rings_weighing_blk)
            imprint_blk_loss = getInsertionLossImprintBlk(insertion_loss_percm, pitch, length_of_imprinting_blk)

            # print("Weigh blk wvg IL ",weigh_blk_loss)
            # print("Imprint blk wvg IL ",imprint_blk_loss)

            #aggregation block
            # calculate summation Mr ring drop loss
            aggr_mr_drop_loss = getDropLossOfRing(aggre_blk[module_name]['mr_w' + str(array_idx)]['Q'],
                                 aggre_blk[module_name]['mr_w' + str(array_idx)]['ER'],
                                 lamdar_per_channel[array_idx], 0)
            # print("Aggr blk Mr loss ",aggr_mr_drop_loss)


            # pd is at the top

            aggre_blk_loss = getAggregateBlkLoss(aggre_blk,wavelength_idx=array_idx,lamdar_per_channel=lamdar_per_channel)

            # print("Total Aggregate block loss :",aggre_blk_loss)
            total_insertion_loss_dB = pre_weigh_loss+weigh_blk_loss+imprint_blk_loss+aggre_blk_loss
            print("Pre Weigh Block : ", pre_weigh_loss)
            print("Weigh Blk Loss : ", weigh_blk_loss)
            print("Imprint Blk Loss : ", imprint_blk_loss)
            print("Aggre_blk_Loss : ", aggre_blk_loss)
            print("Total Insertion losses dB: ", total_insertion_loss_dB)
            #calculate power of a wavelength
            mr_wgt_ER = mr_0['ER']
            pow_wavelength_at_pd_dbm = getPowerOfWavelength(mr_wgt_ER, total_insertion_loss_dB, per_wavelength_power_dbm)
            lamda_power.append(pow_wavelength_at_pd_dbm)
            print("Power of Wavelength at Pd :",pow_wavelength_at_pd_dbm)

            # Resolution calculation:
            resolution = getResolution(mr_0)
            print("Resolution", resolution)
            resoution_of_weight_blk.append(resolution)

            array_idx += 1

    pow_wavelength_at_pd_mW = []
    for wavelength_pw in lamda_power:
        pow_wavelength_at_pd_mW.append(np.power(10,(wavelength_pw/10)))
    # print("pow_wavelength_at_pd_mW ",pow_wavelength_at_pd_mW)
    print("Total power at PD mW",sum(pow_wavelength_at_pd_mW) )
    total_power_for_lamda.append(sum(pow_wavelength_at_pd_mW) )
    # print("Total power at PD dBm", 10*np.log10(sum(pow_wavelength_at_pd_mW)))
    total_pw_at_pd_mW = sum(pow_wavelength_at_pd_mW)
    dectector_sensitivity_mW = np.power(10,(dectector_sensitivity/10))
    print("Detector Sensitivity in mW", dectector_sensitivity_mW)
    # print("Detector Sensitivity in dBm", 10*np.log10(dectector_sensitivity_mW))

    if(total_pw_at_pd_mW>dectector_sensitivity_mW):
        print("Nlamda Supported ",nlamda)
        supported_nlamda.append(nlamda)
    else:
        print("Nlamda Not Supported")
print(total_power_for_lamda)
print(supported_nlamda)
print("Min Resolution ", min(resoution_of_weight_blk))
print("Resolution ", resoution_of_weight_blk)