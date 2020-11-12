import numpy as np
import pandas as pd
from HitlessArchWafferMap import *
import pprint

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
    # return through_loss

def getDropLossOfRing(Q, ER, LamdaR, delta_lamda):
    A = 1 - (1/ER)
    dropport_loss = (A/(1+np.square((2*Q*delta_lamda)/LamdaR)))
    return -10*np.log10(dropport_loss)
    # return through_loss







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


ER_config = np.genfromtxt('InterDie_ER_0.csv', delimiter=',')
Q_config = np.genfromtxt('InterDie_Q_0.csv', delimiter=',')
LAMDA_R_config = np.genfromtxt('InterDie_LAMDA_R_0.csv', delimiter=',')
FINESSE_config = np.genfromtxt('InterDie_FINESSE_0.csv', delimiter=',')
each_block_dim = '200umX200um'
no_of_blocks_x = ER_config.shape[1]
no_of_blocks_y = ER_config.shape[0]
# print(no_of_blocks_y)
# #
# print(getMaxNlamdaPerModule(no_of_blocks_y,no_of_modules,200,pitch))

no_of_arrays = 75  # this equivalent to the N in the matrix
no_of_rings_aggrtion_blk = 2  # this is equivalent to the N in the matrix
total_rings_per_module = no_of_rings_weighing_blk * no_of_arrays + no_of_rings_aggrtion_blk  # N waveguides with 2 MRs each and N rings at the aggregation block
total_rings_per_arch = total_rings_per_module * no_of_modules

no_of_blocks_x_for_preweigh = int(preweighingblock * 1e3 / 200)
no_of_blocks_x_for_vector_imprint = int(vectorimprintblock * 1e3 / 200)
arch_start_block_indx = no_of_blocks_x_for_preweigh + 1
# print(arch_start_block_indx)
# print(Q_config[20][20])
# arch, aggre_blk = getArchConfigParamsFromDie(no_of_modules, no_of_arrays, no_of_rings_weighing_blk,
#                                              arch_start_block_indx, no_of_blocks_x_for_vector_imprint, ER_config,
#                                              Q_config, LAMDA_R_config, FINESSE_config)
# print(arch.keys())
# print(arch)
# pprint.pprint(arch)
# pprint.pprint(aggre_blk)


# per wavelength power budget
# p_lamda_maop = 5  # dBm

# p_lamda_budget = p_lamda_maop - dectector_sensitivity

# per waveguide power budget

p_waveguide_maop = 100  # mW
dectector_sensitivity = -20  # dB

n_lamda_range = np.arange(6, 7)


configuration = {}
for nlamda in n_lamda_range:
    RESONANCE_WAVELENGTH = 1550  # nm
    FSR = 20  # nm
    channel_spacing = (FSR / (1 + nlamda))

    per_wavelength_power = (p_waveguide_maop/nlamda) #mw
    per_wavelength_power_dbm = 10*np.log10(per_wavelength_power) #dBm
    wavelengths_in_wvg = nlamda
    print("Nlamda:",nlamda)
    print("spacing :",channel_spacing)
    print("Per Wavelength Power",per_wavelength_power)


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
        # Each Array is a one lamda
        for array_name in arch[module_name].keys():
            array_result = {}
            if supported == False:
                break

            array = arch[module_name][array_name]
            #weighing block mr selection
            mr_0 = array[list(array.keys())[0]]
            insertion_loss_percm = 3.7  # dB
            print("Check or Lamda ",(array_idx+1))
            # pre weighing blk
            # calculate the pre weighing blk waveguide propagation loss and filter loss
            pre_filter_wvg_length = 300 #um
            post_filter_wvg_length = 600 #um
            filter_drop_loss = getDropLossOfRing(pre_filter[module_name]['mr_w' + str(array_idx)]['Q'],
                                 pre_filter[module_name]['mr_w' + str(array_idx)]['ER'],
                                 lamdar_per_channel[array_idx], 0)
            filter_wvg_loss = getInsertionLoss(insertion_loss_percm,pitch,ring_radius,array_idx+1)
            pre_weigh_blk_wvg_loss = (insertion_loss_percm*1e-4)*(pre_filter_wvg_length+post_filter_wvg_length)+filter_wvg_loss
            pre_weigh_loss = pre_weigh_blk_wvg_loss+filter_drop_loss
            # print("Filter Drop Loss ",filter_drop_loss )
            # print("Pre Weigh block loss", pre_weigh_loss)

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
            # calculate the summation waveguide propagation loss depending on the length a lamda travels
            aggr_wvg_loss = getInsertionLoss(insertion_loss_percm, pitch, ring_radius, array_idx + 1)

            # print("Aggr Wvg propagation loss", aggr_wvg_loss)

            # calculate through loss

            lamda_idx = array_idx
            resonant_wavelength = lamdar_per_channel[lamda_idx]
            aggr_through_loss = 0
            aggr_mr_Q =  aggre_blk[module_name]['mr_w' + str(array_idx)]['Q']
            aggr_mr_ER = aggre_blk[module_name]['mr_w' + str(array_idx)]['ER']
            for lamda in range(lamda_idx+1):

                operating_wavelength = lamdar_per_channel[lamda]
                if resonant_wavelength == operating_wavelength:
                    # print("Matched")
                    break
                else:
                    through_loss = 0
                    delta_lamda = abs(operating_wavelength-resonant_wavelength)
                    print("Delta Lamda",delta_lamda)
                    through_loss = getThroughLossOfRing(aggr_mr_Q,aggr_mr_ER, resonant_wavelength, delta_lamda)
                    print("Through Loss on ring :"+str(array_idx+1)+" due to ring :"+str(lamda+1)+" is :"+str(through_loss))
                    aggr_through_loss += through_loss
            print("Through of Ring ",(array_idx+1))
            print("Aggre Through Loss",aggr_through_loss)

            aggre_blk_loss = aggr_mr_drop_loss+aggr_wvg_loss+aggr_through_loss

            # print("Total Aggregate block loss :",aggre_blk_loss)
            total_insertion_loss = pre_weigh_loss+weigh_blk_loss+imprint_blk_loss+aggre_blk_loss
            print("Pre Weigh Block : ", pre_weigh_loss)
            print("Weigh Blk Loss : ", weigh_blk_loss)
            print("Imprint Blk Loss : ", imprint_blk_loss)
            print("Aggre_blk_Loss : ", aggre_blk_loss)
            print("Total Insertion losses : ", total_insertion_loss)
            weigh_blk_mr_ER = mr_0['ER']
            weigh_blk_mr_ER_dB = 10*np.log10(weigh_blk_mr_ER)
            pow_wavelength_at_pd = per_wavelength_power_dbm -
            array_idx += 1


        #     # calculate insertion loss for each wavelength as length of waveguide(weighing block,imprint, summation until the wavelength)
        #     total_insertion_loss = getInsertionLoss(insertion_loss_percm, pitch, ring_radius,
        #                                                    no_of_rings_weighing_blk) \
        #                            + getInsertionLossImprintBlk(insertion_loss_percm, pitch, length_of_imprinting_blk) \
        #                            + getInsertionLossSummationBlk(insertion_loss_percm, pitch, , ring_radius)
        #
        #     total_coupling_loss = getCouplingLoss()
        #
        #
        #     # total_modulator_detector_loss = getModulatorDetectorPenalty(mr_0['Q'], nlamda)
        #     # total_photodetector_loss = getPhotoDetectorLoss()
        #     total_through_loss = 0
        #
        #     # Weighing block MR through losses
        #     for mr_name in array.keys():
        #         total_through_loss += getThroughLossOfRing(array[mr_name]['Q'], array[mr_name]['ER'],lamdar_per_channel[array_idx],0)
        #     # Summation block mr through loss for particular wavelength
        #     total_through_loss += getThroughLossOfRing(aggre_blk[module_name]['mr_w' + str(array_idx)]['Q'], aggre_blk[module_name]['mr_w' + str(array_idx)]['ER'],
        #                                                     lamdar_per_channel[array_idx],0)
        #     total_loss = total_through_loss + total_insertion_loss + total_coupling_loss
        #     per_wavlength_error = p_lamda_budget - total_loss - mr_0['ER']
        #     per_wavlength_error_list.append(per_wavlength_error)
        #     total_per_wavelength_error += per_wavlength_error
        #     array_result['wavelength_error'] = per_wavlength_error
        #     array_result['waveguide_error'] = 'NA'
        #
        #
        #     print("Weight block W insertion loss ", getInsertionLossWghtBlk(insertion_loss_percm, pitch, ring_radius,
        #                                                                     no_of_rings_weighing_blk))
        #     print("Imprint Weight block insertion loss",
        #           getInsertionLossImprintBlk(insertion_loss_percm, pitch, length_of_imprinting_blk))
        #     print("Summation block W loss ",
        #           getInsertionLossSummationBlk(insertion_loss_percm, pitch, nlamda, ring_radius))
        #
        #     print("Total Insertion Loss :", total_insertion_loss)
        #     print("Total Coupling Loss :", total_coupling_loss)
        #     print("Total through Loss :", total_through_loss)
        #     print("Total Loss :", total_loss)
        #     print("ER ", mr_0['ER'])
        #     print("Per wavelength Error of array/Lamda :" + str(array_idx + 1) + " = " + str(per_wavlength_error))
        #
        #     print("Total Per Wavelength Error after arrays/Lamda"+str(array_idx+1)+" = "+ str(total_per_wavelength_error))
        #
        #     # print(per_wavlength_error)
        #     if per_wavlength_error <= 0:
        #         configuration['status'] = "Per Wavelength not supported"
        #         supported = False
        #         break
        #     else:
        #         print("Per wavelength supported ======")
        #         print("Check Per Waveguide")
        #         total_through_loss = 0
        #         # computing pre weighing block losses
        #
        #         #summation block losses
        #         for mr_no in range(array_idx + 1):
        #             total_through_loss += getThroughLossOfRing(aggre_blk[module_name]['mr_w' + str(mr_no)]['Q'],
        #                                                        aggre_blk[module_name]['mr_w' + str(mr_no)]['ER'],
        #                                                       lamdar_per_channel[mr_no],channel_spacing)
        #         total_insertion_loss = ((array_idx + 1)*(ring_radius * 2 + pitch) + pitch)*(insertion_loss_percm*1e-4)
        #         total_coupling_loss = getCouplingLoss()
        #
        #         total_photodetector_loss = getPhotoDetectorLoss()
        #         print("W Total Insertion Loss :", total_insertion_loss)
        #         print("W Total Coupling Loss :", total_coupling_loss)
        #         total_loss = total_through_loss + total_insertion_loss + total_coupling_loss+total_photodetector_loss
        #         print("Total Loss :", total_loss)
        #         per_waveguide_error = p_waveguide_budget - total_loss - total_per_wavelength_error
        #         print("Per waveguide error after array:"+str(array_idx)+" = " +str(per_waveguide_error))
        #         array_idx += 1
        #         array_result['waveguide_error'] = per_waveguide_error
        #         per_waveguide_error_list.append(per_waveguide_error)
        #         if per_waveguide_error <= 0:
        #             configuration['status'] = "Per Waveguide not supported"
        #             supported = False
        #             break
        #     module_result['array_'+str(array_idx)] = array_result
        # # print("Per Wavelength Error", per_wavlength_error)
        # # print("Per Waveguide Error", per_waveguide_error)
        # if supported == True:
        #     configuration['status'] = supported
        # configuration[module_name] = module_result

print(configuration)