import numpy as np
import pandas as pd

def getArchConfigParamsFromDie(no_of_modules, no_of_ring_pairs_weighing_blk, arch_start_block_indx, ER_config, Q_config, LAMDA_R_config, FINESSE_config):
    arc = {}
    block_row_idx = 0
    block_column_idx = arch_start_block_indx
    block_size = 200  # um
    pitch = 200  # um
    block_increment_in_x = int(block_size / pitch)
    for module_id in range(no_of_modules):
        module = {}
        array = {}
        for ring_pair in range(no_of_ring_pairs_weighing_blk):
            #weighing block
            config_params = {}
            config_params['Q'] = Q_config[block_row_idx][block_column_idx]
            config_params['ER'] = ER_config[block_row_idx][block_column_idx]
            config_params['LAMDA_R'] = LAMDA_R_config[block_row_idx][block_column_idx]
            config_params['FINESSE_R'] = FINESSE_config[block_row_idx][block_column_idx]
            array['mr_w' + str(ring_pair)] = config_params
            block_column_idx = block_column_idx + block_increment_in_x
            module['array' + str(block_row_idx)] = array
        arc.update({'module' + str(module_id): module})
    return arc
#
# ER_config = np.loadtxt("interpvfolder/ER/Block2by2/InterDie_ER_0_block200_200.csv", delimiter=',')
# Q_config = np.genfromtxt("interpvfolder/Q/Block2by2/InterDie_Q_0_block200_200.csv", delimiter=',')
# LAMDA_R_config = np.genfromtxt("interpvfolder/LAMDA_R/Block2by2/InterDie_LAMDA_R_0_block200_200.csv", delimiter=',')
# FINESSE_config = np.genfromtxt("interpvfolder/FINESSE/Block2by2/InterDie_FINESSE_0_block200_200.csv", delimiter=',')
#
# preweighingblock = 600
# pitch = 200
# no_of_modules = 1
# nlamda = 100
# no_of_blocks_x = ER_config.shape[1]
# no_of_blocks_y = ER_config.shape[0]
# no_of_blocks_x_for_preweigh = int(preweighingblock / pitch)
# arch_start_block_indx = no_of_blocks_x_for_preweigh + 1
# arch = getArchConfigParamsFromDie(no_of_modules,nlamda,arch_start_block_indx,ER_config,Q_config,LAMDA_R_config,FINESSE_config)
#
# print(arch)

