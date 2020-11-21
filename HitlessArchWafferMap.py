import numpy as np
import pandas as pd
import pprint


def getMaxNlamdaPerModule(no_of_block_y, no_of_modules,block_size, pitch):
    per_module_y = int(no_of_block_y/no_of_modules)
    print(no_of_block_y)
    print(1+(block_size/pitch))
    per_array_mr_y = (block_size/pitch)
    per_module_supported_array = int(per_module_y/per_array_mr_y)
    return per_module_supported_array

def doesArcFitOnDie(no_of_modules, no_of_arrays, block_size, no_of_block_y):
    max_supported_array = getMaxNlamdaPerModule(no_of_block_y,no_of_modules,block_size,pitch)
    if no_of_arrays>max_supported_array:
        return False
    else:
        return True


def getArchConfigParamsFromDie(no_of_modules,no_of_arrays,no_of_rings_weighing_blk,arch_start_block_indx,no_of_blocks_x_for_vector_imprint,ER_config,Q_config,LAMDA_R_config,FINESSE_config):
    hitlessweight =[]
    arc ={}
    aggr_block = {}
    fltr_block = {}
    block_row_idx = 0
    block_column_idx = arch_start_block_indx
    block_size = 200 #um
    pitch = 200 #um
    block_increment_in_x = int(block_size/pitch)
    no_of_blocks_x_for_vector_imprint = no_of_blocks_x_for_vector_imprint
    for module_id in range(no_of_modules):# M
        module = {}
        aggre_mr = {}
        pre_weigh_filter = {}
        aggregate_mr_no = 0
        pre_weigh_filter_no = 0
        # print('Module----', module_id)
        # print('rowindex ', block_row_idx)
        print(no_of_arrays)
        for array_id in range(no_of_arrays):# N
            block_column_idx = arch_start_block_indx
            array = {}
            #pre weigh block filter
            config_params = {}
            # print("Row ",block_row_idx+1 )
            # print("Column ", block_column_idx-3)
            # print(Q_config.shape)
            # print(ER_config.shape)
            # LAMDA_R_config = np.transpose(LAMDA_R_config)
            # print(LAMDA_R_config.shape)
            # print(FINESSE_config.shape)
            # print("Row", block_row_idx+1)
            # print("Column", block_column_idx-3)

            config_params['Q'] = Q_config[block_row_idx+1][block_column_idx-3]
            config_params['ER'] = ER_config[block_row_idx+1][block_column_idx-3]
            config_params['LAMDA_R'] = LAMDA_R_config[block_row_idx+1][block_column_idx-3]
            config_params['FINESSE_R'] = FINESSE_config[block_row_idx+1][block_column_idx-3]
            pre_weigh_filter['mr_w'+str(pre_weigh_filter_no)] = config_params
            pre_weigh_filter_no+=1
            #weighing block
            for ring in range(no_of_rings_weighing_blk):
                config_params = {}
                config_params['Q'] = Q_config[block_row_idx][block_column_idx]
                config_params['ER'] = ER_config[block_row_idx][block_column_idx]
                config_params['LAMDA_R'] = LAMDA_R_config[block_row_idx][block_column_idx]
                config_params['FINESSE_R'] = FINESSE_config[block_row_idx][block_column_idx]
                array['mr_w' + str(ring)] = config_params
                block_column_idx = block_column_idx+block_increment_in_x
            #vector imprint
            block_column_idx = block_column_idx + no_of_blocks_x_for_vector_imprint
            #summation block
            config_params = {}
            block_row_idx = block_row_idx+1
            config_params['Q'] = Q_config[block_row_idx][block_column_idx]
            config_params['ER'] = ER_config[block_row_idx][block_column_idx]
            config_params['LAMDA_R'] = LAMDA_R_config[block_row_idx][block_column_idx]
            config_params['FINESSE_R'] = FINESSE_config[block_row_idx][block_column_idx]
            # array['mr_a' + str(aggregate_mr_no)] = config_params
            aggre_mr['mr_w'+str(aggregate_mr_no)] = config_params
            aggregate_mr_no += 1
            module['array'+str(array_id)] = array
            # block_row_idx += 1
        fltr_block.update({'module'+str(module_id): pre_weigh_filter})
        aggr_block.update({'module'+str(module_id): aggre_mr})

        # print("module",module_id)
        hitlessweight.append(module)
        arc.update({'module'+str(module_id): module})
        # print(arc.keys())
    return arc,aggr_block,fltr_block

