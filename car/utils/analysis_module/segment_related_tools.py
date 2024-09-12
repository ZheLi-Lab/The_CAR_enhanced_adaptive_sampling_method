import numpy as np
import pandas as pd
import json
import re
import copy
import os
from parse import parse
from glob import glob
from ..input_parser.input_file_parser import InputParser as InputParser
from ..out_parser import read_openmm_out


class Segment_lambda_tools:
    def __init__(self):
        pass

    @staticmethod
    def convert_lambda_list_to_dict(lambda_list):
        '''
        Parameter
        ----------
        lambda_list: list
            2-D list of lambda values, like [[0.0, 0.0, 0.0],[0.0, 0.1, 0.1], [0.0, 0.2, 0.2], ...]

        Return
        ----------
        lambda_dict: dict
            dict of lambda values. Such as:
            {
            "lambda_restraints"     : [0.0, 0.0, 0.0],
            "lambda_electrostatics" : [0.0, 0.1, 0.2],
            "lambda_sterics"        : [0.0, 0.1, 0.2]
            }
        '''
        lambda_matrix = np.array(lambda_list).T.tolist()
        all_lambda_type = ["lambda_restraints", "lambda_electrostatics", "lambda_sterics", "lambda_electrostatics_env", "lambda_electrostatics_chgprod_square"]
        lambda_dict = {}
        for lambda_type in all_lambda_type[:len(lambda_list[0])]:
            lambda_dict[lambda_type] = lambda_matrix[all_lambda_type.index(lambda_type)]
        # lambda_dict = {"lambda_restraints":lambda_matrix[0],"lambda_electrostatics":lambda_matrix[1],"lambda_sterics":lambda_matrix[2]}
        return lambda_dict
    
    @staticmethod
    def find_neighboring_vectors(specific_lambda_dict, all_mbar_lambda_dict, num_neighbors=1, iffindright=True):
        """
        Find the position of the specific_lambda_dict within the all_mbar_lambda_dict and return a 
        result_lambda_dict that contains the neighboring lambda states.

        Parameters:
        - specific_lambda_dict: The lambda values of restraints, electrostatics and sterics. 
            The keys are 'lambda_restraints', 'lambda_electrostatics' and 'lambda_sterics'. 
            The values are three respective lambda list with one value, whose datatype is [float, ].
        - all_mbar_lambda_dict: The lambda values of states whose energy need to be calculated.
            The keys are 'lambda_restraints', 'lambda_electrostatics', and 'lambda_sterics'.
            The values are three respective lambda lists.
        - num_neighbors: Number of neighboring lambda states before and after the specicifc lambda state. Default is 1.
        - iffindright: Bool value, determine whether to find neighboring lambda in the right of the specific lambda dict.
            The value should be False when generate lambda dict for analysing data.

        Returns:
        - result_dict: Dictionary contains the neighboring lambda states.
        """
        ## Convert to NumPy arrays
        target_array = np.array([specific_lambda_dict["lambda_restraints"], specific_lambda_dict["lambda_electrostatics"], specific_lambda_dict["lambda_sterics"]]).T
        reference_array = np.array([all_mbar_lambda_dict["lambda_restraints"], all_mbar_lambda_dict["lambda_electrostatics"], all_mbar_lambda_dict["lambda_sterics"]]).T
        ## Find the index of the target matrix in the reference matrix
        index_start = np.where(np.all(reference_array == target_array[0],axis=1))[0][0]
        index_end = np.where(np.all(reference_array == target_array[-1], axis=1))[0][0]
        ## Extract the specified number of column vectors before and after the target
        start_index = max(0, index_start - num_neighbors)
        end_index = min(reference_array.shape[0], index_end + 1 + num_neighbors*iffindright)
        result_matrix = reference_array[start_index:end_index,:].T
        # print(start_index, end_index)
        # print(reference_array)
        # print(result_matrix)
        ## Convert the result matrix to a dictionary with a similar structure
        result_dict = {
            "lambda_restraints": result_matrix[0].tolist(),
            "lambda_electrostatics": result_matrix[1].tolist(),
            "lambda_sterics": result_matrix[2].tolist()
        }
        # print(f'mbar lambda before:{result_dict}')
        if result_dict['lambda_electrostatics'][0]-result_dict['lambda_electrostatics'][-1] != 0:
            result_dict = Segment_lambda_tools.convert_3Dlambda_to_5Dlambda(result_dict, False)
            specific_lambda_dict = Segment_lambda_tools.convert_3Dlambda_to_5Dlambda(specific_lambda_dict, True)
        # print(f'-------mbar lambda after:{result_dict}')
        return specific_lambda_dict, result_dict

    @staticmethod
    def convert_3Dlambda_to_5Dlambda(lambda_dict, ifSingleWin):
        '''
        TODO: 根据提供的三维度的lambda表，扩充为五维度的
        simulation lambda也需要相应的变化为五位
        '''
        if ifSingleWin:
            lambda_dict["lambda_electrostatics_env"] = [1.0]
            lambda_dict["lambda_electrostatics_chgprod_square"] = np.around(np.array(lambda_dict["lambda_electrostatics"])**2, decimals=6).tolist()
        else:
            lambda_list_new = []
            lambda_list = copy.deepcopy(np.array([lambda_dict["lambda_restraints"], lambda_dict["lambda_electrostatics"], lambda_dict["lambda_sterics"]]).T.tolist())
            for lambda_ in lambda_list:
                lambda_list_new.append(lambda_ + [1.0, np.around(lambda_[1]**2, decimals=6)])
                lambda_list_new.append(lambda_ + [0.0, np.around(lambda_[1]**2, decimals=6)])
            lambda_dict = Segment_lambda_tools.convert_lambda_list_to_dict(lambda_list_new)
        return lambda_dict

    @staticmethod
    def get_segment_lambda_dir_list(segment_lambda):
        '''
        Use segment lambda dict to generate the name list of lambda directory storing data csv.
        
        Parameter
        ----------
        segment_lambda: dict
            A dict of segment lambda setting.
            e.g.    {
                    "lambda_restraints"     : [0.0, 0.0, 0.0],
                    "lambda_electrostatics" : [0.0, 0.1, 0.2],
                    "lambda_sterics"        : [0.0, 0.1, 0.2]
                    }
        
        Return
        ----------
        segment_lambda_dir_list: list of file directory str.
            e.g.    ['0.0_0.0_0.0', '0.0_0.1_0.2', '0.0_0.1_0.2']
        '''
        segment_lambda_dir_list = []
        lambda_list = copy.deepcopy(np.array(list(segment_lambda.values())).T.tolist())
        for lambda_ in lambda_list:
            segment_lambda_dir_list.append(f'{lambda_[0]}_{lambda_[1]}_{lambda_[2]}')
        return segment_lambda_dir_list
    
    @staticmethod
    def generate_data_df(dir_list):
        '''
        Generate dataframe containing each window production MD output data, while windows order is up to given directory order.

        Parameter
        ----------
        dir_list: list
            A list contains names of segment lambda directories.
            e.g. ['0.0_0.0._0.0', '0.0_0.5_0.5', '0.0_1.0_1.0']
        
        Return
        ----------
        data_df: pd.DataFrame
            All data of the segment lambdas prod output.
        '''
        edge_path = os.getcwd()
        file_suffix = 'csv'
        data_df = pd.DataFrame()
        for lambda_dir in dir_list:
            lambda_path = os.path.join(edge_path, str(lambda_dir))
            file_prefix = f'lambda{lambda_dir}'
            read_openmm_obj = read_openmm_out.READ_PROD_OUT(lambda_path, file_prefix, file_suffix)
            files = glob(r'{}'.format(read_openmm_obj.path_pattern))
            single_df = read_openmm_obj.read_file(files[0], read_openmm_obj.index_col, '|')
            data_df = pd.concat([data_df, single_df])
        return data_df
    
    @staticmethod
    def generate_data_dict(dir_list):
        '''
        Generate data dict containing each window production MD output data of a segment.

        Parameter
        ----------
        dir_list: list
            A list contains names of segment lambda directories.
            e.g. ['0.0_0.0._0.0', '0.0_0.5_0.5', '0.0_1.0_1.0']
        
        Return
        ----------
        data_dict: dict
            The dict contains all prod output data. Keies are lambda directory name, values are prod out dataframe.
        '''
        edge_path = os.getcwd()
        file_suffix = 'csv'
        data_dict = {}
        for lambda_dir in dir_list:
            lambda_path = os.path.join(edge_path, str(lambda_dir))
            file_prefix = f'lambda{lambda_dir}'
            read_openmm_obj = read_openmm_out.READ_PROD_OUT(lambda_path, file_prefix, file_suffix)
            files = glob(r'{}'.format(read_openmm_obj.path_pattern))
            single_df = read_openmm_obj.read_file(files[0], read_openmm_obj.index_col, '|')
            data_dict[lambda_dir] = single_df
        return data_dict
    
    @staticmethod
    def generate_data_dict_by_checkfile_frames(edge_path, segment_lambda_dir_list, ana_proportion, frames_max):
        data_dict = {}
        for lambda_dir in segment_lambda_dir_list:
            try:
                data_dict = Segment_lambda_tools.update_data_dict(edge_path, data_dict, lambda_dir, ana_proportion, frames_max, False)
            except:
                pass
        return data_dict

    @staticmethod
    def process_csv(df, ana_type):
        # print('dataframeeeeeeeeeeeeeeeeee:\n',df)
        index_list = ['times(ps)', 'lambda_restraints', 'lambda_electrostatics', 'lambda_sterics'] 
        for i in df.index.names:
            if i not in index_list:
                df = df.reset_index(level=i, drop=True)
        headers = df.columns.tolist()
        # print(headers,'hhhhhhhhhhhhh')
        tuples=[]
        for h in headers:
            if type(h) == tuple:
                tuples.append(h)
            elif type(h)==str and h.startswith('('):
                # print(h)
                h = tuple([].append(float(h[i])) for i in range(len(h))) 
                tuples.append(tuple(h.strip('()').split(', ')))
        # print(tuples,'---------------')
        # tuples = [(float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])) for t in tuples]
        intra_mol_tuples = [t for t in tuples if t[3] == 0.0]
        all_mol_tuples = [t for t in tuples if t[3] == 1.0]
        intra_mol_ene = df[intra_mol_tuples]
        all_mol_ene = df[all_mol_tuples]

        mol_env_ene_data={}
        for intra_t in intra_mol_tuples:
            for all_t in all_mol_tuples:
                if intra_t[:3] == all_t[:3]:
                    new_col = tuple(intra_t[:3])
                    mol_env_ene_data[new_col] = all_mol_ene[all_t] - intra_mol_ene[intra_t]
                    break
        mol_env_ene = pd.DataFrame(mol_env_ene_data, index=df.index, columns=list(mol_env_ene_data.keys()))
        intra_mol_ene.columns = [t[:3] for t in intra_mol_tuples]
        all_mol_ene.columns = [t[:3] for t in all_mol_tuples]
        # print(f'molllll:{mol_env_ene},\n inteeeee:{intra_mol_ene},\n all:{all_mol_ene}\n', all_mol_ene.columns)
        ana_index = ['intra_mol_ene', 'all_mol_ene', 'mol_env_ene'].index(ana_type)
        return [intra_mol_ene, all_mol_ene, mol_env_ene][ana_index]

    @staticmethod
    def update_data_dict(edge_path, data_dict, new_lambda, ana_proportion, frames_max, ifupdate_check=True,ana_type='all_mol_ene'):
        '''
        ana_type: str
            'intra_mol_ene' | 'all_mol_ene' | 'mol_env_ene', determine the analyzed data type of 5-lambda-MD-schedule
        '''
        check_point_file = os.path.join(edge_path, 'analysis_used_segments_frames.json')
        lambda_frame_dict = json.load(open(check_point_file, 'r', encoding='utf-8'))
        lambda_path = os.path.join(edge_path, str(new_lambda))
        read_openmm_obj = read_openmm_out.READ_PROD_OUT(lambda_path, f'lambda{new_lambda}', 'csv')
        files = glob(r'{}'.format(read_openmm_obj.path_pattern))
        single_df = read_openmm_obj.read_file(files[0], read_openmm_obj.index_col, '|')
        # print(f'init_df:\n{single_df}')
        if len(read_openmm_obj.index_col)>4 :
            single_df = Segment_lambda_tools.process_csv(single_df, ana_type)
        for i in range(len(lambda_frame_dict), 0, -1):
            try:
                pre_start_frame = lambda_frame_dict[f"run_{i}"][new_lambda][-1][0]
                break
            except:
                pre_start_frame = 0
        len_df = len(single_df) - pre_start_frame
        ana_df_len = min(len_df*ana_proportion, frames_max)
        cur_start_frame = int(np.floor(len_df - ana_df_len))
        cur_end_frame = int(len(single_df))
        df_ana_use = single_df.iloc[cur_start_frame:cur_end_frame,:]
        data_dict[new_lambda] = df_ana_use
        if ifupdate_check:
            frames_tuple = (cur_start_frame,cur_end_frame)
            lambda_frame_dict_new =  Segment_lambda_tools.update_check_point_dict(lambda_frame_dict, new_lambda, frames_tuple)
            with open(check_point_file, 'w', encoding="utf-8") as f:
                json.dump(lambda_frame_dict_new ,f)
        return data_dict

    @staticmethod
    def init_check_point_file(edge_path, frames_each_simu, frames_max):
        check_point_file = os.path.join(edge_path, 'analysis_used_segments_frames.json')
        if os.path.exists(check_point_file):
            check_dict = json.load(open(check_point_file, 'r', encoding='utf-8'))
            check_dict[f"run_{len(check_dict)+1}"] = {"frames_each_simu_update": frames_each_simu, "frames_to_ana_max": frames_max}
        else:
            check_dict = {"run_1":{"frames_each_simu_update": frames_each_simu, "frames_to_ana_max": frames_max}}
        with open(check_point_file, 'w', encoding="utf-8") as f:
            json.dump(check_dict ,f)

    @staticmethod
    def update_check_point_dict(lambda_frame_dict, lambda_dir, frames_tuple):
        last_dict = lambda_frame_dict[f"run_{len(lambda_frame_dict)}"]
        try:
            last_dict[lambda_dir].append(frames_tuple)
        except:
            last_dict[lambda_dir] = [frames_tuple]
        return lambda_frame_dict

    @staticmethod
    def set_segment(all_lambda_dict, segment_lambda_step=0.1):
        '''
        Parameter
        ----------
        all_lambda_dict: dict
            The dict of all lambda values need to be seperated by given step.
        segment_lambda_step: float
            The lambda step of the segment lambda values. Default: 0.1
        
        Return
        ----------
        segmented_lambda: list
            The list contains segmented lambda dict.
        
        Example usage:
            all_lambda_dict = {
                "lambda_restraints"     : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "lambda_electrostatics" : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "lambda_sterics"        : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                }
            segment_lambda_step = 0.5
            segmented_lambda = set_segment(all_lambda_dict, segment_lambda_step)
            print(segmented_lambda)
            >> ouput: [{"lambda_restraints"     : [0.0, 0.0, 0.0],
                        "lambda_electrostatics" : [0.0, 0.2, 0.4],
                        "lambda_sterics"        : [0.0, 0.2, 0.4]
                        },
                            
                        {"lambda_restraints"     : [0.0, 0.0, 0.0],
                        "lambda_electrostatics" : [0.4, 0.6, 0.8],
                        "lambda_sterics"        : [0.4, 0.6, 0.8]
                        },
                            
                        {"lambda_restraints"     : [0.0, 0.0],
                        "lambda_electrostatics" : [0.8, 1.0],
                        "lambda_sterics"        : [0.8, 1.0]
                        }]
            
        '''
        all_lambda_list = np.array(list(all_lambda_dict.values())).T.tolist()
        segmented_lambda = []
        segment_lambda_0 = all_lambda_list[0]
        cur_segment_lambda = []
        for lambda_ in all_lambda_list:
            lambda_step, label, lambda_info_x = Segment_lambda_tools.cal_delta_lambda(np.array(segment_lambda_0), np.array(lambda_))
            if abs(lambda_step)<=segment_lambda_step:
                cur_segment_lambda.append(lambda_)
            else:
                segmented_lambda.append(Segment_lambda_tools.convert_lambda_list_to_dict(cur_segment_lambda))
                cur_segment_lambda = [cur_segment_lambda[-1]]
                cur_segment_lambda.append(lambda_)
                segment_lambda_0 = cur_segment_lambda[0]
        segmented_lambda.append(Segment_lambda_tools.convert_lambda_list_to_dict(cur_segment_lambda))
        return segmented_lambda

    @staticmethod
    def cal_delta_lambda(lambda_value_1, lambda_value_2):
        lambda_value_1 = np.array(lambda_value_1)
        lambda_value_2 = np.array(lambda_value_2)
        dd_coords = (np.subtract(lambda_value_1,lambda_value_2))
        ABFE_labels = ['restraint', 'charge', 'vdw']
        if type(dd_coords)==np.float64:
            label = 'amber_RBFE'
            delta_lambda = np.around(dd_coords, decimals=3) 
            lambda_info_x = np.around((lambda_value_1+lambda_value_2)/2, decimals=4)
        else:
            nonzero_indices = np.argwhere(dd_coords != 0).flatten()
            nonzero_count = len(nonzero_indices)
            if nonzero_count==1:
                delta_lambda = np.around(dd_coords[nonzero_indices[0]], decimals=3)
                if nonzero_indices[0] < len(ABFE_labels):
                    label = ABFE_labels[nonzero_indices[0]]
                    lambda_info_x = np.around((lambda_value_1[nonzero_indices[0]]+lambda_value_2[nonzero_indices[0]])/2, decimals=4)
                else:
                    label = 'unknown'
                    lambda_info_x = None
            elif nonzero_count==2 and lambda_value_1[1]==lambda_value_1[2] and lambda_value_2[1]==lambda_value_2[2]:
                label = 'openmm_RBFE'
                delta_lambda = np.around(dd_coords[nonzero_indices[0]], decimals=3)
                lambda_info_x = np.around((lambda_value_1[nonzero_indices[0]]+lambda_value_2[nonzero_indices[0]])/2, decimals=4)
            else:
                label = 'mix'
                delta_lambda = np.around(dd_coords.sum(), decimals=3) 
                lambda_info_x = None
        return delta_lambda, label, lambda_info_x

    @staticmethod
    def get_lambda_start_end_coords(delta_A_what_to_what, md_pack=''):
        pattern = '({},{},{}) to ({},{},{})'
        result = parse(pattern, delta_A_what_to_what)
        if result:
            start_coords = (float(result[0]), float(result[1]), float(result[2]))
            end_coords = (float(result[3]), float(result[4]), float(result[5]))
        else:
            pattern = '{} to {}'
            result = parse(pattern, delta_A_what_to_what)
            if result:
                start_coords = float(result[0])
                end_coords = float(result[1])
            else:
                raise ValueError("delta_A_what_to_what can't match pattern '({},{},{}) to ({},{},{})' or '{} to {}'. Please check the format.")
        return start_coords, end_coords

    @staticmethod
    def check_segmented_lambda(segmented_lambda_lst, mbar_lambda_dict):
        '''
        Check if the segmented lambdas can form a continuous and monotonic lambda setting dict, and is part of all mbar lambda dict. 
        If not, raise ValueError.

        Parameter
        ----------
        segmented_lambda_lst: list
            Each element is a dict of one segment lambda setting. 
        mbar_lambda_dict: dict
            The dict contains all mbar lambda.
        '''
        segment_0 = copy.deepcopy(list(segmented_lambda_lst[0].values()))
        all_simulation_lambda_list = [segment_0[0],segment_0[1],segment_0[2]]
        for i in range(len(segmented_lambda_lst) - 1):
            cur_segment_lambda_list = list(segmented_lambda_lst[i].values())
            next_segment_lambda_list = list(segmented_lambda_lst[i+1].values())
            for lambda_group in range(len(cur_segment_lambda_list)):
                if cur_segment_lambda_list[lambda_group][-1] == next_segment_lambda_list[lambda_group][0]:
                    all_simulation_lambda_list[lambda_group].extend(next_segment_lambda_list[lambda_group][1:])
                else:
                    raise ValueError(f"The given segmented lambda groups is not continuous as the last lambda in No.{i} segment is not the first in No.{i+1} segment. Please check your lambdas setting json file.")            
        mbar_lambda_list = copy.deepcopy(list(mbar_lambda_dict.values()))
        # print(f": All simulation lambda list is: {all_simulation_lambda_list}")
        # print(f'mbar_lambda_list is: {mbar_lambda_list}')
        for sublist_simu, sublist_mbar in zip(all_simulation_lambda_list, mbar_lambda_list):
            for i in range(len(sublist_simu)):
                increasing = all(sublist_simu[i] <= sublist_simu[i + 1] for i in range(len(sublist_simu) - 1))
                decreasing = all(sublist_simu[i] >= sublist_simu[i + 1] for i in range(len(sublist_simu) - 1))
            if not increasing and not decreasing:
                raise ValueError("The segmented lambda dicts are not monotonic. Please check your lambdas setting json file.")
            if not set(sublist_simu).issubset(set(sublist_mbar)):
                raise ValueError("The segmented lambda dicts are not entirely included in mbar lambda dict, may cause MD error when simulating specific lambda. Please check your lambdas setting json file.")

    @staticmethod
    def cal_each_simulation_frames_num(soft, md_input_file, prod_md_time):
        '''
        Calculate the frame num of output data obtained from a single simulation.
        openmm: line_num = md_time/timestep/nsteps = niterations
        amber: line_num = md_time/dt/bar_intervall = nstlim/bar_intervall
        '''
        if soft == 'amber':
            input_dict = json.load(open(md_input_file, 'r', encoding='utf-8'))
            prod_setting_list = input_dict["prod.in"]
            dt = float(re.findall(r'\d*\.\d+|\d+', prod_setting_list[17])[0])
            bar_intervall = int(re.findall(r'\d*\.\d+|\d+', prod_setting_list[-9])[0])
            n_line = int(np.floor(prod_md_time/dt/bar_intervall))
        elif soft == 'openmm':
            from ..Alchemd.utils.file_parser import InputParser as openmmInputParser
            input_data = openmmInputParser(md_input_file)
            alchemical_setting = input_data.get_alchemical()
            timestep_in_fs = alchemical_setting['timestep']
            nsteps = alchemical_setting['nsteps']
            n_line = int(np.floor(prod_md_time/(timestep_in_fs*0.001)/nsteps))
        else:
            raise ValueError(f"The specified software {soft} is not supported yet. Can analyse data obtained by 'amber' or 'openmm'.")
        return n_line

    @staticmethod
    def cal_md_time_by_frames(soft, md_input_file, nline):
        '''
        Calculate the md time of a lambda window by frames num of output data obtained.
        openmm: md_time = line_num * nsteps * timestep
        amber: md_time = line_num * dt * bar_intervall
        '''
        if soft == 'amber':
            input_dict = json.load(open(md_input_file, 'r', encoding='utf-8'))
            prod_setting_list = input_dict["prod.in"]
            dt = float(re.findall(r'\d*\.\d+|\d+', prod_setting_list[17])[0])
            bar_intervall = int(re.findall(r'\d*\.\d+|\d+', prod_setting_list[-9])[0])
            md_time = int(nline*dt*bar_intervall)
        elif soft == 'openmm':
            from ..Alchemd.utils.file_parser import InputParser as openmmInputParser
            input_data = openmmInputParser(md_input_file)
            alchemical_setting = input_data.get_alchemical()
            timestep_in_fs = alchemical_setting['timestep']
            nsteps = alchemical_setting['nsteps']
            md_time = int(nline*(timestep_in_fs*0.001)*nsteps)
        else:
            raise ValueError(f"The specified software {soft} is not supported yet. Can analyse data obtained by 'amber' or 'openmm'.")
        return md_time

    @staticmethod
    def copy_last_simulation_frame(cur_lambda, frames_num):
        '''
        Parameter
        ----------
        cur_lambda: string
            like '0.0_0.0._0.0'
        frames_num: int
            The number of frames saved from a single simulation.
        '''
        edge_path = os.getcwd()
        lambda_path = os.path.join(edge_path, str(cur_lambda))
        read_openmm_obj = read_openmm_out.READ_PROD_OUT(lambda_path, f'lambda{cur_lambda}', 'csv')
        prev_df = read_openmm_obj.extract_data(1, read_openmm_obj.index_col, '|')
        cur_df = prev_df.iloc[-frames_num: , :]
        new_df = pd.concat([prev_df, cur_df])
        new_df.to_csv(f'{lambda_path}/lambda{cur_lambda}.csv', '|')