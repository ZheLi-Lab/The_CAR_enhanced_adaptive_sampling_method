import numpy as np
import os
import shutil
import copy
import json
import time
from glob import glob
from .md_run_module import AmberMD, OpenmmMD
from .analysis_module.segment_analyse import SegmentConvergenceAnalysis
from .analysis_module.segment_related_tools import Segment_lambda_tools


class AlchemdControl(Segment_lambda_tools):
    def __init__(self, input_file, complex_coor, complex_topo, soft, custom_time, lambdas_setting_file='lambdas.json'):
        self.input_file=input_file
        self.complex_coor=complex_coor
        self.complex_topo=complex_topo
        self.soft = soft
        self.custom_time = custom_time
        self.lambdas_setting_file = lambdas_setting_file
        self.lambdas_setting_dict = json.load(open(self.lambdas_setting_file, 'r', encoding='utf-8'))
        self.all_mbar_lambda_dict = list(self.lambdas_setting_dict.values())[0]
        self.all_lambda_directories = self.get_segment_lambda_dir_list(self.all_mbar_lambda_dict)

    @staticmethod
    def overwrite_rm_files(edge_path, directories_lst):
        check_point_file = os.path.join(edge_path, 'analysis_used_segments_frames.json')
        if os.path.exists(check_point_file):
            os.remove(check_point_file)
            print("NOTE: you set overwrite as True, so existed lambda directories will be removed.")
        segment_ana_dir = os.path.join(edge_path, 'segment_converge_analysis')
        if os.path.exists(segment_ana_dir):
            shutil.move(segment_ana_dir, os.path.join(edge_path, f'bak_segment_converge_analysis{time.strftime("%Y-%m-%d_%H_%M")}'))
        ana_used_csv_dir = os.path.join(edge_path, 'ana_used_data')
        if os.path.exists(ana_used_csv_dir):
            shutil.move(ana_used_csv_dir, os.path.join(edge_path, f'bak_ana_used_data{time.strftime("%Y-%m-%d_%H_%M")}'))
        for dir_lambda in directories_lst:
            dir_path = os.path.join(edge_path, str(dir_lambda))
            if os.path.exists(dir_path):
                print(f"now remove: {dir_path}")
                shutil.rmtree(dir_path)
        print("All lambda directories were removed.")

    def get_segmented_lambda_list(self, segment_lambda_step):
        ### set segments or check the input segmented lambda dict.
        if len(self.lambdas_setting_dict) == 1:
            ## only one lambda dict, use set segment function to get segmented lambda list.
            segmented_lambda_list = self.set_segment(self.all_mbar_lambda_dict, segment_lambda_step)
        else:
            segmented_lambda_list = copy.deepcopy(list(self.lambdas_setting_dict.values())[1:])
            self.check_segmented_lambda(segmented_lambda_list, self.all_mbar_lambda_dict)
        return segmented_lambda_list

    def new_continuous_run_setting(self, segmented_lambda_list, edge_path, rerun_start_win, ifuse_initial_rst, ifuse_current_win_coor):
        '''
        Set the input state for new simulation, and return the new segmented lambda list if the start win is not the first lambda.
        '''
        if rerun_start_win is None: ### start md simulation from lambda_0 in the first segmented lambda list.
            if not ifuse_initial_rst:
                start_win = self.get_segment_lambda_dir_list(segmented_lambda_list[0])[0]
                start_win_path = os.path.join(edge_path, start_win)
                if glob(os.path.join(start_win_path, 'prod*.rst')) or glob(os.path.join(start_win_path, 'alc_final_state.xml')):
                    self.set_input_state(edge_path, start_win)
                elif self.soft=='openmm':
                    print(f'You set ifuse_initial_rst as False, but there is no prod*.rst or alc_final_state.xml in the {start_win} directory. Will use input_state in {self.input_file}.')
                else:
                    ifuse_initial_rst = True
                    print(f"Warning: You set ifuse_initial_rst as False, but there is no prod*.rst or alc_final_state.xml in the {start_win} directory. The value of ifuse_initial_rst is set to be True automatically.")
        elif rerun_start_win in self.all_lambda_directories: ### start md simulation from the specific lambda.
            simu_lambda_dir_list = []
            for i in range(len(segmented_lambda_list)):
                part_lambda_dir = self.get_segment_lambda_dir_list(segmented_lambda_list[i])
                simu_lambda_dir_list = simu_lambda_dir_list + part_lambda_dir
                if rerun_start_win in part_lambda_dir:
                    segmented_lambda_list = segmented_lambda_list[i:]
                    break
            if rerun_start_win not in simu_lambda_dir_list:
                raise ValueError("The start lambda is in the mbar lambdas list but not in simulation lambdas list. Check the input file.")
            elif not ifuse_initial_rst:
                start_win_index = simu_lambda_dir_list.index(rerun_start_win)
                prev_coor_win = simu_lambda_dir_list[max(0, start_win_index+(int(ifuse_current_win_coor)-1))]
                prev_coor_win_path = os.path.join(edge_path, prev_coor_win)
                if glob(os.path.join(prev_coor_win_path, 'prod*.rst')) or glob(os.path.join(prev_coor_win_path, 'alc_final_state.xml')):
                    self.set_input_state(edge_path, prev_coor_win)
                else:
                    raise ValueError(f"You set ifuse_initial_rst as False, but there is no prod*.rst or alc_final_state.xml in {prev_coor_win} directory. Please check the setting file.")
        else:
            raise ValueError("The start lambda is not in the input mbar lambda setting file. Please check the rerun_start_win or the lambda setting file.")
        return segmented_lambda_list

    def md_control(self, mdobj, prod_md_time, segment_lambda_step, num_neighbors_state,
                    min_reitera_times, max_reitera_times, error_max_edge,
                    ana_proportion, compare_simu_nums, time_serials_num,
                    ifrun_preliminary_md, ifuse_initial_rst,
                    rerun_start_win, ifuse_current_win_coor, ifrun_turnaround, ifoverwrite):
        '''
        Parameter
        -------
        -prod_md_time: float, with unit of ps
        -mdobj: object of AmberMD or OpenmmMD
        -num_neighbors_state: int
            The number of neighboring lambda states before and after the specicifc lambda state. Default is 5.
        -max_reitera_times: int
            The maximum number of reiteration simulations, determining when to terminate the repetition and start the next segment. Default is 10.
        -ifoverwrite: bool
            If need overwrite, the exist lambda directory will be deleted and then overwirted.
        '''
        edge_path = os.getcwd()
        frames_each_simu = self.cal_each_simulation_frames_num(self.soft, self.input_file, prod_md_time)
        frames_max = int(500/prod_md_time*frames_each_simu)
        segmented_lambda_list = self.get_segmented_lambda_list(segment_lambda_step)
        if ifoverwrite:
            self.overwrite_rm_files(edge_path, self.all_lambda_directories)
        else:
            segmented_lambda_list = self.new_continuous_run_setting(segmented_lambda_list, edge_path, rerun_start_win, ifuse_initial_rst, ifuse_current_win_coor)
        self.init_check_point_file(edge_path, frames_each_simu, frames_max)
        for segment_lambda in segmented_lambda_list:
            ifnext_segment = False
            anlysisobj = SegmentConvergenceAnalysis(segment_lambda, frames_each_simu, error_max_edge, ifrun_turnaround, ifoverwrite)
            ana_proportion = np.around(1-1/(2+np.exp(-0.15*(anlysisobj.count-30))), decimals=4)
            segment_lambda_dir_list = self.get_segment_lambda_dir_list(segment_lambda)
            if not ifoverwrite and rerun_start_win in segment_lambda_dir_list:
                start_num = segment_lambda_dir_list.index(rerun_start_win)
                data_dict = self.generate_data_dict_by_checkfile_frames(edge_path, segment_lambda_dir_list, ana_proportion, frames_max)
            else:
                start_num = 0
                data_dict = {}
            while not ifnext_segment:
                ana_proportion = np.around(1-1/(2+np.exp(-0.15*(anlysisobj.count-30))), decimals=4)
                segment_lambda_num = len(segment_lambda["lambda_restraints"]) # assume that the three types of the lambda with the same length.
                for num in range(start_num, segment_lambda_num):
                    specific_lambda_dict = {
                        "lambda_restraints": [segment_lambda["lambda_restraints"][num]],
                        "lambda_electrostatics": [segment_lambda["lambda_electrostatics"][num]],
                        "lambda_sterics": [segment_lambda["lambda_sterics"][num]]
                    }
                    lambda_dir = f'{segment_lambda["lambda_restraints"][num]}_{segment_lambda["lambda_electrostatics"][num]}_{segment_lambda["lambda_sterics"][num]}'
                    # print(f'ifrun_preliminary_md:{ifrun_preliminary_md}, ifuse_initial_rst:{ifuse_initial_rst}')
                    specific_lambda_dict, cal_ene_lambda_dict = self.find_neighboring_vectors(specific_lambda_dict, self.all_mbar_lambda_dict, num_neighbors_state)
                    # print(f'simulation lambda:{specific_lambda_dict}\nmbar_lambda:{cal_ene_lambda_dict}')
                    mdobj.run_md(specific_lambda_dict, cal_ene_lambda_dict, prod_md_time, None, ifrun_preliminary_md, ifuse_initial_rst)
                    data_dict = self.update_data_dict(edge_path, data_dict, lambda_dir, ana_proportion, frames_max)
                    ifrun_preliminary_md, ifuse_initial_rst = False, False
                start_num = 0
                #anlysis
                segment_lambda, ifnext_segment  = anlysisobj.convergence_aly(data_dict, segment_lambda, min_reitera_times, max_reitera_times, compare_simu_nums, time_serials_num)
            run_segment_lambda_dir = self.get_segment_lambda_dir_list(segment_lambda)
            if self.soft == 'amber':
                for cur_lambda in run_segment_lambda_dir:
                    lambda_path = os.path.join(edge_path, str(cur_lambda))
                    for file in os.listdir(lambda_path):
                        if file.startswith('prod') and not (file.endswith('.rst') or file.endswith('.out')):
                            os.remove(f"{lambda_path}/{file}")

    def run(self, segment_lambda_step=0.1, num_neighbors_state=5,
                    min_reitera_times=2, max_reitera_times=50, error_max_edge=0.5,
                    ana_proportion=0.8, compare_simu_nums=3, time_serials_num=10,
                    ifrun_preliminary_md=False, ifuse_initial_rst=False,
                    rerun_start_win=None, ifuse_current_win_coor=False, ifrun_turnaround=True, ifoverwrite=False):
        '''
        -segment_lambda_step: float
            Specify the lambda step to seperate all mbar lambda dict into segments. Default is 0.1.
        '''
        if self.soft == 'amber':
            ambermd = AmberMD(self.input_file, self.complex_coor, self.complex_topo)
            self.md_control(ambermd, self.custom_time, segment_lambda_step, num_neighbors_state,
                    min_reitera_times, max_reitera_times, error_max_edge,
                    ana_proportion, compare_simu_nums, time_serials_num,
                    ifrun_preliminary_md, ifuse_initial_rst,
                    rerun_start_win, ifuse_current_win_coor, ifrun_turnaround, ifoverwrite)

        if self.soft == 'openmm':
            openmmmd = OpenmmMD(self.input_file, self.complex_coor, self.complex_topo)
            openmmmd.runobj.AlchemicalRun.cal_all_ene=True
            self.md_control(openmmmd, self.custom_time, segment_lambda_step, num_neighbors_state,
                    min_reitera_times, max_reitera_times, error_max_edge,
                    ana_proportion, compare_simu_nums, time_serials_num,
                    ifrun_preliminary_md, ifuse_initial_rst,
                    rerun_start_win, ifuse_current_win_coor, ifrun_turnaround, ifoverwrite)
            


    def set_input_state(self, edge_path, lambda_dir):
        if self.soft == 'amber':
            ### copy the prev rst to the edge path
            lambda_path = os.path.join(edge_path, lambda_dir)
            prev_rst_files = sorted([file for file in os.listdir(str(lambda_path)) if file.startswith('prod') and file.endswith('.rst')])
            final_coor = os.path.join(lambda_path, prev_rst_files[-1])
            self.complex_coor = final_coor
            target_coor_path = os.path.join(edge_path, 'prev.rst')
            shutil.copy(final_coor, target_coor_path)
        elif self.soft == 'openmm':
            ### set the inpute state to previous alc_final_state.xml in the input file.
            lambda_path = os.path.join(edge_path, lambda_dir)
            prev_xml_file_path = os.path.join(lambda_path, 'alc_final_state.xml') ## inpute state file path string
            with open(self.input_file, 'r') as f:
                content = f.read()
                new_content = ""
                iffind = False
                for line in content.splitlines():
                    if line.find("input_state") != -1:
                        iffind = True
                        new_line = line.replace(line, f"input_state = {prev_xml_file_path}")
                    elif line.find("[restraint]") != -1 and iffind==False:
                        new_line = f"input_state = {prev_xml_file_path}\n\n{line}"
                    else:
                        new_line = line
                    new_content += new_line + "\n"
            with open(self.input_file, 'w') as fw:
                fw.write(new_content)
            
