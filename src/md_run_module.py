import subprocess
import json
import math
import copy
import shutil
from glob import glob
from copy import deepcopy
import os
import pandas as pd
from contextlib import contextmanager
from .lambda_related_data_convert import from_mbar_lambda_dict_to_lambda_lst, from_lambda_dict_to_lambda_float
from .out_parser import read_amber_out
from .out_parser import read_openmm_out

@contextmanager
def working_directory(path):
    """
    Context manager to temporarily change the working directory.

    Args:
    - path (str): The path of the directory to change to.

    Yields:
    - None

    After the block inside 'with' is executed, it changes back to the original directory.
    """
    current_directory = os.getcwd()  # Get the current working directory
    try:
        os.chdir(path)  # Change to the specified directory
        yield
    finally:
        os.chdir(current_directory)  # Change back to the original directory

#TODO: amber out文件到统一的openmm csv； 帮助文档

# TODO: self.run_md()



def amber_df_to_openmm_df(amberdf, state_lambda):
    '''convert one state's dataframe of amber to the dataframe of openmm
    
    Parameters
    ----------
    - amberdf(pd.DataFrame): 
        one state's dataframe of amber
    - state_lambda(tuple): 
        The lambda values(float) of restraints(element 1), electrostatics(element 2) and sterics(element 3).
    
    Return
    ------
    - openmmdf(pd.DataFrame): 
        one state's dataframe of openmm
    '''
    openmmdf = deepcopy(amberdf)
    ori_multi_index = deepcopy(amberdf.index)
    ori_columns = deepcopy(amberdf.columns)
    new_multi_index_name = ['times(ps)', 'lambda_restraints', 'lambda_electrostatics', 'lambda_sterics']
    times_lambda_tuples = [(one_ori_multi_index[0],)+state_lambda for one_ori_multi_index in ori_multi_index]
    muti_idx = pd.MultiIndex.from_tuples(times_lambda_tuples, names=new_multi_index_name)
    new_columns = [(0.0, float(ori_column), float(ori_column)) for ori_column in ori_columns]
    openmmdf.columns = new_columns
    openmmdf.index = muti_idx
    return openmmdf

class AmberMD:
    '''This class is used to run amber alchemical md simulation.
    '''
    def __init__(self, input_file, complex_coor, complex_topo, custom_time=0):
        '''
        Parameters
        ----------
        - input_file: 
            str, the path of input file.
        - complex_coor: 
            str, the path of complex coordinate file.
        - complex_topo: 
            str, the path of complex topology file.   
        - custom_time: 
            int, the user-specific time of md simulation, unit is ps.
        '''
        self.complex_coor = complex_coor
        self.complex_topo = complex_topo
        cur_path = os.getcwd()
        try:
            shutil.copy(self.complex_coor, cur_path)
            shutil.copy(self.complex_topo, cur_path)
        except:
            pass
        self.custom_time = custom_time
        with open(input_file, encoding="utf-8") as f:
            self.input_dict = json.load(f)

    def read_checking_point_json(self, lambda_value):
        '''Read the checking_point.json file to get the lambda value and the simulation time of the previous md simulation.

        Parameters
        ----------
        - lambda_value (float): 
            The lambda value of the current md simulation.
        
        Return
        ------
        - lambda_frame_dict(dict): 
            The lambda value and the simulation time of the previous md simulation.
        '''
        try:
            with open('check_point.json', encoding="utf-8") as f:
                lambda_frame_dict = json.load(f)
        except FileNotFoundError:
            lambda_frame_dict = {}
        except json.JSONDecodeError:
            lambda_frame_dict = {}
        if str(lambda_value) not in lambda_frame_dict.keys():
            lambda_frame_dict[str(lambda_value)] = {}
        return lambda_frame_dict

    def updata_md_input(self, input_dict, lambda_path, lambda_value, mbar_lambda, simulation_steps, nth, ifrun_preliminary_nvt_npt=False, ifuse_initial_rst=False):
        '''Update the input file of md simulation.
        Parameters
        ----------
        - input_dict (dict): 
            The input data dictionary of md simulation.
        - lambda_path (str): 
            The absolute path of the alchemical MD with a specific lambda.
        - lambda_value (float): 
            The lambda value of the current md simulation.
        - mbar_lambda (list): 
            The lambda values of states whose energy need to be calculated.
        - simulation_steps (int): 
            The number of steps of md simulation.
        - nth (int): 
            The number of the current md simulation times. (start from 0)
        - ifrun_preliminary_nvt_npt (bool): 
            Whether to run the preliminary nvt and npt simulation.
        - ifuse_initial_rst (bool):
            Whether to use the initial rst file. This option comes with the ifrun_preliminary_nvt_npt option. 
            If the ifrun_preliminary_nvt_npt is True and the ifuse_initial_rst is True, we will use the ../*.prmcrd(usually is the protein.prmcrd) as the initial conformation for the preliminary nvt and npt simulation. 
            If the ifrun_preliminary_nvt_npt is True and the ifuse_initial_rst is False, will use the ../prev_cen.rst(which generated by the cpptraj_center.in) as the initial conformation for the preliminary nvt and npt simulation. 
            Under the situation above (when ifrun_preliminary_nvt_npt is True), we will use the equi-5.rst as the initial conformation for the production MD.
            If the ifrun_preliminary_nvt_npt is False, we will use the ../prev_cen.rst(which generated by the cpptraj_center.in) as the initial conformation for the production MD.

        Return
        ------
        - input_dict (dict): 
            The updated input data dictionary of md simulation.
        '''
        mbar_lambda_str = str(mbar_lambda).replace('[','').replace(']','')
        # edge_path = os.path.dirname(lambda_path)
        input_dict = copy.deepcopy(input_dict)
        for file_name, file_content in input_dict.items():
            if ifrun_preliminary_nvt_npt:
                if file_name == 'preliminary_md.sh':
                    strs = file_content[7]
                    if ifuse_initial_rst:
                        strs = strs.replace('_PRELIMINARY_MD_IN_RST_', '$PRMCRD')
                    else:
                        strs = strs.replace('_PRELIMINARY_MD_IN_RST_', 'prev_cen.rst')
                    file_content[7] = strs
                elif file_name == 'submit.sh':
                    strs = file_content[6]
                    strs = strs.replace('_PRODIN_RST_', 'equi-5.rst')
                    file_content[6] = strs
                ins_need_modify_lambda_info = ['min.in', 'heat-cpu.in', 'heat.in', 'equi-pre.in', 'equi.in', 'prod.in']
                files_to_write = ['min.in', 'heat-cpu.in', 'heat.in', 'equi-pre.in', 'equi.in', 'preliminary_md.sh', f'prod{nth}.in', f'submit{nth}.sh', 'cpptraj_center.in', ]
            else:
                if file_name == 'submit.sh':
                    strs = file_content[6]
                    if ifuse_initial_rst:
                        strs = strs.replace('_PRODIN_RST_', '../cM2A.prmcrd')
                        # print(f'md_run use initial rst')
                    else:
                        strs = strs.replace('_PRODIN_RST_', 'prev_cen.rst')
                    file_content[6] = strs
                ins_need_modify_lambda_info = ['prod.in']
                files_to_write = [f'prod{nth}.in', f'submit{nth}.sh', 'cpptraj_center.in']
            if file_name == 'submit.sh':
                strs = file_content[-1]
                strs = strs.replace('prod.in', f'prod{nth}.in')
                strs = strs.replace('prod.out', f'prod{nth}.out')
                strs = strs.replace('prod.rst', f'prod{nth}.rst')
                strs = strs.replace('prod.mdinfo', f'prod{nth}.mdinfo')
                strs = strs.replace('prod.netcdf', f'prod{nth}.netcdf')
                file_content.pop()
                file_content.append(strs)
            elif file_name == 'prod.in':
                file_content.insert(-2, f'  nstlim = {simulation_steps},\n')
            if file_name in ins_need_modify_lambda_info:
                file_content.insert(-2, f'  clambda = {lambda_value},\n')
                file_content.insert(-2, f'  mbar_states = {len(mbar_lambda)},\n')
                file_content.insert(-2, f'  mbar_lambda = {mbar_lambda_str},\n')
            files_with_different_name = ['prod.in', 'submit.sh']
            if file_name in files_with_different_name:
                file_name = file_name.split('.', maxsplit=1)[0] + str(nth) + '.' + file_name.split('.')[1]
            if file_name in files_to_write:
                # print(file_name)
                with open(f'{lambda_path}/{file_name}', 'w', encoding="utf-8") as f:
                    for line in file_content:
                        f.write(line)

        return input_dict

    def submit_md(self, lambda_path, nth, ifcpptraj_center=False, ifrun_preliminary_nvt_npt=False):
        '''
        Parameters
        ----------
        - lambda_path (str): 
            The absolute path of the alchemical MD with a specific lambda.
        - nth (int): 
            The number of the current md simulation times. (start from 0)
        - ifcpptraj_center (bool):
            If run the cpptraj to center the previous rst file.
        - ifrun_prelimnary_nvt_npt (bool):
            If run the minimization, heat(nvt), equilibration(npt) MD first, then run the production MD
        '''
        with working_directory(lambda_path):
            if ifcpptraj_center:
                cpptraj_result = subprocess.run('cpptraj -i cpptraj_center.in > /dev/null 2>&1', shell=True, text=True, check=True)
            if ifrun_preliminary_nvt_npt:
                run_command = f'sh preliminary_md.sh; sh submit{nth}.sh'
            else:
                run_command = f'sh submit{nth}.sh'
            run_result = subprocess.run(run_command, shell=True, text=True, check=True)

    def run_md(self, lambda_dict, mbar_lambda_dict, least_time, lambda_lst = None, ifrun_preliminary_md=False, ifuse_initial_rst=False):
        '''
        Parameters
        ----------
        - lambda_dict (dict): The lambda values of restraints, electrostatics and sterics. 
            The keys are 'lambda_restraints', 'lambda_electrostatics' and 'lambda_sterics'. 
            The values are three respective lambda list with one value, whose datatype is [float, ].
        - mbar_lambda_dict: The lambda values of states whose energy need to be calculated.
            The keys are 'lambda_restraints', 'lambda_electrostatics', and 'lambda_sterics'.
            The values are three respective lambda lists.
        - least_time (int): The user-specific least time of md simulation,unit is ps.
            Usually, the least time is used as the actually simulation time.
        - lambda_lst (list): The list that usually records the previous lambda value (the first element), 
            the current lambda value (the second element) and the next lambda value (the last element).
            The datatype of the elements is float. The default value is None.  
            (TODO: the datatype of lambda_lst should be dict, if the fast_alchem_control workflow is used for the ABFE calculation).
        - ifrun_preliminary_md (bool): 
            Whether to run the preliminary md for this lambda window. Default is False.
        - ifuse_initial_rst (bool): 
            Whether to use the initial rst file. This option comes with the ifrun_preliminary_md option. Default is False.
        '''
        cur_lambda_value = from_lambda_dict_to_lambda_float(lambda_dict, mbar_lambda_dict)
        cur_lambda = f"{lambda_dict['lambda_restraints'][0]}_{lambda_dict['lambda_electrostatics'][0]}_{lambda_dict['lambda_sterics'][0]}"
        mbar_lambda_lst = from_mbar_lambda_dict_to_lambda_lst(mbar_lambda_dict)[0]
        if lambda_lst is not None:
            lambda_step = abs(round(lambda_lst[-1] - lambda_lst[-2], 3)) #TODO: if the lambda_lst is the dict, the lambda_step should be calculated by the dict.
            md_time = least_time if self.custom_time*1000*lambda_step <= least_time else self.custom_time*1000*lambda_step
            if lambda_lst[-2] == 0 and md_time < 100:
                md_time = 100
        else:
            md_time = least_time
        lambda_frame_dict = self.read_checking_point_json(cur_lambda_value)
        print(f"md_lambda: {cur_lambda_value}, simul_time: {md_time}ps")
        steps = math.floor(md_time/0.002) # The unit of timestep is fs. The default value of timestep is 2fs. (May could be changed by passing a variable to the input file in the furture.)
        edge_path = os.getcwd()
        lambda_path = os.path.join(edge_path, str(cur_lambda))
        input_dict = copy.deepcopy(self.input_dict)

        if os.path.exists(str(cur_lambda)):
            out_files = [
                file for file in os.listdir(lambda_path)
                if file.startswith('prod') and file.endswith('.out')
                ]
            num = len(out_files)
        else:
            os.makedirs(str(cur_lambda))
            num=0
        self.updata_md_input(input_dict, lambda_path, cur_lambda_value, mbar_lambda_lst, steps, num,
                                ifrun_preliminary_md, ifuse_initial_rst)
        ifcpptraj_center = not ifuse_initial_rst
        self.submit_md(lambda_path, num, ifcpptraj_center=ifcpptraj_center,
                        ifrun_preliminary_nvt_npt=ifrun_preliminary_md)
        if num != 0:
            lambda_frame_dict[str(cur_lambda_value)]['least_time'] += least_time
            lambda_frame_dict[str(cur_lambda_value)]['md_time'] += md_time
        else:
            lambda_frame_dict[str(cur_lambda_value)]['least_time'] = least_time
            lambda_frame_dict[str(cur_lambda_value)]['md_time'] = md_time
        with open('check_point.json', 'w', encoding="utf-8") as f:
            json.dump(lambda_frame_dict ,f)
        prev_rst_files = sorted([file for file in os.listdir(str(lambda_path)) if file.startswith('prod') and file.endswith('.rst')])
        final_coor = prev_rst_files[-1]
        self.complex_coor = f'{lambda_path}/{final_coor}'
        target_coor_path = os.path.join(edge_path, 'prev.rst')
        shutil.copy(self.complex_coor, target_coor_path)
        self.out_to_csv(lambda_path, cur_lambda, num)


    def out_to_csv(self, lambda_path, cur_lambda, num=''):
        '''Format the output file of amber md simulation to csv file.
        
        Parameters
        ----------
        - lambda_path (str): to specify the absolute path of the alchemical MD with a specific lambda
        - cur_lambda (str): f'{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}' 
                The lambda values(float) of restraints, electrostatics and sterics. 
        
        Recording file
        --------------
        - f'{lambda_path}/lambda{cur_lambda}.csv': 
            The csv file of the alchemical MD with a specific lambda, containing the data of the current MD simulation and the previous MD simulation.
        '''
        csv_path = os.path.join(lambda_path, f'lambda{cur_lambda}.csv')
        print(f"now extract num: {num}, lambda_path is {lambda_path}")
        cur_df = read_amber_out.ReadProdOut(lambda_path, f'prod{num}', 'out').extract_data()
        cur_df = amber_df_to_openmm_df(cur_df, tuple(cur_lambda.split('_')))
        if os.path.exists(csv_path):
            prev_df = read_openmm_out.READ_PROD_OUT(lambda_path, f'lambda{cur_lambda}', 'csv').extract_data(1, [0,1,2,3], '|',False)
            # print(f'--------prev_df is-----------\n{prev_df}')
            # print(f'eeeeeee ------csv_path:\n{csv_path}')
            new_df = pd.concat([prev_df, cur_df])
            lambda_df = new_df
        else:
            # print(f'ffffffffff ------csv_path:\n{csv_path}')
            lambda_df = cur_df
        lambda_df.to_csv(f'{lambda_path}/lambda{cur_lambda}.csv', '|')
        return lambda_df




class OpenmmMD:
    '''This class is used to run openmm alchemical md simulation.
    '''
    def __init__(self, input_file, complex_coor, complex_topo, ):
        '''
        Parameters
        ----------
        - input_file: 
            str, the path of input file.
        - complex_coor: 
            str, the path of complex coordinate file.
        - complex_topo: 
            str, the path of complex topology file.
        - custom_time: 
            int, the user-specific time of md simulation, unit is ps.
        '''
        from .Alchemd.utils.file_parser import InputParser
        from .Alchemd.utils.run import RunAlchemdSimulation
        self.input_data = InputParser(input_file)
        self.complex_coor = complex_coor
        self.complex_topo = complex_topo
        self.runobj = RunAlchemdSimulation(self.input_data, self.complex_coor, self.complex_topo)   #拓扑和坐标最好给绝对路径；

    def run_md(self, lambda_dict, mbar_lambda_dict, least_time, lambda_lst = None, ifrun_preliminary_md=False, ifuse_initial_rst=False):
        '''
        Parameters
        ----------
        - lambda_dict (dict): The lambda values of restraints, electrostatics and sterics. 
            The keys are 'lambda_restraints', 'lambda_electrostatics' and 'lambda_sterics'. 
            The values are three respective lambda list with one value, whose datatype is [float, ].
        - mbar_lambda_dictThe lambda values of states whose energy need to be calculated.
            The keys are 'lambda_restraints', 'lambda_electrostatics', and 'lambda_sterics'.
            The values are three respective lambda lists.
        - least_time (int): The user-specific least time of md simulation,unit is ps.
            Usually, the least time is used as the actually simulation time.
        - lambda_lst (list): The list that usually records the previous lambda value (the first element), 
            the current lambda value (the second element) and the next lambda value (the last element).
            The datatype of the elements is float. The default value is None. Will not use in the Openmm run_md so far. 
            (TODO: the datatype of lambda_lst should be dict, if the fast_alchem_control workflow is used for the ABFE calculation).
        - ifrun_preliminary_md (bool): 
            Whether to run the preliminary md for this lambda window. Default is False.
        - ifuse_initial_rst (bool): 
            Whether to use the initial rst file. This option comes with the ifrun_preliminary_md option. Default is False.
        '''
        alchemical_data = self.input_data.get_alchemical() 
        md_time = least_time
        lambda_restraints = lambda_dict['lambda_restraints'][0]
        lambda_electrostatics = lambda_dict['lambda_electrostatics'][0]
        lambda_sterics = lambda_dict['lambda_sterics'][0]
        check_point_lambda = from_lambda_dict_to_lambda_float(lambda_dict, mbar_lambda_dict)
        try:
            with open('check_point.json', encoding="utf-8") as f:
                lambda_frame_dict = json.load(f)
        except FileNotFoundError:
            lambda_frame_dict = {}
        except json.JSONDecodeError:
            lambda_frame_dict = {}
        lambda_frame_dict[check_point_lambda] = {}
        lambda_frame_dict[check_point_lambda]['least_time'] = least_time
        lambda_frame_dict[check_point_lambda]['md_time'] = md_time # ps

        nsteps = alchemical_data['nsteps']
        timestep_in_fs = alchemical_data['timestep']
        niterations = math.floor(md_time/(nsteps*timestep_in_fs*0.001))

        pwd_path = os.getcwd()
        if os.path.exists(f'{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}'):
            pass
        else:
            os.makedirs(f'{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}')
        os.chdir(f'{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}')
        lambdas_json_data = {}
        lambdas_json_data['cal_mbar_lambda'] = mbar_lambda_dict
        lambdas_json_data['simu_lambda'] = lambda_dict
        with open('lambdas.json','w', encoding="utf-8") as f:
            json.dump(lambdas_json_data, f) # this lambdas.json file is just used for recording.
        ## Update the input file of md simulation.
        self.runobj.AlchemicalRun.lambdas_group = mbar_lambda_dict
        self.runobj.AlchemicalRun.simulation_lambdas = lambda_dict
        self.runobj.AlchemicalRun.niterations = niterations

        if ifrun_preliminary_md:
            self.runobj.AlchemicalRun.if_min_heat_density = True
        else:
            self.runobj.AlchemicalRun.if_min_heat_density = False
        if ifuse_initial_rst:
            self.runobj.AlchemicalRun.state = None
        else:
            print('Use the state updated by the previous md simulation or the initial state file assigned by the "input_state" option of the input.txt file')
        ## Run the md simulation.
        print('------------------------------')
        print(os.getcwd())
        self.runobj.run() # will generate state_s*.xml, state_s*.csv, alc_final_state.xml; alc_final_state.xml as same as state_s*.xml 
        for xml in glob('state_*.xml'):
            os.remove(xml)
        os.chdir(pwd_path)

        with open('check_point.json','w', encoding="utf-8") as f:
            json.dump(lambda_frame_dict ,f)

        lambda_path = os.path.join(pwd_path, f'{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}')
        alchemical_data['input_state'] = os.path.join(lambda_path, 'alc_final_state.xml')
        # print(self.runobj.AlchemicalRun.__dir__())
        # self.runobj.AlchemicalRun.alchem_md.loadState(alchemical_data['input_state']) # not need because the AlchemicalRun will update the state.
        self.out_to_csv(lambda_path, f"{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}")
        if os.path.exists(f'{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}'):
            os.chdir(f'{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}')
            for statecsv in glob('state_*.csv'):
                os.remove(statecsv)
            os.chdir(pwd_path)

    def out_to_csv(self, lambda_path, cur_lambda, ):
        '''Format the output file of amber md simulation to csv file.
        
        Parameters
        ----------
        - lambda_path (str): 
            to specify the absolute path of the alchemical MD with a specific lambda
        - cur_lambda (str): f'{lambda_restraints}_{lambda_electrostatics}_{lambda_sterics}' 
                The lambda values(float) of restraints, electrostatics and sterics. 
        
        Recording file
        --------------
        - f'{lambda_path}/lambda{cur_lambda}.csv': 
            The csv file of the alchemical MD with a specific lambda, containing the data of the current MD simulation and the previous MD simulation.
        '''
        csv_path = os.path.join(lambda_path, f'lambda{cur_lambda}.csv')
        read_out_obj = read_openmm_out.READ_PROD_OUT(lambda_path, 'state_*', 'csv')
        cur_df = read_out_obj.extract_data(1, read_out_obj.index_col, read_out_obj.delimiter, False)
        # cur_df = read_openmm_out.READ_PROD_OUT(lambda_path, 'state_*', 'csv').extract_data(1, [0,1,2,3], '|',False)
        if os.path.exists(csv_path):
            read_out_obj = read_openmm_out.READ_PROD_OUT(lambda_path, f'lambda{cur_lambda}', 'csv')
            prev_df = read_out_obj.extract_data(1, read_out_obj.index_col, read_out_obj.delimiter, False)
            # prev_df = read_openmm_out.READ_PROD_OUT(lambda_path, f'lambda{cur_lambda}', 'csv').extract_data(1, [0,1,2,3], '|',False)
            new_df = pd.concat([cur_df, prev_df])
            lambda_df = new_df
        else:
            lambda_df = cur_df
        lambda_df.to_csv(f'{lambda_path}/lambda{cur_lambda}.csv', '|')