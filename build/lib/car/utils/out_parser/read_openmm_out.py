from glob import glob
from functools import partial
from multiprocessing import Pool
import pandas as pd
import re
import os 
import concurrent.futures
from .re_func import extract_folder_names, extract_numbers_from_path, remove_g_and_get_numbers

# def get_csv_num(file_name, re_pattern):
#     return int(re.findall(re_pattern, file_name)[0])

# def get_csv_num_wrapper(file_name, re_pattern):
#     return get_csv_num(file_name, re_pattern=re_pattern)

def get_state_num_with_group_csv(file):
    file = remove_g_and_get_numbers(file)[0] 
    return extract_numbers_from_path(file)


class READ_PROD_OUT():
    def __init__(self, sys_path, file_prefix, file_suffix, ):
        self.folder_list = extract_folder_names(sys_path)
        self.file_pattern = f'{file_prefix}*{file_suffix}'
        
        try:
            self.grp_num = remove_g_and_get_numbers(file_prefix)[1][0]
        except:
            self.grp_num = None
        self.core_count = os.cpu_count()
        # print(self.grp_num)
        self.path_pattern = os.path.join(*self.folder_list, self.file_pattern)
        self.files = glob(r'{}'.format(self.path_pattern))
        self.u_nk_list = []
        self.index_col = []
        _tmp_csv_file = self.files[0]
        self.delimiter = '|' if '|' in open(_tmp_csv_file).read() else ','
        _tmp_df = pd.read_csv(_tmp_csv_file, self.delimiter)
        known_lambda_time_columns = ['times(ps)', 'lambda_restraints', 'lambda_electrostatics', 'lambda_sterics', 'lambda_electrostatics_env', 'lambda_electrostatics_chgprod_square']
        for idx in range(0, len(_tmp_df.columns)):
            if _tmp_df.columns[idx] in known_lambda_time_columns:
                self.index_col.append(idx)

        
    def read_file(self, filename, index_col, delimiter):
        # print(f'Processing the file {filename}')
        single_df = pd.read_csv(filename, index_col=index_col, delimiter=delimiter)
        old_columns = list(single_df.columns)        
        if isinstance(old_columns[0], str):   
            regex_pattern = r"\((.*?)\)"
            new_columns = [eval(f'({re.findall(regex_pattern, column)[0]})') for column in old_columns]
        elif isinstance(old_columns[0], list):
            new_columns = [tuple(column) for column in old_columns]
        elif isinstance(old_columns[0], tuple):
            new_columns = old_columns
        single_df.columns = new_columns
#         print(f'File {filename} processed')
        return single_df
    
    def read_file_wrapper(self, filename, index_col, delimiter):
        return self.read_file(filename, index_col=index_col, delimiter=delimiter)

    def extract_data(self, processes_num=4, index_col=[0,1,2,3], delimiter='|', if_sort=False):
        # print(self.path_pattern)
        # files = glob(r'{}'.format(self.path_pattern))
        files = self.files
        # print(files, len(files))
        if if_sort:
            if self.grp_num is not None:
                files = sorted(files, key=lambda x: get_state_num_with_group_csv(x))
            else:
                files = sorted(files, key=extract_numbers_from_path)
        # print(files)
        wrapped_read_file = partial(self.read_file_wrapper, index_col=index_col, delimiter=delimiter)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.core_count) as executor:
            # 使用 executor.map 代替 Pool.map
            self.u_nk_list = list(executor.map(wrapped_read_file, files))
          
        u_nk_pd = pd.concat(self.u_nk_list,)
        self.u_nk_pd = u_nk_pd
        return self.u_nk_pd