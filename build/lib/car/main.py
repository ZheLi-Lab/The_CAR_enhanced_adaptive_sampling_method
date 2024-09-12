from optparse import OptionParser
from .utils.input_parser.input_file_parser import InputParser as InputParser
from .utils.car_converge_control import AlchemdControl

class optParser():  
    def __init__(self, fakeArgs):
        parser = OptionParser()
        parser.add_option('-i', '--input', dest='input', help="The file name of input, which recording the analysis settings. Default: 'segment_run_input.txt'", default='car_run_input.txt')
        if fakeArgs:
            self.option, self.args = parser.parse_args(fakeArgs)
        else:
            self.option, self.args = parser.parse_args()

def main():
    opts = optParser('')
    #####################################################################################################
    # keep for jupyter notebook test
    # fakeArgs = '-i input.txt'
    # opts = optParser(fakeArgs.strip().split())
    #####################################################################################################
    input_parser = InputParser(opts.option.input)
    normal_alc_md_settings = input_parser.get_normal_alc_md()
    segmented_md_control_settings = input_parser.get_segmented_md_control()
    # print(normal_alc_md_settings)
    ## normal_alc_md
    simulation_pack = normal_alc_md_settings['simulation_software']
    complex_coor = normal_alc_md_settings['coordinate_file']
    complex_topo = normal_alc_md_settings['topology_file']
    prod_md_time = normal_alc_md_settings['prod_md_time']
    lambda_setting_json_file = normal_alc_md_settings['mbar_lambda_dict_file']
    input_file = normal_alc_md_settings['input_file']
    
    ## segmented_md_control
    segment_lambda_step = segmented_md_control_settings['segment_lambda_step']
    num_mabr_neighbor_states = segmented_md_control_settings['num_neighbors_state']
    min_reitera_times = segmented_md_control_settings['min_reitera_times']
    max_reitera_times = segmented_md_control_settings['max_reitera_times']
    if min_reitera_times>max_reitera_times:
        raise ValueError("The min_reitera_times is larger than the max_reitera_times. Please check the input file setting.")
    error_max_edge = segmented_md_control_settings['error_max_lambda_0to1']
    analysis_data_proportion = segmented_md_control_settings['analysis_data_proportion']
    compare_simu_nums = segmented_md_control_settings['compare_simu_nums']
    time_serials_num = segmented_md_control_settings['time_serials_num']
    ifrun_preliminary_md = segmented_md_control_settings['ifrun_preliminary_md']
    ifuse_initial_rst = segmented_md_control_settings['ifuse_initial_rst']
    rerun_start_win = segmented_md_control_settings['rerun_start_win']
    ifuse_current_win_coor = segmented_md_control_settings['ifuse_current_win_coor']
    ifrun_turnaround_points = segmented_md_control_settings['ifrun_turnaround_points']
    ifoverwrite = segmented_md_control_settings['ifoverwrite']
    print(input_file, complex_coor, complex_topo, simulation_pack, prod_md_time, lambda_setting_json_file)
    # print(f'ifuse_initial_rst:{ifuse_initial_rst}')
    
    alchemd_obj = AlchemdControl(input_file, complex_coor, complex_topo, simulation_pack, prod_md_time, lambda_setting_json_file)
    alchemd_obj.run(segment_lambda_step, num_mabr_neighbor_states,
                    min_reitera_times, max_reitera_times, error_max_edge,
                    analysis_data_proportion, compare_simu_nums, time_serials_num,
                    ifrun_preliminary_md, ifuse_initial_rst,
                    rerun_start_win, ifuse_current_win_coor, ifrun_turnaround_points, ifoverwrite)

if __name__ == "__main__":
    main()
