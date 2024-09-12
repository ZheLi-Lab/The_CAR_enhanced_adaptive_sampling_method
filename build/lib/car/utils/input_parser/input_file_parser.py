import types

'''
###Input file format:
[normal_alc_md]
simulation_software     = amber # should be 'amber' or 'openmm'
coordinate_file         = cM2A.prmcrd # could be '.prmcrd' or '.rst7'
topology_file           = cM2A.prmtop
prod_md_time            = 10 # simulation time of each state with unit in ps
mbar_lambda_dict_file   = lambdas.json
input_file              = input.json # could be 'input.json' for amber, 'input.txt' or 'input.temp' for openmm

[segmented_md_control]
segment_lambda_step     = 0.1
num_neighbors_state     = 5
min_reitera_times       = 2
max_reitera_times       = 50
error_max_lambda_0to1   = 0.1
ifrun_preliminary_md    = False
ifuse_initial_rst       = False
ifoverwrite             = False

'''

class InputParser:
    def __init__(self, filename):
        # Define the expected keys for each section and their default values
        self.def_sections()

        # Initialize variables to hold the parsed data
        self.data = {}
        for section, options in self.sections.items():
            self.data[section] = {}
            for option, default_value in options.items():
                self.data[section][option] = default_value

        self.current_section = None

        # Parse the input file
        with open(filename, 'r') as f:
            for line in f:
                # Remove inline comments
                line = line.split('#')[0].strip()
                if not line:  # Skip empty lines and comments
                    continue
                if line.startswith('[') and line.endswith(']'):
                    self.current_section = line[1:-1]
                elif self.current_section is not None:
                    for item in line.split(','):
                        key, value = item.split('=')
                        key = key.strip()
                        value = value.strip()
                        if key in self.sections[self.current_section]:
                            if value.isdigit():
                                self.data[self.current_section][key] = int(value)
                            elif value.replace('.', '').isdigit():
                                self.data[self.current_section][key] = float(value)
                            elif value.lower() == 'true':
                                self.data[self.current_section][key] = True
                            elif value.lower() == 'false':
                                self.data[self.current_section][key] = False
                            elif value.lower() == 'none':
                                self.data[self.current_section][key] = None
                            else:
                                self.data[self.current_section][key] = value.strip("'").strip('"')


        # Define the section-related methods dynamically
        for section in self.sections:
            def get_section_data(self, section=section):
                return self.data.get(section, {})
            setattr(self, f'get_{section}', types.MethodType(get_section_data, self))

    def def_sections(self):
        # Define the expected keys for each section and their default values
        self.sections = {
            'normal_alc_md': {
                'simulation_software': 'amber', # openmm, amber
                'coordinate_file': 'cM2A.prmcrd', # could be '.prmcrd' or '.rst7'
                'topology_file': 'cM2A.prmtop',
                'prod_md_time': 10, # simulation time of each state with unit in ps
                'mbar_lambda_dict_file': 'lambdas.json',
                'input_file': 'input.json', # could be 'input.json' for amber, 'input.txt' or 'input.temp' for openmm
                'segmented_md_control': False,
                'if_post_analyze_segment': False
                },
            'segmented_md_control': {
                'segment_lambda_step': 0.1,
                'num_neighbors_state': 5,
                'min_reitera_times': 2,
                'max_reitera_times': 50,
                'min_converge_score': 2,
                'error_max_lambda_0to1': 0.1,
                'analysis_data_proportion': 0.8,
                'compare_simu_nums': 4,
                'time_serials_num': 10,
                'ifrun_preliminary_md': False,
                'ifuse_initial_rst': False,
                'rerun_start_win': None,
                'ifuse_current_win_coor': False,
                'ifrun_turnaround_points': True,
                'ifplot_heatmap': False, ###not use in new version
                'ifoverwrite': False
                },
            'segment_post_analysis': {
                'analysis_plan': 'moving',
                'divided_ratio': 0.05,
                'ifplot_heatmap': False,
                'moving_width_times': 5,
                'ifplot_time_serials': False,
                'plot_plan': 'forward|reverse|moving'
            }
        }



if __name__ == '__main__':
    def main():
        filename = 'input.txt'
        parser = InputParser(filename)
        normal_alc_md_setting = parser.get_normal_alc_md()
        print(normal_alc_md_setting)
        if normal_alc_md_setting['segmented_md_control'] == True:
            segmented_md_control_settings = parser.get_segmented_md_control()
            print(segmented_md_control_settings)
        if normal_alc_md_setting['if_post_analyze_segment'] == True:
            post_analyze_segment_setting = parser.get_segment_post_analysis()
            print(post_analyze_segment_setting)
    main()

