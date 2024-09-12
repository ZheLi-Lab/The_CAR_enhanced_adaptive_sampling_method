import decimal
from webbrowser import get
import numpy as np
import pandas as pd
from .plotting import PLOTTING
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from scipy.special import lambertw
try:
    from pymbar import BAR as BAR_
except:
    from pymbar import bar as bar # pymbar 4.0.2
    def BAR_ (ori_du_f, ori_du_b, method='self-consistent-iteration', maximum_iterations=5000, verbose=False):
        res_dict = bar(ori_du_f, ori_du_b, method='self-consistent-iteration', maximum_iterations=5000, verbose=False)
        return res_dict['Delta_f'], res_dict['dDelta_f']
import copy
from alchemlyb.estimators import MBAR
from .resample import RESAMPLE
import alchemlyb
from alchemlyb import visualisation
from alchemlyb.visualisation import plot_mbar_overlap_matrix 
import re



def tuple_to_str(tuple_):
    if type(tuple_) == tuple:        
        str_tmp = str(list(tuple_)).replace('[','(',1)
        str_tmp = str_tmp.replace(']',')',1)
        return str_tmp
#        return str(tuple_)
    else:
        return str(tuple_)

class FEP():
    def __init__(self, d_u):
        self.u_l=np.array(d_u)
        self.u_std=np.std(self.u_l)
        exponential=np.exp(-self.u_l)
        expave=exponential.mean()
        self.ene=-np.log(expave)


class CAL_FREE_ENERGY():
    
    def __init__(self, u_nk_pd, wanted_win_lst=False, scale_f=0.75, force_align_mbar_list=False):
        '''Calculate the free energy according to the multiple lambda internal energy under different force field.

        Parameters
        ----------
        u_nk_pd: pd.DataFrame
            The dataframe obtained by alchemlyb extraction from amber prod.out
        wanted_win_lst: List
            A list assigns simulation windows' data used in the following calculation. If it equals to False, all the windows will be used. Default: False
        scale_f: float
            To tail the percentage of the every dataframe data used for the calculation. Default: 0.75
        force_align_mbar_list: Bool
            To determine if let the mbar_lambda_dict equal to the simulation_lambda_dict. Default: False
     
        Key properties
        ----------
        self.mbar_lambda_dict: dict
            The keys of it are the sequential integers and its values are the calculated lambda for every simlation
        self.lambda_list: list
            The list used to group the u_nk_pd
        self.simulation_lambda: dict
            The keys of it are the sequential integers and its values are the actual lambda of the simulation
        self.lambda_range: int
            The number of the actual simulations
        self.all_data: dict
            The keys of it are the sequential integers and its values are the every dataframe of the single lambda simulation
        self.all_data_unk: pd.DataFrame
            The dataframe obtained by concating the list(self.all_data.values())
        '''
        self.u_nk_pd = u_nk_pd
        # print(f'self.u_nk_pd :{u_nk_pd}')
#         self.temperature = temperature
        self.scale_f = scale_f
#         self.ene_unit = ene_unit
        #get mbar index
        self.mbar_lambda_dict = {}
        idx = 0 
        for i in self.u_nk_pd.columns:
            self.mbar_lambda_dict[idx] = i
            idx+=1
        
        ori_index_list = list(self.u_nk_pd.index.names)
        ori_index_list.pop(0)
        self.lambda_list = ori_index_list
        # print(f'self.lambda_list: {self.lambda_list}')
        a = self.u_nk_pd.groupby(self.lambda_list, sort=False)
        lambda_dict = {}
        K=0
        if wanted_win_lst is False:
            for i,j in a:
                # print(f'Current grp id :{i}')
                # print(j)
                every_dataframe = pd.DataFrame(j)
                lambda_dict[K]=every_dataframe.tail(n=math.floor(every_dataframe.shape[0]*self.scale_f))
                # print(f'___________{K}_____________\n{lambda_dict[K]}')
                # print(i, len(lambda_dict[K]))
                K+=1
        else:
            for i,j in a:
#                 print(i,j)
                if wanted_win_lst[K]==i:
                    every_dataframe = pd.DataFrame(j)
                    lambda_dict[K]=every_dataframe.tail(n=math.floor(every_dataframe.shape[0]*self.scale_f))
                    K+=1
                    if K == len(wanted_win_lst):
                        break
                else:
                    pass
            
        self.all_data = lambda_dict
        # print('lambda_dict len is',len(self.all_data))
        self.all_data_unk = pd.concat(list(lambda_dict.values()),sort=False)
        
        lamb_value_dict = {}
        type_of_column = type(self.mbar_lambda_dict[0])
        for key in self.all_data.keys():
            index_list = list(self.all_data[key].index[0])
            index_list.pop(0)
            if len(index_list) == 1:
                lamb_value_dict[key] = index_list[0]
            else:
                if type_of_column == str:
                    lamb_value_dict[key] = str(tuple(index_list))
                else:
                    lamb_value_dict[key] = tuple(index_list)
        self.simulation_lambda = lamb_value_dict
        # print('simulation lambda is ',self.simulation_lambda)
        if force_align_mbar_list:
            self.mbar_lambda_dict = copy.deepcopy(lamb_value_dict)
        self.lambda_range = len(self.all_data)
        self.plot_obj = PLOTTING()
                
    def get_deltaU_in_lambda_by_tuplekey(self, lambda_idx, delta_U_whominus_tuple_key,filter_=False):
        '''Get the speicific d_U in the specified lambda by a tuple of the two lambda_values, which was not limited to the lambda value in the self.simulation_lambda.

        Parameters
        ----------
        lambda_idx: int, to specify a lambda simulation.
        delta_U_whominus_tuple_key: tuple, a two elements tuple that the first element is the lambda value of reduced U.
        filter_: bool, if it equals to True, do the filteration of d_U based on its mean and std.  

        Return
        ----------
        dU_at_lambda: np.array, float, shape=(N,)
                    N is the flames including in each lambda window.
                    Each value in this np.array is the speicific d_U in the specified lambda. 
                    If filter == True, only remain the d_U in the range (d_U_mean-2*std, d_U_mean+2*std).
        '''
        U_key_1 = delta_U_whominus_tuple_key[0]
        U_key_2 = delta_U_whominus_tuple_key[1]
        U_at_lambda_key1 = np.array(self.all_data[lambda_idx][U_key_1])
        U_at_lambda_key2 = np.array(self.all_data[lambda_idx][U_key_2])
        if filter_ == True:
            result_dict = {}
            dU_at_lambda = U_at_lambda_key1 - U_at_lambda_key2
            d_U_mean = dU_at_lambda.mean()
            d_U_std = dU_at_lambda.std()
            up_edge = d_U_mean+2*d_U_std
            down_edge = d_U_mean-2*d_U_std
            bool_index_less = dU_at_lambda<up_edge
            bool_index_more = dU_at_lambda>down_edge
            bool_index_all = bool_index_less*bool_index_more
            dU_at_lambda = dU_at_lambda[bool_index_all]
            result_dict['dU_at_lambda'] = dU_at_lambda
            result_dict['bool_index'] = bool_index_all 
            return result_dict
        else:
            dU_at_lambda = U_at_lambda_key1 - U_at_lambda_key2
            return dU_at_lambda
   
    def get_deltaU_in_lambda(self, lambda_idx, delta_U_whominus, filter_=True):
        '''Get the speicific d_U in the specified lambda, which was only included in the self.simulation_lambda.

        Parameters
        ----------
        lambda_idx: int, to specify a lambda simulation.
        delta_U_whominus: tuple, int, a two elements tuple that the first element is the lambda index of reduced U.
        filter_: bool, if it equals to True, do the filteration of d_U based on its mean and std. 

        Return
        ----------
        dU_at_lambda: np.array, float, shape=(N,)
                    N is the flames including in each lambda window.
                    Each value in this np.array is the speicific d_U in the specified lambda. 
                    If filter == True, only remain the d_U in the range (d_U_mean-2*std, d_U_mean+2*std).
        '''
        if filter_ == True:
            result_dict = {}
            key_1 = delta_U_whominus[0]
            key_2 = delta_U_whominus[1]
            U_key_1 = self.simulation_lambda[key_1]
            U_key_2 = self.simulation_lambda[key_2]
            U_at_lambda_key1 = np.array(self.all_data[lambda_idx][U_key_1])
            U_at_lambda_key2 = np.array(self.all_data[lambda_idx][U_key_2])
            dU_at_lambda = U_at_lambda_key1 - U_at_lambda_key2
            d_U_mean = dU_at_lambda.mean()
            d_U_std = dU_at_lambda.std()
            up_edge = d_U_mean+2*d_U_std
            down_edge = d_U_mean-2*d_U_std
            bool_index_less = dU_at_lambda<up_edge
            bool_index_more = dU_at_lambda>down_edge
            bool_index_all = bool_index_less*bool_index_more
            dU_at_lambda = dU_at_lambda[bool_index_all]
            result_dict['dU_at_lambda'] = dU_at_lambda
            result_dict['bool_index'] = bool_index_all 
            return result_dict
        else:
            key_1 = delta_U_whominus[0]
            key_2 = delta_U_whominus[1]
            U_key_1 = self.simulation_lambda[key_1]
            U_key_2 = self.simulation_lambda[key_2]
            U_at_lambda_key1 = np.array(self.all_data[lambda_idx][U_key_1])
            U_at_lambda_key2 = np.array(self.all_data[lambda_idx][U_key_2])
            dU_at_lambda = U_at_lambda_key1 - U_at_lambda_key2
            return dU_at_lambda

    def plot_simulation_deltaU(self, bins=50):
        '''To plot the forward(dU_{i+1,i}) and backward(dU_{i-1,i}) delta_U in the lambda with the index of i

        Parameters
        ----------
        bins: int, to specify the number of bins for generating the histogram.

        Generated files
        ----------
        state_{i}_du_b.png: the backward dU distribution under the i lambda window.
        state_{i}_du_f.png: the forward dU distribution under the i lambda window.
        du_std_info.xlsx: recording all the std of the dU.
        '''
        plot_obj = self.plot_obj 
        init_df_ = pd.DataFrame()
        for i in range(0, self.lambda_range):
            std_df = pd.DataFrame(columns=['state_idx', 'lambda_info', 'ori_du_b_std', 'ori_du_f_std'], index =[0,])
            try:
                ori_du_b = self.get_deltaU_in_lambda(i,(i-1,i),False)
                ori_du_b_std = ori_du_b.std()
                png_name_b = f'state_{i}_du_b.png'
            except:
                print(f'Current i is {i}, can not get ori_du_b')
                png_name_b = None
                ori_du_b_std = None
            if png_name_b:
                plot_obj.plot_dU_distribution(ori_du_b, png_name_b, True, bins)
            try:
                ori_du_f = self.get_deltaU_in_lambda(i,(i+1,i),False)
                ori_du_f_std = ori_du_f.std()
                png_name_f = f'state_{i}_du_f.png'
            except:
                print(f'Current i is {i}, can not get ori_du_f')
                png_name_f = None
                ori_du_f_std = None
            if png_name_f:
                plt.clf()
                plot_obj.plot_dU_distribution(ori_du_f, png_name_f, True, bins)
            std_df['state_idx'] = i
            std_df['lambda_info'] = tuple_to_str(self.simulation_lambda[i])
            std_df['ori_du_b_std'] = ori_du_b_std
            std_df['ori_du_f_std'] = ori_du_f_std
            init_df_ = pd.concat([init_df_, std_df], axis = 0 )
        init_df_.to_excel('du_std_info.xlsx', )
        
    def check_cfm_overall(self, png_save=False):
        '''Use the curve-fitting method to check the overlap degree for the pair windows through the all windows.
        '''
        assert self.lambda_range != 1, 'Use_Condition_Error: Only one simulation windows exist! Can not do the curve fitting method check!'
        for i in range(0, self.lambda_range-1):
            lambda_0 = i
            lambda_1 = i+1
            key_0 = i
            key_1 = i+1
            U_key_0 = self.simulation_lambda[key_0]
            U_key_1 = self.simulation_lambda[key_1]
            U_0_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_0])
            U_1_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_1])
            U_0_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_0])
            U_1_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_1])
            delta_U_1to0_in_lambda_0 = U_1_in_lambda_0 - U_0_in_lambda_0
            delta_U_1to0_in_lambda_1 = U_1_in_lambda_1 - U_0_in_lambda_1
            if png_save == True:
                self.check_cfm_convergence(delta_U_1to0_in_lambda_0, delta_U_1to0_in_lambda_1, png_file='cfm_{}.png'.format(i))
            else:
                self.check_cfm_convergence(delta_U_1to0_in_lambda_0, delta_U_1to0_in_lambda_1, png_file=None)

    def get_interpolate(self, fitted_xy):
        '''Get the interpolate function of given xy. The funcion is forcibly through all data points by setting s=0.

        Parameters
        ----------
        fitted_xy: np.array 
            np.array([[x1,x2,x3,,,,], [y1,y2,y3,,,,]])The array for generate interpolate function. 
            
        Retrun
        ----------
        func:the interpolate function generated by given x,y.
        '''
        # func = spi.interp1d(fitted_xy[0], fitted_xy[1], kind='linear')
        func = spi.UnivariateSpline(fitted_xy[0],fitted_xy[1],s=0)
        return func # x->y

    def Boltzmann_weight_PdU(self, lambda_idx, direction, png_file='Boltzmann_weight_PdU.png'):
        plot_obj = self.plot_obj
        if direction == 'forward':
            delta_U_whominus = (lambda_idx+1, lambda_idx)
        elif direction == 'backward':
            delta_U_whominus = (lambda_idx-1, lambda_idx)
        d_u = self.get_deltaU_in_lambda(lambda_idx, delta_U_whominus, filter_=False)

        #generate histogram distribution
        count,bin_edges = np.histogram(d_u,bins=500,density=True)
        bin_mid = (bin_edges[1:]+bin_edges[:-1])/2
        d_u_max = bin_mid.max()
        d_u_min = bin_mid.min()
        # print(bin_mid)
        PdU_func = self.get_interpolate([bin_mid, count])
        input_interpolate_bin = np.linspace(d_u_min, d_u_max, 1000)
        interpolate_PdU = PdU_func(input_interpolate_bin)
        Boltzmann_weight = np.exp(-input_interpolate_bin)
        B_wei_times_PdU = np.multiply(interpolate_PdU, Boltzmann_weight)
        plot_input_interpolate_bin = np.linspace(d_u_min, d_u_max, 100000)
        plot_Boltzmann_weight = np.exp(-plot_input_interpolate_bin)
        plot_y_max = max(B_wei_times_PdU.max(), interpolate_PdU.max())

        plot_obj.plot_Boltzmann_weight_PdU_result(input_interpolate_bin, plot_input_interpolate_bin, B_wei_times_PdU, interpolate_PdU, plot_Boltzmann_weight, (0,plot_y_max+0.1*plot_y_max), png_file)

    def plot_Boltzmann_weight_PdU(self, ):
        for i in range(0, self.lambda_range):
            # self.Boltzmann_weight_PdU(i, 'forward', png_file=f'Boltzmann_weight_PdU_forward_state_{i}.png')
            try:
                self.Boltzmann_weight_PdU(i, 'backward', png_file=f'Boltzmann_weight_PdU_backward_state_{i}.png')
            except:
                print(f'Current i is {i}, can not plot Boltzmann_weight_PdU_backward')
            try:
                self.Boltzmann_weight_PdU(i, 'forward', png_file=f'Boltzmann_weight_PdU_forward_state_{i}.png')
            except:
                print(f'Current i is {i}, can not plot Boltzmann_weight_PdU_forward')

    def check_cfm_convergence(self, delta_U_1to0_in_lambda_0, delta_U_1to0_in_lambda_1, png_file=None, ):
        '''Use the curve-fitting method to do the consistency check on the data generated by either equilibrium or nonequilibrium sampling.

        Parameters
        ----------
        delta_U_1to0_in_lambda_0: np.array, float, (U_1 - U_0)_lambda0
        delta_U_1to0_in_lambda_1: np.array, float, (U_1 - U_0)_lambda1
        '''
        plot_obj = self.plot_obj
        delta_U_0to1_in_lambda_1 = -delta_U_1to0_in_lambda_1
        df, dff = BAR_(delta_U_1to0_in_lambda_0, delta_U_0to1_in_lambda_1, method='self-consistent-iteration', maximum_iterations=1000, verbose=False)
        # print(df)
        if np.isnan(dff):
            raise SystemExit('There are something wrong with the sampled delta_U since the variance of BAR is infinite!')
        else:

            state_0_du_mean = np.mean(delta_U_1to0_in_lambda_0)
            count_0, bin_edges_0 = np.histogram(delta_U_1to0_in_lambda_0,bins=500,density=True,)
            xu_0 = []
            for i in range(len(bin_edges_0)-1):
                xu_0.append((bin_edges_0[i]+bin_edges_0[i+1])/2)
            xu_0 = np.array(xu_0)
            count_1, bin_edges_1 = np.histogram(delta_U_1to0_in_lambda_1,bins=500,density=True,)
            xu_1 = []
            for i in range(len(bin_edges_1)-1):
                xu_1.append((bin_edges_1[i]+bin_edges_1[i+1])/2)
            xu_1 = np.array(xu_1)
            min_0 = xu_0.min()
            max_0 = xu_0.max()
            min_1 = xu_1.min()
            max_1 = xu_1.max()
            min_ = min(min_0, min_1)
            min_plot = max(min_0, min_1)
            max_ = max(max_0, max_1)
            max_plot = min(max_0, max_1)
            input_interpolate_x = np.linspace(min_plot, max_plot, 200)
            # input_interpolate_x = np.linspace(min_plot, 1.5*state_0_du_mean, 200)
            # input_interpolate_x = np.linspace(min_plot, 4+0.1*state_0_du_mean, 200)
            # input_interpolate_x = np.linspace(min_plot, 4+0.011*state_0_du_mean, 200)
            state_0_pdu_func = self.get_interpolate([xu_0, count_0])
            state_1_pdu_func = self.get_interpolate([xu_1, count_1])
            interpolate_y_state_0 = state_0_pdu_func(input_interpolate_x)
            interpolate_y_state_1 = state_1_pdu_func(input_interpolate_x)

            data = pd.DataFrame()
            data['d_u_in_0_P'] = interpolate_y_state_0
            data['d_u_in_1_P'] = interpolate_y_state_1
            data['d_u_bin'] = input_interpolate_x
            data_ = copy.deepcopy(data)
            data_ = data_[(True^data_['d_u_in_0_P'].isin([0]))]
            data_ = data_[(True^data_['d_u_in_1_P'].isin([0]))]
            log_left = np.log(data_['d_u_in_0_P']) - 0.5*data_['d_u_bin']
            log_righ = np.log(data_['d_u_in_1_P']) + 0.5*data_['d_u_bin']
            diff = log_righ-log_left
            diff_filter_nan = diff[~np.isnan(diff)]
            d_u_bin_filter_nan = data_['d_u_bin'][~np.isnan(diff)]
            
            #plotting
            plot_obj.plot_cfm_checking_result(df, dff, d_u_bin_filter_nan, diff_filter_nan, data_['d_u_bin'], png_file)
            
    def get_reweight_entropy(self, lambda_0, lambda_1):
        '''Reference:  J. Chem. Inf. Model. 2017, 57, 2476-2489
        Calculate the reweight_entropy S: S=-\frac{1}{\ln N}\sum_{j=1}^{N}W_{j}\ln W_{j}
        where \mathcal{W}_j=\frac{\exp(-\beta\Delta U_j)}{\sum_{i=1}^N\exp(-\beta\Delta U_i)}
        N is the number of frames that are harvested under the specific state.  As the reweighting entropy increases, the free energy calculation becomes more reliable. S has a maximum value of 1.0, where all the samples have equal weights. "If the reweighting entropy is below a certain threshold (0.65 for the solvation free energy calculations in this work: J. Chem. Inf. Model. 2017, 57, 2476-2489), this MM-to-QM correction is not reliable."
        The maximum of W_{j} of N frames is called the "maximal weight", W_{max}

        Parameters
        ----------
        lambda_0: float(in amber rbfe), the key in the simulation_lambda(dict) to assign the first window
        lambda_1: float(in amber rbfe), the key in the simulation_lambda(dict) to assign the second window

        Return
        ----------
        W_max_1to0_0: float, the maximal weight of the dU_{1-0} in the state 0 
        W_max_0to1_0: float, the maximal weight of the dU_{0-1} in the state 0 
        W_max_1to0_1: float, the maximal weight of the dU_{1-0} in the state 1
        W_max_0to1_1: float, the maximal weight of the dU_{0-1} in the state 1
        S_1to0_0: float, the reweight_entropy calculated by W_max_1to0_0
        S_0to1_0: float, the reweight_entropy calculated by W_max_0to1_0
        S_1to0_1: float, the reweight_entropy calculated by W_max_1to0_1
        S_0to1_1: float, the reweight_entropy calculated by W_max_0to1_1 
        '''
        U_key_0 = self.simulation_lambda[lambda_0]
        U_key_1 = self.simulation_lambda[lambda_1]
        U_0_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_0])
        U_1_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_1])
        U_0_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_0])
        U_1_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_1])
        delta_U_1to0_in_lambda_0 = U_1_in_lambda_0 - U_0_in_lambda_0
        delta_U_0to1_in_lambda_0 = U_0_in_lambda_0 - U_1_in_lambda_0
        delta_U_1to0_in_lambda_1 = U_1_in_lambda_1 - U_0_in_lambda_1
        delta_U_0to1_in_lambda_1 = U_0_in_lambda_1 - U_1_in_lambda_1
        def from_du_to_reweight_entropy(du):
            N = len(du)
            W = np.exp(-du)/np.exp(-du).sum()
            W_max = W.max()
            S = (-1/np.log(N))*np.sum(W*np.log(W))
            return W_max, S
        W_max_1to0_0, S_1to0_0 = from_du_to_reweight_entropy(delta_U_1to0_in_lambda_0)
        W_max_0to1_0, S_0to1_0 = from_du_to_reweight_entropy(delta_U_0to1_in_lambda_0)
        W_max_1to0_1, S_1to0_1 = from_du_to_reweight_entropy(delta_U_1to0_in_lambda_1)
        W_max_0to1_1, S_0to1_1 = from_du_to_reweight_entropy(delta_U_0to1_in_lambda_1)
        return W_max_1to0_0, W_max_0to1_0, W_max_1to0_1, W_max_0to1_1, S_1to0_0, S_0to1_0, S_1to0_1, S_0to1_1
        
    def check_reweight_entropy_overall(self, filename="reweight_entropy_check.csv", ):
        '''Calculate the reweight_entropy values between two adjacent windows through all the windows.
        ''' 
        init_data_df = pd.DataFrame()
        for i in range(0, self.lambda_range-1):
            lambda_0 = i
            lambda_1 = i + 1
            lambda_0_value = tuple_to_str(self.simulation_lambda[lambda_0])
            lambda_1_value = tuple_to_str(self.simulation_lambda[lambda_1])
            single_df = pd.DataFrame(columns=['lambda_0_value', 'lambda_1_value', 'W_max_1to0_0', 'W_max_0to1_0', 'W_max_1to0_1', 'W_max_0to1_1', 
                                              'S_1to0_0', 'S_0to1_0', 'S_1to0_1', 'S_0to1_1'], index=[lambda_0,])
            single_df['lambda_0_value'], single_df['lambda_1_value'] = lambda_0_value, lambda_1_value
            single_df['W_max_1to0_0'], single_df['W_max_0to1_0'], single_df['W_max_1to0_1'], single_df['W_max_0to1_1'], single_df['S_1to0_0'], single_df['S_0to1_0'], single_df['S_1to0_1'], single_df['S_0to1_1'] = self.get_reweight_entropy(lambda_0, lambda_1)
            init_data_df = pd.concat([init_data_df, single_df], axis = 0 )
        init_data_df.to_csv(filename)

    def get_pai_values(self, lambda_0, lambda_1,):
        '''Use the equations from the paper J. Chem. Phys. 123, 054103 (2005) to calculate the pai value for checking if the free energy calculation is free from the bias.
        Should be greater than zero, better greater than 0.5.
        \Pi_{0\to 1}=\sqrt{\frac{s_{0}}{s_{1}}\mathbf{W}_{L}\bigg[\frac{1}{2\pi}(M-1)^{2}\bigg]}-\sqrt{2s_{0}}
        \Pi_{1\to 0}=\sqrt{\frac{s_{1}}{s_{0}}\mathbf{W}_{L}\bigg[\frac{1}{2\pi}(M-1)^{2}\bigg]}-\sqrt{2s_{1}}

        which: 
        s_{0}=\beta\overline{\Delta U_{1-0}}-\beta\Delta F_{1-0}
        s_{1}=\beta\overline{\Delta U_{0-1}}+\beta\Delta F_{1-0}
        \mathbf{W}_{L}(x) is the the Lambert W function, defined as the solution for w in x=wexp(w).
        M is the number of samples
        
        Parameters
        ----------
        lambda_0: float(in amber rbfe), the key in the simulation_lambda(dict) to assign the first window
        lambda_1: float(in amber rbfe), the key in the simulation_lambda(dict) to assign the second window
        
        Generated key variables
        ----------
        delta_U_1to0_in_lambda_0: np.array, float, shape=(N,). \Delta U_{back-front} in front lambda window.
        delta_U_1to0_in_lambda_1: np.array, float, shape=(N,). \Delta U_{back-front} in back lambda window.
            N is the number of the flames in each lambda window.
            
        Return
        ----------
        relative_entrophy_0: float, the value of s_{0}
        relative_entrophy_1: float, the value of s_{1}
        pai_0to1:            float, the value of \Pi_{0\to 1}
        pai_1to0:            float, the value of \Pi_{1\to 0}
        '''
        U_key_0 = self.simulation_lambda[lambda_0]
        U_key_1 = self.simulation_lambda[lambda_1]
        U_0_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_0])
        U_1_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_1])
        U_0_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_0])
        U_1_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_1])
        delta_U_1to0_in_lambda_0 = U_1_in_lambda_0 - U_0_in_lambda_0
        delta_U_1to0_in_lambda_0_std = np.std(delta_U_1to0_in_lambda_0)
        delta_U_1to0_in_lambda_1 = U_1_in_lambda_1 - U_0_in_lambda_1
        delta_U_1to0_in_lambda_1_std = np.std(delta_U_1to0_in_lambda_1)
        delta_U_0to1_in_lambda_1 = -delta_U_1to0_in_lambda_1
        delta_F_1to0_in_lambda_0_forward = FEP(delta_U_1to0_in_lambda_0).ene
        delta_F_0to1_in_lambda_0_forward = FEP(delta_U_0to1_in_lambda_1).ene
        # df, dff = BAR_(delta_U_1to0_in_lambda_0, delta_U_0to1_in_lambda_1, method='self-consistent-iteration', maximum_iterations=1000, verbose=False)
        M = 1/2*(len(delta_U_1to0_in_lambda_0)+len(delta_U_0to1_in_lambda_1)) 
        W = float(lambertw((M-1)**2/(2*np.pi)))
        relative_entrophy_0 = np.mean(delta_U_1to0_in_lambda_0)-delta_F_1to0_in_lambda_0_forward #df is f1-f0 
        relative_entrophy_1 = np.mean(delta_U_0to1_in_lambda_1)-delta_F_0to1_in_lambda_0_forward #df is f1-f0 

        pai_0to1 = (relative_entrophy_0*W/relative_entrophy_1)**0.5-(2*relative_entrophy_0)**0.5
        pai_1to0 = (relative_entrophy_1*W/relative_entrophy_0)**0.5-(2*relative_entrophy_1)**0.5

        return relative_entrophy_0, relative_entrophy_1, pai_0to1, pai_1to0

    def check_pai_overall(self, filename="pai_check.csv", ):
        '''calculate the pai values between two adjacent windows through all the windows.
        '''
        init_data_df = pd.DataFrame()
        for i in range(0, self.lambda_range-1):
            lambda_0 = i
            lambda_1 = i + 1
            lambda_0_value = tuple_to_str(self.simulation_lambda[lambda_0])
            lambda_1_value = tuple_to_str(self.simulation_lambda[lambda_1])
            single_df = pd.DataFrame(columns=['lambda_0_value', 'lambda_1_value', 'relative_entrophy_0', 'relative_entrophy_1', 'pai_0to1', 'pai_1to0', ], index=[lambda_0,])
            single_df['lambda_0_value'] = lambda_0_value
            single_df['lambda_1_value'] = lambda_1_value
            single_df['relative_entrophy_0'], single_df['relative_entrophy_1'], single_df['pai_0to1'], single_df['pai_1to0'] = self.get_pai_values(lambda_0, lambda_1)
            init_data_df = pd.concat([init_data_df, single_df], axis = 0 )
        init_data_df.to_csv(filename)

    # TODO: doc_string need
    def get_MeasurementOfConvergence(self, lambda_0, lambda_1, growing_step_ratio):
        U_key_0 = self.simulation_lambda[lambda_0]
        U_key_1 = self.simulation_lambda[lambda_1]
        U_0_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_0])
        U_1_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_1])
        U_0_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_0])
        U_1_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_1])
        delta_U_1to0_in_lambda_0 = U_1_in_lambda_0 - U_0_in_lambda_0
        delta_U_1to0_in_lambda_1 = U_1_in_lambda_1 - U_0_in_lambda_1
        N_0 = len(delta_U_1to0_in_lambda_0)
        N_1 = len(delta_U_1to0_in_lambda_1)
        init_data_df = pd.DataFrame()
        cal_time = 0
        filename = f'MeasurementOfConvergence_{lambda_0}_{lambda_0}.csv'
        for used_data_percentage in np.arange(growing_step_ratio, 1+growing_step_ratio, growing_step_ratio):
            cal_time+=1
            single_df = pd.DataFrame(columns=['lambda_0', 'lambda_1', 'The top X percent of data used', 'BAR_estimate_df', 'fermi_first_moment_in_lambda_0', 'fermi_first_moment_in_lambda_1', 'fermi_second_moment', 'MeasurementOfConvergence_in_lambda_0', 'MeasurementOfConvergence_in_lambda_1'], index=[cal_time,])
            single_df['lambda_0'], single_df['lambda_1'] = lambda_0, lambda_1
            single_df['The top X percent of data used'] = np.around(used_data_percentage*100, 2)
            used_delta_U_1to0_in_lambda_0 = delta_U_1to0_in_lambda_0[0:int(np.floor(N_0*used_data_percentage))]
            used_delta_U_1to0_in_lambda_1 = delta_U_1to0_in_lambda_1[0:int(np.floor(N_1*used_data_percentage))]
            df, dff = BAR_(used_delta_U_1to0_in_lambda_0, -used_delta_U_1to0_in_lambda_1, method='self-consistent-iteration', 
                          maximum_iterations=1000, verbose=False)
            alpha, beta = N_0/(N_0+N_1), N_1/(N_0+N_1)
            fermi_first_moment_array_in_lambda_0 = 1/(alpha+beta*np.exp(-used_delta_U_1to0_in_lambda_1+df))
            fermi_first_moment_array_in_lambda_1 = 1/(alpha*np.exp(used_delta_U_1to0_in_lambda_0-df)+beta)
            fermi_first_moment_in_lambda_0 = np.mean(fermi_first_moment_array_in_lambda_0)
            fermi_first_moment_in_lambda_1 = np.mean(fermi_first_moment_array_in_lambda_1)
            fermi_second_moment =  alpha*((fermi_first_moment_array_in_lambda_0**2).mean()) + beta*((fermi_first_moment_array_in_lambda_1**2).mean())
            MeasurementOfConvergence_in_lambda_0 = (fermi_first_moment_in_lambda_0-fermi_second_moment)/fermi_first_moment_in_lambda_0
            MeasurementOfConvergence_in_lambda_1 = (fermi_first_moment_in_lambda_1-fermi_second_moment)/fermi_first_moment_in_lambda_1
            single_df['BAR_estimate_df'], single_df['fermi_first_moment_in_lambda_0'], single_df['fermi_first_moment_in_lambda_1'], single_df['fermi_second_moment'], single_df['MeasurementOfConvergence_in_lambda_0'], single_df['MeasurementOfConvergence_in_lambda_1'] = df, fermi_first_moment_in_lambda_0, fermi_first_moment_in_lambda_1, fermi_second_moment, MeasurementOfConvergence_in_lambda_0, MeasurementOfConvergence_in_lambda_1
            init_data_df = pd.concat([init_data_df, single_df], axis = 0 )
        # init_data_df.to_csv(filename)
        return init_data_df
    
    def check_MeasurementOfConvergence_overall(self, filename="MeasurementOfConvergence.csv", ):
        '''calculate the MeasurementOfConvergence between two adjacent windows through all the windows.
        '''
        df_list = []
        for i in range(0, self.lambda_range-1):
            lambda_0 = i
            lambda_1 = i + 1
            df_list.append(self.get_MeasurementOfConvergence(lambda_0, lambda_1, 0.01))
        init_data_df = pd.concat(df_list, axis = 0 )
        init_data_df.to_csv(filename)

    def cal_FE(self, filename="free_ene.csv",unit='kbT', ifbootstrap_std=False):
        '''
        ene_unit: str
            To determine the final output csv energy unit, 'kbT' or 'kcal/mol', default is 'kbT'
        '''
        init_data_df = pd.DataFrame()
        last_lambda_key = self.mbar_lambda_dict[len(self.mbar_lambda_dict)-1]
        first_lambda_key = self.mbar_lambda_dict[0]
        if self.lambda_range == 1:
            i=0
            esti_ene_0, esti_std_0 = self.cal_FE_first_window_bar(ifbootstrap_std)
            index_info_0=tuple_to_str(first_lambda_key)+' to '+ tuple_to_str(self.simulation_lambda[i])
            esti_ene_1, esti_std_1 = self.cal_FE_last_window_bar(ifbootstrap_std)
            index_info_1=tuple_to_str(self.simulation_lambda[i])+' to '+ tuple_to_str(last_lambda_key)
            single_df = pd.DataFrame(columns=['delta_A_what_to_what', 'free_energy(kbT)', 'estimated std'], index =[0,1] )
            single_df.iloc[0,0]=index_info_0
            single_df.iloc[0,1]=esti_ene_0
            single_df.iloc[0,2]=esti_std_0
            single_df.iloc[1,0]=index_info_1
            single_df.iloc[1,1]=esti_ene_1
            single_df.iloc[1,2]=esti_std_1
            init_data_df = pd.concat([init_data_df, single_df], axis = 0 )
        else:
            for i in range(self.lambda_range):
                if i == self.lambda_range-1:
                    if last_lambda_key in self.simulation_lambda.values():
                        single_df = pd.DataFrame()
                    else:
                        esti_ene, esti_std = self.cal_FE_last_window_bar(ifbootstrap_std)
                        index_info=tuple_to_str(self.simulation_lambda[i])+' to '+ tuple_to_str(last_lambda_key)
                        single_df = pd.DataFrame(columns=['delta_A_what_to_what', 'free_energy(kbT)', 'estimated std'], index =[0,] )
                        single_df['delta_A_what_to_what'] = index_info
                        single_df['free_energy(kbT)'] = esti_ene
                        single_df[ 'estimated std'] = esti_std
                elif i == 0:
                    if first_lambda_key in self.simulation_lambda.values():
                        esti_ene, esti_std= self.cal_FE_middle_window_bar(i, ifbootstrap_std)
                        index_info=tuple_to_str(self.simulation_lambda[i])+' to '+ tuple_to_str(self.simulation_lambda[i+1])
                        single_df = pd.DataFrame(columns=['delta_A_what_to_what', 'free_energy(kbT)', 'estimated std'], index =[0,] )
                        single_df['delta_A_what_to_what'] = index_info
                        single_df['free_energy(kbT)'] = esti_ene
                        single_df[ 'estimated std'] = esti_std                    
                    else:
                        esti_ene_0, esti_std_0 = self.cal_FE_first_window_bar(ifbootstrap_std)
                        index_info_0=tuple_to_str(first_lambda_key)+' to '+ tuple_to_str(self.simulation_lambda[i])
                        esti_ene_1, esti_std_1 = self.cal_FE_middle_window_bar(i, ifbootstrap_std)
                        index_info_1=tuple_to_str(self.simulation_lambda[i])+' to '+ tuple_to_str(self.simulation_lambda[i+1])
                        single_df = pd.DataFrame(columns=['delta_A_what_to_what', 'free_energy(kbT)', 'estimated std'], index =[0,1] )
                        single_df.iloc[0,0]=index_info_0
                        single_df.iloc[0,1]=esti_ene_0
                        single_df.iloc[0,2]=esti_std_0
                        single_df.iloc[1,0]=index_info_1
                        single_df.iloc[1,1]=esti_ene_1
                        single_df.iloc[1,2]=esti_std_1
                else:
                    esti_ene, esti_std= self.cal_FE_middle_window_bar(i, ifbootstrap_std)
                    index_info=tuple_to_str(self.simulation_lambda[i])+' to '+ tuple_to_str(self.simulation_lambda[i+1])
                    single_df = pd.DataFrame(columns=['delta_A_what_to_what', 'free_energy(kbT)', 'estimated std'], index =[0,] )
                    single_df['delta_A_what_to_what'] = index_info
                    single_df['free_energy(kbT)'] = esti_ene
                    single_df['estimated std'] = esti_std
                init_data_df = pd.concat([init_data_df, single_df], axis = 0 )
        # print(init_data_df)
        sum_df = pd.DataFrame(columns=['delta_A_what_to_what', 'free_energy(kbT)', 'estimated std'], index =[0,] )
        sum_df['delta_A_what_to_what']=tuple_to_str(first_lambda_key)+' to '+tuple_to_str(last_lambda_key)
        sum_df['free_energy(kbT)']=init_data_df.iloc[:,1].sum()
        sum_df['estimated std']=((init_data_df.iloc[:,2]**2).sum())**0.5
        init_data_df=pd.concat([init_data_df,sum_df],axis=0)
        init_data_df.index = init_data_df['delta_A_what_to_what']
        init_data_df=init_data_df.drop(columns=['delta_A_what_to_what',])
        #init_data_df['free_energy(kbT)'].astype(float)
        #init_data_df['estimated std'].astype(float)
        if unit=='kbT':
            pass
#             print('aaaa')
#             init_data_df.to_csv(filename)
        elif unit=='kcal/mol':
#             print('bbbb')
            init_data_df.columns=['free_energy(kcal/mol)', 'estimated std']
            init_data_df['free_energy(kcal/mol)']=init_data_df['free_energy(kcal/mol)'].astype(float)
            init_data_df['estimated std']=init_data_df['estimated std'].astype(float)
            #print(init_data_df.dtypes)
            init_data_df[init_data_df.select_dtypes(include=['float64']).columns] *= 0.5922
            #print(init_data_df)
        if filename:
            init_data_df.to_csv(filename)
        
        return init_data_df

    def cal_FE_FEP(self, filename='free_ene_fep.csv', unit='kbT'):
        '''calculate the free energy difference between the simulaion lambda by FEP
        Parameters
        ----------
        filename: str
            The output csv filename of the result dataframe, default is 'free_ene_fep.csv'
        ene_unit: str
            To determine the final output csv energy unit, 'kbT' or 'kcal/mol', default is 'kbT'
        
        Return    
        ----------
        init_data_df: pd.DataFrame
            The final output of calculated free energy
        '''
        init_data_df = pd.DataFrame()
        last_lambda_key = self.mbar_lambda_dict[len(self.mbar_lambda_dict)-1]
        first_lambda_key = self.mbar_lambda_dict[0]
        if self.lambda_range == 1:
            i=0
            lambda_value = self.simulation_lambda[i]
            single_df = pd.DataFrame(columns=['lambda_value', 'FEP_forward_bythislambda(kbT)', 'FEP_reverse_bythislambda(kbT)'], index =[0,] )
            single_df.iloc[0,0]=str(lambda_value)
            if lambda_value == last_lambda_key:
                esti_ene_0, esti_std_0 = self.cal_FE_first_window_bar()#use fep_backward calculation
                single_df.iloc[0,1] = pd.NA
                single_df.iloc[0,2] = -esti_ene_0
                # print("For the convergence check, set FEP_forward_bythislambda to be the value of FEP_reverse_bythislambda.")
                single_df.iloc[0,1] = esti_ene_0
            elif lambda_value == first_lambda_key:
                esti_ene_0, esti_std_0 = self.cal_FE_last_window_bar()#use fep_forward calculation
                single_df.iloc[0,1] = esti_ene_0
                single_df.iloc[0,2] = pd.NA
                # print("For the convergence check, set FEP_backward_bythislambda to be the value of FEP_forward_bythislambda.")
                single_df.iloc[0,2] = -esti_ene_0
            else:
                esti_ene_0, esti_std_0 = self.cal_FE_first_window_bar()#use fep_backward calculation
                esti_ene_1, esti_std_1 = self.cal_FE_last_window_bar()#use fep_forward calculation
                single_df.iloc[0,1] = esti_ene_1
                single_df.iloc[0,2] = -esti_ene_0
            init_data_df = pd.concat([init_data_df, single_df], axis = 0 )
        else:
            for i in range(self.lambda_range):
                lambda_value = self.simulation_lambda[i]
                if i == 0:
                    if first_lambda_key in self.simulation_lambda.values():
                        fe_forward = self.cal_FE_FEP_forward(i)
                        fe_reverse = pd.NA
                        # print("For the convergence check, set FEP_backward_bythislambda to be the value of FEP_forward_bythislambda.")
                        fe_reverse = - fe_forward
                    else:
                        fe_forward = self.cal_FE_FEP_forward(i)
                        fe_reverse = self.cal_FE_first_window_bar()[0]
                elif i == self.lambda_range-1:
                    if last_lambda_key in self.simulation_lambda.values():
                        fe_forward = pd.NA
                        fe_reverse = -self.cal_FE_FEP_reverse(i)
                        # print("For the convergence check, set FEP_forward_bythislambda to be the value of FEP_reverse_bythislambda.")
                        fe_forward = -fe_reverse
                    else:
                        fe_forward = self.cal_FE_last_window_bar()[0]
                        fe_reverse = -self.cal_FE_FEP_reverse(i)
                else:
                    fe_forward = self.cal_FE_FEP_forward(i)
                    fe_reverse = -self.cal_FE_FEP_reverse(i)
                single_df = pd.DataFrame(columns=['lambda_value', 'FEP_forward_bythislambda(kbT)', 'FEP_reverse_bythislambda(kbT)'], index=[0,])
                single_df['lambda_value'] = str(lambda_value)
                single_df['FEP_forward_bythislambda(kbT)'] = fe_forward
                single_df['FEP_reverse_bythislambda(kbT)'] = fe_reverse
                init_data_df = pd.concat([init_data_df,single_df], axis = 0)
        # Disregard the sum_df for FEP, as FEP is solely appropriate for examining the temporal convergence of each window.
        # sum_df = pd.DataFrame(columns=['lambda_value', 'FEP_forward_bythislambda(kbT)', 'FEP_reverse_bythislambda(kbT)'], index=[0,])
        # sum_df['lambda_value'] = 'sum_of_all'
        # sum_df['FEP_forward_bythislambda(kbT)'] = init_data_df.iloc[:,1].sum()
        # sum_df['FEP_reverse_bythislambda(kbT)'] = init_data_df.iloc[:,2].sum()
        init_data_df.index = init_data_df['lambda_value']
        init_data_df=init_data_df.drop(columns=['lambda_value',])
        if unit=='kbT':
            pass
        elif unit=='kcal/mol':
            init_data_df.columns=['FEP_forward_bythislambda(kcal/mol)','FEP_reverse_bythislambda(kcal/mol)']
            init_data_df['FEP_forward_bythislambda(kcal/mol)']=init_data_df['FEP_forward_bythislambda(kcal/mol)'].astype(float)
            init_data_df['FEP_reverse_bythislambda(kcal/mol)']=init_data_df['FEP_reverse_bythislambda(kcal/mol)'].astype(float)
            init_data_df[init_data_df.select_dtypes(include=['float64']).columns] *= 0.5922
        if filename:
            init_data_df.to_csv(filename)   
        return init_data_df

    def bootstrap_std(self, statistic_func, num_samples, data):
        '''
        statistic_func: a function only require one or two variable to give statistical value.
        num_samples: times for resampling.
        data: could be one-dimensional or two-dimensional. One-dimensional means only one array of data should be required by statistic_func. 
        '''
        arry_num = len(data)
        if arry_num == 1:
            used_data = data[0]
            resamples = np.random.choice(used_data, (num_samples, len(used_data)), replace=True)
            stat = np.array([statistic_func(r) for r in resamples])
        elif arry_num == 2:
            used_data_1 = data[0]
            used_data_2 = data[1]
            resamples_1 = np.random.choice(used_data_1, (num_samples, len(used_data_1)), replace=True)
            resamples_2 = np.random.choice(used_data_2, (num_samples, len(used_data_2)), replace=True)
            stat = np.array([statistic_func(resamples_1[r], resamples_2[r]) for r in range(num_samples)])
        bootstrap_std = np.std(stat)
        return bootstrap_std

    def cal_FE_FEP_forward(self, lambda_idx, ifbootstrap_std=False, ):
        '''paicheck: use the recipe for determining if a free energy calculation is free of bias in the literature: THE JOURNAL OF CHEMICAL PHYSICS 123, 054103 (2005)
        Pai value must be greater than zero. And suggested pai value should be greater than 0.5. 
        '''
        i=lambda_idx
        ori_du_f = self.get_deltaU_in_lambda(i,(i+1,i),False)
        def fep_forward(du_):
            obj_f = FEP(du_)
            return obj_f.ene
        fe_fep = fep_forward(ori_du_f)
        boot_strap_data = [ori_du_f]
        if ifbootstrap_std:
            fe_std = self.bootstrap_std(fep_forward, 200, boot_strap_data)
        else:
            fe_std = 0
        return fe_fep
    
    def cal_FE_FEP_reverse(self, lambda_idx, ifbootstrap_std=False):
        i=lambda_idx
        ori_du_b = self.get_deltaU_in_lambda(i, (i-1,i),False)
        def fep_backward(du_):
            obj_b = FEP(du_)
            return -obj_b.ene
        fe_fep = fep_backward(ori_du_b)
        boot_strap_data = [ori_du_b]
        if ifbootstrap_std:
            fe_std = self.bootstrap_std(fep_backward, 200, boot_strap_data)
        else:
            fe_std = 0
        return fe_fep

    def cal_FE_first_window_bar(self, ifbootstrap_std=False):
        i=0
        first_key = self.mbar_lambda_dict[0]
        second_key = self.simulation_lambda[0]
        delta_U_tuplekey = (first_key, second_key)
        ori_du_b = self.get_deltaU_in_lambda_by_tuplekey(0,delta_U_tuplekey)
        def fep_backward(du_):
            obj_b = FEP(du_)
            return -obj_b.ene
        fep_exp_fe_b = fep_backward(ori_du_b)
        boot_strap_data = [ori_du_b]
        if ifbootstrap_std:
            fe_std = self.bootstrap_std(fep_backward, 200, boot_strap_data)
        else:
            fe_std = 0.0
        return fep_exp_fe_b, fe_std

    def cal_FE_last_window_bar(self,ifbootstrap_std=False):
        i=self.lambda_range-1
        first_key = self.mbar_lambda_dict[len(self.mbar_lambda_dict)-1]
        second_key = self.simulation_lambda[i]
        delta_U_tuplekey = (first_key, second_key)
        ori_du_f = self.get_deltaU_in_lambda_by_tuplekey(i,delta_U_tuplekey)
        def fep_forward(du_):
            obj_f = FEP(du_)
            return obj_f.ene
        fep_exp_fe_f = fep_forward(ori_du_f)
        boot_strap_data = [ori_du_f]
        if ifbootstrap_std:
            fe_std = self.bootstrap_std(fep_forward, 200, boot_strap_data)
        else:
            fe_std = 0
        return fep_exp_fe_f, fe_std
    
    def cal_FE_middle_window_bar(self,lambda_idx, ifbootstrap_std=False):
        i=lambda_idx
        ori_du_f = self.get_deltaU_in_lambda(i,(i+1,i),False)
        ori_du_b = self.get_deltaU_in_lambda(i+1,(i,i+1),False)
        fe_bar, fe_std = BAR_(ori_du_f, ori_du_b, method='self-consistent-iteration', maximum_iterations=1000, verbose=False)
        def bar_estimate(du_f, du_b,):
            fe_bar, fe_std = BAR_(du_f, du_b, method='self-consistent-iteration', maximum_iterations=1000, verbose=False)
            return fe_bar
        boot_strap_data = [ori_du_f, ori_du_b]
        if ifbootstrap_std:
            fe_std = self.bootstrap_std(bar_estimate, 200, boot_strap_data)
        # print(fe_bar)
        return fe_bar, fe_std
        
    def plot_overlap_matrix(self, png_file=None):
        '''
        Plot the overlap matrix using MBAR weight values.
        '''
        if png_file == None: 
            mbar__ = MBAR()
            mbar__.fit(self.all_data_unk)
            #overlap_matrix
            overlap_matx=mbar__.overlap_matrix
        else:
            mbar__ = MBAR()
            mbar__.fit(self.all_data_unk)
            #overlap_matrix
            overlap_matx=mbar__.overlap_matrix
            ax1=alchemlyb.visualisation.plot_mbar_overlap_matrix(matrix=overlap_matx)
            ax1.figure.savefig(png_file, dpi=600, format='png', transparent=True)
            plt.show()
            
        return overlap_matx

    def DC_MBAR_overlap_values(self, lambda_0, lambda_1,):
        '''Use the relationship 
        \int \frac{\rho_{0} \times \rho_{1}}{\rho_{1}+\rho_{0}} d q^{N}=overlap_0=overlap_1 to calculate the overlap_0 and overlap_1 for checking the overlap degree between two windows,
        which: 
        overlap_0=\left\langle\frac{1}{1+\exp \left(\Delta U_{1-0}-\Delta G_{1-0}\right)}\right\rangle_{0}
        overlap_1=\left\langle\frac{1}{1+\exp \left(-\Delta U_{1-0}+\Delta G_{1-0}\right)}\right\rangle_{1}
        Both of two overlap values should be between 0 and 0.5. 
        The closer the value is to 0.5, the better the overlap of the corresponding two distributions. 
        The closer the value is to 0, the two distributions corresponding to the surface have almost no overlap.
        Note that two lambda windows sample point must be same!
        
        Parameters
        ----------
        lambda_0: float(in amber rbfe), the key in the simulation_lambda(dict) to assign the first window
        lambda_1: float(in amber rbfe), the key in the simulation_lambda(dict) to assign the second window
        
        Generated key variables
        ----------
        delta_U_1to0_in_lambda_0: np.array, float, shape=(N,). \Delta U_{back-front} in front lambda window.
        delta_U_1to0_in_lambda_1: np.array, float, shape=(N,). \Delta U_{back-front} in back lambda window.
            N is the number of the flames in each lambda window.
            
        Return
        ----------
        overlap_0: float, the overlap value calculated from lambda 0
        overlap_1: float, the overlap value calculated from lambda 1
        dff:       float, the bar std calculated bwteen lambda 0 and lambda 1
        '''
        U_key_0 = self.simulation_lambda[lambda_0]
        U_key_1 = self.simulation_lambda[lambda_1]
        U_0_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_0])
        U_1_in_lambda_0 = np.array(self.all_data[lambda_0][U_key_1])
        U_0_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_0])
        U_1_in_lambda_1 = np.array(self.all_data[lambda_1][U_key_1])
        delta_U_1to0_in_lambda_0 = U_1_in_lambda_0 - U_0_in_lambda_0
        delta_U_1to0_in_lambda_0_std = np.std(delta_U_1to0_in_lambda_0)
        delta_U_1to0_in_lambda_1 = U_1_in_lambda_1 - U_0_in_lambda_1
        delta_U_1to0_in_lambda_1_std = np.std(delta_U_1to0_in_lambda_1)
        delta_U_0to1_in_lambda_1 = -delta_U_1to0_in_lambda_1
        df, dff = BAR_(delta_U_1to0_in_lambda_0, delta_U_0to1_in_lambda_1, method='self-consistent-iteration', maximum_iterations=1000, verbose=False)
        
        delta_U01_minus_df_0 = delta_U_1to0_in_lambda_0-df
        df_minus_delta_U01_1 = df-delta_U_1to0_in_lambda_1
        fermi_0 = 1/(1+np.exp(delta_U01_minus_df_0))
        fermi_1 = 1/(1+np.exp(df_minus_delta_U01_1))
        overlap_0 = np.mean(fermi_0)
        overlap_1 = np.mean(fermi_1)
        return overlap_0, overlap_1, delta_U_1to0_in_lambda_0_std, delta_U_1to0_in_lambda_1_std
    
    def check_DC_MBAR_overlap_overall(self, ):
        '''
        Calculate the overlap value between two adjacent windows through all the windows.
        '''
        init_data_df = pd.DataFrame()
        for i in range(0, self.lambda_range-1):            
            lambda_0 = i
            lambda_1 = i+1
            single_df = pd.DataFrame(columns=['lambda_value', 'overlap_0', 'overlap_1', 'delta_U_1to0_in_lambda_0_std', 'delta_U_1to0_in_lambda_1_std'], index=[lambda_0,])
            single_df['lambda_value'] = lambda_0
            single_df['overlap_0'], single_df['overlap_1'], single_df['delta_U_1to0_in_lambda_0_std'], single_df['delta_U_1to0_in_lambda_1_std'] = self.DC_MBAR_overlap_values(lambda_0, lambda_1)
            init_data_df = pd.concat([init_data_df, single_df], axis = 0 )
        init_data_df.to_csv('dc_overlap.csv')

    def get_weight(self, original_lambda, target_lambda, bool_index=None):
        '''Get the Boltzmann factor weight for changing the distribution of d_U in the original_lambda to the target_lambda.

        Parameters
        ----------
        original_lambda: int, original_lambda index
        target_lambda: int, target_lambda index
        bool_index: np.array, bool, shape=(N,) or None, if it is a np.array, it will be used to extract the weight value with the respective bool value of True.
        
        Return
        ----------
        weight_ori_to_target: np.array, float, shape=(N,) or shape=(N_true,)
                N is the flames including in each lambda window.
                N_true is the flames after filteration in each lambda window.
        '''
        if bool_index is None:
            U_key_ori = self.simulation_lambda[original_lambda]
            U_key_target = self.simulation_lambda[target_lambda]
            U_ori = np.array(self.all_data[original_lambda][U_key_ori])
            U_target = np.array(self.all_data[original_lambda][U_key_target])
        else:
            U_key_ori = self.simulation_lambda[original_lambda]
            U_key_target = self.simulation_lambda[target_lambda]
            U_ori = np.array(self.all_data[original_lambda][U_key_ori])
            U_ori = U_ori[bool_index]
            U_target = np.array(self.all_data[original_lambda][U_key_target])
            U_target = U_target[bool_index]
        dU_ori_sub_target = U_ori-U_target
        weight_ori_to_target = np.exp(dU_ori_sub_target)
        # print("The weights' ene_unit is kbT.\n")
        return weight_ori_to_target

    def get_weighted_xy(self, original_lambda, target_lambda, delta_U_tuplekey,bins=100):
        '''
        original_lambda: int, original_lambda index, used to reweight.
        target_lambda: int, target_lambda index, the lambda reweighting to.
        delta_U_tuplekey: 
        '''
        ## get original dU in original_lambda by tuple key
        origin_dU = self.get_deltaU_in_lambda_by_tuplekey(original_lambda,delta_U_tuplekey)
        ## get weight for changing the distribution of dU in original_lambda to target_lambda
        weight_factor = self.get_weight(original_lambda, target_lambda)
        ## get weighted dU distribution in target_lambda
        weight_prob_y, weight_bins_x = np.histogram(origin_dU, bins, weights= weight_factor, density=True)
        # weight_bins_x = []
        # for j in range(len(bin_edges) - 1):
        #     weight_bins_x.append((bin_edges[j] + bin_edges[j + 1]) / 2)
        return weight_bins_x, weight_prob_y
    
    def get_resmapled_dU(self, bins_x, probs_y, size=100000, pngfile=None, origin_dU=None, bins=100):
        '''
        resample dU by weighted x_bins and y_probs
        '''
        resample_obj = RESAMPLE()
        resample_dU = resample_obj.generate_resample_dU(bins_x, probs_y, size, pngfile, origin_dU, bins)
        return resample_dU

    def reweight_one_win(self,original_lambda_lst, target_lambda, delta_U_tuplekey):
        '''
        Reweighting the distribution of dU in original_lambda to target_lambda, and calculate the free energy with FEP.
        original_lambda_lst: list of int, original_lambda indexs.
        '''
        def fep_(du_):
            obj_fe = FEP(du_)
            return obj_fe.ene
        tar_win_du = self.get_deltaU_in_lambda_by_tuplekey(target_lambda, delta_U_tuplekey)
        tar_win_fe_fep = float(fep_(tar_win_du))
        reweight_fe_fep = {'win_0':[tar_win_fe_fep]}
        # tar_win_index = original_lambda_lst.index(target_lambda)
        original_lambda_lst_ = list(filter(lambda x: x != target_lambda, original_lambda_lst)) ##remove target lambda from the list
        if len(original_lambda_lst_) > 0:
            for original_lambda in original_lambda_lst_:
                # win_step = tar_win_index-original_lambda.index(original_lambda)
                win_step = original_lambda - target_lambda
                weight_bins_x, weight_prob_y = self.get_weighted_xy(original_lambda, target_lambda, delta_U_tuplekey)
                resample_dU = self.get_resmapled_dU(weight_bins_x, weight_prob_y)
                rewei_fe_fep = float(fep_(resample_dU))
                reweight_fe_fep[f'win_{win_step}'] = [rewei_fe_fep]
        return reweight_fe_fep

    def cal_delta_lambda(self, lambda_value_1, lambda_value_2):
        lambda_value_1 = np.array(lambda_value_1)
        lambda_value_2 = np.array(lambda_value_2)
        dd_coords = abs(np.subtract(lambda_value_1,lambda_value_2))
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
                try:
                    label = ABFE_labels[nonzero_indices[0]]
                    lambda_info_x = np.around((lambda_value_1[nonzero_indices[0]]+lambda_value_2[nonzero_indices[0]])/2, decimals=4)
                except:
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

    def gen_ori_lambda_lst(self, target_lambda, use_wins_step, use_wins_num):
        '''
        Generate the lambda list used to reweight according to specified use_wins_step.
        use_wins_step: float, specify the max delta lambda of forward and reverse windows to do reweighting.
                if use_wins_step is 0.1, original lambda list will consist of nearing lambdas with delta_lambda less than 0.1. 
        use_wins_num: int, specify the number of windows of each direction used to reweighting.
                if use_wins_num is 3, target lambda is 6, original lambda list will be [3,4,5, 6, 7,8,9]
        ** the target lambda will also in the lst **
        '''
        target_value = np.array(self.simulation_lambda[target_lambda])
        # print(target_value)
        ori_lambda_lst = []
        if use_wins_step:
            for lambda_index,lambda_value in self.simulation_lambda.items():
                lambda_value = np.array(lambda_value)
                delta_lambda, label, lambda_info_x = self.cal_delta_lambda(target_value, lambda_value)
                if delta_lambda <= use_wins_step:
                    ori_lambda_lst.append(lambda_index)
        elif use_wins_num:
            start_idx = target_lambda - use_wins_num
            if start_idx < 1:
                if type(self.simulation_lambda[0]) is tuple and self.simulation_lambda[0][0] == 0.0 and self.simulation_lambda[1][0] == 1.0:
                    start_idx = 1
                else:
                    start_idx = 0
            end_idx = target_lambda + use_wins_num
            if end_idx > self.lambda_range - 1:
                end_idx = self.lambda_range - 1
            ori_lambda_lst = list(np.arange(start_idx, end_idx+1,1))
        else:
            raise ValueError('use_wins_step and use_wins_num can not be both None!')
        return ori_lambda_lst

    def cal_reweight_all(self, use_wins_step=0.0, use_wins_num=0, unit='kbT', forward=True, ifcal_sum=False, output=False, ifdiagnum=False,postfix='timeall', ifplot=False, error_max=2):
        '''
        1. if len(simulation_lambda) == 1, can not do reweighting.
        2. cal each lambda: obtain a dict
                 {0:[init_dG], -n:[backward_n_dG], ..., -1:[backward_1_dG], 1:[forward_1_dG], ..., m:[forward_m_dG]}.
        3. concat results into result_df.
        4. fill nan with dG colunms.(暂未执行)
        5. output in specified unit.
        '''
        first_lambda_key = self.mbar_lambda_dict[0]
        last_lambda_key = self.mbar_lambda_dict[len(self.mbar_lambda_dict)-1]       
        def cal_(lambda_index_, delta_U_tuplekey_):
            index_info = tuple_to_str(delta_U_tuplekey_[1])+' to '+tuple_to_str(delta_U_tuplekey_[0])
            ori_lambda_lst=self.gen_ori_lambda_lst(lambda_index_,use_wins_step,use_wins_num)
            esti_ene_dict=self.reweight_one_win(ori_lambda_lst, lambda_index_, delta_U_tuplekey_)
            df_=pd.DataFrame(esti_ene_dict, index=[index_info])
            return df_
        if self.lambda_range<=1:
            print("There is only one simulation lambda, so reweighting is not supported! ")
            result_df = pd.DataFrame()
        else:
            if first_lambda_key in self.simulation_lambda.values():
                if forward:
                    delta_U_tuplekey = (self.simulation_lambda[1], self.simulation_lambda[0])
                    result_df = cal_(0, delta_U_tuplekey)
                else:
                    result_df = pd.DataFrame()
            else:
                if forward:
                    delta_U_tuplekey_0 = (self.simulation_lambda[0], first_lambda_key)
                    df_0 = cal_(0, delta_U_tuplekey_0)
                    delta_U_tuplekey_1 = (self.simulation_lambda[1], self.simulation_lambda[0])
                    df_1 = cal_(0, delta_U_tuplekey_1)
                    result_df = pd.concat([df_0, df_1])
                else:
                    delta_U_tuplekey = (first_lambda_key, self.simulation_lambda[0])
                    result_df = cal_(0, delta_U_tuplekey)
            for i in range(1, self.lambda_range-1):
                if forward:
                    delta_U_tuplekey = (self.simulation_lambda[i+1],self.simulation_lambda[i])
                else:
                    delta_U_tuplekey = (self.simulation_lambda[i-1],self.simulation_lambda[i])
                single_df = cal_(i, delta_U_tuplekey)
                result_df = pd.concat([result_df, single_df])
            if last_lambda_key in self.simulation_lambda.values():
                if forward:
                    pass
                else:
                    i = self.lambda_range-1
                    delta_U_tuplekey = (self.simulation_lambda[i-1],self.simulation_lambda[i])
                    single_df = cal_(i, delta_U_tuplekey)
                    result_df = pd.concat([result_df, single_df])
            else:
                i = self.lambda_range-1
                if forward:
                    delta_U_tuplekey = (last_lambda_key, self.simulation_lambda[i])
                    single_df = cal_(i, delta_U_tuplekey)
                    result_df = pd.concat([result_df, single_df])
                else:
                    delta_U_tuplekey_0 = (self.simulation_lambda[i-1],self.simulation_lambda[i])
                    df_0 = cal_(i, delta_U_tuplekey_0)
                    delta_U_tuplekey_1 = (self.simulation_lambda[i], last_lambda_key)
                    df_1 = cal_(i, delta_U_tuplekey_1)
                    single_df = pd.concat([df_0, df_1])
                    result_df = pd.concat([result_df, single_df])
            # print(result_df.columns)
            if ifcal_sum:
                sum_index ={True:[tuple_to_str(first_lambda_key)+' to '+tuple_to_str(last_lambda_key)],\
                            False:[tuple_to_str(last_lambda_key)+' to '+tuple_to_str(first_lambda_key)]}
                to_cal_sum_df = pd.DataFrame()
                for column in result_df.columns:
                    to_cal_sum_df[column]= result_df[column].fillna(result_df.iloc[:,0])
                sum_df = pd.DataFrame([to_cal_sum_df.sum()], columns=result_df.columns, index=sum_index[forward])
                result_df = pd.concat([result_df, sum_df])
            if unit=='kbT':
                print('unit is kbT!')
                result_df = result_df.add_suffix('(kbT)')
            elif unit == 'kcal/mol':
                # print('unit is kcal/mol!')
                result_df[result_df.select_dtypes(include=['float64']).columns] *= 0.5922
                result_df = result_df.add_suffix('(kcal/mol)')
            else:
                print('the specified unit is not known, output as kbT!')
            result_df.index.names=['delta_A_what_to_what']
            dG_df_diagonal, dG_diff_df, df_diff_precent = self.conver_reweight_df_for_heatmap(result_df, use_wins_num, ifdiagnum)
            if forward:
                file_direct = 'f_'+postfix
            else:
                file_direct = 'b_'+postfix
            if output:
                result_df.to_csv(f'rewei_use{use_wins_step}wins{use_wins_num}_{file_direct}fe.csv', sep='|')
                dG_df_diagonal.to_csv(f'diag_rewei_use{use_wins_step}wins{use_wins_num}_{file_direct}fe.csv', sep='|')
                dG_diff_df.to_csv(f'dG_diff_rewei_use{use_wins_step}wins{use_wins_num}_{file_direct}fe.csv', sep='|')
                df_diff_precent.to_csv(f'df_diff_precent_rewei_use{use_wins_step}wins{use_wins_num}_{file_direct}fe.csv', sep='|')
            if ifplot:
                plotting_obj = self.plot_obj
                plotting_obj.plot_heatmap_cmap(df=dG_diff_df, error_max=error_max, png_file=f'dG_diff_rewei_use{use_wins_step}wins{use_wins_num}_{file_direct}.png')#df, error_max=2, png_file=None
                plotting_obj.plot_heatmap_cmap(df=df_diff_precent, error_max=error_max, png_file=f'df_diff_precent_rewei_use{use_wins_step}wins{use_wins_num}_{file_direct}.png')
        return result_df, dG_diff_df, df_diff_precent

    def conver_reweight_df_for_heatmap(self, reweight_df, use_wins_num, ifdiagnum):
        
        first_lambda_key = self.mbar_lambda_dict[0]
        last_lambda_key = self.mbar_lambda_dict[len(self.mbar_lambda_dict)-1]
        if str(first_lambda_key) in reweight_df.index[-1] and str(last_lambda_key) in reweight_df.index[-1]: 
            reweight_df = reweight_df.iloc[0:len(reweight_df.index)-1,:]
        dG_df_diagonal = pd.DataFrame(columns=reweight_df.index, index=[str(i) for i in self.simulation_lambda.values()])
        dG_diff_df = pd.DataFrame(columns=reweight_df.index, index=[str(i) for i in self.simulation_lambda.values()])
        df_diff_precent = pd.DataFrame(columns=reweight_df.index, index=[str(i) for i in self.simulation_lambda.values()])
        d_index = [] ## get delta_lambda_indexs(like 0,1,2,...,-1,...) from reweight_df columns (win_0, win_1, win_2, ..., win_-1, ...). 
        use_wins_num_columns_idx = []
        column_idx=0
        for column in reweight_df.columns:
            d_lambda_index=int(re.search('-{0,1}\d{1,}',column).group(0))
            d_index.append(d_lambda_index)
            if ifdiagnum:
                if abs(d_lambda_index) <= use_wins_num:
                    use_wins_num_columns_idx.append(column_idx)
            else:
                use_wins_num_columns_idx.append(column_idx)
            column_idx = column_idx+1
        for i in range(len(reweight_df.index)):
            delta_what_to_what = reweight_df.index[i]
            win_0 = re.match('.*(?= to)',delta_what_to_what).group()
            win_1 = re.search('(?<=to ).*',delta_what_to_what).group()
            if win_0 in dG_df_diagonal.index:
                index_0 = list(dG_df_diagonal.index).index(win_0)
            else:
                index_0 = list(dG_df_diagonal.index).index(win_1)
            ##get the columns_index where not nan.
            not_nan = np.where(reweight_df.iloc[i].notna())[0]
            not_nan = np.intersect1d(not_nan, np.array(use_wins_num_columns_idx))
            index_ = index_0 + np.array(d_index)[not_nan]
            dG_df_diagonal.iloc[index_,i] = reweight_df.iloc[i,not_nan]
            dG_diff_df.iloc[index_,i] = dG_df_diagonal.iloc[index_,i] - dG_df_diagonal.iloc[index_0,i]
            df_diff_precent.iloc[index_,i] = (dG_df_diagonal.iloc[index_,i]-dG_df_diagonal.iloc[index_0,i]) / abs(dG_df_diagonal.iloc[index_0,i])
        dG_df_diagonal.index.names=['lambda_info']
        dG_diff_df.index.names=['lambda_info']
        df_diff_precent.index.names=['lambda_info']
        return dG_df_diagonal, dG_diff_df, df_diff_precent
    
