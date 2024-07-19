import numpy as np
from numpy import concatenate, diff,cumsum
from .plotting import PLOTTING

class RESAMPLE():
    def __init__(self, ):
        pass

    def generate_resample_dU(self, bins_x, probs_y, size=100000, pngfile=None, origin_dU=None, bins=100):
        '''
        Do the resampling according to the given d_u distribution.
        Parameters
        -------------
        bins: int, the number of bins when plotting d_u distribution.
        size: int, the amount of the samples when resampling.
        pngfile: str, name of the pngfile showing the original and resample d_u distribution. 
                    If None, pngfile will not be saved.
        Return
        -------------
        resample_dU: array-like, float, of size 'size'. 
        '''
        # import scipy.interpolate as spi
        cdf = self.init_cdf(probs_y,bins_x)
        resample_dU = self.sample_cdf(cdf, bins_x,size)
        if pngfile:
            plot_obj = PLOTTING()
            plot_obj.plot_resample_dU_distribution(origin_dU,resample_dU, pngfile, bins=bins)
        else:
            pass    
        return resample_dU
    
    def init_cdf(self, hist,bins):
        """Initialize CDF from histogram

        Parameters
        -------------
        hist : array-like, float of size N
                Histogram height 
        bins : array-like, float of size N+1
                Histogram bin boundaries 

        Returns:
        -------------
        cdf : array-like, float of size N+1
        """
        steps  = diff(bins) / 2  # Half bin size
        slopes = diff(hist) / (steps[:-1]+steps[1:]) 
        ends = concatenate(([hist[0] - steps[0] * slopes[0]], 
                            hist[:-1] + steps[:-1] * slopes,
                            [hist[-1] + steps[-1] * slopes[-1]]))
        sum = cumsum(ends)
        sum -= sum[0]
        sum /= sum[-1]
        return sum

    def sample_cdf(self, cdf,bins,size):
        """Sample a CDF defined at specific points.
        Linear interpolation between defined points 

        Parameters
        -------------
           cdf : array-like, float, size N
               CDF evaluated at all points of bins. First and 
               last point of bins are assumed to define the domain
               over which the CDF is normalized. 
           bins : array-like, float, size N
               Points where the CDF is evaluated.  First and last points 
               are assumed to define the end-points of the CDF's domain
           size : integer, non-zero
               Number of samples to draw 
        Returns
        -------------
            sample : array-like, float, of size ``size``
                 Random sample
        """
        random_sample = np.interp(np.random.random(size), cdf, bins)
        return random_sample
