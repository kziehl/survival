import numpy as np


def round_to_closest_even(x):
    """ Calculates the closes even number for two significant digits. Necessary for Nair and Wellner table for confidence bands
    If a number is exactly uneven such as 0.31, the highest round number is chosen, in this case 0.32.
    """
    up = np.ceil(x*100)
    down = np.floor(x*100)
    if up%2 == 0 : return up/100
    elif down%2 == 0 : return down/100
    else: return x+0.01
    
    
def get_confidence_interval_bounds(kmf_obj, conf_int, technique, alpha):
    """ Calculates the lower and upper confidence bounds based on the chosen method.
    Parameters:
        kmf_obj: KaplanMeierFitter Instance
        conf_int str: linear, log, arcsin or linearb, logb, or arcsinb
        technique str: if bands are calculated option between "nair" and "wellner". The former constructs bands proportional to pointwise confidence intervals.
        alpha: significance level
   Returns:
        conf_low, conf_up arr: lower and upper bounds of the confidence interval
    """

    if(conf_int=="linear"):
        conf_int = [kmf_obj.linear_conf_int(t, alpha) for t in kmf_obj.eventtimes]
        lower, upper = zip(*conf_int)
        return list(lower), list(upper)

    elif(conf_int=="log"):
        conf_int = [kmf_obj.log_conf_int(t, alpha) for t in kmf_obj.eventtimes]
        lower, upper = zip(*conf_int)
        return list(lower), list(upper)

    elif(conf_int=="arcsin"):
        conf_int = [kmf_obj.arcsin_conf_int(t, alpha) for t in kmf_obj.eventtimes]
        lower, upper = zip(*conf_int)
        return list(lower), list(upper)

    elif(conf_int=="linearb" and technique=="nair"):
        kmf_obj.confidence_bands_survival(t_low=min(kmf_obj.eventtimes),t_up=max(kmf_obj.eventtimes), alpha=alpha, method="linear")
        return kmf_obj.linear_band_low_, kmf_obj.linear_band_up_
    elif(conf_int=="linearb" and technique=="wellner"):
        kmf_obj.confidence_bands_survival(min(kmf_obj.eventtimes),max(kmf_obj.eventtimes), alpha, method="linear", technique="wellner")
        return kmf_obj.linear_band_low_, kmf_obj.linear_band_up_
    elif(conf_int=="logb" and technique=="nair"):
        kmf_obj.confidence_bands_survival(min(kmf_obj.eventtimes),max(kmf_obj.eventtimes), alpha, method="log")
        return kmf_obj.log_band_low_, kmf_obj.log_band_up_
    elif(conf_int=="logb" and technique=="wellner"):
        kmf_obj.confidence_bands_survival(min(kmf_obj.eventtimes),max(kmf_obj.eventtimes), alpha, method="log", technique="wellner")
        return kmf_obj.log_band_low_, kmf_obj.log_band_up_
        
    elif(conf_int=="arcsinb" and technique=="nair"):
        kmf_obj.confidence_bands_survival(min(kmf_obj.eventtimes),max(kmf_obj.eventtimes), alpha, method="arcsin")
        return kmf_obj.arcsin_band_low_, kmf_obj.arcsin_band_up_
    elif(conf_int=="arcsinb" and technique=="wellner"):
        kmf_obj.confidence_bands_survival(min(kmf_obj.eventtimes),max(kmf_obj.eventtimes), alpha, method="arcsin", technique="wellner")
        return kmf_obj.arcsin_band_low_, kmf_obj.arcsin_band_up_
    elif(conf_int==None):
        return None, None
    else:
        print("Entered confidence interval calculation method is not accepted. Please choose between linear, log, and arcsin and the respective band calculations.")
        return None, None



