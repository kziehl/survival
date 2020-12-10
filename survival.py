import numpy as np
import statsmodels.api as sm 
import statsmodels.formula as smf
from scipy.stats import norm, chi2
import scipy.stats
import pandas as pd
import data_tables
import itertools
import numpy.linalg 
import matplotlib.pyplot as plt
import math
import helper 


class KaplanMeierFitter:
    """ Create a KM object """

    def __init__(self,time ,event ,**kwargs):
        """Transform a table that contains time under study to include number of events, eventtimes, and number of individuals at risk.

            Parameters:
                time array: for right censored data, this is the follow up time. 
                event array: either 0 or 1. If 1, event of interest occured. 0 means that the data is right-censored.
                
            Optional Paramters:
                entry_time: time at which participants entered the study. Helpful if one conducts a cohort study. time then is the time at which subject exists the study.

            Returns:
                pd.DataFrame: transformed dataframe with eventtimes, number of events, and the number of individuals at risk.
            
        """
        entry_time = kwargs.setdefault("entry_time", 0)

        
        time_under_study = time - entry_time
        data = {"time_under_study": time_under_study, "indicator": event, "entry_time": entry_time}
        df = pd.DataFrame(data)
        df = df.sort_values(by="time_under_study")
        df_number_of_death_events= df[["time_under_study","indicator"]].groupby("time_under_study").sum().reset_index()
        df["drop_out"] = 1 - df["indicator"]
        df_number_of_dropouts = df[["time_under_study","drop_out"]].groupby("time_under_study").sum().reset_index()
        df_combined = pd.merge(df_number_of_death_events, df_number_of_dropouts, on="time_under_study")
        number_at_risk = np.zeros(df_combined.shape[0])
        number_at_risk[0] = df.shape[0]
        for index in range(1, number_at_risk.size):
            number_at_risk[index] = number_at_risk[index-1] - df_combined["indicator"][index-1] - df_combined["drop_out"][index-1]

        df_combined["number_at_risk"] = number_at_risk
        self.df_censored = df_combined[["time_under_study","indicator","number_at_risk"]]
        df_combined = df_combined[df_combined["indicator"]>0][["time_under_study","indicator","number_at_risk"]]
        self.df = df_combined #remnant for old code

        # experiment characeteristics
        self.eventtimes = np.asarray(df_combined["time_under_study"])
        self.events = np.asarray(df_combined["indicator"])
        self.risk = np.asarray(df_combined["number_at_risk"])

        # data characteristics
        self.survival_times_ = [self.estimator(t) for t in self.eventtimes]
        self.chf_ = [self.chf(t) for t in self.eventtimes]
        self.survival_variance_ = [self.variance(t) for t in self.eventtimes] # be careful some of them will be inherited...

    def survplot(self, **kwargs):
        """ Plots survival curve.
        Optional Parameters:
            risk_table bool: True or False specifying whether to show or not show a table containing the number of people at risk at any given timepoint. Default is False.
            conf_int str: Specify which confidence interval should be used according to selected method. Possible values for pointwise confidence intervals "arcsin", "linear", and "log". Possible confidence bands are "linearb", "logb", and "arcsinb". Default is "None".
            t_max double: specify the maximum timepoint up to which the survival curve should be plotted. Default is maximum observed event time.
            figsize (float, float): width, height in inches
            xlabel str: Set the label for the x-axis. Default is "Time".
            color str: Set the color of the graph. Default is "black".
            ax axes.Axes: Pass an axes object. Default is "None".
            linestyle: 	{'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            label str: Specify the label for the graph
            technique: If confidence bands be plotted, decided if "nair" or "wellner". Default "nair".
            alpha: size of confidence interval for point estimation or confidence bands. For bands, only values of 1%, 5% and 10% are allowed. Default 0.05.

            Note: This is more of a quick plot tool and you should use directly with the data to produce the results that you want.
            TODO: might want to change this again such that the survival function extends to the largest value which could be censored.
        """
        #risk_table = kwargs.setdefault("risk_table", False)
        conf_int = kwargs.setdefault("conf_int", None)
        t_max = kwargs.setdefault("t_max",max(self.eventtimes))
        figsize = kwargs.setdefault("figsize", (10,8))
        xlabel = kwargs.setdefault("xlabel", "Time")
        color = kwargs.setdefault("color", "black")
        ax = kwargs.setdefault("ax", None)
        linestyle = kwargs.setdefault("linestyle", "-")
        label = kwargs.setdefault("label", "")
        alpha = kwargs.setdefault("alpha", 0.05)
        technique = kwargs.setdefault("technique", "nair")


        if (ax is None):
            fig, ax = plt.subplots(figsize=figsize)
        
        conf_low, conf_up = helper.get_confidence_interval_bounds(self, conf_int, technique, alpha)

        ax.set_xlim(0, t_max) #should be maximum observed time, should be censored or larger than observed event time
        ax.set_ylim(0,1.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Survival Function")
        x_axes = [0]
        x_axes.extend(self.eventtimes)
        x_axes.extend([t_max])
        y_axes = [1]
        y_axes.extend(self.survival_times_)
        y_axes.extend([self.survival_times_[-1]])
        ax.plot(x_axes, y_axes, drawstyle="steps-post", color=color, linestyle=linestyle, label=label)
        if (conf_low == None or conf_up == None):
            ax.legend(loc=0)
            return None
        else:
            ax.plot(self.eventtimes, conf_low, drawstyle="steps-post", linestyle = "-.", color = "black", label="Confidence Interval")
            ax.plot(self.eventtimes, conf_up, drawstyle="steps-post", linestyle = "-.", color = "black")
            ax.legend(loc=0)
            return None
        

    def chfplot(self, **kwargs):
        """ Plots cumulative hazard function
        Optional Parameters:
            risk_table bool: True or False specifying whether to show or not show a table containing the number of people at risk at any given timepoint. Default is False.
            conf_int str: Specify which confidence interval should be used according to selected method. Possible values are "arcsin", "linear", and "log". Default is "None".
            t_max double: specify the maximum timepoint up to which the survival curve should be plotted. Default is maximum observed event time.
            figsize (float, float): width, height in inches
            xlabel str: Set the label for the x-axis. Default is "Time".
            color str: Set the color of the graph. Default is "black".
            ax axes.Axes: Pass an axes object. Default is "None".
            linestyle: 	{'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            label str: Specify the label for the graph
            TODO: confidence intervals and pass axes and figure as arguments
            Tested: yes
        """
        #risk_table = kwargs.setdefault("risk_table", False)
        #conf_int = kwargs.setdefault("conf_int", "None")
        t_max = kwargs.setdefault("t_max",max(self.eventtimes))
        figsize = kwargs.setdefault("figsize", (10,8))
        xlabel = kwargs.setdefault("xlabel", "Time")
        color = kwargs.setdefault("color", "black")
        ax = kwargs.setdefault("ax", None)
        linestyle = kwargs.setdefault("linestyle", "-")
        label = kwargs.setdefault("label", "")

        if (ax is None):
            fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlim(0, t_max) #should be maximum observed time, should be censored or larger than observed event time
        ax.set_ylim(0,1.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative Hazard Function")
        x_axes = [0]
        x_axes.extend(self.eventtimes)
        x_axes.extend([t_max])
        y_axes = [0]
        y_axes.extend(self.chf_)
        y_axes.extend([self.chf_[-1]])
        ax.plot(x_axes, y_axes, drawstyle="steps-post", color=color, linestyle=linestyle, label=label)
        ax.legend(loc=0)

        return None
        



    def two_sample_comparison(self, *args, **kwargs):
        """Conduct a hypothesis test by comparing the hazard rates of 2 populations such that
        H0: h1(t) = h2(t) for all t<=tau
        H1: at least one of the hj's is different for some t<=tau
        Here tau is the largest time at which all of the groups have at least one subject at risk.
        Parameters:
            args: array of all KaplanMeierFitter objects for comparison
            kwargs:
                alpha: default = 0.05, significance level
                tau: maximum time to up to which hazard rates should be compared, default is largest observed event time
                method: logrank, fleming: specify p and q parameters (klein, moeschberger (7.3.9))Default: p=1, q=0
        Returns: 
            tuple: test_statistic, p-value

        TODO: other methods to implement such as gehan, tarone-ware, peto-peto, modified peto
        """
        sample_size = len(args) + 1
        alpha = kwargs.setdefault("alpha", 0.05)
        method = kwargs.setdefault("method", "logrank")
        tau_max = kwargs.setdefault("tau", 0)
        

        list_of_kmf_objects = [self.df_censored]
        for kmf_object in args:
            list_of_kmf_objects.append(kmf_object.df_censored)
        
        # this only works for 2 object so far, the problem are the indicies
        merged_dataframes = pd.merge(self.df_censored, list_of_kmf_objects[1], how="outer", on="time_under_study", sort=True)
        merged_dataframes["indicator_x"] = merged_dataframes["indicator_x"].fillna(0)
        merged_dataframes["indicator_y"] = merged_dataframes["indicator_y"].fillna(0)
        merged_dataframes = merged_dataframes.fillna(method="bfill")
        merged_dataframes = merged_dataframes.ffill()
        merged_dataframes = merged_dataframes[(merged_dataframes["indicator_x"] > 0) | (merged_dataframes["indicator_y"] > 0)].copy()

        number_of_people_at_risk_per_time = merged_dataframes["number_at_risk_x"] + merged_dataframes["number_at_risk_y"]
        number_of_events_per_time = merged_dataframes["indicator_x"] + merged_dataframes['indicator_y']
        self.merged_dataframes = merged_dataframes

        # this further implementation only works for K=2
        
        if(method == "logrank"):
            term1 = merged_dataframes["number_at_risk_x"] * (number_of_events_per_time/number_of_people_at_risk_per_time)
            term2 = merged_dataframes["indicator_x"] - term1
            prod1 = merged_dataframes["number_at_risk_x"] / number_of_people_at_risk_per_time
            prod2 = (1 - prod1)
            prod3 = ((number_of_people_at_risk_per_time - number_of_events_per_time) / (number_of_people_at_risk_per_time -1)) * number_of_events_per_time
            term3 = prod1 * prod2 * prod3
            chi2_statistic = (sum(term2)/np.sqrt(sum(term3)))**2
            pval = 1 - chi2.cdf(chi2_statistic, 1)
            return chi2_statistic, pval

        elif(method =="fleming"):

            p = kwargs.setdefault("p", 1)
            q = kwargs.setdefault("q", 0)
            term1 = merged_dataframes["number_at_risk_x"] * (number_of_events_per_time/number_of_people_at_risk_per_time)
            term2 = merged_dataframes["indicator_x"] - term1
            prod1 = merged_dataframes["number_at_risk_x"] / number_of_people_at_risk_per_time
            prod2 = (1 - prod1)
            prod3 = ((number_of_people_at_risk_per_time - number_of_events_per_time) / (number_of_people_at_risk_per_time -1)) * number_of_events_per_time
            term3 = prod1 * prod2 * prod3
            survival = [1]
            survival.extend(list(itertools.accumulate((1-number_of_events_per_time/number_of_people_at_risk_per_time), lambda x,y: x*y)))
            survival = np.asarray(survival[:-1])
            weights = survival**p * (1-survival)**q
            chi2_statistic = (sum(term2 * weights) / np.sqrt(sum(weights**2 * term3)))**2
            pval = 1 - chi2.cdf(chi2_statistic, 1)
            return chi2_statistic, pval

   
  

    def mean_survival(self, alpha=0.05):
        """Calculate the mean survival time of an individual in the provided interval
        Parameters:
        Returns:
            mean surival time in the given interval
        """
        survival = []
        for t in self.eventtimes:
            survival.append(self.estimator(t))
        
        # this only works if our last event time is censored
        a = list(self.eventtimes)
        b = a[:-1]
        b = [0] + b
        diff = np.flip(np.asarray(a) - np.asarray(b))
        c = survival[:-1]
        c = list(np.flip(np.asarray(c))) + [1]
        d = c * diff
        self.mean_survival_times_ = list(itertools.accumulate(d))
        self.mean_survival_time = self.mean_survival_times_[-1]
        self.mean_survival_times_ = self.mean_survival_times_[:-1]

        # compute the vairance estimate, this requires to declare a bunch of nonsense variables, this how thing only works if last observation is not an event

        flipit= np.flip(self.mean_survival_times_)
        num = self.events[:-1] * flipit**2
        den = self.risk[:-1] * (self.risk[:-1] - self.events[:-1]) # the last value is right-censored
        sol = num / den
        self.mean_variance_ = sum(sol) # standard error is the square root

        self.mean_lower_ = self.mean_survival_time - norm.ppf(1-alpha/2) * np.sqrt(self.mean_variance_)
        self.mean_upper_ = self.mean_survival_time + norm.ppf(1-alpha/2) * np.sqrt(self.mean_variance_)

    def linear_conf_int(self,t , alpha = 0.05):
        """Calculate the linear confindence interval for the survival function at time t
        Parameters: 
            t double: timepoint at which confidence interval should be constructed. t < t_end
            alpha double: Type I error size. Determines the length of the confidence interval
        Returns:
            tuple: returns lower confidence interval and upper confidence interval
        Tested: yes
        """
        
        sigma_sq = self.variance(t)/self.estimator(t)**2 
        linear_conf_low_ = self.estimator(t) - norm.ppf(1-alpha/2) * np.sqrt(sigma_sq) * self.estimator(t)
        linear_conf_up_ = self.estimator(t) + norm.ppf(1-alpha/2) * np.sqrt(sigma_sq) * self.estimator(t)
        return linear_conf_low_, linear_conf_up_ 


    def log_conf_int(self,t , alpha=0.05):
        """Calculate the log-transformed confidence interval for the survival function at time t. Proposed by Borgan and Liestol(1990).
        Parameters: 
            t double: timepoint at which confidence interval should be constructed. t < t_end
            alpha double: Type I error size. Determines the length of the confidence interval
        Returns:
            tuple: returns lower confidence interval and upper confidence interval
        Tested: yes
        """
        
        sigma_sq = self.variance(t)/self.estimator(t)**2 
        theta = np.exp(norm.ppf(1-alpha/2) * np.sqrt(sigma_sq) / np.log(self.estimator(t)))
        log_conf_low_ = self.estimator(t)**(1/theta)
        log_conf_up_ = self.estimator(t)**(theta)
        return log_conf_low_, log_conf_up_

    def arcsin_conf_int(self,t , alpha=0.05):
        """Calculate the arcsin-square root transformation confidence interval for the survival function at time t. Proposed by Borgan and Liestol(1990).
        Parameters: 
            t double: timepoint at which confidence interval should be constructed. t < t_end
            alpha double: Type I error size. Determines the length of the confidence interval
        Returns:
            tuple: returns lower confidence interval and upper confidence interval
        Tested: yes
        """
        
        sigma_sq = self.variance(t)/self.estimator(t)**2 
        _ = np.arcsin(np.sqrt(self.estimator(t))) - 0.5 * norm.ppf(1-alpha/2) * np.sqrt(sigma_sq) * np.sqrt(self.estimator(t)/(1-self.estimator(t)))
        lower = max(0, _)
        _ = np.arcsin(np.sqrt(self.estimator(t))) + 0.5 * norm.ppf(1-alpha/2) * np.sqrt(sigma_sq) * np.sqrt(self.estimator(t)/(1-self.estimator(t)))
        upper = min(np.pi/2, _)
        arcsin_conf_low_ = (np.sin(lower))**2
        arcsin_conf_up_ = (np.sin(upper))**2
        return arcsin_conf_low_, arcsin_conf_up_

    

    def variance(self, t):
        """Calculate the variance of the Kaplan-Meier estimator estimated by Greenwood's formula. For the standard error, the square root has to be taken.
        Parameters:
            t double: time at which estimator should be calculated
        Returns:
            variance double: Variance
        Tested:
            using dataset from chapter 4.2 Klein
        """
        

        if (t < min(self.eventtimes)):
            print("Variance cannot be computed. Observation time is lower than the first eventtime. The first event has been observed at ", min(self.eventtimes),".")
            return None
        else:
            sum = 0
            _ = self.eventtimes[self.eventtimes <= t]
            length = len(_)
            events = self.events[:length]
            risk = self.risk[:length]
            for index in range(length):
                if((risk[index]*(risk[index]-events[index]))==0): sum += 0 #not sure if that is correct
                else:
                    sum += events[index]/(risk[index]*(risk[index]-events[index]))
            variance_ = self.estimator(t)**2 * sum
            return variance_
        
        


    def estimator(self, t):
        """Calculate the Kaplan-Meier estimate of the survival function at a given time t. The estimator is not defined for time t > t_max
        Parameters:
            t double: time at which estimator should be calculated
        Returns:
            double: Kaplan-Meier estimate of the survival function at time t
        Tested:
            using dataset from chapter 4.2 Klein
        """
        if (t < min(self.eventtimes)):
            print("Kaplan-Meier estimate of the survival function is not defined. Return default value of 1.")
            return 1
        else:
            kaplan_meier_estimate_ = 1
            _ = self.eventtimes[self.eventtimes <= t]
            length = len(_)
            events = self.events[:length]
            risk = self.risk[:length]
            for index in range(length):
                kaplan_meier_estimate_ *= (1 - events[index]/risk[index])
            
            return kaplan_meier_estimate_

    def chf(self, t):
        """Calculate the cumulative hazard function at the time based on the Product-Limit estimator
        Parameters:
            t double: time at which estimator should be calculated
        Returns:
            Cumulative hazard function
        Tested:
            using dataset from chapter 4.2 Klein
        """
        if (self.estimator(t) == 0): return 1 #check what to do with KM is zero
        else:
            return - np.log(self.estimator(t))

    def one_sample_test(self, chf_reference_truncated, chf_reference_time_of_event, **kwargs):
        """One sample hypothesis test, to compare the sum of weighted differences between observed and expected hazard rates. Want to compare that the hazard rates are equal for all t <= tau. 
        Tau is typically the maximum observed time. The statistic follows a chi-square distribution.
        Parameters:
            chf_reference_truncatd np.array: cumulative hazard rate at the age of entry. This time is left-truncated. This has to be from a reference population.
            chf_reference_time_of_event np.array: cumulative hazard rate at the age of event. This can also be understood as the time under study. This has to be from a reference population.
            kwargs: 
                max_time: time up to which the hazard rate should be compared. Defaults to maximum observed time.
                method str: describes the weights that should be used to calculate the weighted difference. (see Klein and Moeschberger e.q 7.21). Defaults to the log-rank test
            where the weights are equal to the number at people at risk.
        Returns: chi-squared statistic and p-value
        """

        #include if it is one or two-sided
        method = kwargs.setdefault("method", "logrank")

        if (method=="logrank"):
            observed_deaths = sum(self.events)
            expected_death = sum(chf_reference_time_of_event - chf_reference_truncated)
            #max_time = kwargs.setdefault("max_time", max(self.eventtimes))
            chi_sq_statistic = (observed_deaths - expected_death)**2 / expected_death
            return chi_sq_statistic, 1 - scipy.stats.chi2.cdf(chi_sq_statistic, df=1)
        if (method=="fleming"):
            p = kwargs.setdefault("p", 0)
            q = kwargs.setdefault("q", 0)
            survival = np.exp(-chf_reference_time_of_event) #not sure if this should be the difference
            weights = self.df_censored["number_at_risk"] * survival**p * (1 - survival)**q
            observed = sum(weights * self.df_censored["indicator"] / self.df_censored["number_at_risk"])
            expected = sum(weights * 0.045)
            z = observed - expected
            var = sum(weights**2 * 0.045 / self.df_censored["number_at_risk"])
            chi_sq_statistic = (z/var)**2
            return chi_sq_statistic, 1 - scipy.stats.chi2.cdf(chi_sq_statistic, df=1)


    def confidence_bands_survival(self, t_low, t_up, alpha=0.05, method="linear", technique="nair"):
        """Calculate the confidence band of the survival function between two times points. These bands are called equal probability bands and t_l must be greater or equal than the smallest observed 
        event time. t_up must be equal or lower than the largest observed time. It allows to specify which method should be used to calculate the confidence bands bounderies.
        Parameters:
            t_low double: lower time point where to start the confidence band
            t_up double: upper bound where the confidence band should end
            alpha double: alpha-level to construct confidence interval
            method str: Choose between "linear", "log", and "arcsin"
            technique str: Default Nair. Provide confidence bounds which are proportional to the pointwise confidence intervals. Other value would be "wellner". This allows a lower limit for t_l, 
            namely 0. Both technique have three different methods.
        Returns:
            confidence band array
        Notes:
            confidence bans are only calculated from the minimum observed time up to the maximum observed time. If changes should be made, please work directly
            with the data.
        TODO: confidence bands for chf
        """
        # This function is not working properly. The values from the table are not correctly determined! Solution are almost correct. I am
        # a custom function to determine to closest even number.

        n = self.df["number_at_risk"][0]

        sigma_sq_low = self.variance(t_low) / self.estimator(t_low)**2
        a_l = (n * sigma_sq_low) / (1 + n * sigma_sq_low)
        sigma_sq_up = self.variance(t_up) / self.estimator(t_up)**2
        a_u = (n * sigma_sq_up) / (1 + n * sigma_sq_up)

        # I am adding the lower and upper time to the band. If t_low and t_up are the min and max, those values are contained twice
        if(t_low == min(self.eventtimes) and t_up == max(self.eventtimes)):
            band_times = self.eventtimes
        else:
            band_times = [t_low]
            band_times.extend(self.eventtimes[np.logical_and((t_low<=self.eventtimes),(self.eventtimes<=t_up))])
            band_times.append(t_up)

        if(technique=="nair"):

            if (min(self.eventtimes) <= t_low and t_up <= max(self.eventtimes)):

                if (alpha==0.05): confidence_table = data_tables.ep_band_5
                if (alpha==0.01): confidence_table = data_tables.ep_band_1
                if (alpha==0.1): confidence_table = data_tables.ep_band_10

                conf_coeff = confidence_table.loc[helper.round_to_closest_even(a_u), helper.round_to_closest_even(a_l)] 

                if(method=="linear"):
                    self.linear_band_low_ = []
                    self.linear_band_up_ = []
                    for t in band_times:
                        kaplan_meier_estimate_ = self.estimator(t)
                        sigma_sqrt = np.sqrt(self.variance(t) / self.estimator(t)**2)
                        self.linear_band_low_.append(kaplan_meier_estimate_ - conf_coeff * sigma_sqrt * kaplan_meier_estimate_)
                        self.linear_band_up_.append(kaplan_meier_estimate_ + conf_coeff * sigma_sqrt * kaplan_meier_estimate_)
                    return None
                elif(method=="log"):
                    self.log_band_low_ = []
                    self.log_band_up_ = []
                    for t in band_times:
                        kaplan_meier_estimate_ = self.estimator(t)
                        sigma_sqrt = np.sqrt(self.variance(t) / self.estimator(t)**2)
                        theta = np.exp((conf_coeff * sigma_sqrt) / np.log(kaplan_meier_estimate_))
                        self.log_band_low_.append(kaplan_meier_estimate_**(1/theta))
                        self.log_band_up_.append(kaplan_meier_estimate_**(theta))
                    return None
                elif(method=="arcsin"):
                    self.arcsin_band_low_ = []
                    self.arcsin_band_up_ = []
                    for t in band_times:
                        kaplan_meier_estimate_ = self.estimator(t)
                        sigma_sqrt = np.sqrt(self.variance(t) / self.estimator(t)**2)
                        term1 = np.arcsin(kaplan_meier_estimate_**(1/2))
                        term2 = 0.5 * conf_coeff * sigma_sqrt * (kaplan_meier_estimate_/(1-kaplan_meier_estimate_))**(1/2)
                        self.arcsin_band_low_.append(np.sin(max(0,term1-term2))**2)
                        self.arcsin_band_up_.append(np.sin(min(np.pi/2,term1+term2))**2)
                    return None
                else:
                    print("Specified method is not valid. Please choose between linear, log, and arcsin.")
                    
            else: 
                print("Confidence bands cannot be calculated. Invalid range of eventtimes.")
                return None

        elif(technique=="wellner"):

            if (min(self.eventtimes) <= t_low and t_up <= max(self.eventtimes)): #apparently t_low can also take on the value 0

                if (alpha==0.05): confidence_table = data_tables.hallwellner_5
                if (alpha==0.01): confidence_table = data_tables.hallwellner_1
                if (alpha==0.1): confidence_table = data_tables.hallwellner_10

                conf_coeff = confidence_table.loc[helper.round_to_closest_even(a_u), helper.round_to_closest_even(a_l)]

                if(method=="linear"):
                    self.linear_band_low_ = []
                    self.linear_band_up_ = []
                    for t in band_times:
                        kaplan_meier_estimate_ = self.estimator(t)
                        sigma_sq = self.variance(t)/kaplan_meier_estimate_**2
                        self.linear_band_low_.append(kaplan_meier_estimate_ - (conf_coeff * (1+n*sigma_sq) * kaplan_meier_estimate_)/np.sqrt(n))
                        self.linear_band_up_.append(kaplan_meier_estimate_ + (conf_coeff * (1+n*sigma_sq) * kaplan_meier_estimate_)/np.sqrt(n))
                    return None
                elif(method=="log"):
                    self.log_band_low_ = []
                    self.log_band_up_ = []
                    for t in band_times:
                        kaplan_meier_estimate_ = self.estimator(t)
                        sigma_sq = self.variance(t)/kaplan_meier_estimate_**2
                        theta = np.exp((conf_coeff * (1+n*sigma_sq)) / (np.sqrt(n)*np.log(kaplan_meier_estimate_)))
                        self.log_band_low_.append(kaplan_meier_estimate_**(1/theta))
                        self.log_band_up_.append(kaplan_meier_estimate_**(theta))
                    return None
                elif(method=="arcsin"):
                    self.arcsin_band_low_ = []
                    self.arcsin_band_up_ = []
                    for t in band_times:
                        kaplan_meier_estimate_ = self.estimator(t)
                        sigma_sq = self.variance(t)/kaplan_meier_estimate_**2
                        term1 = np.arcsin(kaplan_meier_estimate_**(1/2))
                        term2 = 0.5 * (conf_coeff * (1+n*sigma_sq))/np.sqrt(n) * (kaplan_meier_estimate_/(1-kaplan_meier_estimate_))**(1/2)
                        self.arcsin_band_low_.append(np.sin(max(0,term1-term2))**2)
                        self.arcsin_band_up_.append(np.sin(min(np.pi/2,term1+term2))**2)
                    return None
                else:
                    print("Specified method is not valid. Please choose between linear, log, and arcsin.")
                    
            else: 
                print("Confidence bands cannot be calculated. Invalid range of eventtimes.")
                return None
        
        else:
            print("Technique is not defined. Please choose between 'wellner' and 'nair'.")

class NelsonAalenFitter(KaplanMeierFitter):

    def __init__(self ,time , event, **kwargs):
        KaplanMeierFitter.__init__(self,time, event)

        # data characteristics
        self.survival_times_ = [self.survival(t) for t in self.eventtimes]
        self.chf_ = [self.chf(t) for t in self.eventtimes]
        self.chf_variance_ = [self.chf_variance(t) for t in self.eventtimes]
        self.chf_linear_conf_int_ = [self.chf_linear_conf_int(t) for t in self.eventtimes]
        self.chf_log_conf_int_ = [self.chf_log_conf_int(t) for t in self.eventtimes]
        self.chf_arcsin_conf_int_ = [self.chf_arcsin_conf_int(t) for t in self.eventtimes]


    def chf(self, t):
        """Calculate the cumulative hazard function at the time based on the Nelson-Aalon estimator
        Parameters:
            t double: time at which estimator should be calculated
        Returns:
            Cumulative hazard function
        Tested:
            using dataset from chapter 4.2 Klein
        """

        if ( t < min(self.eventtimes)):
            print("Observed time is below or equal the first event time. Default value is being used.")
            return 0
        else:
            chf_ = 0
            _ = self.eventtimes[self.eventtimes <= t]
            length = len(_)
            events = self.events[:length]
            risk = self.risk[:length]
            for index in range(length):
                chf_ += events[index]/risk[index]
            return chf_

    def survival(self, t):
        """Calculate an estimate for the survival function at time t based on the Nelson-Aalen estimator.
        Parameters:
            t double: time at which estimator should be calculated
        Returns:
            survival function at time t
        Tested:
            using dataset from chapter 4.2 Klein
        """
        
        survival_ = np.exp(-self.chf(t))
        return survival_

    

    def chf_variance(self, t):
        """Calculate the variance of the cumulative hazard function at time t based on the Nelson-Aalen estimator. Based on Aalen(1978b). Standard Error can be calculated by taking the square root.
        Parameters:
            t double: time at which estimator should be calculated
        Returns:
            variance of the cumulative hazard function at time t
        Tested:
            using dataset from chapter 4.2 Klein
        """

        if ( t < min(self.eventtimes)):
            print("Observed time is below the first event time. Variance cannot be determined.")
            return None
        else:
            variance_ = 0
            _ = self.eventtimes[self.eventtimes <= t]
            length = len(_)
            events = self.events[:length]
            risk = self.risk[:length]
            for index in range(length):
                variance_ += events[index]/(risk[index]**2)
            
            return variance_

    def chf_linear_conf_int(self,t , alpha = 0.05):
        """Calculate the linear confindence interval for the cumulative hazard function at time t.
        Parameters: 
            t double: timepoint at which confidence interval should be constructed. t < t_end
            alpha double: Type I error size. Determines the length of the confidence interval
        Returns:
            tuple: returns lower confidence interval and upper confidence interval
        """
        
        linear_conf_low_ = self.chf(t) - norm.ppf(1-alpha/2) * self.chf_variance(t)
        linear_conf_up_ = self.chf(t) + norm.ppf(1-alpha/2) * self.chf_variance(t)
        return linear_conf_low_, linear_conf_up_ 


    def chf_log_conf_int(self,t , alpha=0.05):
        """Calculate the log-transformed confidence interval for the cumulative hazard function at time t. 
            t double: timepoint at which confidence interval should be constructed. t < t_end
            alpha double: Type I error size. Determines the length of the confidence interval
        Returns:
            tuple: returns lower confidence interval and upper confidence interval
        """
        
        phi = np.exp(norm.ppf(1-alpha/2) * self.chf_variance(t) / self.chf(t))
        log_conf_low_ = self.chf(t)/phi
        log_conf_up_ = self.chf(t)*(phi)
        return log_conf_low_, log_conf_up_

    def chf_arcsin_conf_int(self,t , alpha=0.05):
        """Calculate the arcsin-square root transformation confidence interval for the cumulative hazard function at time t. 
        Parameters: 
            t double: timepoint at which confidence interval should be constructed. t < t_end
            alpha double: Type I error size. Determines the length of the confidence interval
        Returns:
            tuple: returns lower confidence interval and upper confidence interval
        """
        
        lower_parent = min(np.pi/2, np.arcsin(np.exp(-self.chf(t)/2)) + 0.5 * norm.ppf(1-alpha/2) * self.chf_variance(t) * (np.exp(self.chf(t)-1))**(-1/2))
        upper_parent = max(0, np.arcsin(np.exp(-self.chf(t)/2)) - 0.5 * norm.ppf(1-alpha/2) * self.chf_variance(t) * (np.exp(self.chf(t)-1))**(-1/2))
        arcsin_conf_low_ = -2 * np.log(np.sin(lower_parent))
        arcsin_conf_up_ = -2 * np.log(np.sin(upper_parent))
        return arcsin_conf_low_, arcsin_conf_up_

   
        
        
    





    
