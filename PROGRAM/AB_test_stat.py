from scipy import stats
import pandas as pd
import numpy as np

def generate_abtest_data(N_control, N_experiment, p_control, p_experiment, 
                         control_label='Control',test_label='Experiment',random_size=True):
    """Returns a pandas dataframe with fake CTR data
    Example:
    Parameters:
        N_control (int): sample size for control group
        N_experiment (int): sample size for test group
        p_control (float): conversion rate of control group
        p_experiment (float): conversion rate of test group
        random_size(boolean): Decide whether the experiment/control group size is drawn randomly
        control_label (str)
        test_label (str)
    Returns:
        df (df)
    """
    # initiate empty container
    data = []
    # total amount of rows in the data
    N = N_control + N_experiment
    # distribute events based on proportion of group size
    group_bern = stats.bernoulli(N_control / (N_control + N_experiment))
    # initiate bernoulli distributions from which to randomly sample
    control_bern = stats.bernoulli(p_control)
    experiment_bern = stats.bernoulli(p_experiment)
    if random_size:
        for idx in range(N):
            # initite empty row
            row = {}
            # for 'ts' column
            # assign group based on 50/50 probability
            row['group'] = group_bern.rvs()
            if row['group'] == 0:
                # assign conversion based on provided parameters
                row['converted'] = control_bern.rvs()
            else:
                row['converted'] = experiment_bern.rvs()
            # collect row into data container
            data.append(row)    
    else:
        for idx in range(N_control):
            # initite empty row
            row = {}
            row['group'] = 0
            row['converted']=control_bern.rvs()
            data.append(row)
        for idx in range(N_experiment):
            # initite empty row
            row = {}
            row['group'] = 1
            row['converted']=experiment_bern.rvs()
            data.append(row)
    # convert data into pandas dataframe
    df = pd.DataFrame(data)

    # transform group labels of 0s and 1s to user-defined group labels
    df['group'] = df['group'].apply(
        lambda x: control_label if x == 0 else test_label)

    return df


def pooled_prob(N_A, N_B, X_A, X_B):
    """Returns pooled probability for two samples"""
    return (X_A + X_B) / (N_A + N_B)

def pooled_SE(N_A, N_B, X_A, X_B):
    """Returns the pooled standard error for two samples"""
    p_hat = pooled_prob(N_A, N_B, X_A, X_B)
    SE = np.sqrt(p_hat * (1 - p_hat) * (1 / N_A + 1 / N_B))
    return SE

def confidence_interval(sample_mean=0, sample_std=1, sample_size=1,
                        sig_level=0.05):
    """Returns the confidence interval as a tuple"""
    z = z_val(sig_level)
    left = sample_mean - z * sample_std / np.sqrt(sample_size)
    right = sample_mean + z * sample_std / np.sqrt(sample_size)
    return (left, right)

def z_val(sig_level=0.05, two_tailed=True):
    """Returns the z value for a given significance level"""
    z_dist = stats.norm(0,1)
    if two_tailed:
        sig_level = sig_level/2
        area = 1 - sig_level
    else:
        area = 1 - sig_level
    z = z_dist.ppf(area)
    return z

def p_val(N_A, N_B, p_A, p_B):
    """Returns the p-value for an A/B test"""
    return stats.binom(N_A, p_A).pmf(p_B * N_B)

def ab_dist(stderr, d_hat=0, group_type='control'):
    """Returns a distribution object depending on group type
    Examples:
    Parameters:
        stderr (float): pooled standard error of two independent samples
        d_hat (float): the mean difference between two independent samples
        group_type (string): 'control' and 'test' are supported
    Returns:
        dist (scipy.stats distribution object)
    """
    if group_type == 'control':
        sample_mean = 0
    elif group_type == 'test':
        sample_mean = d_hat
    # create a normal distribution which is dependent on mean and std dev
    dist = stats.norm(sample_mean, stderr)
    return dist

def min_sample_size(bcr, mde, power=0.8, sig_level=0.05):
    """Returns the minimum sample size to set up a split test
    Arguments:
        bcr (float): probability of success for control, sometimes
        referred to as baseline conversion rate
        mde (float): minimum change in measurement between control
        group and test group if alternative hypothesis is true, sometimes
        referred to as minimum detectable effect
        power (float): probability of rejecting the null hypothesis when the
        null hypothesis is false, typically 0.8
        sig_level (float): significance level often denoted as alpha,
        typically 0.05
    Returns:
        min_N: minimum sample size
    """
    # standard normal distribution to determine z-values
    standard_norm = stats.norm(0,1)
    # find Z_beta from desired power
    Z_beta = standard_norm.ppf(power)
    # find Z_alpha
    Z_alpha = standard_norm.ppf(1-sig_level/2)
    # Calculate minimum sample size
    min_N = (Z_alpha*np.sqrt(2*bcr*(1-bcr))+Z_beta*np.sqrt(bcr*(1-bcr)+(bcr+mde)*(1-bcr-mde)))**2 / mde**2
    return min_N