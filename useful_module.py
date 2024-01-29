import numpy as np

def DS_1sample_ztest_props(y, popmean, k=1, alternative ='two-sided', alpha=0.05):
    """
    *
    Function DS_1sample_ztest_props(y, popmean, k=1, alternative ='two-sided', alpha=0.05)
     
       This function performs a 1-sample z-test (Null Hypothesis Significance Test)
       in the spirit of R, testing 1 proportion using a normal approximation, 
       with or without the continuity correction, and the exact binomial calculation,
       assuming a Binomial(k, p)-distribution. For Bernoulli data, set k = 1 (default).
       The function also evaluates the effect size (Cramer's V2, Cohen's h).
    
    Requires:          scipy.stats.norm, scipy.stats.binom
    
    Usage:             DS_1sample_ztest_props(y, popmean = p*, k, 
                            alternative=['two-sided']/'less'/'greater', alpha = 0.05)
     
                         Note 1: y is an array with Binomial(k, p) data
                         Note 2: If y is an array with BINARY data (0, 1), 
                                 set k = 1 (Bernoulli data)
     
                         k:       Binomial(k, p) parameter = number of Bernoulli repetitions
                         alternative = 'two-sided' [default]  H1: p != p*
                                       'less'                 H1: p < p*
                                       'greater'              H1: p > p*
                         alpha:   significance level of test [default: 0.05]
     
    Return:            z, p-value, z.crit.L, z.crit.R  [ + print interpretable output to stdout ]
                       where z.crit.L and z.crit.R are the lower and upper critical values, 
                       z is the test statistic and p-value is the p-value of the test.    
     
    Author:            M.E.F. Apol
    Date:              2022-01-28, rev. 2022_08_26, 2024_01_02
    Validation:
    """
    
    import numpy as np
    from scipy.stats import norm
    from scipy.stats import binom
    
    n = len(y)
    p_ML = np.mean(y)/k
    p_star = popmean
    N = k * n
    O_1 = np.sum(y)
    O_0 = N - O_1   
    
    print(80*'-')
    print('1-sample z-test for 1 proportion:')
    if k == 1:
        print('     assuming Bernoulli(p) data for dataset')
    else:
        print('     assuming Binomial(' + str(k) + ', p) data for dataset')
    print('Observed dataset: O.1 = {:d}, O.0 = {:d}, N = {:d}'.format(O_1, O_0, N))
    print('p.ML = {:.3g}, p* = {:.3g}, alpha = {:.3g}'.format(p_ML, popmean, alpha))
    print('H0: p  = p*')
    
    if alternative == 'two-sided':
        print('H1: p != p*')
        # Assuming normal approximation (no continuity correction):
        z = (p_ML - popmean)/np.sqrt(p_star*(1-p_star)/N)
        p_value_z = 2 * norm.cdf(-np.abs(z), 0, 1)
        # Assuming normal approximation with continuity correction:
        z_c = (np.abs(p_ML - p_star) - 1/(2*N))/np.sqrt(p_star*(1-p_star)/N)
        p_value_zc = 2*norm.cdf(-np.abs(z_c), 0, 1)
        # General:
        z_crit_L = norm.ppf(alpha/2, 0, 1)
        z_crit_R = norm.ppf(1-alpha/2, 0, 1)
        # Exact Binomial calculation:
        O = p_ML*N
        E = p_star*N
        Delta = np.abs(O - E)
        O_L = E - Delta
        O_R = E + Delta
        p_value_Bin = binom.cdf(O_L, N, p_star) + binom.sf(O_R-1, N, p_star)  
        # subtract 1 from Observed_R, because of discrete distribution
    elif alternative == 'less':
        print('H1: p  < p*')
        # Assuming normal approximation (no continuity correction):
        z = (p_ML - popmean)/np.sqrt(p_star*(1-p_star)/N)
        p_value_z = norm.cdf(z)
        # Assuming normal approximation with continuity correction:
        z_c = (p_ML - p_star + 1/(2*N))/np.sqrt(p_star*(1-p_star)/N)
        p_value_zc = norm.cdf(z_c, 0, 1)
        # General:
        z_crit_L = norm.ppf(alpha, 0, 1)
        z_crit_R = float('inf')
        # Exact Binomial calculation:
        O = p_ML*N
        p_value_Bin = binom.cdf(O, N, p_star)
    elif alternative == 'greater':
        print('H1: p  > p*')
        # Assuming normal approximation (no continuity correction):
        z = (p_ML - popmean)/np.sqrt(p_star*(1-p_star)/N)
        p_value_z = 1 - norm.cdf(z, 0, 1)
        # better precision, use the survival function:
        p_value_z = norm.sf(z, 0, 1)
        # Assuming normal approximation with continuity correction:
        z_c = (p_ML - p_star - 1/(2*N))/np.sqrt(p_star*(1-p_star)/N)
        p_value_zc = norm.sf(z_c, 0, 1)  # better accuracy!
        # General:
        z_crit_L = float('-inf')
        z_crit_R = norm.ppf(1-alpha, 0, 1)
        # Exact Binomial calculation:
        O = p_ML*N
        p_value_Bin = binom.sf(O-1, N, p_star)  # subtract 1 from Observed, 
                                        # because of discrete distribution
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        z, p_value_z, z_crit_L, z_crit_R = np.nan, np.nan, np.nan, np.nan
        return(z, p_value_z, z_crit_L, z_crit_R)
    
    # Effect size (Cramer's V2):
    V2 = z**2 / N
    # Effect size (Cohen's h, see Cohen 1988, pp. 181-185):
    h = 2*np.arcsin(np.sqrt(p_ML)) - 2*np.arcsin(np.sqrt(p_star))
    
    print('* Normal approximation:')
    print('z   = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z, p_value_z, z_crit_L, z_crit_R))
    print('* Normal approximation with continuity correction:')
    print('z.c = {:.4g}, p-value = {:.4g}, z.crit.L = {:.4g}, z.crit.R = {:.4g}'.format(z_c, p_value_zc, z_crit_L, z_crit_R))
    print('* Exact Binomial calculation:')
    print('p-value = {:.4g}'.format(p_value_Bin))
    print(80*'.')
    print('Effect size: Cramer\'s V2 = {:.3g}; benchmarks V2: 0.01 = small, 0.09 = medium, 0.25 = large'.format(V2))
    print('Effect size: Cohens\' h = {:.3g}; benchmarks |h|:  0.2 = small, 0.5 = medium, 0.8 = large'.format(h))
    print(80*'-' + '\n')
    return(z, p_value_z, z_crit_L, z_crit_R)


def DS_2sample_MannWhitney_test_medians(y1, y2, alternative='two-sided', alpha=0.05):
    """
    *
    Function DS_2sample_MannWhitney_test_medians(y1, y2, alternative='two-sided', alpha=0.05)
    
       This function tests two medians eta.1 and eta.2 of data y1 and y2 (Null Hypothesis Significance Test).
       The distributions are assumed to be identical, but not necessarily normal.
    
    Requires:            scipy.stats.mannwhitneyu
    
    Usage:               DS_2sample_MannWhitney_test_medians(y1, y2, 
                              alternative=['two-sided']/'less'/'greater', alpha=0.05)
    
    Arguments:
      y1, y2             data arrays
      alternative        'two-sided' [default]   H1: eta.1 != eta.2
                         'less'                  H1: eta.1 <  eta.2
                         'greater'               H1: eta.1 >  eta.2
      alpha              significance level of test [default: 0.05]     
 
 
    Returns:             U.1, U.2, U, p_value [ + print interpretable output to stdout ]
    where
      U.1, U.2, U        Mann-Whitney statistics
      p_value            p-value of Mann-Whitney U-test
      
    Validation:          against SPSS v. 28
      
    Author:            M.E.F. Apol
    Date:              2023-12-20, revision 2024_01_02
    """

    from scipy.stats import mannwhitneyu
    
    # Additional statistics:
    y_med_1 = np.median(y1)
    y_med_2 = np.median(y2)
    n_1 = len(y1)
    n_2 = len(y2)
    
    print(80*'-')
    print('2-sample Mann-Whitney U-test for 2 medians:')
    print('     assuming identical distributions for both datasets, that may differ in location')
    print('y.med.1 = {:.4g}, y.med.2 = {:.4g}, n.1 = {:d}, n.2 = {:d}, alpha = {:.3g}'.format(y_med_1, y_med_2, n_1, n_2, alpha))
    print('H0: eta.1  = eta.2')
    
    if alternative == 'two-sided':
        print('H1: eta != eta*')
    elif alternative == 'greater':
        print('H1: eta  > eta*')
    elif alternative == 'less':
        print('H1: eta  < eta*')
    else:
        print('Wrong alternative hypothesis chosen!')
        print(80*'-' + '\n')
        U_1, U_2, U, p_value = np.nan, np.nan, np.nan, np.nan
        return(U_1, U_2, U, p_value)
    
    #res = mannwhitneyu(y1, y2, alternative=alternative, use_continuity=True)
    res = mannwhitneyu(y1, y2, alternative=alternative, use_continuity=False)
    U_1 = res.statistic
    U_2 = n_1*n_2 - U_1
    U = np.min([U_1, U_2])
    p_value = res.pvalue
    
    # -- TO DO -- Find formula for U.crit
    U_crit = np.nan
    
    # To compute an effect size for the signed-rank test, one can use the rank-biserial correlation?
    
    # Correlation coefficient:
    # From: datatab.net (dd 2023_12_19), https://maths.shu.ac.uk/mathshelp/ (dd 2023_12_19)
    # Normal approximation:
    # Expectation value:
    mu_U = n_1*n_2/2
    # Standard deviation:
    sigma_U = np.sqrt(n_1*n_2*(n_1+n_2+1)/12)
    # Standard normal statistic:
    z = (U - mu_U) / sigma_U
    # Effect size:
    r = z / np.sqrt(n_1+n_2)
    
    # Point biserial correlation r.pb
    # From: https://www.andrews.edu/~calkins/math/edrm611/edrm13.htm
    # Additional statistics:
    y_av_1 = np.mean(y1)
    y_av_2 = np.mean(y2)
    s2_1 = np.var(y1, ddof=1)
    s2_2 = np.var(y2, ddof=1)
    s2_p = ((n_1-1)*s2_1 + (n_2-1)*s2_2)/(n_1+n_2-2)
    s_p = np.sqrt(s2_p)
    r_pb = (y_av_1 - y_av_2)/s_p * np.sqrt(n_1*n_2)/(n_1 + n_2)
    
    print('U.1 = {:.3g}, U.2 = {:.3g}, U = {:.3g}, p-value = {:.3g}, z = {:.3g}, U.crit = {:.3g}'.format(U_1, U_2, U, p_value, z, U_crit))
    print(80*'.')
    print('Effect size: r    = {:.3g}; benchmarks |r|: 0.1 = small, 0.3 = medium, 0.5 = large'.format(r))
    # print('Effect size: r.pb = {:.3g}; benchmarks r: 0.1 = small, 0.3 = medium, 0.5 = large (?)'.format(r_pb))
    print(80*'-')
    
    return(U_1, U_2, U, p_value);

def DS_Q_Q_Plot(y, est = 'robust', title = 'Q-Q plot', **kwargs):
    """
    *
    Function DS_Q_Q_Plot(y, est = 'robust', **kwargs)
    
       This function makes a normal quantile-quantile plot (Q-Q-plot), also known
       as a probability plot, to visually check whether data follow a normal distribution.
    
    Requires:            numpy, scipy.stats.iqr, scipy.stats.norm, matplotlib.pyplot
    
    Arguments:
      y                  data array
      est                Estimation method for normal parameters mu and sigma:
                         either 'robust' (default), or 'ML' (Maximum Likelihood),
                         or 'preset' (given values)
      N.B. If est='preset' than the *optional* parameters mu, sigma must be provided:
      mu                 preset value of mu
      sigma              preset value of sigma
      title              Title for the graph (default: 'Q-Q plot')
      
    Returns:
      Estimated mu, sigma, n, and expected number of datapoints outside CI in Q-Q-plot.
      Q-Q-plot
      
    Author:            M.E.F. Apol
    Date:              2020-01-06, revision 2022-08-30, 2023-12-19
    """
    
    import numpy as np
    from scipy.stats import iqr # iqr is the Interquartile Range function
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    # First, get the optional arguments mu and sigma:
    mu_0 = kwargs.get('mu', None)
    sigma_0 = kwargs.get('sigma', None)
    
    n = len(y)
    
    # Calculate order statistic:
    y_os = np.sort(y)
  
    # Estimates of mu and sigma:
    # ML estimates:
    mu_ML = np.mean(y)
    sigma2_ML = np.var(y)
    sigma_ML = np.std(y) # biased estimate
    s2 = np.var(y, ddof=1)
    s = np.std(y, ddof=1) # unbiased estimate
    # Robust estimates:
    mu_R = np.median(y)
    sigma_R = iqr(y)/1.349

    # Assign values of mu and sigma for z-transform:
    if est == 'ML':
        mu, sigma = mu_ML, s
    elif est == 'robust':
        mu, sigma = mu_R, sigma_R
    elif est == 'preset':
        mu, sigma = mu_0, sigma_0
    else:
        print('Wrong estimation method chosen!')
        return()
        
    print('Estimation method: ' + est)
    print('n = {:d}, mu = {:.4g}, sigma = {:.4g}'.format(n, mu,sigma))
    
    # Expected number of deviations (95% confidence level):
    n_dev = np.round(0.05*n)
    
    print('Expected number of data outside CI: {:.0f}'.format(n_dev))
         
    # Perform z-transform: sample quantiles z.i
    z_i = (y_os - mu)/sigma

    # Calculate cumulative probabilities p.i:
    i = np.array(range(n)) + 1
    p_i = (i - 0.5)/n

    # Calculate theoretical quantiles z.(i):
    z_th = norm.ppf(p_i, 0, 1)

    # Calculate SE or theoretical quantiles:
    SE_z_th = (1/norm.pdf(z_th, 0, 1)) * np.sqrt((p_i * (1 - p_i)) / n)

    # Calculate 95% CI of diagonal line:
    CI_upper = z_th + 1.96 * SE_z_th
    CI_lower = z_th - 1.96 * SE_z_th

    # Make Q-Q plot:
    plt.plot(z_th, z_i, 'o', color='k', label='experimental data')
    plt.plot(z_th, z_th, '--', color='r', label='normal line')
    plt.plot(z_th, CI_upper, '--', color='b', label='95% CI')
    plt.plot(z_th, CI_lower, '--', color='b')
    plt.xlabel('Theoretical quantiles, $z_{(i)}$')
    plt.ylabel('Sample quantiles, $z_i$')
    plt.title(title + ' (' + est + ')')
    plt.legend(loc='best')
    plt.show()
    pass;
