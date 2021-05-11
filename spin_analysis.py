# File:     dice_analysis.py
# Author:   Kurt Hamblin
# Description:  Analyze outputs from the dice roll simulation 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from Random import Random
import matplotlib

# Import my custom matplotlib config and activate it
import my_params
custom_params = my_params.params()
matplotlib.rcParams.update(custom_params)

    
# main function for this Python code
if __name__ == "__main__":

    
    data = np.loadtxt('data/spin_exps.txt', delimiter = ',')
    data_fair = np.loadtxt('data/true_fermion_exps.txt', delimiter = ',')

    data_shape = np.shape(data)


    Nexp = data_shape[0]
    Nmeas = data_shape[1]
    
    LLR_H0 = np.zeros(Nexp)
    LLR_H1 = np.zeros(Nexp)
    
    # First We will find the probabilities according to H1
    # Then we calcualte the LLRs
    H1_probs = np.zeros((Nexp, 2), dtype = float)
    for e in np.arange(Nexp):
        #First count the frequencies of each side per experiment
        for i in np.arange(Nmeas):
            H1_probs[e, int(data[e,i])] += 1/Nmeas
        #H1_probs[e, 0] /= Nmeas
        #H1_probs[e, 1] /= Nmeas

        llr0 = 0
        llr1 = 0
        for i in np.arange(Nmeas):
            if H1_probs[e , int(data_fair[e,i])] != 0.0:
                llr0 += np.log(   (H1_probs[e , int(data_fair[e,i])] ) / (1/2) )
            if H1_probs[e , int(data[e,i]) ] != 0.0:
                llr1 += np.log( (H1_probs[e , int(data[e,i])] ) / (1/2) )
        LLR_H0[e] = llr0
        LLR_H1[e] = llr1
    
    LLR_H0 = np.sort(LLR_H0)
    LLR_H1 = np.sort(LLR_H1)
    
    # Position of 95th percentile
    index_95 = int(0.95*Nexp)
    
    # Find alpha value
    alpha = LLR_H0[index_95]
    print('Alpha: ', alpha)
    # Now we find the corresponding beta value
    # We subtract alpha from every value in LLR_H1, and the beta will then correspond to the index of the smallest value
    H1_abs_array = np.abs(LLR_H1 - abs(alpha))
    beta_index = np.nanargmin(H1_abs_array) #.argmin()
    print(H1_abs_array[beta_index])
    beta = beta_index / Nexp
    print('beta: ', beta)
    # Figure
    fig, ax = plt.subplots()
    #ax.bar(x_plot, obs_dist/Ngames, color = 'b', alpha = 0.6, label = 'observed')
    #ax.bar(x_plot, expected_dist/Ngames, color = 'r', alpha = 0.6, label = 'expected')
    BINS = 15
    ax.hist(LLR_H0, bins = BINS, color = 'b', density = True, histtype = 'step', zorder = 1, linewidth = 1.5, label = '$P(\lambda | H0)$')#, hatch = '\\\ ' , linewidth = 2, alpha = 0.7)
    ax.hist(LLR_H1, bins = BINS, color = 'g', density = True, histtype = 'step', zorder = 1, linewidth = 1.5, label = '$P(\lambda | H1)$')#, hatch = '\\\ ' , linewidth = 2, alpha = 0.7)
    
    ax.hist(LLR_H0, bins = BINS, color = 'b', density = True, histtype = 'step', hatch = '\\\\ ' , linewidth = 0.5, alpha = 0.7, zorder = 0)
    ax.hist(LLR_H1, bins = BINS, color = 'g', density = True, histtype = 'step', hatch = '\\\\ ' , linewidth = 0.5, alpha = 0.7, zorder = 0)
    
    # Draw alpha line
    ax.vlines(x = alpha, ymin = 10**-5, ymax = 1, color = 'k', lw = 2)
    
    ax.text(0.5, 6*10**-1, r'$\lambda_\alpha =$' + f'{alpha:.3f}', rotation = 'horizontal', fontsize = 12)
    ax.text(-29, 4.5*10**-1, r'$\alpha = 0.05$', color = 'b', fontsize = 16, fontweight = 'bold')
    ax.text(-29, 2.3*10**-1, r'$\beta =$' + f'{beta:.3f}', color = 'g', fontsize = 16, fontweight = 'bold')
    
    
    ax.set_xlabel(r'$\lambda = \log{\left[L(H1)/L(H0)\right]}$')
    ax.legend(loc='upper left')
    ax.set_ylabel('Probability')
    ax.set_yscale('log')
    #ax.set_xlim([-30, 30])
    ax.set_ylim([10**-4, 1])

    

    plt.show()

