# File:     dice_experiment.py
# Author:   Kurt Hamblin
# Description:  Utitlize the Random Class to:
# Simulate dice rolls where the dice weights are sampled from a Rayleigh Distribution

from Random import Random
import numpy as np
import argparse

# main function for this Python code
if __name__ == "__main__":
    
    # Set up parser to handle command line arguments
    # Run as 'python monopoly_experiment.py -h' to see all available commands
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", help="Seed number to use")
    parser.add_argument("--Nmeas",  help="Number of measurements")
    parser.add_argument("--Nexp",  help="Number of experiments")
    args = parser.parse_args()

    # default seed
    seed = 5555
    if args.seed:
        print("Set seed to %s" % args.seed)
        seed = args.seed
    random = Random(seed)
    
    Nmeas = 100
    if args.Nmeas:
        print("Number of measurements per experiment: %s" % args.Nmeas)
        Nmeas = int(args.Nmeas)
     
    Nexp = 1000
    if args.Nexp:
        print("Number of experiments: %s" % args.Nexp)
        Nexp = int(args.Nexp)

    # We hardcode in a down spin state bias of 0.4, to test against a true fermion
    unfair_bias = 0.4
    
    exp_test = np.zeros((Nexp, Nmeas))
    exp_fair = np.zeros((Nexp, Nmeas))
    for i in np.arange(Nexp):
        for j in np.arange(Nmeas):
            exp_test[i,j] = random.fermion(bias = unfair_bias)
            exp_fair[i,j] = random.fermion()

    
    # Save the normalized results
    np.savetxt('data/spin_exps.txt', exp_test, delimiter = ',', fmt="%d")
    np.savetxt('data/true_fermion_exps.txt', exp_fair, delimiter = ',', fmt="%d")
    
