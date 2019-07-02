import numpy as np
# parameters
params = {
        "outpath" : "test", # prefix for output files
        "eta" : 0.00005, # competition factor
        "mrate" : 0.01, # chance of mutation occurance per birth event
        "mrate_mean" : 0., # mutation expected value (gaussian), negative for skewed toward deleterious mutations
        "mrate_std" : 0.25, # mutation standard deviation as fraction of trait (gaussian) (absolute)
        "number_of_jumps" : 1000, # number of jumps to simulate
        "init_pop_size" : 10000, # initial population size
        "number_of_ages" : 11, # number of biological ages
        "max_age" : 10., # maximal biological age
        "init_death" : 0.5, # initial death trait ['random'/float]
        "init_repr": 1.2, # initial reproduction trait ['random'/float]
        "bound_death": [0.,2.], # bound for death rate
        "bound_repr": [0.,2,], # bound for birth rate
        "prng_init" : "" # seed value for pseudo-random number generator, generated automatically if ""
    }
