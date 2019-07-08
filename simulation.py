import os, time, sys, imp
import numpy as np

import pandas as pd

# auxiliary functions

def my_argmax(arr, axis, _max):
    if axis == 1:
        where = np.sum(arr, axis=1) == 0
        res = (np.argmax(arr, axis=1)-1).clip(0)
        res[where] = _max
        return res
    elif axis == 0:
        if np.sum(arr) == 0:
            return _max
        else:
            return max(0,np.argmax(arr, axis=0)-1)

# TODO add check that init is in bound
def init_trait(dim, bound, init="random", prng=""):
        if isinstance(init, (int, long, float)):
            trait = np.zeros(dim) + init
        elif isinstance(init, np.ndarray):
            if init.shape != dim: raise ValueError("Shape of array passed to init must be equal to dim.")
            trait = init
        elif init == "random":
            trait = bound[0] + prng.rand(dim[0], dim[1]) * (bound[1] - bound[0])
        else:
            raise ValueError("Invalid input.")
        return trait

# simulation class

class population_process:

    def __init__(self, params):
        self.time = 0.
        # inherit parameters as attributes
        for k,v in params.items():
            setattr(self, k, v)
        # initialize pseudo-random number generator
        if not hasattr(self, "prng"):
            if self.prng_init == "":
                self.prng_init = np.random.randint(1000000)
            self.prng = np.random.RandomState(self.prng_init)
        # number of living individuals
        self.N = self.init_pop_size
        # individual = (death, reproduction, birth_date, death_date, id, parent_id, alive)
        self.death = init_trait((self.N, self.number_of_ages), init=self.init_death, bound=self.bound_death, prng=self.prng)
        self.reproduction = init_trait((self.N, self.number_of_ages), init=self.init_repr, \
                bound=self.bound_repr, prng=self.prng)
        self.birth_date = np.zeros(self.N)
        self.death_date = np.full(self.N, np.nan)
        self.id = np.zeros(self.N).astype(int)
        self.max_id = 0
        self.parent_id = np.full(self.N, np.nan)
        self.alive = np.ones(self.N)
        self.agebin_mask = np.linspace(0, self.max_age, self.number_of_ages)

    def new_id(self):
        self.max_id += 1
        return self.max_id

    def get_agecard(self, ix):
        return my_argmax(self.agebin_mask >= (self.time - self.birth_date[ix]), axis=0, _max=self.number_of_ages-1)

    def get_death(self, ix):
        return self.death[ix, self.get_agecard(ix)]

    def get_reproduction(self, ix):
        return self.reproduction[ix, self.get_agecard(ix)]

    def mutate(self, x, mean, bound, dim):
        return np.clip(x + self.prng.normal(mean, self.mrate_std, dim),\
                bound[0], bound[1])

    def AR_clone(self, ix, norm):
        return self.get_reproduction(ix) * (1 - (self.mrate * self.mrate + 2 * self.mrate * (1-self.mrate))) / norm

    def AR_1mut(self, ix, norm):
        return self.AR_clone(ix, norm) + self.get_reproduction(ix) * 2 * self.mrate * (1-self.mrate) / norm

    def AR_2mut(self, ix, norm):
        return self.AR_1mut(ix, norm) + self.get_reproduction(ix) * self.mrate * self.mrate/ norm

#    def AR_death(self, ix, norm):
#        return self.AR_2mut(ix, norm) + (self.get_death(ix) + self.N * self.eta) / norm

    # this corresponds to intmax in the bd-paper
    def jump_rate_sum(self, where_alive):
        # find cardinal age of living individuals
        age = self.time - self.birth_date[where_alive]
        agecard = my_argmax(np.tile(self.agebin_mask,self.N).reshape(self.N, self.number_of_ages) >= \
                np.repeat(age,self.number_of_ages).reshape(self.N, self.number_of_ages), axis=1, _max=self.number_of_ages-1)
        # sum the jump rates
        death_rates = self.death[where_alive][np.arange(self.N), agecard]
        reproduction_rates = self.reproduction[where_alive][np.arange(self.N), agecard]
        # death due to competition is same for every individual (logistic)
        return death_rates + reproduction_rates + self.N*self.eta, np.sum((death_rates.sum(), reproduction_rates.sum(), self.N*self.N*self.eta))

    # individual = (death, reproduction, birth_date, death_date, id, parent_id, alive)
    def add(self, death, reproduction, birth_date, _id, parent_id):
        self.death = np.vstack((self.death, death))
        self.reproduction = np.vstack((self.reproduction, reproduction))
        self.birth_date = np.append(self.birth_date, birth_date)
        self.death_date = np.append(self.death_date, np.nan)
        self.id = np.append(self.id, _id)
        self.parent_id = np.append(self.parent_id, parent_id)
        self.alive = np.append(self.alive, 1)
        self.N += 1

    def transition(self):
        # get indices of live individuals
        where_alive = np.where(self.alive==1)
        # get indiividual and total jump rates
        rates, rtot = self.jump_rate_sum(where_alive)
        probs = rates / np.sum(rates)
        # gillespie: get time of next jump and then choose which individual jumped,
        # whereby the probability that an individual has jumped is proportional to its rate
        jump_time = self.prng.exponential(1) / rtot
        # NOTE ix is an index from the living population, ix_in_all is the index from the complete population object
        ix = self.prng.choice(np.arange(self.N), p=probs)
        ix_in_all = np.arange(self.id.size)[where_alive][ix]

        # choose which event took place in the jump
        u = self.prng.rand()
        a1 = self.AR_clone(ix_in_all, rates[ix])
        a2 = self.AR_1mut(ix_in_all, rates[ix])
        a3 = self.AR_2mut(ix_in_all, rates[ix])

        # if clonal birth
        if u <= a1:
            self.add(self.death[where_alive][ix], self.reproduction[where_alive][ix], self.time + jump_time, self.id[where_alive][ix], self.id[where_alive][ix])

        # if one mutation
        elif a1 < u <= a2:
            traits = [self.death[where_alive][ix], self.reproduction[where_alive][ix]]
            choice = self.prng.choice([0,1])
            # NOTE abs here makes sure that when mutation mean is negative (more detrimental mutations), mutation mean for death rate is positive as this corresponds to detrimental mutations
            traits_mut = [traits[0] if choice else \
                    self.mutate(traits[0], abs(self.mrate_mean), self.bound_death, self.number_of_ages),\
                    self.mutate(traits[1], self.mrate_mean, self.bound_repr, self.number_of_ages)\
                    if choice else traits[1]]
            self.add(traits_mut[0], traits_mut[1], self.time + jump_time, self.new_id(), self.id[where_alive][ix])

        # if two mutations
        elif a2 < u <= a3:
            self.add(self.mutate(self.death[where_alive][ix], abs(self.mrate_mean), self.bound_death, self.number_of_ages),\
                    self.mutate(self.reproduction[where_alive][ix], self.mrate_mean, self.bound_repr, self.number_of_ages),\
                    self.time + jump_time, self.new_id(), self.id[where_alive][ix])

        # if death
        else:
            self.death_date[ix_in_all] = self.time + jump_time
            self.alive[ix_in_all] = 0
            self.N -= 1

        # update time
        self.time += jump_time

    def trajectory(self):
        print "Pseudo-random number generator seed: {0}.".format(self.prng_init)
        print "Jump: {0}. Time: {1}. Population size: {2}.".format(0, round(self.time, 2), self.N)
        for j in range(self.number_of_jumps):
            if self.N > 0:
                self.transition()
                print "Jump: {0}. Time: {1}. Population size: {2}.".format(j+1, round(self.time, 2), self.N)
            else:
                print "Dieout."
                break

    def to_csv(self):
        df = pd.DataFrame()
        df["age"] = np.tile(np.arange(self.number_of_ages), len(self.id))
        df["death"] = self.death.flatten()
        df["reproduction"] = self.reproduction.flatten()
        df["birth_date"] = np.repeat(self.birth_date, self.number_of_ages)
        df["death_date"] = np.repeat(self.death_date, self.number_of_ages)
        df["id"] = np.repeat(self.id, self.number_of_ages).astype(int)
        df["parent_id"] = np.repeat(self.parent_id, self.number_of_ages).astype(int)
        df["alive"] = np.repeat(self.alive, self.number_of_ages).astype(int)
        df.to_csv(os.path.join(self.outpath+".csv"), index=False)

    # TODO maybe add output method and meta as dict

# main

# start timing
starttime = time.time()
# import parameters
params = imp.load_source("params_module", sys.argv[1]).params
# init population
pop = population_process(params)
# simulate
pop.trajectory()
# save output
pop.to_csv()
# finish timing
print 'Time taken: {} hours.'.format((time.time()-starttime)/3600.)
