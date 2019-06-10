import os
import numpy as np
import pandas as pd

# parameters
params = {
        "outpath" : "test",
        "competition" : 0.0005,
        "mrate" : 0.05,
        "mrate_mean" : 0.05,
        "mrate_std" : np.sqrt(0.05),
        "number_of_jumps" : 1000,
        "init_pop_size" : 1000,
        "number_of_ages" : 10,
        "max_age" : 3.,
        "maturity" : 0,
        "init_surv" : 1.2,
        "init_repr": 1.2,
        "bound_surv": [0.5,1.5],
        "bound_repr": [0.5,1.5],
        # maximal jump intensity by individual
        "intmax" : 2. + 0.0005,
        "prng_init" : ""
    }

class population_process:

    def init_trait(self, dim, bound, init="random", prng=""):
        # dim is an int-array of length 2
        # bound is a double-array of length 2
        # init is either 'random' or a double
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

    def __init__(self, params):
        self.time = 0.
        # inherit parameters as attributes
        for k,v in params.items():
            setattr(self, k, v)
        # initialize pseudo-random number generator
        if not hasattr(self, "prng"):
            if self.prng_init != "": self.prng = np.random.RandomState(self.prng_init)
            else: self.prng = np.random.RandomState()
        # number of living individuals
        self.N = self.init_pop_size
        # individual = (survival, reproduction, birth_date, death_date, id, parent_id, alive)
        self.survival = self.init_trait((self.N, self.number_of_ages), init=self.init_surv, bound=self.bound_surv, prng=self.prng)
        self.reproduction = self.init_trait((self.N, self.number_of_ages - self.maturity), init=self.init_repr, \
                bound=self.bound_repr, prng=self.prng)
        self.birth_date = np.zeros(self.N)
        self.death_date = np.zeros(self.N)
        self.id = np.zeros(self.N)
        self.parent_id = np.full(self.N, np.nan)
        self.alive = np.ones(self.N)

    def new_id(self):
        return np.max(self.id)+1

    def get_age(self, ix):
        return np.where(np.linspace(0, self.max_age, self.number_of_ages) <= self.time - self.birth_date[ix])[0][-1]

    def get_survival(self, ix):
        return self.survival[ix, self.get_age(ix)]

    def get_reproduction(self, ix):
        return self.reproduction[ix, self.get_age(ix) - self.maturity]

    def mutate(self, x):
        return (x + self.prng.normal(self.mrate_mean, self.mrate_std, self.number_of_ages)).clip(min=0)

    def AR_clone(self, ix):
        return self.get_reproduction(ix) * (np.square(self.mrate) + 2*self.mrate*(1-self.mrate)) / (self.intmax * self.N)

    def AR_1mut(self, ix):
        return self.AR_clone(ix) + self.get_reproduction(ix) * 2 * self.mrate * (1-self.mrate) / (self.intmax * self.N)

    def AR_2mut(self, ix):
        return self.AR_1mut(ix) + self.get_reproduction(ix) * np.square(self.mrate) / (self.intmax * self.N)

    def AR_death(self, ix):
        return self.AR_2mut(ix) + (1 - self.get_survival(ix) + self.N * self.competition) / (self.intmax * self.N)

    # individual = (survival, reproduction, birth_date, death_date, id, parent_id, alive)
    def add(self, survival, reproduction, birth_date, _id, parent_id):
        self.survival = np.vstack((self.survival, survival))
        self.reproduction = np.vstack((self.reproduction, reproduction))
        self.birth_date = np.append(self.birth_date, birth_date)
        self.death_date = np.append(self.death_date, 0.)
        self.id = np.append(self.id, _id)
        self.parent_id = np.append(self.parent_id, parent_id)
        self.alive = np.append(self.alive, 1)

    def transition(self):
        jump_time = self.prng.exponential(1) / (self.intmax * np.square(self.N))
        u = self.prng.rand()
        ix = self.prng.choice(np.arange(self.id.size)[np.where(self.alive==1)])

        # acceptance/rejection method
        while u > self.AR_death(ix):
            jump_time += self.prng.exponential(1) / (self.intmax * np.square(self.N))
            u = self.prng.rand()
            ix = self.prng.choice(np.arange(self.id.size)[np.where(self.alive==1)])

        # when accepted
        a1 = self.AR_clone(ix)
        a2 = self.AR_1mut(ix)
        a3 = self.AR_2mut(ix)

        # if clonal birth
        if u <= a1:
            self.add(self.survival[ix], self.reproduction[ix], self.time + jump_time, self.id[ix], self.id[ix])

        # if one mutation
        elif a1 < u <= a2:
            traits = [self.survival[ix], self.reproduction[ix]]
            choice = self.prng.choice([0,1])
            traits_mut = [traits[0] if choice else self.mutate(traits[0]),\
                    self.mutate(traits[1]) if choice else traits[1]]
            self.add(traits_mut[0], traits_mut[1], self.time + jump_time, self.new_id(), self.id[ix])

        # if two mutations
        elif a2 < u <= a3:
            self.add(self.mutate(self.survival[ix]), self.mutate(self.reproduction[ix]),\
                    self.time + jump_time, self.new_id(), self.id[ix])

        # if death
        else:
            self.death_date[ix] = self.time + jump_time
            self.alive[ix] = 0

        # update time and number of living
        self.time += jump_time
        self.N = int(np.sum(self.alive))

    def trajectory(self):
        print "Jump: {0}. Time: {1}. Population size: {2}.".format(0, round(self.time, 2), self.N)
        for j in range(self.number_of_jumps):
            if self.N > 0:
                self.transition()
                print "Jump: {0}. Time: {1}. Population size: {2}.".format(j, round(self.time, 2), self.N)
            else:
                print "Dieout."
                break

    def to_csv(self):
        df = pd.DataFrame()
        df["age"] = np.tile(np.arange(self.number_of_ages), len(self.id))
        df["survival"] = self.survival.flatten()
        df["reproduction"] = self.reproduction.flatten()
        df["birth_date"] = np.repeat(self.birth_date, self.number_of_ages)
        df["death_date"] = np.repeat(self.death_date, self.number_of_ages)
        df["id"] = np.repeat(self.id, self.number_of_ages).astype(int)
        df["parent_id"] = np.repeat(self.parent_id, self.number_of_ages).astype(int)
        df["alive"] = np.repeat(self.alive, self.number_of_ages).astype(int)
        df.to_csv(os.path.join(self.outpath+".csv"), index=False)

    # TODO maybe add output method and meta as dict

# main
pop = population_process(params)
pop.trajectory()
pop.to_csv()
