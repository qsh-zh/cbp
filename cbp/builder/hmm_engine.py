import numpy as np
import hmmlearn.hmm as hmm

# pylint: skip-file

class HMMEngine(hmm.MultinomialHMM):
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        hmm.MultinomialHMM.__init__(self, n_components,
                                    startprob_prior=startprob_prior,
                                    transmat_prior=transmat_prior,
                                    algorithm=algorithm,
                                    random_state=random_state,
                                    n_iter=n_iter, tol=tol, verbose=verbose,
                                    params=params, init_params=init_params)

    def set_simprob(self, hmm_simulator):
        self.startprob_ = hmm_simulator.get_init_potential()
        self.transmat_ = hmm_simulator.get_transition_potential()
        self.emissionprob_ = hmm_simulator.get_emission_potential()

    def set_dictprob(self, dict_prob):
        self.startprob_ = dict_prob.pi
        self.transmat_ = dict_prob.transition
        self.emissionprob_ = dict_prob.emission

    def score(self, traj, time_length):
        num_seq = traj.size // time_length
        length = [time_length] * num_seq
        return super().score(traj, length) / num_seq

    def fit(self, traj, time_length):
        num_seq = traj.size // time_length
        length = [time_length] * num_seq
        super().fit(traj, length)

    def register_init(self, dict_prob):
        self.init_param = dict_prob

    def _init(self, X, lengths=None):
        self.set_dictprob(self.init_param)
