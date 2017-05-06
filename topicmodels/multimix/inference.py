from __future__ import division
import numpy as np

# functions for EM algorithm #


class EM():

    def __init__(self, feature_counts, K):

        """
        feature_counts: D x V document-term matrix
        K: number of latent types to estimate
        """

        self.feature_counts = feature_counts
        self.K = K

        self.N = feature_counts.shape[0]
        self.M = feature_counts.shape[1]
        self.observations = self.feature_counts.sum(axis=1)

        # seed parameters

        self.rho = np.full(self.K, 1/self.K)  # equal probability of all types
        self.mu = np.random.dirichlet(self.M*[1], self.K)

    def set_seed(self, rho_seed, mu_seed):

        """
        set seeds manually (should add dimensionality check)
        """

        self.rho = rho_seed
        self.mu = mu_seed

    def E_step(self):

        """
        compute type probabilities given current parameter estimates.
        """

        # on the first iteration of estimation, this will be called
        if not hasattr(self, 'type_prob'):
                self.type_prob = np.empty((self.N, self.K))

        temp_probs = np.zeros((self.N, self.K))

        for i in range(self.N):
            for k in range(self.K):
                temp_probs[i, k] = \
                    np.log(self.rho[k]) + np.dot(self.feature_counts[i, :],
                                                 np.log(self.mu[k, :]))

        temp_probsZ = temp_probs - np.max(temp_probs, axis=1)[:, np.newaxis]
        self.type_prob = np.exp(temp_probsZ) / \
            np.exp(temp_probsZ).sum(axis=1)[:, np.newaxis]

        return np.log(np.exp(temp_probsZ).sum(axis=1)).sum() + \
            np.max(temp_probs, axis=1).sum()

    def M_step(self):

        """
        generate new parameter estimates given updated type distribution
        """

        for k in range(self.K):
            self.rho[k] = self.type_prob[:, k].sum() / self.N

        for k in range(self.K):
            for m in range(self.M):
                temp_prob = np.dot(self.type_prob[:, k],
                                   self.feature_counts[:, m])
                if temp_prob < 1e-99:
                    temp_prob = 1e-99
                self.mu[k, m] = temp_prob / np.dot(self.type_prob[:, k],
                                                   self.observations)

    def estimate(self, maxiter=250, convergence=1e-7):

        """
        run EM algorithm until convergence, or until maxiter reached
        """

        self.loglik = np.zeros(maxiter)

        iter = 0

        while iter < maxiter:

            self.loglik[iter] = self.E_step()
            if np.isnan(self.loglik[iter]):
                    print("undefined log-likelihood")
                    break
            self.M_step()

            if self.loglik[iter] - self.loglik[iter - 1] < 0 and iter > 0:
                    print("log-likelihood decreased by %f at iteration %d"
                          % (self.loglik[iter] - self.loglik[iter - 1],
                             iter))
            elif self.loglik[iter] - self.loglik[iter - 1] < convergence \
                    and iter > 0:
                    print("convergence at iteration %d, loglik = %f" %
                           (iter, self.loglik[iter]))
                    self.loglik = self.loglik[self.loglik < 0]
                    break

            iter += 1
