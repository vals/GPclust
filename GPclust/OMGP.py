# Copyright (c) 2012, 2013, 2014 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from .collapsed_mixture import CollapsedMixture
import GPy
from GPy.util.linalg import mdot, pdinv, backsub_both_sides, dpotrs, jitchol, dtrtrs
from GPy.util.linalg import tdot_numpy as tdot

class OMGP(CollapsedMixture):
    """
    Overlapping mixtures of Gaussian processes
    """
    def __init__(self, X, Y, K=2, kernels=None, variance=1., alpha=1., prior_Z='symmetric', name='OMGP'):

        N, self.D = Y.shape
        self.Y = Y
        self.YYT = tdot(self.Y)

        self.X = X

        if kernels == None:
            self.kern = []
            for i in range(K):
                self.kern.append(GPy.kern.RBF(input_dim=1))
        else:
            self.kern = kernels

        CollapsedMixture.__init__(self, N, K, prior_Z, alpha, name)

        self.link_parameter(GPy.core.parameterization.param.Param('variance', variance, GPy.core.parameterization.transformations.Logexp()))
        self.link_parameters(*self.kern)


    def parameters_changed(self):
        """ Set the kernel parameters
        """
        self.update_kern_grads()

    def do_computations(self):
        """
        Here we do all the computations that are required whenever the kernels
        or the variational parameters are changed.
        """
        if len(self.kern) < self.K:
            self.kern.append(self.kern[-1].copy())
            self.link_parameter(self.kern[-1])

        if len(self.kern) > self.K:
            for kern in self.kern[self.K:]:
                self.unlink_parameter(kern)

            self.kern = self.kern[:self.K]

    def update_kern_grads(self):
        """
        Set the derivative of the lower bound wrt the (kernel) parameters
        """
        grad_Lm_variance = 0.0

        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            B_inv = np.diag(1. / (self.phi[:, i] / self.variance))

            # Numerically more stable version using cholesky decomposition
            #alpha = linalg.cho_solve(linalg.cho_factor(K + B_inv), self.Y)
            #K_B_inv = pdinv(K + B_inv)[0]
            #dL_dK = .5*(tdot(alpha) - K_B_inv)

            # Make more stable using cholesky factorization:
            Bi, LB, LBi, Blogdet = pdinv(K+B_inv)

            tmp = dpotrs(LB, self.YYT)[0]
            GPy.util.diag.subtract(tmp, 1)
            dL_dB = dpotrs(LB, tmp.T)[0]

            kern.update_gradients_full(dL_dK=.5*dL_dB, X=self.X)

            # variance gradient

            #for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            #I = np.eye(self.N)

            B_inv = np.diag(1. / ((self.phi[:, i] + 1e-6) / self.variance))
            #alpha = np.linalg.solve(K + B_inv, self.Y)
            #K_B_inv = pdinv(K + B_inv)[0]
            #dL_dB = tdot(alpha) - K_B_inv
            grad_B_inv = np.diag(1. / (self.phi[:, i] + 1e-6))

            grad_Lm_variance += 0.5 * np.trace(np.dot(dL_dB, grad_B_inv))
            grad_Lm_variance -= .5*self.D * np.einsum('j,j->',self.phi[:, i], 1./self.variance)

        self.variance.gradient = grad_Lm_variance

    def bound(self):
        """
        Compute the lower bound on the marginal likelihood (conditioned on the
        GP hyper parameters).
        """
        GP_bound = 0.0

        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            B_inv = np.diag(1. / ((self.phi[:, i] + 1e-6) / self.variance))

            # Make more stable using cholesky factorization:
            Bi, LB, LBi, Blogdet = pdinv(K + B_inv)

            # Data fit
            # alpha = linalg.cho_solve(linalg.cho_factor(K + B_inv), self.Y)
            # GP_bound += -0.5 * np.dot(self.Y.T, alpha).trace()
            GP_bound -= .5 * dpotrs(LB, self.YYT)[0].trace()

            # Penalty
            # GP_bound += -0.5 * np.linalg.slogdet(K + B_inv)[1]
            GP_bound -= 0.5 * Blogdet

            # Constant, weighted by  model assignment per point
            #GP_bound += -0.5 * (self.phi[:, i] * np.log(2 * np.pi * self.variance)).sum()
            GP_bound -= 0.5 * self.D * np.einsum('j,j->', self.phi[:, i], np.log(2 * np.pi * self.variance))

        return  GP_bound + self.mixing_prop_bound() + self.H

    def vb_grad_natgrad(self):
        """
        Natural Gradients of the bound with respect to phi, the variational
        parameters controlling assignment of the data to GPs
        """
        grad_Lm = np.zeros_like(self.phi)
        for i, kern in enumerate(self.kern):
            K = kern.K(self.X)
            I = np.eye(self.N)

            B_inv = np.diag(1. / ((self.phi[:, i] + 1e-6) / self.variance))
            K_B_inv, L_B, _, _ = pdinv(K + B_inv)
            alpha, _ = dpotrs(L_B, self.Y)
            dL_dB_diag = np.sum(np.square(alpha), 1) - np.diag(K_B_inv)

            grad_Lm[:,i] = -0.5 * self.variance * dL_dB_diag / (self.phi[:,i]**2 + 1e-6)

        grad_phi = grad_Lm + self.mixing_prop_bound_grad() + self.Hgrad

        natgrad = grad_phi - np.sum(self.phi * grad_phi, 1)[:, None]
        grad = natgrad * self.phi

        return grad.flatten(), natgrad.flatten()

    def predict(self, Xnew, i):
        """ Predictive mean for a given component
        """
        kern = self.kern[i]
        K = kern.K(self.X)
        kx = kern.K(self.X, Xnew)

        # Predict mean
        # This works but should Cholesky for stability
        B_inv = np.diag(1. / (self.phi[:, i] / self.variance))
        K_B_inv = pdinv(K + B_inv)[0]
        mu = kx.T.dot(np.dot(K_B_inv, self.Y))

        # Predict variance
        kxx = kern.K(Xnew, Xnew)
        va = self.variance + kxx - kx.T.dot(np.dot(K_B_inv, kx))

        return mu, va

    def predict_components(self, Xnew):
        """The predictive density under each component"""
        mus = []
        vas = []
        for i in range(len(self.kern)):
            mu, va = self.predict(Xnew, i)
            mus.append(mu)
            vas.append(va)

        return np.array(mus)[:, :, 0].T, np.array(vas)[:, :, 0].T

    def sample(self, Xnew, gp=0, size=10, full_cov=True):
        ''' Sample the posterior of a component
        '''
        mu, va = self.predict(Xnew, gp)

        samples = []
        for i in range(mu.shape[1]):
            if full_cov:
                smp = np.random.multivariate_normal(mean=mu[:, i], cov=va, size=size)
            else:
                smp = np.random.multivariate_normal(mean=mu[:, i], cov=np.diag(np.diag(va)), size=size)

            samples.append(smp)

        return np.stack(samples, -1)

    def get_X_linspace(self, gp_num=0, n=50, assignment_limit=None):
        ''' Get linspace X values for a given GP component.
        '''
        if not assignment_limit:
            XX = np.linspace(self.X.min(), self.X.max(), 50)[:, None]
        else:
            mask = self.phi[:, gp_num] > assignment_limit
            XX = np.linspace(self.X[mask, 0].min(), self.X[mask, 0].max(), 50)[:, None]

        return XX

    def plot_component(self, gp_num, x_vals, col, out_dim=0, style='1d'):
        ''' Plot an individual GP component.
        '''
        from matplotlib import pylab as plt
        from matplotlib import cm

        YY_mu, YY_var = self.predict(x_vals, gp_num)
        if style == '1d':
            plt.fill_between(x_vals[:, 0],
                             YY_mu[:, out_dim] - 2 * np.sqrt(YY_var[:, out_dim]),
                             YY_mu[:, out_dim] + 2 * np.sqrt(YY_var[:, out_dim]),
                             alpha=0.1,
                             facecolor=col)
            plt.plot(x_vals, YY_mu[:, out_dim], c=col, lw=2);

        elif style =='2d':
            plt.plot(YY_mu[:, 0], YY_mu[:, 1], c='w', lw=4);
            plt.plot(YY_mu[:, 0], YY_mu[:, 1], c=col, lw=2);

    def plot(self, gp_num=0, plot_data=True, plot_dims=[], subplot_dims=False, assignment_limit=None):
        """
        Plot the mixture of Gaussian Processes.
        Supports plotting 1d and 2d regression.

        Arguments
        ---------

        gp_num (int) When plotting data, which GP assignment should data
                     points be colored by.
        plot_data (bool) Whether data observations should be plotted.
        plot_dims (list) A list of output dimensions to plot for each GP.
        subplot_dims (bool) Whether to plot each output dimension in an
                            individual subplot.
        assignment_limit (float) Sets the X-limits when plotting so that plotting
                                 range is only in a region where there are
                                 observatiosn with at least assignment_limit
                                 assignment probability for each GP.
        """
        from matplotlib import pylab as plt
        from matplotlib import cm

        try:
            Tango = GPy.plotting.Tango
        except:
            Tango = GPy.plotting.matplot_dep.Tango
        Tango.reset()

        scatter_kwargs = {'c': self.phi[:, gp_num], 'cmap': cm.RdBu, 'vmin': 0., 'vmax': 1., 'lw': 0.5}

        if len(plot_dims) == 0:
            plot_dims = range(self.Y.shape[1])

        if (len(plot_dims) == 1) & (not subplot_dims):
            if plot_data:
                plt.scatter(self.X, self.Y, **scatter_kwargs)
                plt.colorbar(label='GP {} assignment probability'.format(gp_num))

            for i in range(self.phi.shape[1]):
                XX = self.get_X_linspace(i, assignment_limit=assignment_limit)
                col = Tango.nextMedium()
                self.plot_component(i, XX, col)

        if (len(plot_dims) == 2) & (not subplot_dims):
            if plot_data:
                plt.scatter(self.Y[:, 0], self.Y[:, 1], **scatter_kwargs)
                plt.colorbar(label='GP {} assignment probability'.format(gp_num))

            for i in range(self.phi.shape[1]):
                XX = self.get_X_linspace(i, assignment_limit=assignment_limit)
                col = Tango.nextMedium()
                self.plot_component(i, XX, col, style='2d')

        if (len(plot_dims) > 2) & (not subplot_dims):
            raise NotImplementedError('More than 2 dimensions only supported on individual subplots. '\
                                      'Use the subplot_dims=True keyword.')

        if subplot_dims:
            n_total = len(plot_dims)
            n_x = np.floor(np.sqrt(n_total))
            n_y = int(np.ceil(n_total / n_x))
            n_x = int(n_x)
            for d in plot_dims:
                Tango.reset()
                plt.subplot(n_x, n_y, d + 1)

                if plot_data:
                    plt.scatter(self.X, self.Y[:, d], **scatter_kwargs)
                    plt.colorbar(label='GP {} assignment probability'.format(gp_num))


                for i in range(self.phi.shape[1]):
                    XX = self.get_X_linspace(i, assignment_limit=assignment_limit)
                    col = Tango.nextMedium()
                    self.plot_component(i, x_vals=XX, col=col, out_dim=d)


    def plot_probs(self, gp_num=0):
        """
        Plot assignment probabilities for each data point of the OMGP model
        """
        from matplotlib import pylab as plt
        plt.scatter(self.X, self.phi[:, gp_num])
        plt.ylim(-0.1, 1.1)
        plt.ylabel('GP {} assignment probability'.format(gp_num))
