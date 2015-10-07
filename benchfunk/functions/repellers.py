from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

__all__ = ['RepellersModel']

class RepellersModel:
    def __init__(
            self,
            weights,
            centers,
            scales,
            gravity=0.03,
            viscosity=0.1,
            sigma=0.02,
            npaths=100):

        # parameters controlling the initial state distribution.
        self.initRangeX = np.r_[-.5, .5]
        self.initRangeY = np.r_[-.2, .3]

        # parameters controlling the transition model.
        self.gravity = gravity
        self.viscosity = viscosity
        self.sigma = sigma
        self.npaths = npaths

        # parameters controlling the rewards.
        self.weights = np.array(weights, ndmin=1, copy=True)
        self.centers = np.array(centers, ndmin=2, copy=True)
        self.scales = np.array(scales, ndmin=2, copy=True)

    def __call__(self, x):
        return self.getExpectedReward(x, self.npaths)

    def sampleInit(self, N):
        """Return N samples from the initial state model."""
        return np.random.uniform(
            low  = np.r_[self.initRangeX[0], self.initRangeY[0], 0., 0.],
            high = np.r_[self.initRangeX[1], self.initRangeY[1], 0., 0.],
            size = [N, 4])

    def sampleTransition(self, x, u):
        """Sample transitions given N states and actions x[i] and u[i]."""
        xnew = x.copy()
        xnew[:, 2:4] *= (1.0 - self.viscosity)
        xnew[:, 2:4] += u + np.r_[0, -self.gravity]
        xnew[:, 0:2] += xnew[:, 2:4]
        xnew[:, 2:4] += np.random.normal(scale=self.sigma, size=(x.shape[0], 2))
        return xnew

    def samplePolicy(self, theta, x):
        """
        Sample actions u[i] for each of N states x[i], where the policy is
        parameterized by theta. In particular theta is a collection of 3-tuples
        where theta[3n] is the force of the nth repulsor and theta[3n+1:3n+2]
        are the x and y positions.
        """
        u = 0.0
        for w, xpos, ypos in np.asarray(theta).reshape(-1,3):
            d = x[:,0:2] - np.r_[xpos, ypos] + .001
            d2 = np.sum(d**2, axis=1).reshape(-1, 1)
            u += w * d / (d2**1.5)
        return u

    def sampleReward(self, x):
        """Sample rewards for each given state x[i]."""
        r = 0.0
        for w, center, scale in zip(self.weights, self.centers, self.scales):
            a = (x[:,0:2] - center) / scale
            r += w * np.exp(-0.5 * np.sum(a**2, axis=1))
        return r

    def samplePaths(self, theta, N, horizon):
        """Sample paths (position only) under the given policy."""
        paths = []
        x = self.sampleInit(N)
        for t in xrange(horizon):
            paths.append(x[:, 0:2])
            u = self.samplePolicy(theta, x)
            x = self.sampleTransition(x,u)
        return np.swapaxes(paths, 0, 1)

    def getExpectedReward(self, theta, N=1000, horizon=100, gamma=1.0):
        """
        Get the expected reward for the given policy where we use N sample
        paths, each with the given time horizon and use a discount factor of
        gamma.
        """
        x = self.sampleInit(N)
        r = 0.0
        for n in xrange(horizon):
            r += (gamma**n) * np.sum(self.sampleReward(x)) / N
            u = self.samplePolicy(theta, x)
            x = self.sampleTransition(x, u)
        return r

    def plot(self, ax, theta=None, N=0, horizon=20):
        """
        Make an informative plot of the model. If theta is not None display the
        repellers and if N is also greater than zero then plot some sample paths
        with the given time horizon.
        """
        ax.cla()

        for loc, weight, (width, height) in zip(self.centers, self.weights, self.scales):
            ellipse = matplotlib.patches.Ellipse(
                loc,
                3 * width,
                3 * height,
                color='g',
                alpha=0.2*weight,
                ec='none')
            ax.add_patch(ellipse)

        # draw the initial state distribution.
        ax.add_patch(plt.Rectangle(
            np.r_[self.initRangeX[0], self.initRangeY[0]],  # corner of the rect.
            self.initRangeX[1] - self.initRangeX[0],     # width.
            self.initRangeY[1] - self.initRangeY[0],     # height.
            edgecolor=(0.0, 0.0, 0.0, 0.5),
            linewidth=2,
            facecolor=(1.0, 1.0, 1.0, 0.0)),
            )

        # draw the repellers if any are given.
        if theta is not None:
            for w, xpos, ypos in np.asarray(theta).reshape(-1,3):
                ax.plot([xpos],
                        [ypos],
                        'ro',
                        markersize=max([2, np.sqrt(1000*w)]),
                        alpha=0.3)

        # draw some paths.
        if N>0 and theta is not None:
            paths = self.samplePaths(theta, N, horizon)
            for i in xrange(len(paths)):
                ax.plot(paths[i,:,0],
                        paths[i,:,1],
                        'b-',
                        linewidth=2,
                        alpha=1./np.sqrt(N))

        # draw everything
        ax.set_xbound(-10, 10)
        ax.set_ybound(-10, 2)
