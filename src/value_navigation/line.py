import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt


def make_composite_input(*input_funcs):
    def composite_input(t):
        inputs = list(inp_f(t) for inp_f in input_funcs)
        return np.sum(inputs, axis=0)

    return composite_input


def make_noise_input(sig, n_neurs):
    def noise_input(t):
        return sts.norm(0, sig).rvs(n_neurs)

    return noise_input


def make_alternating_occupancy_func(n_lines, alternation_period=2000):
    def occupancy_func(t):
        ind = int(np.floor(t / alternation_period)) % n_lines
        occ = np.zeros(n_lines)
        occ[ind] = 1
        return occ

    return occupancy_func


def make_depletion_repletion_input(
    occupancy_func, d, r, kick_period=500, value_dim=(1, -1)
):
    value_dim = np.array(value_dim)

    def depletion_repletion_input(t):
        occupied = occupancy_func(t)
        inps = list(np.zeros_like(value_dim) for _ in occupied)
        depletion = -d * value_dim
        repletion = r * value_dim
        inps = []
        if t % kick_period:
            for occupy in occupied:
                if occupy == 1:
                    inps.append(depletion)
                else:
                    inps.append(repletion)
            inps = np.concatenate(inps)
        else:
            inps = np.zeros(len(occupied) * len(value_dim))
        return inps

    return depletion_repletion_input


def value_readout(traj, n_lines=2, val_dim=(1, -1)):
    val_dim = np.array(val_dim)
    ls = np.split(traj, n_lines, axis=1)
    return list(val_dim @ li.T for li in ls)


class SimpleLineAttractor:
    def __init__(self, s, n_neurs=2, tau=50):
        if n_neurs % 2 > 0:
            raise IOError("n_neurs should be even, but is {}".format(n_neurs))
        self.block_size = int(n_neurs / 2)
        self.n_neurs = n_neurs
        self.weights = np.block(
            [
                [
                    np.zeros((self.block_size, self.block_size)),
                    -np.ones((self.block_size, self.block_size)),
                ],
                [
                    -np.ones((self.block_size, self.block_size)),
                    np.zeros((self.block_size, self.block_size)),
                ],
            ]
        )
        self.bias = np.ones(n_neurs) * s
        self.ref0 = np.zeros(n_neurs)
        self.tau = tau

    def dxdt(self, x, inp):
        dxdt = -x + np.max((self.weights @ x + self.bias + inp, self.ref0), axis=0)
        return dxdt / self.tau

    def integrate(self, duration, input_func=None, init=None, dt=1):
        if init is None:
            init = np.zeros(self.n_neurs)
        else:
            init = np.array(init)
        if input_func is None:

            def input_func(t):
                return np.zeros(self.n_neurs)

        ts = np.arange(0, duration + dt, dt)
        x = init
        traj = np.zeros((len(ts), self.n_neurs))
        for i, t in enumerate(ts[:-1]):
            traj[i] = x
            x = x + dt * self.dxdt(x, input_func(t))
        traj[i + 1] = x
        return traj, ts


class MultiLineAttractor(SimpleLineAttractor):
    def __init__(self, n_lines, s, n_neurs=None, **kwargs):
        try:
            len(s)
        except TypeError:
            s = (s,) * n_lines
        if n_neurs is None:
            n_neurs = 2 * n_lines
        self.neurs_per_line = int(n_neurs / n_lines)

        self.lines = list(
            SimpleLineAttractor(si, n_neurs=self.neurs_per_line, **kwargs) for si in s
        )
        self.n_neurs = n_neurs
        self.n_lines = n_lines

    def dxdt(self, x, inp):
        x_lines = np.split(x, self.n_lines)
        inp_lines = np.split(inp, self.n_lines)
        return np.concatenate(
            list(self.lines[i].dxdt(xl, inp_lines[i]) for i, xl in enumerate(x_lines)),
            axis=0,
        )


def simulate_multiline(
    depletion,
    repletion,
    n_lines=2,
    strength=2,
    noise_std=0.5,
    n_neurs_per_line=2,
    initial_condition=None,
    integ_time=20000,
):
    n_neurs = n_neurs_per_line * n_lines
    if initial_condition is None:
        initial_condition = np.array(strength) * np.ones(n_neurs) / 2

    mla = MultiLineAttractor(n_lines, strength)
    occupancy = make_alternating_occupancy_func(n_lines)
    dr_inp = make_depletion_repletion_input(occupancy, depletion, repletion)
    noise_inp = make_noise_input(noise_std, n_neurs)
    inp = make_composite_input(dr_inp, noise_inp)

    traj, ts = mla.integrate(integ_time, init=initial_condition, input_func=inp)
    return traj, ts


def plot_multiline(traj, axs=None, fwid=4, label="", **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, 2, figsize=(fwid * 2, fwid))
    ax1, ax2 = axs
    ax1.plot(*traj[:, :2].T, **kwargs)
    ax1.set_xlabel("x1")
    ax1.set_xlabel("x2")
    ax1.set_title("single line attractor dynamics")

    v1, v2 = value_readout(traj)
    ax2.plot(v1, v2, label=label, **kwargs)
    ax2.set_xlabel("value 1")
    ax2.set_xlabel("value 2")
    ax2.set_title("value dynamics")
    ax2.legend(frameon=False)
