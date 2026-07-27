import numpy as np  # noqa: I001
import scipy.stats as sts
import matplotlib.pyplot as plt
import general.plotting as gpl


def make_composite_input(*input_funcs):
    def composite_input(t):
        inputs = [inp_f(t) for inp_f in input_funcs]
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


def make_simultaneous_depletion_repletion_input(
    occupancy_func, d, r, kick_period=500, value_dim=(1, -1)
):
    value_dim = np.array(value_dim)

    def depletion_repletion_input(t):
        occupied = occupancy_func(t)
        inps = [np.zeros_like(value_dim) for _ in occupied]
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


def make_switch_repletion_input(
    occupancy_func, d, r, kick_period=200, value_dim=(1, -1)
):
    value_dim = np.array(value_dim)

    def depletion_repletion_input(t):
        occupied = occupancy_func(t)
        prev_occupied = occupancy_func(t - 1)
        inps = [np.zeros_like(value_dim) for _ in occupied]
        depletion = -d * value_dim
        zs = 0 * value_dim
        repletion = r * value_dim
        inps = np.zeros(len(occupied) * len(value_dim))
        switch = np.any(occupied != prev_occupied) and t - 1 >= 0
        if t % kick_period == 0:
            inps_add = []
            for occupy in occupied:
                if occupy == 1:
                    inps_add.append(depletion)
                else:
                    inps_add.append(zs)

            inps = inps + np.concatenate(inps_add)
        if switch:
            inps_add = []
            for occupy in occupied:
                if occupy == 1:
                    inps_add.append(zs)
                else:
                    inps_add.append(repletion)
            inps = inps + np.concatenate(inps_add)
        return inps

    return depletion_repletion_input


def make_switch_repletion_corr_input(
    occupancy_func, d, r, kick_period=200, value_dim=(1, -1)
):
    value_dim = np.array(value_dim)

    def depletion_repletion_input(t):
        occupied = occupancy_func(t)
        prev_occupied = occupancy_func(t - 1)
        inps = [np.zeros_like(value_dim) for _ in occupied]
        depletion = -d * value_dim
        zs = 0 * value_dim
        repletion = r * value_dim
        inps = np.zeros(len(occupied) * len(value_dim))
        switch = np.any(occupied != prev_occupied) and t - 1 >= 0
        common = np.zeros_like(value_dim)
        if t % kick_period == 0:
            inps_add = []
            for occupy in occupied:
                if occupy == 1:
                    inps_add.append(depletion)
                    common = common + depletion
                else:
                    inps_add.append(zs)

            inps = inps + np.concatenate(inps_add)
        if switch:
            inps_add = []
            for occupy in occupied:
                if occupy == 1:
                    inps_add.append(zs)
                else:
                    inps_add.append(repletion)
                    common = common + repletion
            inps = inps + np.concatenate(inps_add)
        inps = np.concatenate((inps, common))
        return inps

    return depletion_repletion_input


def value_readout(traj, n_lines=2, val_dim=(1, -1)):
    val_dim = np.array(val_dim)
    ls = np.split(traj, n_lines, axis=1)
    return [val_dim @ li.T for li in ls]


def common_value_readout(traj, weight, val_dim=(1, -1)):
    val_dim = np.array(val_dim)
    n_lines = int(traj.shape[1] / 2)
    ls = np.split(traj, n_lines, axis=1)
    vals = [val_dim @ li.T for li in ls]
    common_val = vals[-1]
    vals = vals[:-1]
    w_vals = []
    for v in vals:
        w_vals.append(weight * v + (1 - weight) * common_val)
    return w_vals



class SimpleLineAttractor:
    def __init__(self, s, n_neurs=2, tau=50):
        if n_neurs % 2 > 0:
            raise OSError(f"n_neurs should be even, but is {n_neurs}")
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

        self.lines = [
            SimpleLineAttractor(si, n_neurs=self.neurs_per_line, **kwargs) for si in s
        ]
        self.n_neurs = n_neurs
        self.n_lines = n_lines

    def dxdt(self, x, inp):
        x_lines = np.split(x, self.n_lines)
        inp_lines = np.split(inp, self.n_lines)
        return np.concatenate(
            [self.lines[i].dxdt(xl, inp_lines[i]) for i, xl in enumerate(x_lines)],
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
    input_maker=make_switch_repletion_input,
    integ_time=20000,
):
    n_neurs = n_neurs_per_line * n_lines
    if initial_condition is None:
        initial_condition = np.array(strength) * np.ones(n_neurs) / 2

    mla = MultiLineAttractor(n_lines, strength)
    occupancy = make_alternating_occupancy_func(n_lines)
    dr_inp = input_maker(occupancy, depletion, repletion)
    noise_inp = make_noise_input(noise_std, n_neurs)
    inp = make_composite_input(dr_inp, noise_inp)

    traj, ts = mla.integrate(integ_time, init=initial_condition, input_func=inp)
    occup = np.stack([occupancy(t) for t in ts])
    return traj, ts, occup


def simulate_correlated_multiline(
    depletion,
    repletion,
    n_lines=2,
    strength=2,
    noise_std=0.5,
    n_neurs_per_line=2,
    initial_condition=None,
    input_maker=make_switch_repletion_corr_input,
    integ_time=20000,
):
    n_lines = n_lines + 1
    n_neurs = n_neurs_per_line * n_lines
    if initial_condition is None:
        initial_condition = np.array(strength) * np.ones(n_neurs) / 2
    
    mla = MultiLineAttractor(n_lines, strength)
    occupancy = make_alternating_occupancy_func(n_lines - 1)
    dr_inp = input_maker(occupancy, depletion, repletion)
    noise_inp = make_noise_input(noise_std, n_neurs)
    inp = make_composite_input(dr_inp, noise_inp)

    traj, ts = mla.integrate(integ_time, init=initial_condition, input_func=inp)
    occup = np.stack([occupancy(t) for t in ts])
    return traj, ts, occup


def plot_multiline(
    traj,
    occup,
    axs=None,
    fwid=4,
    gray_color=(0.5,) * 3,
    label="",
    colors=("Blues", "Oranges"),
    common=False,
    weight=.5,
    **kwargs,
):
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(fwid * 2, fwid))
    ax1, ax2 = axs
    ax1.plot(*traj[:, :2].T, color=gray_color, lw=0.5)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("single line attractor dynamics")

    if common:
        v1, v2 = common_value_readout(traj, weight)
    else:
        v1, v2 = value_readout(traj)
    ax2.plot(v1, v2, color=gray_color, lw=0.5)
    breaks = np.where(np.any(np.abs(np.diff(occup, axis=0)) > 0, axis=1))[0] + 1
    breaks = np.concatenate(((0,), breaks, (len(occup),)))
    for i, b in enumerate(breaks[:-1]):
        cmap = colors[int(occup[b, 1])]
        traj_section = traj[b + 1 : breaks[i + 1], :2]
        gpl.plot_colored_line(*traj_section.T, ax=ax1, cmap=cmap, **kwargs)

        v1_section = v1[b + 1 : breaks[i + 1]]
        v2_section = v2[b + 1 : breaks[i + 1]]
        gpl.plot_colored_line(v1_section, v2_section, ax=ax2, cmap=cmap, **kwargs)

    ax2.set_xlabel("value 1")
    ax2.set_ylabel("value 2")
    ax2.set_title("value dynamics")
    ax2.set_aspect("equal")
    gpl.clean_plot(ax1, 0)
    gpl.clean_plot(ax2, 0)
