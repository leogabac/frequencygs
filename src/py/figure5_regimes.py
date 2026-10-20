"""Descriptive fits to the existing Fig. 5 switching-width estimates."""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import linregress


def analyze_widths(omegas, widths_deg, thermal_time_s, n_bootstrap=1000):
    """Fit log residuals with equal weight per sampled frequency.

    Bootstrap errors describe scatter of the reduced curve conditional on the
    fitted model. They do not resample trajectories or propagate width errors.
    """
    x = np.asarray(omegas)
    y = np.deg2rad(widths_deg) / (x * thermal_time_s)
    use = (x > 0) & (x <= 1) & np.isfinite(y) & (y > 0)
    x, y = x[use], y[use]
    z, q = np.log(x), np.log(y)

    def predict(p, z):
        log_c, log_a, p1, p2, log_b = p
        dz = z - log_b
        return np.where(dz <= 0, np.logaddexp(log_c, log_a) - p1 * dz,
                        np.logaddexp(log_c, log_a - p2 * dz))

    bounds = ([-12, -10, 0, 0, np.log(.01)],
              [-2, 4, 5, 8, np.log(.15)])
    candidates = [least_squares(lambda p: predict(p, z) - q,
                                [np.log(.002), np.log(.2), 1, 2, np.log(b)],
                                bounds=bounds) for b in [.025, .035, .05]]
    fit = min(candidates, key=lambda f: np.sum(f.fun**2))
    p = fit.x

    def quantities(p):
        return [np.exp(p[0]), p[2], p[3], np.exp(p[4]),
                np.exp(p[4] + (p[1] - p[0]) / p[3])]

    rng = np.random.default_rng(3912)
    residual = fit.fun - fit.fun.mean()
    draws = []
    for _ in range(n_bootstrap):
        sample = predict(p, z) + rng.choice(residual, len(z), replace=True)
        boot = least_squares(lambda pp: predict(pp, z) - sample, p, bounds=bounds)
        draws.append(quantities(boot.x))
    keys = ['C', 'p_slow', 'p_decay', 'omega_break', 'omega_equal']
    result = {key: {'estimate': float(value), 'bootstrap_se': float(error)}
              for key, value, error in zip(keys, quantities(p),
                                           np.std(draws, axis=0, ddof=1))}
    result['A'] = float(np.exp(p[1]))
    result['n_points'] = len(x)
    result['log_sse'] = float(np.sum(fit.fun**2))

    # Same five parameters, but an exponential approach to the plateau.
    def exponential(pp):
        c, a, p1, b, log_scale = pp
        return np.where(z <= b, np.logaddexp(c, a) - p1 * (z - b),
                        np.logaddexp(c, a - np.maximum(x - np.exp(b), 0)
                                     / np.exp(log_scale)))
    exp_fit = least_squares(lambda pp: exponential(pp) - q,
                            [np.log(.004), np.log(.3), 1, np.log(.03), np.log(.025)],
                            bounds=([-12, -10, 0, np.log(.005), -10],
                                    [-2, 4, 5, np.log(.15), 0]))
    result['exponential_log_sse'] = float(np.sum(exp_fit.fun**2))

    # Check the high-frequency tail without imposing a constant.
    tail = x >= .224
    slope = linregress(z[tail], q[tail])
    result['tail_slope'] = {'estimate': float(slope.slope),
                            'regression_se': float(slope.stderr),
                            'n_points': int(tail.sum())}
    return result
