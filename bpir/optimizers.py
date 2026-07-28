"""Ask/tell optimiser engines shared by SPE and HDR: CMA-ES (evolution
strategy), PPO (policy-gradient RL with a clipped surrogate) and an LSTM
meta-optimiser (a learned recurrent update rule). Engines maximise the
composite score; interfaces and update equations are distilled here.
"""
import numpy as np


class AskTell:
    """ask() proposes candidate weight vectors; tell() reports their composite
    scores (higher is better); best() returns (theta, score) seen so far."""
    converged = False

    def ask(self): raise NotImplementedError
    def tell(self, candidates, scores): raise NotImplementedError
    def best(self): raise NotImplementedError


class CMAES(AskTell):
    """Wraps the `cma` package; scores are negated because cma minimises. The
    search box matches the log-weight bounds; x0 allows warm-starting."""

    def __init__(self, dim, popsize, sigma0, bounds, seed=0, x0=None):
        import cma
        opts = {"popsize": popsize, "bounds": list(bounds), "seed": seed, "verbose": -9}
        self._es = cma.CMAEvolutionStrategy(list(x0) if x0 is not None else [0.0] * dim,
                                            sigma0, opts)
        self._best = (np.zeros(dim), -np.inf)

    def ask(self):
        return [np.asarray(x, float) for x in self._es.ask()]

    def tell(self, candidates, scores):
        self._es.tell([np.asarray(s).tolist() for s in candidates],
                      [-float(v) for v in scores])
        i = int(np.argmax(scores))
        if scores[i] > self._best[1]:
            self._best = (np.asarray(candidates[i]).copy(), float(scores[i]))

    def best(self): return self._best

    @property
    def converged(self): return bool(self._es.stop())


class _GaussianPolicy(AskTell):
    """Diagonal-Gaussian search distribution N(mu, diag(exp(logstd))^2) with
    candidates clipped to the weight box and best-so-far tracking."""

    def __init__(self, dim, popsize, bounds, init_logstd, logstd_bounds, seed=0):
        self._dim, self._pop, self._bounds, self._ls_bounds = dim, popsize, bounds, logstd_bounds
        self._rng = np.random.default_rng(seed)
        self._mu, self._logstd = np.zeros(dim), np.full(dim, float(init_logstd))
        self._best, self._last = (np.zeros(dim), -np.inf), None

    def ask(self):
        a = self._mu + self._rng.normal(size=(self._pop, self._dim)) * np.exp(self._logstd)
        self._last = np.clip(a, *self._bounds)
        return [x.copy() for x in self._last]

    def _track(self, scores):
        i = int(np.argmax(scores))
        if scores[i] > self._best[1]:
            self._best = (self._last[i].copy(), float(scores[i]))

    def best(self): return self._best

    @staticmethod
    def _logprob(a, mu, logstd):
        std = np.exp(logstd)
        return -0.5 * (((a - mu) / std) ** 2 + 2 * logstd + np.log(2 * np.pi)).sum(-1)


class PPO(_GaussianPolicy):
    """An action is a candidate weight vector, the reward is the composite.
    Updates ascend the PPO clipped surrogate over probability ratios, with
    advantages against a learned value baseline and an entropy bonus."""

    def __init__(self, dim, popsize, bounds, clip_eps, lr, epochs, value_lr,
                 entropy_coef, init_logstd, logstd_bounds, seed=0):
        super().__init__(dim, popsize, bounds, init_logstd, logstd_bounds, seed)
        self._clip_eps, self._lr, self._epochs = clip_eps, lr, epochs
        self._value_lr, self._ent, self._baseline = value_lr, entropy_coef, 0.0

    def tell(self, candidates, scores):
        r = np.asarray(scores, float)
        self._track(r)
        adv = r - self._baseline                    # advantage vs value baseline
        self._baseline += self._value_lr * (r.mean() - self._baseline)
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        a = self._last
        logp_old = self._logprob(a, self._mu.copy(), self._logstd.copy())
        for _ in range(self._epochs):
            std = np.exp(self._logstd)
            ratio = np.exp(np.clip(self._logprob(a, self._mu, self._logstd) - logp_old, -10, 10))
            clipped = ((adv >= 0) & (ratio > 1 + self._clip_eps)) | \
                      ((adv < 0) & (ratio < 1 - self._clip_eps))
            coef = np.where(clipped, 0.0, adv * ratio)          # clipped surrogate
            g_mu = (coef[:, None] * (a - self._mu) / std ** 2).mean(0)
            g_ls = (coef[:, None] * (((a - self._mu) / std) ** 2 - 1.0)).mean(0) + self._ent
            self._mu = np.clip(self._mu + self._lr * g_mu, *self._bounds)
            self._logstd = np.clip(self._logstd + self._lr * g_ls, *self._ls_bounds)


class LSTMMeta(_GaussianPolicy):
    """A small coordinate-wise LSTM, meta-trained offline on a family of
    objectives by evolution strategies, emits the next search distribution.
    `cell(features) -> (d_mu, d_logstd)` wraps the trained recurrent cell,
    which carries hidden state across calls; per-coordinate features are
    [population gradient estimate, direction to best, logstd, trend, progress]."""

    def __init__(self, dim, popsize, bounds, cell, mu_scale, ls_scale, horizon,
                 init_logstd, logstd_bounds, seed=0):
        super().__init__(dim, popsize, bounds, init_logstd, logstd_bounds, seed)
        self._cell, self._mu_scale, self._ls_scale = cell, mu_scale, ls_scale
        self._horizon, self._iter, self._hist = horizon, 0, []

    def tell(self, candidates, scores):
        r = np.asarray(scores, float)
        self._track(r)
        self._iter += 1
        self._hist.append(float(r.max()))
        rs = (r - r.mean()) / (r.std() + 1e-8)
        grad = np.tanh(((self._last - self._mu) / np.exp(self._logstd)
                        * rs[:, None]).mean(0))     # score-function gradient estimate
        trend = np.tanh(self._hist[-1] - self._hist[-4]) if len(self._hist) >= 4 else 0.0
        feat = np.column_stack([
            grad, np.tanh(self._best[0] - self._mu), np.tanh(self._logstd),
            np.full(self._dim, trend),
            np.full(self._dim, min(self._iter / self._horizon, 1.0))])
        d_mu, d_ls = self._cell(feat)
        self._mu = np.clip(self._mu + self._mu_scale * d_mu, *self._bounds)
        self._logstd = np.clip(self._logstd + self._ls_scale * d_ls, *self._ls_bounds)


def make_optimizer(name, *args, **kwargs):
    """Instantiate "cmaes", "ppo" or "lstm" behind the shared interface."""
    return {"cmaes": CMAES, "ppo": PPO, "lstm": LSTMMeta}[name.lower().strip()](*args, **kwargs)
