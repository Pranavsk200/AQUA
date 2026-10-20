import numpy as np

from prune.criteria import CRITERIA

NEVER = -10 ** 9


def _top(scores, idx, n):
    # n best entries of idx by score, ties broken by position so runs are repeatable
    if n <= 0 or len(idx) == 0:
        return idx[:0]
    order = np.argsort(-scores[idx], kind="stable")
    return idx[order[:n]]


class SelfPruner:
    def __init__(self, layers, criterion, plan, seed=0, ema=0.9, min_density=0.01,
                 regrow=0.0, cooldown=200, grace=200, normalize=True, optimizer=None):
        self.layers = layers
        self.crit = CRITERIA[criterion]
        self.plan = plan
        self.rng = np.random.default_rng(seed)
        self.ema = ema
        self.min_density = min_density
        self.regrow = regrow
        self.cooldown, self.grace = cooldown, grace
        self.normalize = normalize
        self.opt = optimizer
        self.steps = sorted(plan)

        self.sizes = [l.weight.data.size for l in layers]
        self.total = sum(self.sizes)
        self.gbar = [np.zeros_like(l.weight.data) for l in layers]
        self.g2bar = [np.zeros_like(l.weight.data) for l in layers]
        self.pruned_at = [np.full(l.weight.data.shape, NEVER, dtype=np.int64) for l in layers]
        self.grown_at = [np.full(l.weight.data.shape, NEVER, dtype=np.int64) for l in layers]
        self.log = []

    def current_sparsity(self):
        active = sum(int(l.mask.sum()) for l in self.layers)
        return 1.0 - active / self.total

    def observe(self):
        b = self.ema
        for l, gb, g2 in zip(self.layers, self.gbar, self.g2bar):
            g = l.dense_grad()
            if g is None:
                continue
            gb *= b
            gb += (1 - b) * g
            g2 *= b
            g2 += (1 - b) * g * g

    def maybe_update(self, step):
        if step in self.plan:
            self._update(step, self.plan[step])

    def _regrow_count(self, step, k):
        if self.regrow <= 0 or len(self.steps) < 2:
            return 0
        progress = (step - self.steps[0]) / (self.steps[-1] - self.steps[0])
        return int(round(self.regrow * 0.5 * (1 + np.cos(np.pi * progress)) * k))

    def _flat(self, arrays):
        return np.concatenate([a.ravel() for a in arrays])

    def _layer_ids(self):
        return np.concatenate([np.full(n, i) for i, n in enumerate(self.sizes)])

    def _scores(self, active):
        out = []
        for l, gb, g2, a in zip(self.layers, self.gbar, self.g2bar, active):
            s = self.crit(l.weight.data, gb, g2, self.rng)
            if self.normalize:
                s = s / (s[a].mean() + 1e-12) if a.any() else s
            out.append(s)
        return self._flat(out)

    def _update(self, step, target):
        k = int(round((1.0 - target) * self.total))
        was_active = [l.mask > 0 for l in self.layers]
        act = self._flat(was_active)
        lid = self._layer_ids()
        scores = self._scores(was_active)

        # regrowth candidates: currently off, and not just pruned
        r = 0
        cand = np.array([], dtype=np.int64)
        if self.regrow > 0:
            cooled = self._flat([step - p >= self.cooldown for p in self.pruned_at])
            cand = np.flatnonzero(~act & cooled)
            r = min(self._regrow_count(step, k), len(cand))

        protected = self._flat([step - g < self.grace for g in self.grown_at]) & act
        forced = protected.copy()
        for i, n in enumerate(self.sizes):
            idx = np.flatnonzero(act & (lid == i))
            forced[_top(scores, idx, int(np.ceil(self.min_density * n)))] = True

        k_drop = k - r
        if forced.sum() > k_drop:
            forced[:] = False
        keep = np.flatnonzero(forced)
        rest = np.flatnonzero(act & ~forced)
        keep = np.concatenate([keep, _top(scores, rest, k_drop - len(keep))])

        new = np.zeros(self.total, dtype=bool)
        new[keep] = True
        if r > 0:
            gscore = np.abs(self._flat(self.gbar))
            if self.normalize:
                for i in range(len(self.sizes)):
                    sel = cand[lid[cand] == i]
                    if len(sel):
                        gscore[sel] = gscore[sel] / (gscore[sel].mean() + 1e-12)
            new[_top(gscore, cand, r)] = True

        assert new.sum() == k, f"mask has {new.sum()} active weights, wanted {k}"
        self._commit(step, act, new)
        self.log.append({"step": step, "target": target, "sparsity": self.current_sparsity(), "regrown": r})

    def _commit(self, step, old, new):
        start = 0
        for l, pa, ga, n in zip(self.layers, self.pruned_at, self.grown_at, self.sizes):
            sl = slice(start, start + n)
            shape = l.weight.data.shape
            dropped = (old[sl] & ~new[sl]).reshape(shape)
            added = (~old[sl] & new[sl]).reshape(shape)
            pa[dropped] = step
            ga[added] = step
            l.mask[...] = new[sl].reshape(shape)
            l.apply_mask()
            start += n
        if self.opt is not None:
            self.opt.apply_masks()
