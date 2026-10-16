"""Access schemes compared in the paper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMES = ("MF-SIC", "ARJOA", "IOJOA", "FDMA", "ALCA")


@dataclass
class Scheme:
    name: str
    offload_decision: np.ndarray | None = None   # IOJOA only, shape (N_ul,)
    _big: float = field(default=1e14, repr=False)

    # -- population seeding --------------------------------------------- #
    def seed(self, assoc_shape: tuple[int, int], n_ul: int, m_dl: int, rng: np.random.Generator, n_agents: int) -> np.ndarray:
        rows, k = assoc_shape
        pop = np.zeros((n_agents, rows, k), dtype=np.int8)

        for s in range(n_agents):
            if self.name == "FDMA":
                chans = rng.permutation(k)
                for m in range(m_dl):
                    pop[s, n_ul + m, chans[m % k]] = 1
                used = {int(chans[m % k]) for m in range(m_dl)}
                free = [c for c in range(k) if c not in used]
            else:
                for m in range(m_dl):
                    pop[s, n_ul + m, rng.integers(k)] = 1
                free = list(range(k))

            if self.name == "ALCA":
                continue
            if self.name == "FDMA":
                if free and n_ul:
                    take = min(len(free), n_ul)
                    users = rng.permutation(n_ul)[:take]
                    for u, c in zip(users, rng.permutation(free)[:take]):
                        pop[s, u, c] = 1
                continue
            for n in range(n_ul):
                if self.name == "ARJOA":
                    offload = True
                elif self.name == "IOJOA":
                    offload = bool(self.offload_decision[n]) if self.offload_decision is not None else True
                else:
                    offload = True if s < max(1, n_agents // 5) else rng.random() > 0.5
                if offload:
                    pop[s, n, rng.integers(k)] = 1
        return pop

    def penalty(self, assoc: np.ndarray, n_ul: int) -> float:
        a_ul, a_dl = assoc[:n_ul], assoc[n_ul:]
        if self.name == "ARJOA":
            g = a_ul.sum(axis=1) - 1.0
            return self._big * float(np.sum(g ** 2))
        if self.name == "IOJOA":
            g = a_ul.sum(axis=1) - (self.offload_decision if self.offload_decision is not None else 1.0)
            return self._big * float(np.sum(g ** 2))
        if self.name == "ALCA":
            return self._big * float(np.sum(a_ul))
        if self.name == "FDMA":
            load = a_ul.sum(axis=0) + a_dl.sum(axis=0)
            g = load - 1.0
            return self._big * float(np.sum(np.where(g > 0, g ** 2, 0.0)))
        return 0.0

    @property
    def optimises_uplink(self) -> bool:
        return self.name != "ALCA"


def make_scheme(name: str, n_ul: int, rng: np.random.Generator) -> Scheme:
    name = name.upper().replace("_", "-")
    canonical = {s.upper(): s for s in SCHEMES}
    if name not in canonical:
        raise ValueError(f"unknown scheme {name!r}; choose from {SCHEMES}")
    name = canonical[name]
    decision = None
    if name == "IOJOA":
        decision = (rng.random(n_ul) > 0.5).astype(float)
    return Scheme(name=name, offload_decision=decision)
