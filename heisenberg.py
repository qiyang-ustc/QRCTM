import argparse
import time
from typing import List, Tuple
import numpy as np
import torch
from torch import optim

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "chi": 24,
    "d": 2,
    "D": 4,
    "seed": 7,
    "ADiter": 16,
    "warmup": 2,
    "warmupiter": 40,
    "maxoptiter": 120,
    "checkpoint": False,
}

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x)

class C4Symmetrizer:
    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        A = A + A.permute(0, 1, 4, 3, 2)
        A = A + A.permute(0, 3, 2, 1, 4)
        A = A + A.permute(0, 4, 1, 2, 3)
        A = A + A.permute(0, 3, 4, 1, 2)
        return normalize(A)

class EnvironmentInitializer:
    def __call__(self, M: torch.Tensor, chi: int) -> List[torch.Tensor]:
        D = M.shape[1]
        assert chi > D**2, "Strongly suggest chi > D**2"
        with torch.no_grad():
            M = normalize(M)
            T = torch.einsum("xabcd,xaefg->becfgd", M, M.conj())
            C = torch.einsum("beccgd->begd", T)
            C, T = C.reshape(D**2, D**2), T.reshape(D**2, D, D, D**2)

            pad_c = chi - C.shape[0]
            C = torch.nn.functional.pad(C, (0, pad_c, 0, pad_c))
            T = torch.nn.functional.pad(T, (0, pad_c, 0, 0, 0, 0, 0, pad_c))
        dev = M.device
        return [normalize(C.to(device=dev, dtype=DTYPE)), normalize(T.to(device=dev, dtype=DTYPE))]

class CTMRG:
    def __init__(self, checkpoint=True):
        self.checkpoint = checkpoint

    @staticmethod
    def _qr(C: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        chi, D = C.shape[0], T.shape[1]
        CT = torch.einsum("ij,jklm->iklm", C, T).reshape(chi * D * D, chi)
        q, r = torch.linalg.qr(CT, mode="reduced")
        return q.reshape(chi, D, D, chi), r

    @staticmethod
    def _update_T(T: torch.Tensor, v: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.einsum("ibfj,iaep,xabcd,xefgh,jcgq->pdhq", T, v, M, M.conj(), v)

    def __call__(self, M: torch.Tensor, env: List[torch.Tensor], warmup=0, ADiter=0) -> List[torch.Tensor]:
        C, T = env
        with torch.no_grad():
            for _ in range(warmup):
                v, r = self._qr(C, T)
                T = self._update_T(T, v, M)
                C = torch.einsum("ijkl,la,ajkx->ix", T, r, v)
                C, T = normalize(C+C.T.conj()), normalize(T+T.permute(3, 1, 2, 0).conj())

        for _ in range(ADiter):
            if self.checkpoint and ADiter > 0 and M.requires_grad:
                v, r = torch.utils.checkpoint.checkpoint(self._qr, C, T, use_reentrant=True)
                T = torch.utils.checkpoint.checkpoint(self._update_T, T, v, M, use_reentrant=True)
            else:
                v, r = self._qr(C, T)
                T = self._update_T(T, v, M)
            C = torch.einsum("ijkl,la,ajkx->ix", T, r, v)
            C, T = normalize(C+C.T.conj()), normalize(T+T.permute(3, 1, 2, 0).conj())
        return [C, T]


class HeisenbergEnergy:
    def __init__(self):
        sp = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=DTYPE, device=DEVICE)
        sm = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=DTYPE, device=DEVICE)
        sz = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=DTYPE, device=DEVICE)
        rot = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE, device=DEVICE)
        sp_r, sm_r, sz_r = rot @ sp @ rot.T, rot @ sm @ rot.T, rot @ sz @ rot.T
        Ez = torch.einsum("ij,kl->ijkl", sz, sz_r) / 4
        Ep = torch.einsum("ij,kl->ijkl", sp, sm_r) / 2
        Em = torch.einsum("ij,kl->ijkl", sm, sp_r) / 2
        self.E = (Ez + Ep + Em).to(DEVICE)

    def _rho(self, env: List[torch.Tensor], M: torch.Tensor) -> torch.Tensor:
        C, T = env
        Csym, Tsym = C + C.conj(), T + T.permute(3, 1, 2, 0).conj()
        L = torch.einsum("ij,jklm,mn->ikln", Csym, Tsym, Csym)
        L = torch.einsum("ibfj,iaep,xabcd,yefgh,jcgq->pdhqxy", L, Tsym, M, M.conj(), Tsym)
        return torch.einsum("pdhqxy,pdhqzw->xyzw", L, L)

    def __call__(self, M: torch.Tensor, env: List[torch.Tensor]) -> torch.Tensor:
        rho = self._rho(env, M)
        norm = torch.einsum("aacc->", rho)
        energy = torch.einsum("abcd,abcd->", rho, self.E)
        return 2.0 * torch.real(energy / norm)

def bulk_tensor(params: torch.Tensor, d: int, D: int) -> torch.Tensor:
    return normalize(params.reshape(d, D, D, D, D).to(DTYPE))


def mwe_main(config=None):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    chi = cfg["chi"]
    d = cfg["d"]
    D = cfg["D"]
    seed = cfg["seed"]
    ADiter = cfg["ADiter"]
    warmup = cfg["warmup"]
    warmupiter = cfg["warmupiter"]
    maxoptiter = cfg["maxoptiter"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    sym = C4Symmetrizer()
    energy_fn = HeisenbergEnergy()
    initializer = EnvironmentInitializer()
    ctmrg = CTMRG()

    nparams = d * D * D * D * D
    params = (torch.rand(nparams, device=DEVICE, dtype=torch.float64) - 0.5).requires_grad_(True)

    M0 = sym(bulk_tensor(params, d, D))
    env = initializer(M0, chi)
    env = ctmrg(M0, env, warmup=warmupiter, ADiter=0)

    optimizer = optim.LBFGS([params], max_iter=maxoptiter, tolerance_grad=1e-8, tolerance_change=1e-12, history_size=200, line_search_fn="strong_wolfe")
    loss_history: List[float] = []
    start = time.time()

    def closure():
        nonlocal env
        optimizer.zero_grad(set_to_none=True)
        M = sym(bulk_tensor(params, d, D))
        env2 = ctmrg(M, env, warmup=0, ADiter=ADiter)
        E = energy_fn(M, env2)
        E.backward()
        with torch.no_grad():
            env = ctmrg(sym(bulk_tensor(params, d, D)), env, warmup=warmup, ADiter=0)
            loss_history.append(E.item())
            elapsed = time.time() - start
            print(f"[{len(loss_history):03d}] E={E.item():.8f} | t={elapsed:.2f}s")
        return E

    optimizer.step(closure)
    print(f"Final energy: {loss_history[-1]:.8f}")
    return loss_history[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Heisenberg C4 PEPS (PyTorch)")
    parser.add_argument("--chi", type=int, default=DEFAULT_CONFIG["chi"])
    parser.add_argument("--d", type=int, default=DEFAULT_CONFIG["d"])
    parser.add_argument("--D", type=int, default=DEFAULT_CONFIG["D"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--ADiter", type=int, default=DEFAULT_CONFIG["ADiter"])
    parser.add_argument("--warmup", type=int, default=DEFAULT_CONFIG["warmup"])
    parser.add_argument("--warmupiter", type=int, default=DEFAULT_CONFIG["warmupiter"])
    parser.add_argument("--maxoptiter", type=int, default=DEFAULT_CONFIG["maxoptiter"])
    args = vars(parser.parse_args())
    mwe_main(args)