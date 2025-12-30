import argparse
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Any, Tuple, List


# ============================ Config ============================
DTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x)


DEFAULT_CONFIG: Dict[str, Any] = {
    "chi": 100,
    "d": 2,
    "D": 4,
    "seed": 7,
    "ADiter": 20,
    "warmup": 2,
    "warmupiter": 50,
    "maxoptiter": 400,
}


# ============================ C3v Symmetrizer ============================
class SymmetrizedTensorKitaev:
    def __init__(self, d=2, device=None, S=None):
        self.device = device if device is not None else DEVICE
        self.dtype = DTYPE
        self.S = S if S is not None else (d - 1) / 2
        Sval = self.S
        dphys = d

        m_vals = torch.linspace(Sval, -Sval, steps=dphys, dtype=torch.float64, device=self.device).to(self.dtype)
        self.sigma_z = torch.diag(m_vals)
        coeffs = torch.sqrt((Sval - m_vals[1:]) * (Sval + m_vals[1:] + 1))
        S_plus = torch.zeros((dphys, dphys), dtype=self.dtype, device=self.device)
        idx = torch.arange(dphys - 1, device=self.device)
        S_plus[idx, idx + 1] = coeffs
        S_minus = S_plus.conj().t()
        self.sigma_x = (S_plus + S_minus) / 2
        self.sigma_y = (S_plus - S_minus) / (2.0j)

        n = torch.tensor([1, 1, 1], dtype=self.dtype, device=self.device) / math.sqrt(3)
        sigma_n = n[0] * self.sigma_x + n[1] * self.sigma_y + n[2] * self.sigma_z
        theta = 2 * math.pi / 3
        phase = np.exp(1j * 2 * math.pi / 3)
        phase2 = np.exp(1j * 4 * math.pi / 3)
        self.U_C3 = (-1) ** (dphys - 1) * phase * torch.matrix_exp(1j * theta * sigma_n)
        self.U_C3_2 = self.U_C3 @ self.U_C3

        u = torch.tensor([1, -1, 0], dtype=self.dtype, device=self.device) / math.sqrt(2)
        v = torch.tensor([1, 1, -2], dtype=self.dtype, device=self.device) / math.sqrt(6)
        m1 = u
        m2 = -0.5 * u + (math.sqrt(3) / 2) * v
        m3 = -0.5 * u - (math.sqrt(3) / 2) * v

        def Up(m):
            return torch.matrix_exp(1j * math.pi * (m[0] * self.sigma_x + m[1] * self.sigma_y + m[2] * self.sigma_z))

        self.Up_x = Up(m2) * phase2
        self.Up_y = Up(m3) * phase
        self.Up_z = Up(m1)

        self.i_sigma_y = torch.zeros((dphys, dphys), dtype=self.dtype, device=self.device)
        for i in range(dphys):
            self.i_sigma_y[i, dphys - 1 - i] = (-1) ** (int(2 * Sval) - i)

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        A0 = A
        A1 = torch.einsum('ab,bkij->aijk', self.U_C3, A)
        A2 = torch.einsum('ab,bjki->aijk', self.U_C3_2, A)
        A_px = torch.einsum('ab,bikj->aijk', self.Up_x, A)
        A_py = torch.einsum('ab,bkji->aijk', self.Up_y, A)
        A_pz = torch.einsum('ab,bjik->aijk', self.Up_z, A)
        A_ref = A_px + A_py + A_pz
        A3 = torch.einsum('ab,bijk->aijk', self.i_sigma_y, A_ref.conj())
        A_sym = A0 + A1 + A2 + A3
        return A_sym / torch.norm(A_sym)


# ============================ CTMRG (standalone) ============================
class EnvironmentInitializer:
    def __init__(self):
        pass

    def __call__(self, M: torch.Tensor, chi: int) -> List[torch.Tensor]:
        d, D = M.shape[0:2]
        with torch.no_grad():
            M = M / M.norm()
            # Create double layer tensor
            M2 = torch.einsum("iabc,idef->adbecf", M, M.conj())
            M2layer = M2.reshape(D * D, D * D, D * D)
            # Initial boundary condition (identity on virtual space)
            C = torch.einsum("adbecc->adbe", M2).reshape(D * D, D * D)
            R = torch.einsum("ij,ajk->ika", C, M2layer)

            # Pad to target chi dimensions
            C = F.pad(C, (0, chi - C.shape[1], 0, chi - C.shape[0]))
            R = F.pad(R, (0, chi - R.shape[2], 0, 0, 0, chi - R.shape[0]))

        assert C.shape == (chi, chi)
        assert R.shape == (chi, D * D, chi)
        return [C.to(DTYPE), R.reshape(chi, D, D, chi).to(DTYPE)]


class QRCTMRG:
    def __init__(self):
        pass

    @staticmethod
    def _cheap_forward(C: torch.Tensor, R: torch.Tensor):
        chi, D = R.shape[0], R.shape[1]
        CR = torch.einsum("iq,qjkm->ijkm", C, R)
        v, r = torch.linalg.qr(CR.reshape(D * D * chi, chi), mode='reduced')
        v = v.reshape(chi, D, D, chi)
        return v, r

    @staticmethod
    def _update_R(v: torch.Tensor, R: torch.Tensor, M: torch.Tensor):
        # R': iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc -> tJKc
        Rnew = torch.einsum("iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc", # potential slicing along t
                            v.conj(), R, M, M.conj(), M, M.conj(), v)
        return Rnew

    def __call__(self, M: torch.Tensor, env: List[torch.Tensor], warmup=0, ADiter=0) -> List[torch.Tensor]:
        C, R = env[0].detach(), env[1].detach()
        for _ in range(warmup + ADiter):
            v, r = self._cheap_forward(C, R)
            Rnew = self._update_R(v, R, M)
            Cnew = torch.einsum("tJKc,tb,bJKr->cr", Rnew.conj(), r, v) 
            # normalize
            C = Cnew / torch.linalg.norm(Cnew)
            R = Rnew / torch.linalg.norm(Rnew)
        return [C, R]


# ============================ Operators and Energy ============================
class PhysicalOperators:
    def __init__(self, d=2, device=None, dtype=DTYPE):
        self.device = device if device is not None else DEVICE
        self.dtype = dtype
        self.d = d
        self.S = (d - 1) / 2
        m_vals = torch.linspace(self.S, -self.S, steps=d, dtype=torch.float64, device=self.device).to(self.dtype)
        self.Sz = torch.diag(m_vals)
        self.i = torch.eye(d, dtype=self.dtype, device=self.device)


class KitaevEnergy:
    def __init__(self, operators):
        self.ops = operators

    @staticmethod
    def _rc_crc(env: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        C, R = env  # C: (chi,chi), R: (chi,D,D,chi)
        RC = torch.einsum("ijkl,lm->ijkm", R, C)
        CRC = torch.einsum("ij,jklm->iklm", C, RC)
        return RC, CRC

    @staticmethod
    def _L(RC: torch.Tensor, CRC: torch.Tensor, M: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        # ibca, ijkm, xjbp, xy, ykcq -> apqm
        return torch.einsum("ibca,ijkm,xjbp,xy,ykcq->apqm", # potential slicing along a
                            RC.conj(), CRC, M, op, M.conj())

    @staticmethod
    def _contract(Li: torch.Tensor, Lj: torch.Tensor) -> torch.Tensor:
        return torch.sum(Li * Lj.permute(3, 1, 2, 0))

    def __call__(self, M: torch.Tensor, env: List[torch.Tensor]) -> torch.Tensor:
        RC, CRC = self._rc_crc(env)
        Li = self._L(RC, CRC, M, self.ops.i)
        Lz = self._L(RC, CRC, M, self.ops.Sz.to(DTYPE))
        norm = self._contract(Li, Li)
        Ez = self._contract(Lz, Lz) / norm
        return -1.5 * torch.real(Ez)


# ============================ MWE main ============================
def mwe_main(config: Optional[Dict[str, Any]] = None):
    full = DEFAULT_CONFIG.copy()
    if config:
        full.update(config)

    print("==== Kitaev C3v MWE (single-file, PyTorch) ====")
    for k, v in full.items():
        print(f"{k}: {v}")

    chi = full["chi"]
    d = full["d"]
    D = full["D"]
    seed = full["seed"]
    ADiter = full["ADiter"]
    warmup = full["warmup"]
    warmupiter = full["warmupiter"]
    maxoptiter = full["maxoptiter"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    symmetrizer = SymmetrizedTensorKitaev(d=d, device=DEVICE)
    initializer = EnvironmentInitializer()
    ctmrg_module = QRCTMRG()
    operators = PhysicalOperators(d=d, device=DEVICE, dtype=DTYPE)
    energy_module = KitaevEnergy(operators)

    print("-- Initialization --")
    t0 = time.time()
    P = (torch.randn(d, D, D, D, dtype=DTYPE, device=DEVICE) * 0.1).requires_grad_(True)
    with torch.no_grad():
        M0 = normalize(symmetrizer(P))
        env = initializer(M0, chi)
        env = ctmrg_module(M0, env, warmup=warmupiter, ADiter=0)
        E0 = energy_module(M0, env).item()
    print(f"Initial energy: {E0:.8f} (init took {time.time()-t0:.2f}s)")

    optimizer = optim.LBFGS([P], max_iter=maxoptiter, tolerance_grad=1e-8, tolerance_change=1e-12, history_size=200, line_search_fn='strong_wolfe')

    loss_history = []
    time_history = []
    grad_norms = []
    param_changes = []
    prev_params = P.detach().clone()
    wall0 = time.time()
    fun_evals = 0
    iteration_count = 0

    def closure():
        nonlocal fun_evals, iteration_count, env, prev_params
        optimizer.zero_grad(set_to_none=True)
        M = normalize(symmetrizer(P))
        env = ctmrg_module(M, env, warmup=warmup, ADiter=ADiter)
        E = energy_module(M, env)
        E.backward()
        fun_evals += 1
        
        with torch.no_grad():
            # Only print and record on actual LBFGS iterations (not line search evaluations)
            current_iter = optimizer.state[P].get('n_iter', 0)
            if current_iter > iteration_count:
                iteration_count = current_iter
                loss_history.append(E.item())
                elapsed = time.time() - wall0
                time_history.append(elapsed)
                grad_norm = torch.norm(P.grad) if P.grad is not None else torch.tensor(0.0, dtype=torch.float64)
                grad_norms.append(grad_norm.item())
                dparam = torch.norm(P.detach() - prev_params).item()
                param_changes.append(dparam)
                prev_params = P.detach().clone()
                print(f"[Iter {iteration_count:4d}/{maxoptiter}] E={E.item():.8f} | grad={grad_norm.item():.3e} | dP={dparam:.3e} | t={elapsed:.2f}s | fevals={fun_evals}")
        
        return E

    # Single LBFGS optimization call (like main.py)
    optimizer.step(closure)

    print("-- Summary --")
    print(f"Final Loss: {loss_history[-1]:.8f}")
    print(f"Total Time: {time_history[-1]:.2f}s")
    print(f"Final Param Norm: {P.detach().norm().item():.6f}")
    print(f"Final Grad Norm: {grad_norms[-1] if grad_norms else 'N/A'}")
    return {"loss_history": loss_history, "time_history": time_history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kitaev C3v MWE (single-file)")
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
