import torch
import numpy as np
import scipy.integrate
import scipy.linalg

# Set default tensor type to float64 for precision
torch.set_default_dtype(torch.float64)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Analytical Ising 2D functions ---
def ising_2d_analytical_logZ(beta, J1, J2):
    """Calculate analytical log partition function for 2D Ising model"""
    def integrand(theta2, theta1, beta, J1, J2):
        return np.log(np.abs(
            np.cosh(2 * beta * J1) * np.cosh(2 * beta * J2) -
            np.sinh(2 * beta * J1) * np.cos(theta1) -
            np.sinh(2 * beta * J2) * np.cos(theta2)
        ))
    result, _ = scipy.integrate.dblquad(
        lambda t2, t1: integrand(t2, t1, beta, J1, J2),
        0, 2 * np.pi, lambda x: 0, lambda x: 2 * np.pi,
        epsabs=1.49e-13, epsrel=1.49e-13,
    )
    return np.log(2) + result / (8 * np.pi ** 2)

def ising_2d_analytical_magnetization(beta):
    """Calculate analytical magnetization for 2D Ising model"""
    critical_beta = np.log(1 + np.sqrt(2)) / 2.0
    if beta < critical_beta: 
        return 0.0
    sinh_term = np.sinh(2 * beta)
    return (1.0 - (1.0 / sinh_term**4)) ** (1.0/8)

# --- Tensor Network Functions ---
def classic_ising_tensor(beta, mag_op=False):
    """Create Ising tensor for given beta"""
    expJ_np = np.exp(beta * np.array([[1.0, -1.0], [-1.0, 1.0]]))
    sqrtJ = torch.tensor(scipy.linalg.sqrtm(expJ_np), device=_DEVICE)
    M_core = torch.zeros((2,2,2,2), device=_DEVICE)
    M_core[0, 0, 0, 0] = 1.0
    M_core[1, 1, 1, 1] = -1.0 if mag_op else 1.0
    return torch.einsum("ijkl,ai,bj,ck,dl->abcd", M_core, sqrtJ, sqrtJ, sqrtJ, sqrtJ)

def init_CT(M_tensor, chi):
    """Initialize corner transfer matrices"""
    D = M_tensor.shape[0]
    C = torch.eye(chi, device=_DEVICE)
    T = torch.ones((chi, D, chi), device=_DEVICE)
    return C + C.T, T + T.permute(2,1,0)

def qr_step(C, T):
    """QR decomposition step"""
    chi, D = T.shape[:2]
    Q, R = torch.linalg.qr(torch.einsum("ij,jkl->ikl", C, T).reshape(D * chi, chi))
    signs = torch.sign(torch.diag(R))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    Q = Q * signs.unsqueeze(0)
    return Q.reshape(chi, D, chi)

def step_ctmrg(C, T, M):
    """Single CTMRG step"""
    v = qr_step(C, T)
    C_new = torch.einsum("ibj,ik,kap,abcd,jcq,pdr->qr", T, C, T, M, v, v)
    T_new = torch.einsum("ibj,iap,abcd,jcq->pdq", T, v, M, v)
    C_new, T_new = C_new + C_new.T, T_new + T_new.permute(2,1,0)
    return C_new / torch.linalg.norm(C_new), T_new / torch.linalg.norm(T_new)

def calculate_logZ(M, C, T):
    """Calculate log partition function"""
    ctr9 = torch.einsum("ab,fga,bcd,cghi,de,eij,kf,lhk,jl->", C, T, T, M, C, T, C, T, C)
    ctrh6 = torch.einsum("ab,la,bgd,egl,df,fe", C, C, T, T, C, C)
    ctr4 = torch.trace(C @ C @ C @ C)
    return torch.log(torch.abs(ctr9 / ctrh6 / ctrh6 * ctr4))

def calculate_magnetization(M, MM, C, T):
    """Calculate magnetization"""
    i = torch.einsum("ab,fga,bcd,cghi,de,eij,kf,lhk,jl->", C, T, T, M, C, T, C, T, C)
    m = torch.einsum("ab,fga,bcd,cghi,de,eij,kf,lhk,jl->", C, T, T, MM, C, T, C, T, C)
    return m / i

def run_ctmrg(M, MM, chi, maxiter=1000):
    """Run CTMRG algorithm"""
    C, T = init_CT(M, chi)
    
    for i in range(1, maxiter + 1):
        C, T = step_ctmrg(C, T, M)
            
    return calculate_logZ(M, C, T), calculate_magnetization(M, MM, C, T)

if __name__ == "__main__":
    # Parameters
    beta_c = np.log(1 + np.sqrt(2)) / 2.0  # Critical temperature
    beta_shift = 0.0001
    beta = beta_c * (1 + beta_shift)
    chi = 100
    max_iterations = 1000

    print("="*50)
    print("MINIMAL ISING MODEL CALCULATION")
    print("="*50)
    print(f"Beta: {beta:.6f}")
    print(f"Chi: {chi}")
    print(f"Max iterations: {max_iterations}")
    print()

    # Calculate analytical values
    analytical_logZ = ising_2d_analytical_logZ(beta, 1.0, 1.0)
    analytical_mag = ising_2d_analytical_magnetization(beta)
    
    print("ANALYTICAL RESULTS:")
    print(f"logZ: {analytical_logZ:.10f}")
    print(f"Magnetization: {analytical_mag:.10f}")
    print()

    # Create Ising tensors
    M_bulk = classic_ising_tensor(beta, mag_op=False)
    M_mag = classic_ising_tensor(beta, mag_op=True)

    # Run CTMRG calculation
    print("Running CTMRG calculation...")
    ctmrg_logZ, ctmrg_mag = run_ctmrg(M_bulk, M_mag, chi, maxiter=max_iterations)
    
    print("CTMRG RESULTS:")
    print(f"logZ: {ctmrg_logZ.item():.10f}")
    print(f"Magnetization: {ctmrg_mag.item():.10f}")
    print()
    
    # Calculate and display errors
    logZ_error = abs(analytical_logZ - ctmrg_logZ.item())
    mag_error = abs(analytical_mag - ctmrg_mag.item())
    
    print("ERROR COMPARISON:")
    print(f"logZ error: {logZ_error:.2e}")
    print(f"Magnetization error: {mag_error:.2e}")
    print()
    
    # Relative errors
    logZ_rel_error = logZ_error / abs(analytical_logZ) if analytical_logZ != 0 else 0
    mag_rel_error = mag_error / abs(analytical_mag) if analytical_mag != 0 else 0
    
    print("RELATIVE ERRORS:")
    print(f"logZ relative error: {logZ_rel_error:.2e}")
    print(f"Magnetization relative error: {mag_rel_error:.2e}")
    print("="*50)
