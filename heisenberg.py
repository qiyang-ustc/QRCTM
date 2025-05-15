# %% Import Modules
import fire
import jax
import numpy as np
import jax.numpy as jnp
import time
from scipy import optimize
from functools import partial
from jaxtyping import Num, Array
from opt_einsum import contract
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
print("Runing on Device:", jax.devices())


def init_env(M: Array, chi: int) -> tuple[Array, Array]:
    D = M.shape[1]
    T = jnp.einsum("ialqr, iampn -> lmqprn", M, M).reshape((D * D, D, D, D * D))
    C = jnp.linalg.eigvalsh(jnp.einsum("lqqr->lr", T))

    nC = contract("i,ibfk,iael,xabcd,xefgh->kcgldh", C, T, T, M, M)
    nC = nC.reshape(nC.shape[0] * D * D, nC.shape[0] * D * D)
    w, v = jnp.linalg.eigh(nC)
    v = v[:, jnp.argsort(jnp.abs(w))[-chi:]].reshape(-1, D, D, chi)
    C = contract("ibfj,i,iaep,xabcd,xefgh,jcgq,pdhr->qr", T, C, T, M, M, v, v)
    C, q = jnp.linalg.eigh(C)
    T = contract("ibfj,iaep,xabcd,xefgh,jcgq->pdhq", T, v, M, M, v)
    T = jnp.einsum("ijkl,ia,lb->ajkb", T, q, q)
    return [C, T]


# noinspection DuplicatedCode
def symmetrize_C4(T: Num[Array, "dims"]) -> Num[Array, "ndims"]:
    T = T + T.transpose(0, 1, 4, 3, 2)  # U-D reflection
    T = T + T.transpose(0, 3, 2, 1, 4)  # L-R reflection
    T = T + T.transpose(0, 4, 1, 2, 3)  # 90 deg CCW rotation
    T = T + T.transpose(0, 3, 4, 1, 2)  # 180deg CCW rotation
    return T


@partial(jax.jit, static_argnums=(2, 3))  # D and chi are static
def qr_step(C, T, D, chi):
    reshaped = jnp.reshape(jnp.einsum("i,iklm->iklm", C, T), (D * D * chi, chi))
    q, r = jax.lax.linalg.qr(reshaped, full_matrices=False)
    return jnp.reshape(q, (chi, D, D, chi))


@jax.jit
def update_T(T, v, Mu, Md):
    return contract("ibfj,iaep,xabcd,xefgh,jcgq->pdhq", T, v, Mu, Md, v)


@jax.jit
def update_C(T, C, v, Mu, Md):
    extended_C = contract(
        "ibfj,i,iaep,xabcd,xefgh,jcgq,pdhr->qr", T, C, T, Mu, Md, v, v
    )
    C, q = jnp.linalg.eigh(extended_C)
    return C, q


@jax.jit
def normalize_without_gradient(T):
    return T / jax.lax.stop_gradient(jnp.linalg.norm(T))


# noinspection DuplicatedCode
@jax.jit
def next_TC(env, Mu, Md):
    C, T = env
    D, chi = Mu.shape[1], T.shape[0]

    v = qr_step(C, T, D, chi)
    C, q = update_C(T, C, v, Mu, Md)

    T = update_T(T, v, Mu, Md)
    T = jnp.einsum("ijkl,ia,lb->ajkb", T, q, q)

    return [normalize_without_gradient(C), normalize_without_gradient(T)]


# noinspection DuplicatedCode
@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def ctmrg(env, Mu, Md, miniter=20, maxiter=30, tol=1e-12, verbosity=0):
    for i in range(maxiter):
        env = [env[0], env[1] + env[1].transpose(3, 1, 2, 0)]
        env = next_TC(env, Mu, Md)
    return env


# noinspection DuplicatedCode
def traspose_basis(basis, params):
    basisT = np.zeros((params.size, basis.size))
    for i in range(basis.size):
        basisT[basis[i], i] = 1
    return jnp.array(basisT)


@jax.jit
def get_rho(env, M1u, M1d, M2u, M2d):
    C, T = env
    C = C + C.T
    T = T + T.transpose(3, 1, 2, 0)
    L = contract("i,iklm,m->iklm", C, T, C)
    L1 = contract("ibfj,iaep,xabcd,yefgh,jcgq->pdhqxy", L, T, M1u, M1d, T)
    L2 = contract("ibfj,iaep,xabcd,yefgh,jcgq->pdhqxy", L, T, M2u, M2d, T)
    return jnp.einsum("ijklab,ijklcd->abcd", L1, L2)


# noinspection DuplicatedCode
sigma_x = jnp.array([[0, 1.0], [1.0, 0]])
sigma_y = jnp.array([[0, -1.0j], [1.0j, 0]])
sigma_z = jnp.array([[1.0, 0], [0, -1.0]])
sigma_p = jnp.array([[0, 1.0], [0, 0]])
sigma_m = jnp.array([[0, 0], [1.0, 0]])
rot = jnp.array([[0, 1.0], [-1.0, 0]])

sigma_p_rot = rot @ sigma_p @ rot.T
sigma_m_rot = rot @ sigma_m @ rot.T
sigma_z_rot = rot @ sigma_z @ rot.T

Ez = jnp.einsum("ij,kl->ijkl", sigma_z, sigma_z_rot) / 4
Ep = jnp.einsum("ij,kl->ijkl", sigma_p, sigma_m_rot) / 2
Em = jnp.einsum("ij,kl->ijkl", sigma_m, sigma_p_rot) / 2
Eterm = Ez + Ep + Em


@partial(jax.jit, static_argnums=(3, 4))
def Heisenberg_energy(params, env, basis, d, D):
    """
    res=tprod({SIGMAZ,SIGMAZ})/4 + tprod({SIGMAP,SIGMAM})/2 + tprod({SIGMAM,SIGMAP})/2;
    res=fscon({res,rot,conj(rot)},{[-1 2 -3 4],[2 -2],[4 -4]});
    """

    M = params[basis].reshape(d, D, D, D, D)
    rho = get_rho(env, M, M, M, M)
    I = jnp.einsum("aacc->", rho)
    E = jnp.einsum("abcd,abcd->", rho, Eterm)
    return 2.0 * E / I


# noinspection DuplicatedCode
@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def loss(params, env, basis, d, D, maxiter=12, warmupiter=2):
    params = normalize_without_gradient(params)
    Mu, Md = (
        params[basis].reshape(d, D, D, D, D),
        params[basis].reshape(d, D, D, D, D).conj(),
    )

    env = jax.lax.stop_gradient(ctmrg(env, Mu, Md, maxiter=warmupiter))
    env = ctmrg(env, Mu, Md, maxiter=maxiter)
    return Heisenberg_energy(params, env, basis, d, D)


# noinspection DuplicatedCode
def plot_fig(
    data,
    file="energy",
    yscale="log",
    figsize=(10, 6),
    scatter_size=30,
    scatter_color="#1f77b4",
    xlabel="Iterations",
    ylabel=None,
    title=None,
    title_fontsize=14,
    label_fontsize=12,
    submin=False,
):

    data = np.array(data)
    if submin:
        shift = np.min(data)
        title = title + f" Shift:{shift:.8f}"
    else:
        shift = 0.0
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        np.arange(len(data)),
        data - shift,
        s=scatter_size,
        color=scatter_color,
    )
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    fig.tight_layout()
    plt.savefig(file + ".png", dpi=600)
    return None


# noinspection DuplicatedCode
def generate_basis(a, symmetrize):
    a_size = np.size(a)
    q = -np.ones(a_size, dtype=np.int32)
    t = np.zeros(a_size, dtype=np.int32)
    tot = 0  # total number of basis
    for i in range(a_size):
        if q[i] == -1:  # if q[i]!=0 means this number is occupied, and linear dependent
            t.fill(0)
            t[i] = 1
            t_sym = symmetrize(t.reshape(a.shape)).flatten()
            q[~np.isclose(t_sym, 0)] = tot
            tot += 1
    return q, tot


# noinspection DuplicatedCode
def loss_and_grads(**kwargs):
    local_loss = partial(loss, **kwargs)  # loss(params, env)

    def cal_loss(params, env):
        value = local_loss(jnp.array(params), env)
        return np.array(value)

    def cal_grads(params, env):
        params = jnp.array(params)
        grads = jax.grad(local_loss)(params, env)
        return grads

    return cal_loss, cal_grads


# noinspection DuplicatedCode
def main(
    chi: int = 80,
    d: int = 2,
    D: int = 4,
    seed: int = 52,
    maxiter: int = 12,
    init=None,
    maxoptiter=10000,
):

    basis, Nparams = generate_basis(
        jax.random.uniform(jax.random.PRNGKey(seed), (d, D, D, D, D)), symmetrize_C4
    )
    basis = jnp.array(basis)

    # init params
    if init is not None:
        params = init
    else:
        params = jax.random.uniform(jax.random.PRNGKey(seed), Nparams) - 0.5

    # A careful initialization
    M = params[basis].reshape(d, D, D, D, D)
    env = init_env(M, chi)
    C, T = ctmrg(env, M, M, maxiter=100)
    env[0] = C
    env[1] = T

    loss_history = []
    time_history = []
    start = time.time()
    numpy_loss, numpy_grads = loss_and_grads(D=D, d=d, basis=basis, maxiter=maxiter)

    def callback(params):
        nonlocal env
        loss_history.append(numpy_loss(params, env))
        time_history.append(time.time() - start)
        print(loss_history[-1])
        Mu, Md = (
            params[basis].reshape(d, D, D, D, D),
            params[basis].reshape(d, D, D, D, D).conj(),
        )
        C, T = ctmrg(env, Mu, Md, maxiter=10)
        env[0] = C
        env[1] = T
        # print(C) # spectrum
        return None

    res = optimize.fmin_l_bfgs_b(
        func=lambda x: numpy_loss(x, env),
        x0=params,
        fprime=lambda x: numpy_grads(x, env),
        callback=callback,
        factr=1e0,
        m=200,
        pgtol=1e-10,
        maxiter=maxoptiter,
        maxls=10,
    )

    print(res)
    plot_fig(loss_history, title="energy", file="energy", submin=True)
    plot_fig(time_history, title="time", file="time", yscale="linear")
    return res, env


if __name__ == "__main__":
    res, env = main(seed=93)
