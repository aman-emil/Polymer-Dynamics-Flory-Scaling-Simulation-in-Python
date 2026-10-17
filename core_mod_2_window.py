import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
from numba import njit, prange


################# Simulation Parameter #################
kbT = 1
dt = 0.005
m = 1
eta = 0.05
r = 1
alpha = 6*np.pi*eta*r
a = (1.0 - alpha * dt / (2.0 * m)) / (1.0 + alpha * dt / (2.0 * m))
b = 1.0 / (1.0 + alpha * dt / (2.0 * m))
beta_std = np.sqrt(2.0 * alpha * kbT * dt)
b0 = 1.2
record_every = 200
sigma = 1
epsilon = 1
k_bond = 100



################# Random Number Initialization ############
rng = np.random.default_rng(1)

################# Position Initialization #################
n_part = 100
bond_length = 1.2
center = np.array([0, 0, 0])
print(f"no of particles in simulation {n_part}")

def initialize_polymer_chain(n_part, bond_length=1.2, center=None):
    """Initializes a linear polymer chain centered at a specified 3D coordinate,
    grown as a random walk with non-overlapping monomers.

    Parameters:
    - n_part: Number of particles in the polymer (default: 100)
    - bond_length: Distance between adjacent monomers (default: 2.0)
    - center: Array-like of shape (3,) specifying the center of the chain (default: [0.0, 0.0, 0.0])

    Returns:
    - positions: NumPy array of shape (n_part, 3) containing particle coordinates.
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        center = np.asarray(center, dtype=float)

    positions = np.zeros((n_part, 3), dtype=float)
    positions[0] = center

    for i in range(1, n_part):
        while True:
            theta = np.arccos(2.0 * np.random.rand() - 1.0)
            phi = 2.0 * np.pi * np.random.rand()
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            new_pos = positions[i - 1] + bond_length * direction

            if i == 1 or np.min(np.linalg.norm(positions[:i] - new_pos, axis=1)) > 0.9 * sigma:
                positions[i] = new_pos
                break

    return positions


r0 = initialize_polymer_chain(n_part)

################# Velocity Initialization #################

def initialize_polymer_velocities(n_part, kbT=kbT):
    sigma_std = np.sqrt(kbT)
    velocities = np.random.normal(0.0, sigma_std, (n_part, 3))
    return velocities

v0 = initialize_polymer_velocities(n_part)

# ---------------------------------------------------------------
# Noise
# ---------------------------------------------------------------
def noise_3D(shape, beta_std):
    return np.random.normal(0.0, beta_std, size=shape)


################################# Forces Calculation ##############################
@njit
def bond_force(r, k_bond, b0):
    """Harmonic springs between consecutive monomers i, i+1.
    U_bond = 0.5*k_bond*(|r_i - r_{i-1}| - b0)^2
    JIT-compiled for speed."""
    n_part = r.shape[0]
    f = np.zeros_like(r)
    
    for i in range(n_part - 1):
        dr = r[i + 1] - r[i]
        dist = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
        if dist > 1e-10:  # Avoid division by zero
            fmag = k_bond * (dist - b0) / dist
            f[i + 1, 0] -= fmag * dr[0]
            f[i + 1, 1] -= fmag * dr[1]
            f[i + 1, 2] -= fmag * dr[2]
            f[i, 0] += fmag * dr[0]
            f[i, 1] += fmag * dr[1]
            f[i, 2] += fmag * dr[2]
    
    return f


@njit
def wca_force(r, epsilon, sigma):
    """Purely repulsive WCA excluded-volume force between ALL monomer pairs.
    JIT-compiled for O(N²) efficiency improvement.
    """
    n_part = r.shape[0]
    f = np.zeros_like(r)
    rcut = 2.0 ** (1.0 / 6.0) * sigma
    rcut_sq = rcut ** 2
    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    
    for i in range(n_part):
        for j in range(i + 1, n_part):
            dx = r[j, 0] - r[i, 0]
            dy = r[j, 1] - r[i, 1]
            dz = r[j, 2] - r[i, 2]
            
            dist_sq = dx**2 + dy**2 + dz**2
            
            if dist_sq < rcut_sq and dist_sq > 1e-10:
                inv_r = 1.0 / np.sqrt(dist_sq)
                inv_r6 = (sigma * inv_r) ** 6
                inv_r12 = inv_r6 ** 2
                
                # Force magnitude: F = 24*epsilon/r * (2*r^12 - r^6) / sigma^12
                fmag = 24.0 * epsilon * inv_r * (2.0 * inv_r12 - inv_r6)

                # Apply force
                fx = fmag * dx * inv_r
                fy = fmag * dy * inv_r
                fz = fmag * dz * inv_r
                
                f[i, 0] -= fx
                f[i, 1] -= fy
                f[i, 2] -= fz
                f[j, 0] += fx
                f[j, 1] += fy
                f[j, 2] += fz
    
    return f


@njit
def total_force(r, k_bond, b0, epsilon, sigma):
    """Combined force calculation: bond + WCA"""
    force = bond_force(r, k_bond, b0) + wca_force(r, epsilon, sigma)
    return force


##########################################################################
############ Grønbech-Jensen–Farago (GJF) Langevin integrator #############
##########################################################################

# ---------------------------------------------------------------
# Chunked integrator: runs N steps, returns updated state
# ---------------------------------------------------------------
@njit
def run_chunk(r, v, f_n, n_steps, beta_std, k_bond, b0, epsilon, sigma,
              dt, m, a, b):
    n_part = r.shape[0]
    for n in range(n_steps):
        beta_np1 = np.random.normal(0.0, beta_std, (n_part, 3))

        r_new = r + b * (
            dt * v
            + (dt ** 2 / (2.0 * m)) * f_n
            + (dt / (2.0 * m)) * beta_np1
        )

        f_np1 = total_force(r_new, k_bond, b0, epsilon, sigma)

        v_new = (
            a * v
            + (dt / (2.0 * m)) * (a * f_n + f_np1)
            + (b / m) * beta_np1
        )

        r = r_new
        v = v_new
        f_n = f_np1

    return r, v, f_n


def equilibrate(r0, v0, n_steps_eq, chunk, beta_std,
                k_bond, b0, epsilon, sigma,
                dt, m, a, b, n_part):
    r = r0.copy()
    v = v0.copy()
    f_n = total_force(r, k_bond, b0, epsilon, sigma)

    n_chunks = n_steps_eq // chunk
    for _ in range(n_chunks):
        r, v, f_n = run_chunk(r, v, f_n, chunk, beta_std,
                              k_bond, b0, epsilon, sigma,
                              dt, m, a, b)

    return r, v, f_n


# ---------------------------------------------------------------
# Radius of gyration (single frame)
# ---------------------------------------------------------------
@njit
def radius_of_gyration(r):
    com = r.sum(axis=0) / r.shape[0]
    diff = r - com
    return np.mean(np.sum(diff * diff, axis=1))


# ---------------------------------------------------------------
# End-to-End Vector Autocorrelation (FFT) & Relaxation Times
# ---------------------------------------------------------------
def compute_e2e_relaxation(R_traj, dt_chunk):
    """
    Computes the normalized End-to-End Vector Autocorrelation Function C_R(t)
    using FFT, and extracts both the integral and exponential relaxation times.

    Parameters:
    - R_traj   : NumPy array of shape (M, 3) containing R(t) = r_{N-1}(t) - r_0(t)
    - dt_chunk : Physical time between recorded chunks (chunk * dt)

    Returns:
    - t_lags   : Time lag array
    - C_R      : Normalized autocorrelation array C_R(t)
    - tau_int  : Integral relaxation time (up to first zero-crossing)
    - tau_exp  : Exponential relaxation time (fitted on 0.15 <= C_R <= 0.85)
    """
    M = R_traj.shape[0]

    # Zero-pad to next power of 2 (>= 2*M) to prevent circular convolution wrap-around
    n_fft = 1 << (2 * M - 1).bit_length()

    # FFT along the time axis (axis=0) for all 3 spatial components at once
    F = np.fft.rfft(R_traj, n=n_fft, axis=0)
    S = np.fft.irfft(F * np.conj(F), n=n_fft, axis=0)[:M]

    # Sum x, y, z dot-product contributions and divide by number of time origins (M - k)
    norm_counts = np.arange(M, 0, -1, dtype=np.float64)
    C_raw = np.sum(S, axis=1) / norm_counts

    # Normalize by <R^2> = C_raw[0]
    C_R = C_raw / C_raw[0]
    t_lags = np.arange(M, dtype=np.float64) * dt_chunk

    # 1. Integral relaxation time: integrate up to first zero/noise crossing (C_R <= 0.0)
    zero_crossings = np.where(C_R <= 0.0)[0]
    cutoff = zero_crossings[0] if len(zero_crossings) > 0 else M

    # Use np.trapezoid (NumPy 2.0+) with fallback to np.trapz (older NumPy)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    tau_int = trapz_fn(C_R[:cutoff], t_lags[:cutoff])

    # 2. Exponential relaxation time: linear fit of ln(C_R) vs t in [0.15, 0.85]
    # Only fit using data before the first zero crossing
    C_valid = C_R[:cutoff]
    t_valid = t_lags[:cutoff]
    mask = (C_valid >= 0.15) & (C_valid <= 0.85)

    if np.count_nonzero(mask) >= 3:
        slope, _ = np.polyfit(t_valid[mask], np.log(C_valid[mask]), 1)
        tau_exp = -1.0 / slope if slope < 0 else np.nan
    else:
        # Fallback: 1/e crossing (~0.3679)
        idx_1e = np.where(C_valid <= np.exp(-1.0))[0]
        tau_exp = t_valid[idx_1e[0]] if len(idx_1e) > 0 else np.nan

    return t_lags, C_R, tau_int, tau_exp


# ---------------------------------------------------------------
# production
# ---------------------------------------------------------------
def production_until_converged(r0, v0, f0, max_steps_prod, chunk, beta_std,
                               k_bond, b0, epsilon, sigma,
                               dt, m, a, b, n_part):
    """
    Runs full production for max_steps_prod, recording ROG and R(t) at every chunk.
    """
    r = r0.copy()
    v = v0.copy()
    f_n = f0.copy()

    n_chunks = max_steps_prod // chunk

    rog_arr = np.zeros(n_chunks, dtype=np.float64)
    step_axis = np.zeros(n_chunks, dtype=np.int64)
    R_traj = np.zeros((n_chunks, 3), dtype=np.float64)

    for i in range(n_chunks):
        r, v, f_n = run_chunk(r, v, f_n, chunk, beta_std,
                              k_bond, b0, epsilon, sigma,
                              dt, m, a, b)

        rog_arr[i] = radius_of_gyration(r)
        step_axis[i] = (i + 1) * chunk
        R_traj[i] = r[-1] - r[0]

        if (i + 1) % 500 == 0 or i == 0:
            running_mean = np.mean(rog_arr[:i + 1])
            print(f"step {step_axis[i]:>9d}/{max_steps_prod} | ROG = {rog_arr[i]:.4f} | running mean = {running_mean:.4f}")

    return step_axis, rog_arr, R_traj, r, v


def main(r0, v0, beta_std,
         k_bond, b0, epsilon, sigma,
         dt, m, a, b,
         n_steps_eq = int(500 * (n_part ** 2.0)),
         max_steps_prod = int(1500 * (n_part ** 2.0)),
         chunk=200):

    n_part = r0.shape[0]
    dt_chunk = chunk * dt

    print("Equilibrating...")
    r_eq, v_eq, f_eq = equilibrate(
        r0, v0, n_steps_eq, chunk, beta_std,
        k_bond, b0, epsilon, sigma,
        dt, m, a, b, n_part
    )
    print("Equilibration done. Starting production...\n")

    steps, rogs, R_traj, r_final, v_final = production_until_converged(
        r_eq, v_eq, f_eq, max_steps_prod, chunk, beta_std,
        k_bond, b0, epsilon, sigma,
        dt, m, a, b, n_part)

    running_mean = np.cumsum(rogs) / np.arange(1, len(rogs) + 1)

    # Calculate E2E Autocorrelation and Relaxation Times
    t_lags, C_R, tau_int, tau_exp = compute_e2e_relaxation(R_traj, dt_chunk)
    total_sim_time = steps[-1] * dt
    n_indep = total_sim_time / (2.0 * tau_int) if tau_int > 0 else 0.0

    print("\n================ E2E Relaxation Summary ================")
    print(f"Mean Squared E2E Distance <R^2> : {np.mean(np.sum(R_traj**2, axis=1)):.4f}")
    print(f"Integral Relaxation Time (tau_int): {tau_int:.4f} (in simulation time units)")
    print(f"Exponential Fit Time     (tau_R)  : {tau_exp:.4f} (in simulation time units)")
    print(f"Total Production Time             : {total_sim_time:.1f} (~{n_indep:.1f} independent samples)")
    print("========================================================\n")

    return steps, rogs, running_mean, t_lags, C_R, tau_int, tau_exp, r_final, v_final


if __name__ == "__main__":
    steps, rogs, running_mean, t_lags, C_R, tau_int, tau_exp, r_final, v_final = main(
        r0, v0, beta_std,
        k_bond, b0, epsilon, sigma,
        dt, m, a, b,
        n_steps_eq = int(500 * (n_part ** 2.0)),
        max_steps_prod = int(1500 * (n_part ** 2.0)),
        chunk=500,
    )

    # Plot up to 5 * tau_int so you see the clean decay region
    max_plot_t = 5.0 * tau_int if tau_int > 0 else t_lags[-1] * 0.1
    mask_plot = t_lags <= max_plot_t

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Linear plot of C_R(t)
    ax1.plot(t_lags[mask_plot], C_R[mask_plot], label=r"$C_R(t)$")
    ax1.axhline(np.exp(-1), color='gray', linestyle='--', label=r"$1/e \approx 0.368$")
    ax1.axvline(tau_int, color='r', linestyle=':', label=rf"$\tau_{{\mathrm{{int}}}} = {tau_int:.2f}$")
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel(r"$C_R(t)$")
    ax1.set_title("End-to-End Vector Autocorrelation")
    ax1.legend()

    # Semi-log plot of ln C_R(t) vs t
    pos_mask = mask_plot & (C_R > 0.05)
    ax2.plot(t_lags[pos_mask], np.log(C_R[pos_mask]), 'o', markersize=2, label=r"$\ln C_R(t)$")
    if not np.isnan(tau_exp):
        ax2.plot(t_lags[pos_mask], -t_lags[pos_mask] / tau_exp, 'r-',
                 label=rf"Fit slope $-1/\tau_R$ ($\tau_R = {tau_exp:.2f}$)")
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel(r"$\ln C_R(t)$")
    ax2.set_title("Log-Linear Relaxation Decay")
    ax2.legend()

    plt.tight_layout()
    plt.show()
