# 3D Polymer Dynamics & Flory Scaling Simulation

    I built this project to simulate the conformational dynamics of a single linear polymer chain in three dimensions using Langevin dynamics. My primary goal with this code is to explore how excluded-volume interactions cause a real polymer chain to swell in a good solvent, and to verify the classic Flory scaling law (where the root-mean-square radius of gyration scales with the number of monomers as Rg ~ N^v, with the Flory exponent v ≈ 0.588) as well as Rouse relaxation dynamics across different chain lengths.

### How the Physics is Modeled

In this simulation, I model the polymer as a bead-spring chain where consecutive monomers are connected by harmonic springs:

    U_bond = 0.5 * k_bond * (r - b0)^2

To prevent the chain from passing through itself and to capture good-solvent excluded-volume effects, every pair of monomers interacts via a purely repulsive Weeks-Chandler-Andersen (WCA) potential, which is a Lennard-Jones potential shifted and truncated at its minimum (r_cut = 2^(1/6) * sigma). Rather than simulating explicit solvent molecules, I coupled the monomers to a thermal bath using the Gronbech-Jensen-Farago (GJF) Langevin integrator. I chose the GJF thermostat because it provides accurate configurational sampling and stable temperature control even at relatively large integration timesteps.

### Performance & Memory Optimization

Because calculating pairwise excluded-volume forces scales as O(N^2) with the number of monomers and conformational relaxation times scale roughly as tau_R ~ N^2.18, simulating longer chains (N >= 100) requires tens of millions of timesteps. Running that in pure Python is impossibly slow, so I used Numba's `@njit` compiler to compile the inner force loops and the GJF integration chunks down to fast machine code.

When I added the calculation for the End-to-End Vector Autocorrelation Function (E2E-ACF) to measure how quickly the polymer loses memory of its initial orientation, I quickly ran into a memory bottleneck: storing full 3D trajectories over millions of steps eats up gigabytes of RAM. To solve this without sacrificing statistical accuracy, I designed the production loop to generate thermal noise on the fly inside Numba and only record the single 3D end-to-end vector, R(t) = r_N(t) - r_1(t), at the end of each simulation chunk. Once the production run finishes, the script uses Fast Fourier Transforms (FFT) via `numpy.fft` to compute the autocorrelation across all time origins in a fraction of a second:

    C_R(t) = < R(t0 + t) . R(t0) > / < |R(t0)|^2 >

From this decay curve, the code automatically extracts both the exponential relaxation time (tau_R, by fitting the linear region of ln(C_R(t)) vs. t) and the integral relaxation time (tau_int, by integrating C_R(t) up to its first zero crossing).

### Scaling the Simulation for Longer Chains

One big lesson I learned while debugging the scaling behavior beyond N = 90 is just how sensitive large polymers are to their initial configuration and equilibration time. If the initial random walk is generated with a bond length that doesn't match the equilibrium spring length b0, the springs snap violently on the first step and trap the chain in artificial knots. Furthermore, because the Rouse relaxation time grows roughly with the square of the chain length, a fixed number of timesteps that easily equilibrates a 50-mer will leave a 100-mer severely under-equilibrated. 

In the current version of the code, the chain is initialized as a non-overlapping random walk directly at the equilibrium bond length, and both the equilibration and production step counts scale dynamically with N^2. At the end of each run, the script compares the total simulation time against the measured integral relaxation time (tau_int) to report how many statistically independent configurations were actually sampled.

### Getting Started

To run the simulation on your machine, you just need Python 3 along with NumPy, Numba, and Matplotlib:

    pip install numpy numba matplotlib

You can adjust the number of monomers (`n_part`), spring constant (`k_bond`), solvent viscosity (`eta`), and timestep (`dt`) directly at the top of the script and run it via:

    python core_mod_1.py
