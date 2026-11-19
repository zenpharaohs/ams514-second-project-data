# Data and evaluation conventions

## Arrays

Load an NPZ with `numpy.load(path, allow_pickle=False)`. With the original storage layout:

| Key | Shape | Meaning |
| --- | --- | --- |
| `loss` | `(512, 8, 2049)` | Initialization followed by indexed losses |
| `initial_parameters` | `(512, 8, 2)` | Initial `(a,b)` |
| `objective_evaluations` | `(512, 8, 2049)` | Cumulative objective costs after initialization |
| `gradient_evaluations` | `(512, 8, 2049)` | Cumulative gradient costs |
| `terminal_status` | `(512, 8)` | Recording-cap or numerical stopping status |
| `stop_at` | `(512, 8)` | Local stopping pull; horizon plus one if still active |
| `geometry` | `(512, 8, snapshots, 2)` | Saved parameter positions |
| `geometry_pulls` | `(snapshots,)` | Local pulls corresponding to geometry |

The geometry, costs, endpoints, and labels are available for analysis after playback. The allocation rule observes only the prefix interface. When regrouping diagnostic arrays, use the same flat `(pool,arm)` order as for `loss`; leave `geometry_pulls` unchanged.

## Numerical descent

GD uses maximum step 0.05. The damped NGD variant uses the population Fisher/Gauss-Newton matrix `E[J Jᵀ]`, where `J=(g(bX), a X g′(bX))`, diagonal equilibration, damping 0.001 in correlation coordinates, and maximum step 1. Both use Armijo coefficient `1e-4`, at most 48 dyadic halvings, and twice the last accepted step as the next trial, capped at the maximum. These are the stated numerical optimizers, not a claim that every NGD or stochastic optimizer behaves identically.

Armijo loss differences use Horner divided differences to avoid subtracting nearly equal large plateau losses. Tiny negative evaluation roundoff in squared losses is clipped to zero. The degree-nine arrays have upward roundoff of order `6e-8` against a plateau level of order `3.5e8`; bitwise monotonicity is not an appropriate correctness test.

If no representable decreasing step is found, the tape holds its last finite state. This is a numerical precision limit, not a certified local minimum. Such continuations still consume allocation slots and loss-observation costs, but no additional gradient updates. No hidden stopping label is passed to an allocator. The final column is a finite-horizon observation, not an asserted asymptotic limit.

## Reference allocation rules

The fixed scale is the median positive initialization loss in the pool, using only common observed information. Scores are `z_i=L_i/(c+L_i)`. This transformation changes allocation scores; it does not rescale the objective used to generate a trajectory.

* **Round robin:** cyclic equal allocation.
* **Descent UCB:** choose the smallest `z_i - sqrt(2 log(1+t(log t)^2)/(N_i+1))`, with `t` the next pull. This is the declared transformed current-loss variant.
* **CB:** independently sample `η_i` with density proportional to `exp(ν_i[z_i η_i − B(η_i)])`, where `B(η)=log ∫₀¹ exp(ηu) du`, and choose the smallest sample. Fixed-scale CB uses `ν_i=N_i`. After a pull, statistics are replaced using the current loss rather than averaging past losses.
* **CB with exploration:** choose an independent uniform arm with probability 0.1 after warm-up; otherwise use CB.

Adaptive CB keeps actual pull counts separate from effective concentrations `ν_i`. It maintains the number of held losses below `c`. When an observed crossing brings the count to at least `ceil(0.75K)`, it proposes the current held-loss median as the new scale. It waits for common warm-up to finish, but not for an arbitrary fixed interval between checks. After rescaling it rebuilds the count. A pool with an observed zero loss uses the sampler's boundary limit and skips rescaling.

The matcher adjusts positive concentrations within the same conjugate family to preserve selection probabilities approximately. It uses 129-node probability quadrature for fitting, then a 257-node acceptance check with total variation at most `2e-4` and relative error at most 5% for probabilities at least `1e-4`. Concentrations are bounded between 0.02 and `1e6`. Proposals may be backtracked or rejected; a partial change must rearm the count trigger. These are numerical tolerances, not an exact representation theorem.

## Suggested measurements

Let `K` be the arm count, `T` the total pull budget including one warm-up pull per arm, and `H=2048` the recorded local horizon. Let `V_t` be the best checkpoint actually revealed by total pull `t`. After warm-up, the same-tape single-arm oracle is

\[
O_t=\min_i\min_{0\le n\le t-K+1}L_i(n),\qquad t\ge K.
\]

This benchmark chooses an arm in hindsight after giving every arm its warm-up pull. It is computed after playback. An allocation-quality measure is the cumulative shortfall

\[
A_T=\frac1S\sum_{t=K}^{T}(V_t-O_t),
\]

where the JSON sidecar supplies the fixed display scale `S=normalization_scale`. The oracle and the sum's start time change with `K`; keep that in mind when interpreting comparisons across arm counts.

A different statistic is finite-tape endpoint pseudo-regret:

\[
R_T^{(H)}=\frac1S\sum_i\big[L_i(H)-\min_j L_j(H)\big]N_i(T).
\]

This measures allocation to inferior recorded endpoints. It can disagree with achieved training quality, and it is not classical stationary-bandit regret or regret against certified limiting losses. The project is an experimental investigation; a stationary-bandit theorem should not be assumed to cover these descent tapes.

The supplied target is loss at most `1e-4` for the quadratic and 99% of the plateau-to-infimum drop for degree nine. A target is **available** in a pool at budget `T` if some tape reaches it by local pull `T-K+1`. Distinguish missing an available target from a pool in which the target is unavailable. Treat unreached hitting times as right censored.

Use distributions as well as means. A Q-Q plot against a baseline in common nonnegative shortfall units avoids separate centering and standardization. A paired difference plot answers a different question by retaining pool identity. Report replication counts, initialization law, allocation seeds, arm count, budget, actual work, and any conditioning on success or availability. An empirical upper tail in a finite experiment does not by itself identify an asymptotic tail law.
