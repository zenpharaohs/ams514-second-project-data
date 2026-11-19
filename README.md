# AMS514 second-project data

Precomputed neural-network training losses for an open-ended project in Monte Carlo and simulation. Each arm is a fixed descent trajectory from one initialization. An allocator decides which trajectory receives the next unit of work and observes only that arm's next recorded loss.

The data were generated on **18 November 2026** using spong. Students can investigate allocation, exploration, target-reaching behavior, and distributions of performance without generating trajectories or installing spong. The project does not prescribe a winning algorithm or an expected tail law.

## Get started

```sh
git clone https://github.com/zenpharaohs/ams514-second-project-data.git
cd ams514-second-project-data
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_data.py
python student_example.py --case near_saddle_gd --arms 32 --budget 2048
```

The final command runs a **greedy starter example**. Edit `choose_arm(observation)` in [student_example.py](student_example.py) to investigate a different allocation rule. Use `--pools 4 --budget 128` for a smaller check, and `--out runs/example.npz` to save revealed outcomes and decisions. The default seed is 514; `--seed` controls stochastic allocators.

All data are ordinary NumPy `.npz` files stored directly in Git, about 181 MB in total. A standard clone or GitHub's Download ZIP includes them; Git LFS is not required. The NumPy-only example and integrity checks do not require a C compiler.

## The four tape banks

Each file contains 4,096 trajectories, stored as `loss[pool, arm, local_pull]` with shape **`(512, 8, 2049)`**. Column zero is the initialization loss. Columns 1 through 2,048 follow successive chunks of **four attempted optimizer updates**.

| Case | Initialization | Descent method |
| --- | --- | --- |
| [quadratic_gd](corpus/quadratic_gd.npz) | IID uniform on `[-2,2] × [-2,2]` | Population GD |
| [near_saddle_gd](corpus/near_saddle_gd.npz) | Designed near-saddle offsets along the unstable eigendirection; equiprobable signs, log-uniform distances | Population GD |
| [plateau_gd](corpus/plateau_gd.npz) | IID uniform on `[-50,50] × [-0.6,0.6]` | Population GD |
| [plateau_ngd](corpus/plateau_ngd.npz) | The same initializations as `plateau_gd` | Damped population natural gradient |

All objectives have the one-neuron form

$$
L(a,b)=\mathbb{E}\big[(f(X)-a\,g(bX))^2\big],\qquad X\sim N(0,1).
$$

The quadratic cases use `f = g = 1 + x + x²`. The plateau cases use spong's `normal-symmetric-small-integer-d9`. Each [JSON sidecar](corpus) records the exact polynomial coefficients, initialization law, optimizer settings, generation seed, targets, and source commit.

These are deterministic **population-gradient** trajectories. Randomness comes from initialization and stochastic allocation, not minibatch noise. The near-saddle law is deliberately designed; it is not a claim about typical random starts.

## Choose an arm count

The storage layout does not fix the experiment at eight arms. [dataset.py](dataset.py) regroups the same flat sequence of starts without sorting by outcomes or sampling with replacement:

| Arms per pool | Disjoint pools | Average pulls per arm at total budget 2,048 |
| ---: | ---: | ---: |
| 8 | 512 | 256 |
| 16 | 256 | 128 |
| 32 | 128 | 64 |
| 64 | 64 | 32 |

```python
from dataset import load_losses
from playback import PrefixPlayback

loss = load_losses("near_saddle_gd", arms=32)  # harness only
environment = PrefixPlayback(loss)
observation = environment.observation()
# Pass observation, not loss or environment, to the allocation rule.
```

The flat identity of an original arm is `8 * original_pool + original_arm`. Regrouping to `K` arms maps it to `(identity // K, identity % K)`. Use the same pools for policy comparisons. Experiments at different arm counts reuse the same bank, and the two plateau cases share initializations; these comparisons are not independent samples.

## The information and budget contract

An observation contains `total_pulls`, current per-arm `counts`, and current held `losses`. Pulling an arm reveals exactly its next entry. A policy may retain its observed history. All initialization losses are common information; the examples then allocate one warm-up pull to each arm before adaptive choices. Warm-up counts toward the total pull budget.

Full future tapes, geometric snapshots, termination labels, optima, and target availability belong to the harness or evaluator. They are not policy inputs. The interface returns read-only observation copies and rejects tape overruns. This is an experimental information contract, not a security sandbox: the raw data are intentionally downloadable.

The starter retains the best checkpoint ever revealed. Equal pulls do not imply equal gradient evaluations, objective evaluations, or wall time. Initial losses cost one objective evaluation per arm; work arrays record subsequent cumulative costs. See [the data and evaluation notes](docs/data_and_evaluation.md) for numerical conventions and suggested metrics.

## Supplied reference allocators

[playback.py](playback.py) includes `round_robin`, `descent_ucb`, `cb`, and `cb_floor`. These are the declared rules used to check the dataset, with current-loss statistics. They are starting points for experimentation, not a required ordering of performance.

```sh
python student_example.py --policy round_robin --arms 32
python student_example.py --policy descent_ucb --arms 32
python student_example.py --policy cb --arms 32 --seed 514
python student_example.py --policy cb_floor --arms 32 --seed 514
```

CB uses the included continuous-Bernoulli conjugate sampler. It requires a C99 compiler (`cc`, Clang, or GCC); the wrapper builds a temporary shared library on macOS/Linux. The `cb_floor` variant adds 10% uniform exploration after warm-up.

Optional adaptive CB uses the 75%-below-scale trigger and numerical matching of selection probabilities within the conjugate family:

```sh
python -m pip install -r requirements-optional.txt
python student_example.py --policy adaptive_matched --arms 32
python student_example.py --policy adaptive_matched_floor --arms 32
```

These variants additionally require SciPy and the C compiler. The supplied matcher supports at most 64 arms and can be much slower than ordinary playback. Accepted scale changes have numerical probability-matching tolerances; arbitrary rescalings are not guaranteed to be feasible. Details are in [the data and evaluation notes](docs/data_and_evaluation.md).

## Integrity and provenance

[DATA_MANIFEST.json](DATA_MANIFEST.json) contains SHA-256 hashes, file sizes, array shapes, and dtypes. The published corpus files are byte-for-byte copies of the generated banks. Record the repository commit along with your seeds and experimental settings.

```sh
python verify_data.py
# Optional deeper numerical and playback checks; needs mpmath and a compiler:
python -m pip install -r requirements-optional.txt
python verify.py --corpus corpus --out runs/validation.json
```

The original numerical validation and tested environment are retained in [validation](validation). Only regeneration needs a separate spong checkout, at the source commit recorded in the metadata:

```sh
python generate.py --spong /path/to/spong --out regenerated
```

The generator uses seed `20261118` by default. Compiler and floating-point differences can change regenerated trajectories; the hashes identify the distributed data independently of regeneration. The bundled continuous-Bernoulli code is from commit `179b0db51fb5943c04073694127a9e09a14f0704`; its [MIT license](vendor/LICENSE.continuous-bernoulli) is retained.
