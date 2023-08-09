# COMParative Imitation LEarning with Risk rewriting (COMPILER) and COMPILER with Estimation (COMPILER-E)

This repository contains the implementation of the COMParative Imitation LEarning with Risk-rewriting (COMPILER) algorithm and its variant with Estimation (COMPILER-E). These algorithms are designed for Vaguely Pairwise Imitation Learning (VPIL) problems. The RL part is borrowed from @openai/baselines.

## Structure of the Repository

The repository contains the following files and directories:

- `datasets/`: This directory contains scripts for handling datasets and utilities.
- `demonstrations/`: This directory contains the demonstrations $\Gamma^+$ and $\Gamma^-$ used in the experiments.
- `baselines/`: This directory contains the implementations of the openai baselines.
- `meansures/`: This directory contains scripts of PU learning measurements.
- `PU/`: This directory contains scripts of algorithms for estimating mixture proportion.
- `utils/`: This directory contains scripts of utils for estimating mixture proportion.
- `main.py`: The main entry point for executing the COMPILER and COMPILER-E algorithms.
- `adversary_COMPILER.py`: The risk-rewriting version of discriminator.
- `expert_ratio_estimator.py`: Estimating the value of $\alpha$ given $\Gamma^+$ and $\Gamma^-$.
- `ppo.py`: The tensorflow implementation of PPO algorithm. Mainly refer to @openai/baselines.
- `runner.py`: The interaction part between the agent and the environment. Mainly refer to @openai/baselines.


## Requirements
* python==3.6.5
* tensorflow-gpu==1.14.0
* mujoco-py==1.50.1.68
* gym==0.15.7
* tqdm==4.25.0
* numpy==1.16.4
* mpi4py==3.0.0
* cvxopt==1.3.0
* scikit-learn==0.24.2
* IPython==7.16.3
* matplotlib==3.3.4
* pynvml==11.5.0

## Demonstrations
We provide all demonstrations used in the experiment as in the anonymous link: `https://drive.google.com/file/d/1-LWep6U_FHdEWuEJxDaluQXBQWGC086L/view`. Then unzip the demonstrations into `demonstrations/`.

## Usage
Take `HalCheetah` with `$\alpha$=0.1` in `demonstrations/` as an example.

1. Install the environment based on requirements above.
2. For expert ratio $\alpha$ unknown situation, we need to first estimate $\alpha$ using `expert_ratio_estimator.py`.

    ```bash
    python expert_ratio_estimator.py --estimator=BBE --env=HalfCheetah-v2 --alpha=0.1 --demo_path=./demonstrations/mixture_3policies_ppo2.HalfCheetah.Reward_313.78_1563.28
    ```
3. Run the `main.py` script. Note that the input alpha will not be used if you are using COMPILER-E:

    ```bash
    python main.py --env=HalfCheetah-v2 --alg=COMPILER-E-BBE --alpha=0.1 --expert_path=./demonstrations/mixture_3policies_ppo2.HalfCheetah.Reward_313.78_1563.28 --num_timesteps=1e7 --seed=0 
    ```

Note: If you are using COMPILER-E, the algorithm will first estimate the expert ratio `α` using `expert_ratio_estimator.py`. For COMPILER, you need to provide the expert ratio `α` as an input.

## Algorithm Details

Please refer to the original paper for detailed explanations and theoretical analysis of the COMPILER and COMPILER-E algorithms.
