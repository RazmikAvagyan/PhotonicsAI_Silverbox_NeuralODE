# Task 5: Neural-ODE Model of the Silverbox Oscillator

## Overview

This project models the Silverbox oscillator using a neural ordinary differential equation (Neural ODE). A neural network learns the system's acceleration from the output voltage, its velocity, and the input voltage. A fourth-order Runge–Kutta (RK4) integrator then uses the learned dynamics and measured input to predict the output waveform.

The implementation uses a 4,096-sample excerpt from the Silverbox dataset, beginning at index 42,650. The signals are centered by removing their means. The first 3,072 samples are used for training, and the remaining 1,024 samples are reserved for testing.

## Model

The system is represented as

$$
\dot y = v, \qquad \dot v = \alpha_\theta(y, v, u),
$$

where `y` is the centered output voltage, `v` is its velocity, `u` is the centered input voltage, and the neural network $\alpha_\theta$ predicts acceleration.

The network is a fully connected multilayer perceptron with three inputs, two hidden layers, and one output. The width (32 or 64 units per hidden layer) and the activation (Tanh, SiLU or GELU) are compared; the kept model is SiLU with 64 units per hidden layer (4,481 parameters). Signals are scaled using statistics from the training segment to make optimization more stable. RK4 advances the state through time.

## Files

- `data.py` — loads `SNLS80mV.mat`, extracts and centers the data, creates the train/test split, scales the signals, and prepares overlapping training windows.
- `model.py` — defines `AccelerationNet`, the neural network that predicts acceleration.
- `integrator.py` — implements one fourth-order Runge–Kutta integration step.
- `train.py` — trains the network on short trajectories and saves the checkpoint with the lowest training loss.
- `evaluate.py` — compares network architectures and training settings, evaluates free-run predictions, prints RMSE values, and saves plots.
- `SNLS80mV.mat` — required source dataset; it must be in the same directory as the Python scripts.

## Requirements

Python 3.9 or newer is recommended. The project requires:

- NumPy
- SciPy
- PyTorch
- Matplotlib

Install the dependencies with:

```bash
pip install numpy scipy torch matplotlib
```

## How to Run

Place `data.py`, `model.py`, `integrator.py`, `train.py`, `evaluate.py`, and `SNLS80mV.mat` in the same directory.

1. Check the dataset and preprocessing information:

   ```bash
   python data.py
   ```

2. Run the architecture and hyperparameter experiments:

   ```bash
   python evaluate.py
   ```

The evaluation script trains multiple models, prints training and test RMSE values, and creates these figures:

- `test_prediction.png` — measured output compared with the selected model's free-run prediction on the test segment.
- `architecture_comparison.png` — comparison of the six tested architectures and their absolute errors over the first 250 test samples.

Note: `evaluate.py` performs several training runs, so it may take some time to complete. The `.mat` dataset must be available locally before running the scripts.

## Training and Evaluation

Training uses overlapping windows from the first 3,072 samples. For each window, the model starts from the measured initial position and estimated velocity, then predicts the trajectory using RK4. The loss is the mean squared error between predicted and measured output positions.

For final evaluation, the model makes one continuous free run over all 4,096 samples. After the initial state is set, the rollout uses the measured input `u` but does not use the measured output `y` to correct predictions. This tests whether the learned dynamics can reproduce the output over time.

The evaluation compares:

- Activations: Tanh, SiLU, and GELU.
- Hidden-layer widths: 32 and 64 units.
- Rollout lengths: 10, 20, and 40 steps.
- Learning rates: 0.001, 0.002, and 0.005.
- Training durations: 10, 20, and 60 epochs.
- Optimizers: Adam, AdamW, and RMSprop.

The rollout length, learning rate, number of epochs and optimizer are tuned for every architecture, one change at a time around a base run (Adam, learning rate 0.002, 20 epochs, rollout 20). The kept model is the run with the lowest free-run test RMSE over all runs.

## Result

All numbers are free-run RMSE in volts, from one run of `python evaluate.py` (seed 0).

| Quantity | Value |
|---|---:|
| Kept model | SiLU, 64-64 hidden units, Adam, learning rate 0.002, 60 epochs, rollout length 20 |
| Free-run training RMSE (3072 samples) | 0.0135 V |
| Free-run test RMSE (1024 samples) | 0.0158 V |
| Zero-prediction test baseline RMSE | 0.0605 V |

The test RMSE is about 3.8 times lower than the zero-prediction baseline, and only 1.17 times the training RMSE, so the free run does not drift on the test segment.

Best result of each architecture after tuning (all six reached it with 60 epochs):

| Architecture | Train RMSE (V) | Test RMSE (V) |
|---|---:|---:|
| SiLU, 64 units (kept) | 0.0135 | 0.0158 |
| GELU, 64 units | 0.0134 | 0.0158 |
| GELU, 32 units | 0.0130 | 0.0168 |
| Tanh, 64 units | 0.0136 | 0.0211 |
| Tanh, 32 units | 0.0147 | 0.0225 |
| SiLU, 32 units | 0.0152 | 0.0232 |

Findings from the tuning runs (one setting changed at a time, for every architecture):

- 60 epochs was the best setting for every architecture. It is also the longest one tried, so longer training may help further.
- A learning rate of 0.005 and a rollout length of 10 beat the base setting (learning rate 0.002, 20 epochs, rollout 20) in all six architectures. A learning rate of 0.001, 10 epochs and a rollout length of 40 were worse than the base in all six.
- With a rollout length of 40, the SiLU and GELU models are no better than predicting zero (test RMSE 0.0608 V to 0.0637 V).
- AdamW behaves like Adam. RMSprop was only run for 20 epochs: it beats Adam at 20 epochs, but is behind Adam at 60 epochs.
- The tie between SiLU 64 and GELU 64, and the gap between Tanh and the other activations, are smaller than the run-to-run variation (an unseeded run of one setting differed by about 0.01 V), so they should not be over-interpreted.

## Notes

- `SNLS80mV.mat` (Wigren and Schoukens, https://www.nonlinearbenchmark.org/) must be in the same folder as the scripts.
- `evaluate.py` saves the best checkpoint of each run as `tmp_model.pt`, replacing it during the next run, so this file is not part of the result. The figures `test_prediction.png` and `architecture_comparison.png` are the saved outputs of the kept model and of the six architectures with the default training settings.
- All runs use seed 0, so a run can be repeated. Differences of about 0.01 V between runs can come from the seed alone.
