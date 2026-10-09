# Task 5: Neural-ODE Model of the Silverbox Oscillator

## Overview

This project models the Silverbox oscillator using a neural ordinary differential equation (Neural ODE). A neural network learns the system's acceleration from the output voltage, its velocity, and the input voltage. A fourth-order Runge–Kutta (RK4) integrator then uses the learned dynamics and measured input to predict the output waveform.

The implementation uses a 4,096-sample excerpt from the Silverbox dataset, beginning at index 42,650. The signals are centered by removing their means. The first 3,072 samples are used for training, and the remaining 1,024 samples are reserved for testing.

## Model

The system is represented as

\[
\dot y = v, \qquad \dot v = \alpha_\theta(y, v, u),
\]

where `y` is the centered output voltage, `v` is its velocity, `u` is the centered input voltage, and the neural network \(\alpha_\theta\) predicts acceleration.

The network is a fully connected multilayer perceptron with three inputs, two hidden layers, and one output. The selected model uses 64 units per hidden layer and SiLU activation. Signals are scaled using statistics from the training segment to make optimization more stable. RK4 advances the state through time.

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

## Reported Result

According to the project summary, the selected model uses SiLU activation, 64 units in each hidden layer, Adam with a learning rate of `0.002`, 60 epochs, and a rollout length of 20.

| Metric | Reported value |
|---|---:|
| Free-run training RMSE | 0.0135 V |
| Free-run test RMSE | 0.0158 V |
| Zero-prediction test baseline RMSE | 0.0605 V |

The reported test RMSE is about 3.8 times lower than the zero-prediction baseline. Results can vary if the dataset, code, random seed, or training settings change.

## Notes

- The dataset file `SNLS80mV.mat` is not included in this README and must be obtained separately.
- `evaluate.py` saves the best checkpoint from each run as `tmp_model.pt`, replacing it during subsequent runs. The final selected model's weights are not separately saved under a permanent descriptive filename by the current evaluation script.
- The reported results above are taken from the supplied project summary; run the scripts to reproduce and verify them in your environment.
