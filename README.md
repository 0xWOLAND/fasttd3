# `fasttd3` 

![Pendulum](pendulum.gif)
*Pendulum environment trained using this JAX port of FastTD3.*

A pure JAX port of [FastTD3](https://github.com/younggyoseo/FastTD3) — a high-performance variant of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm — tested on the [Farama Gymnasium Pendulum task](https://gymnasium.farama.org/environments/classic_control/pendulum/) for demonstration purposes.

This implementation replicates the FastTD3 paper's approach using JAX/Flax with distributional critics and vectorized training.

## Usage

```bash
uv run train.py
```