#!/usr/bin/env python3
"""Test mujoco playground"""

import os
import time
import jax
import jax.numpy as jnp

# XLA flags for performance
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import mujoco_playground
from mujoco_playground import registry

print(registry.locomotion.ALL_ENVS)