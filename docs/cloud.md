# Cloud Usage

This page covers cloud-specific setup and usage for this repo. The cloud backend is managed by an adjacent project called [coalesce](https://github.com/brysonjones/coalesce), and uses **GCP Vertex AI** infra for GPU nodes.

## Cloud Config

Cloud settings live in the main config file under `cloud.gcp`. They are ignored by local training and local sweeps, but will be validated when you run using the cloud

Install the optional cloud dependencies before using the cloud launchers:

```bash
uv sync --extra cloud
```

## Cloud Training

Launch training on GCP Vertex AI:

```bash
# Default: T4
python -m open_value_estimator.cloud.cloud_launcher \
  config=configs/config.yaml \
  run_name=my_run

# Specify GPU type and count
python -m open_value_estimator.cloud.cloud_launcher \
  config=configs/config.yaml \
  run_name=my_run \
  gpu=a100x4

# Launch in detached mode
python -m open_value_estimator.cloud.cloud_launcher \
  config=configs/config.yaml \
  run_name=my_run \
  gpu=h100 \
  detached=true
```

Available GPU options: `t4`, `t4x2`, `t4x4`, `a100`, `a100x2`, `a100x4`, `a100x8`, `h100`, `h100x2`, `h100x4`, `h100x8`.

Multi-GPU configurations automatically enable Accelerate distributed training.

## Cloud Eval

Use the dedicated eval launcher with a minimal eval config:

```bash
python -m open_value_estimator.cloud.cloud_eval \
  config=configs/eval.yaml
```

Override eval settings as needed:

```bash
python -m open_value_estimator.cloud.cloud_eval \
  config=configs/eval.yaml \
  episodes=[0,5,10] \
  gpu=h100 \
  detached=true
```

## Cloud Sweeps

Local sweeps are documented in the main README. This section covers distributed sweep agents on GCP.

### Prerequisites

- Set `WANDB_API_KEY` in your environment.
- Set `HF_TOKEN` in your environment.
- Make sure `cloud.gcp` is set in `configs/config.yaml`.

### GCP Sweep Usage

Create a new sweep and launch multiple GCP agents:

```bash
python -m open_value_estimator.cloud.cloud_launcher \
  config=configs/config.yaml \
  run_name=gcp_sweep_run \
  sweep_config=configs/sweeps/lr_batch_sweep.yaml \
  sweep_count=10 \
  agents=4 \
  gpu=a100 \
  detached=true
```

Join an existing sweep and launch agents:

```bash
python -m open_value_estimator.cloud.cloud_launcher \
  config=configs/config.yaml \
  run_name=gcp_sweep_run \
  sweep_id=<SWEEP_ID> \
  sweep_count=10 \
  agents=2 \
  gpu=h100
```

Sweep behavior:

- `agents` is the number of parallel jobs run across different machines.
- `sweep_count` is the number of trials each agent runs sequentially.
- Total trials launched is `agents * sweep_count`.
- For multi-GPU choices (e.g., `a100x4`, `h100x8`), Accelerate is auto-configured to match GPU count.
