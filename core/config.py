from __future__ import annotations

import os
import uuid
from typing import Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"

    @computed_field
    @property
    def train_files(self) -> str:
        return os.path.join(self.data_path, "fineweb_train_*.bin")

    @computed_field
    @property
    def val_files(self) -> str:
        return os.path.join(self.data_path, "fineweb_val_*.bin")


class RunConfig(BaseSettings):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resume_from: str = Field("", validation_alias="RESUME")
    checkpoint_every: int = 500
    seed: int = 1337


class ScheduleConfig(BaseSettings):
    val_batch_size: int = 524_288
    val_loss_every: int = 1000
    train_log_every: int = 200
    iterations: int = 500
    warmdown_iters: int = 1200
    warmup_steps: int = 20
    train_batch_tokens: int = 524_288
    train_seq_len: int = 1024
    max_wallclock_seconds: Optional[float] = None


class ModelCommonConfig(BaseSettings):
    model_version: str = "v3"
    vocab_size: int = 1024
    num_steps: int = 8
    n_fourier_basis: int = 16
    n_channels: int = 128
    activation: str = "gelu"
    logit_softcap: float = 30.0
    decay_init: float = 3.0


class V1Config(BaseSettings):
    num_heads: int = 8
    num_kv_heads: int = 4
    rope_base: float = 10000.0
    qk_gain_init: float = 1.5


class V2Config(BaseSettings):
    kernel_size: int = 16


class V4Config(BaseSettings):
    n_heads: int = 4
    transform_rank: int = 8
    unique_steps: int = 5
    invocations_per_step: int = 2


class WaveConfig(BaseSettings):
    slow_decay_init: float = 4.0
    fast_decay_init: float = 2.0
    band_split: str = "4,4,8"


class LgpConfig(BaseSettings):
    n_ops: int = 8


class GraphConfig(BaseSettings):
    interaction_rank: int = 64


class MetaStateConfig(BaseSettings):
    state_dim: int = 64
    inner_dim: int = 128


class TpgConfig(BaseSettings):
    gumbel_tau: float = 1.0
    halt_threshold: float = 0.5
    ponder_lambda: float = 0.01


class SparseRegisterConfig(BaseSettings):
    k_active: int = 256
    inner_mul: int = 2
    parallel_waves: bool = True
    grad_checkpoint: bool = False
    embed_dim: int = 128


class PredictiveConfig(BaseSettings):
    sparsity_k: int = 128
    aux_loss_weight: float = 0.1
    aux_loss_decay: float = 0.9


class ColumnarConfig(BaseSettings):
    num_columns: int = 4
    steps_per_column: int = 3
    n_branches: int = 4


class OptimizerConfig(BaseSettings):
    lr: float = 0.03
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


class DistributedConfig(BaseSettings):
    nccl_p2p_disable: str = "1"
    grad_accum_steps: int = 16
    torch_compile: bool = False
    roundtrip_eval: bool = False


class Hyperparameters:
    def __init__(self) -> None:
        self.data = DataConfig()
        self.run = RunConfig()
        self.schedule = ScheduleConfig()
        self.model_common = ModelCommonConfig()
        self.v1 = V1Config()
        self.v2 = V2Config()
        self.v4 = V4Config()
        self.wave = WaveConfig()
        self.lgp = LgpConfig()
        self.graph = GraphConfig()
        self.meta = MetaStateConfig()
        self.tpg = TpgConfig()
        self.sparse = SparseRegisterConfig()
        self.predictive = PredictiveConfig()
        self.columnar = ColumnarConfig()
        self.optimizer = OptimizerConfig()
        self.distributed = DistributedConfig()
        self._groups = [
            self.data, self.run, self.schedule, self.model_common,
            self.v1, self.v2, self.v4, self.wave, self.lgp,
            self.graph, self.meta, self.tpg, self.sparse,
            self.predictive, self.columnar,
            self.optimizer, self.distributed,
        ]

    def __getattr__(self, name: str):
        for group in self._groups:
            try:
                return getattr(group, name)
            except AttributeError:
                continue
        raise AttributeError(f"Hyperparameters has no field {name!r}")

    def to_dict(self) -> dict:
        result = {}
        for group in self._groups:
            result[type(group).__name__] = group.model_dump()
        return result
