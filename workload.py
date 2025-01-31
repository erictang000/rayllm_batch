"""The workload."""
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import ray
from ray.data.dataset import Dataset

try:
    from ray.anyscale.data.checkpoint.interfaces import (
        CheckpointBackend,
        CheckpointConfig,
    )
except ImportError:
    CheckpointConfig = None
    CheckpointBackend = None

from .logging import get_logger
from .tokenizer import ChatTemplateTokenizer
from .utils import has_materialized
from pathlib import Path
import yaml

logger = get_logger(__name__)


def read_parquet(path: str, max_batch_size: int) -> Dataset:
    """Read the parquet file and return the dataset.
    Different from ray.data.read_parquet, this function controls
    the concurrency and the number of blocks. This is for the following
    reasons:
    - We have to set concurrency explicitly; otherwise
      read_parquet by default asking for all CPU cores to run the task,
      which leads insufficient resources for other stages.
    - The number of blocks determines the fault tolerance granularity.

    Args:
        path: The path to the parquet file.
        max_batch_size: The maximum batch size per block.

    Returns:
        The dataset.
    """
    ds = ray.data.read_parquet(
        path,
        concurrency=2,
    )
    # The .count() would be fast only if the dataset is read from
    # parquet and we are using RayTurbo 2.39+. In this case Ray Data
    # only reads the metadata and does not materialize the dataset.
    num_rows = ds.count()
    num_blocks = math.ceil(num_rows / max_batch_size)

    return ray.data.read_parquet(
        path,
        concurrency=2,
        override_num_blocks=num_blocks,
    )

def load_rayllm_config(config_path: str) -> Dict[str, Any]:
    if isinstance(config_path, str):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Engine config file {config} not found.")
        with open(config_path, "r") as filep:
            config = yaml.safe_load(filep)

    assert isinstance(config, dict)
    return config

@dataclass
class WorkloadBase:
    """The base class for a workload."""

    # The ray.data.Dataset. If None, the Worklod must initialize the dataset
    # in __post_init__().
    dataset: Optional[Dataset]
    # Sampling a fraction of dataset for benchmarking and testing. If the value
    # is greater than one, it means to take the first N rows from the dataset.
    dataset_fraction: float
    # Tokenizer class for the workload.
    tokenizer_cls: Any

    # Sampling parameters for the workload, such as max_tokens, temperature, etc.
    # It can only be None when the workload is used for embedding.
    sampling_params: Optional[Dict[str, Any]] = None
    # Pooling parameters for the workload, such as pooling_type, etc.
    # It can only be None when the workload is used for auto-regressive generation.
    pooling_params: Optional[Dict[str, Any]] = None

    need_tokenize: bool = True
    # The column name to be used as the checkpoint ID.
    ckpt_col_name: Optional[str] = None
    # When specified, the tokenization will be async because we don't need to
    # materialize an entire tokenized dataset to get the maximum tokens in prompt.
    # With the default value of -1, the actual value will be set after tokenization.
    max_tokens_in_prompt: int = -1

    # Do we want to carry over input keys that are not in the output?
    carryover_inputs: bool = True

    def validate(self):
        if not ((self.sampling_params is None) ^ (self.pooling_params is None)):
            raise ValueError(
                "Either sampling_params or pooling_params must be specified."
            )

    def get_preprocessed_dataset(
        self,
        max_batch_size: int = 256,
        ckpt_path: Optional[str] = None,
        repartition_by_batch_size: bool = False,
    ) -> Tuple[Dataset, Optional[int]]:
        """Load the dataset and process it.

        Args:
            max_batch_size: The batch size. This determines the number of rows per
            block. Note that if some rows have already processed (checkpointed),
            the actual batch size may be smaller than this value.
            ckpt_path: The path to the checkpoint directory.
            repartition_by_batch_size: Whether to repartition the dataset by the
                batch size for fault tolerance granularity. You should enable
                this when the dataset is not from parquet and checkpointing is
                disabled.

        Returns:
            The processed dataset and the number of blocks. If checkpointing is
            enabled, then the number of blocks is unknown.
        """
        self.validate()
        if self.dataset is None:
            raise ValueError(
                "dataset must be specified or initialized before calling "
                "get_preprocessed_dataset()."
            )

        self.max_batch_size = max_batch_size

        # Setup checkpointing.
        enable_ckpt = False
        if self.ckpt_col_name is not None:
            if ckpt_path is not None:
                if CheckpointConfig is None:
                    logger.warning(
                        "Checkpoint is not available. Please install Ray Turbo 2.39+"
                    )
                else:
                    enable_ckpt = True

                    if has_materialized(self.dataset):
                        curr_ds_stats = self.dataset.stats()
                        raise RuntimeError(
                            "The checkpointing is specified in this workload, but "
                            "the dataset has already been materialized so checkpointing "
                            "cannot be applied. Please either disable checkpointing "
                            "or avoid to materialize the dataset before running this "
                            f"workload. Current dataset stats: {curr_ds_stats}"
                        )

                    # Determine the checkpoint backend.
                    ckpt_path = os.path.expandvars(ckpt_path)
                    if ckpt_path.startswith("s3://"):
                        ckpt_backend = CheckpointBackend.S3
                    else:
                        ckpt_backend = CheckpointBackend.DISK

                    ckpt_config = CheckpointConfig(
                        enabled=True,
                        backend=ckpt_backend,
                        id_col=self.ckpt_col_name,
                        output_path=ckpt_path,
                    )
                    ray.data.DataContext.get_current().checkpoint_config = ckpt_config
            else:
                logger.warning(
                    "This workload specifies checkpoint column name but checkpoint "
                    "path is not provided, so checkpointing will be disabled."
                )

        ds = self.dataset
        if enable_ckpt and self.dataset_fraction != 1.0:
            raise ValueError(
                "Cannot process dataset fraction when checkpoint is enabled"
            )
        if self.dataset_fraction < 1.0:
            logger.info("Sampling %f dataset", self.dataset_fraction)
            ds = ds.random_sample(self.dataset_fraction, seed=0)
        elif self.dataset_fraction > 1.0:
            n_rows = int(self.dataset_fraction)
            logger.info("Taking the first %d rows from dataset", n_rows)
            ds = ds.limit(n_rows)

        if repartition_by_batch_size:
            if enable_ckpt:
                raise ValueError(
                    "Cannot repartition the dataset by batch size when "
                    "checkpointing is enabled."
                )
            num_requests = ds.count()
            num_blocks = math.ceil(num_requests / max_batch_size)
            ds = ds.repartition(num_blocks)

            logger.info("#Requests: %d (%d blocks)", num_requests, num_blocks)
        else:
            # When checkpointing is enabled, the number of blocks is unknown
            # at this point.
            num_blocks = None

        mapper_fn = (
            self.parse_row_with_carryover_input
            if self.carryover_inputs
            else self.parse_row
        )
        return ds.map(mapper_fn), num_blocks

    def parse_row_with_carryover_input(self, row: dict[str, Any]) -> dict[str, Any]:
        """Same as parse_row but carries over the input keys that are not in the output row.

        This is useful when we want to keep the input keys in the output.
        This method assumes if user returns the same output keys as
        input keys they have already copied input over and there is
        no need to do it again for those keys. We will just copy the input_keys that
        are not in the output row.

        Args:
            row: The row to be parsed.

        Returns:
            The parsed row.
        """
        input_row_keys = set(row.keys())
        output_row = self.parse_row(row)
        output_row_keys = set(output_row.keys())
        return {
            **{k: row[k] for k in input_row_keys if k not in output_row_keys},
            **output_row,
        }

    def tokenizer_constructor_kwargs(
        self, model: str
    ):
        """Return the keyword arguments for tokenizer constructor.

        Args:
            model: The model name.

        Returns:
            The keyword arguments for tokenizer constructor.
        """
        return {"model": model}

    def parse_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Parse a row in the dataset.

        Args:
            row: The row in the dataset.

        Returns:
            The parsed row.
        """
        return row

    def postproc_after_tokenize(self, ds: Dataset) -> Dataset:
        """Post-process the dataset after tokenization.

        Args:
            ds: The tokenized dataset.

        Returns:
            The post-processed dataset.
        """
        return ds

@dataclass
class ChatWorkloadBase(WorkloadBase):
    """The base class for a chat workload."""

    tokenizer_cls: Any = ChatTemplateTokenizer

@dataclass
class EvalWorkload(ChatWorkloadBase):
    dataset_fraction: float = 1.0
    sampling_params: Dict[str, Any] = field(
        default_factory=lambda: {"max_tokens": 4096}
    )

    def parse_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Parse each row in the dataset to make them compatible with
        OpenAI chat API messages. Specifically, the output row should only
        include a single key "messages" with type Dict[str, Union[str, List[Dict]]].      
        """
        return {"messages": row["item"][1], "index": row["item"][0]}