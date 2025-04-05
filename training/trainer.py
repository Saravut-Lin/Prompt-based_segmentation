import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import all_reduce_max, barrier, get_rank

from training.utils.logger import Logger, setup_logging

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)

CORE_LOSS_KEY = "core_loss"


def unwrap_ddp_if_wrapped(model: nn.Module) -> nn.Module:
    """
    Unwraps a DistributedDataParallel (DDP)-wrapped model to return the base
    model. If the model is not wrapped in DDP, it simply returns the input.

    Args:
        model (nn.Module): A PyTorch model, possibly wrapped in DDP.

    Returns:
        nn.Module: The unwrapped underlying model if wrapped in DDP,
                   otherwise the original model.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


@dataclass
class OptimAMPConf:
    """
    Configuration for Automatic Mixed Precision (AMP).

    Attributes:
        enabled (bool): Whether AMP is enabled.
        amp_dtype (str): AMP data type to use, e.g. "float16".
    """
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    """
    Configuration for the optimizer and AMP behavior.

    Attributes:
        optimizer (torch.optim.Optimizer): The optimizer class constructor.
        options (Dict[str, Any]): Options for the optimizer (learning rate, etc.).
        param_group_modifiers (List): Modifications to param groups (e.g. layer-wise LR).
        amp (OptimAMPConf): AMP configuration.
        gradient_clip (Any): Gradient clipping configuration.
        gradient_logger (Any): Logging configuration for gradients.
    """
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        # If amp is provided as a dictionary, convert it to an OptimAMPConf instance.
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    """
    Configuration for distributed training.

    Attributes:
        backend (str): The distributed backend (e.g., "nccl" or "gloo").
        comms_dtype (str): The data type for distributed communications (e.g., fp16).
        find_unused_parameters (bool): Whether to find unused params in DDP.
        timeout_mins (int): Timeout for torch.distributed.init_process_group.
    """
    backend: Optional[str] = None
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30


@dataclass
class CudaConf:
    """
    CUDA-related configuration for training performance and determinism.

    Attributes:
        cudnn_deterministic (bool): Enables deterministic algorithms in CUDNN.
        cudnn_benchmark (bool): Enables benchmarking in CUDNN for faster runs.
        allow_tf32 (bool): Allows TensorFloat-32 on capable GPUs (A100+).
        matmul_allow_tf32 (Optional[bool]): Overrides allow_tf32 for matmul ops.
        cudnn_allow_tf32 (Optional[bool]): Overrides allow_tf32 for cudnn ops.
    """
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    matmul_allow_tf32: Optional[bool] = None
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    """
    Configuration related to checkpointing (saving, loading, skipping parameters, etc.).

    Attributes:
        save_dir (str): Directory to store checkpoints.
        save_freq (int): Frequency (in epochs) at which to save checkpoints.
        save_list (List[int]): Specific epochs at which to save a checkpoint.
        model_weight_initializer (Any): A callable/instantiable for initializing model weights.
        save_best_meters (List[str]): Meters for which we track best values to save.
        skip_saving_parameters (List[str]): Patterns for parameters to exclude from checkpoint.
        initialize_after_preemption (bool): Whether to (re)initialize after preemption.
        resume_from (str): Path to a checkpoint to resume from.
    """
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    resume_from: Optional[str] = None

    def infer_missing(self) -> "CheckpointConf":
        """
        Infers missing attributes if they are not set explicitly.
        Particularly sets 'initialize_after_preemption' if not defined.
        """
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    """
    Configuration for logging and monitoring.

    Attributes:
        log_dir (str): Directory to store logs.
        log_freq (int): How often (in iterations) to print logging info.
        tensorboard_writer (Any): TensorBoard writer instance.
        log_level_primary (str): Logging level for the main log.
        log_level_secondary (str): Logging level for lower-priority logs.
        log_scalar_frequency (int): Frequency (in iterations) to log scalar metrics.
        log_visual_frequency (int): Frequency (in iterations) to log visual data.
        scalar_keys_to_log (Dict[str, Any]): Specific scalar keys to log.
        log_batch_stats (bool): Whether to log batch-level stats.
    """
    log_dir: str
    log_freq: int
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False


class Trainer:
    """
    A flexible trainer for PyTorch models supporting (Distributed) DataParallel training.

    Attributes:
        data_conf (Dict[str, Any]): Data configuration dictionary.
        model_conf (Dict[str, Any]): Model configuration dictionary.
        logging_conf (LoggingConf): Logging configuration.
        checkpoint_conf (CheckpointConf): Checkpoint configuration.
        max_epochs (int): Maximum number of epochs to train.
        mode (str): One of ["train", "train_only", "val"].
        val_epoch_freq (int): Frequency (in epochs) at which to run validation.
        optim_conf (OptimConf): Optimizer configuration.
        meters_conf (Dict[str, Any]): Configuration for performance meters.
        loss_conf (Dict[str, Any]): Configuration dict for the loss function(s).
        distributed_rank (int): The global rank of this process in the world.
        local_rank (int): The local rank of this process on the machine (for CUDA).
        device (torch.device): The device on which to run computations.
        epoch (int): Current epoch number (0-indexed).
        steps (Dict[str, int]): Dictionary of step counters per Phase (TRAIN/VAL).
        where (float): Fraction of the training completed (0.0 to 1.0).
        best_meter_values (Dict[str, float]): Tracks the best meter values encountered.
        time_elapsed_meter (DurationMeter): Timer meter for tracking total time elapsed.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the Trainer class with given configurations.

        Args:
            data (Dict[str, Any]): Configuration for dataset(s).
            model (Dict[str, Any]): Model configuration dict.
            logging (Dict[str, Any]): Logging configuration dict.
            checkpoint (Dict[str, Any]): Checkpointing configuration dict.
            max_epochs (int): Maximum number of epochs to train.
            mode (str): Training mode, one of ["train", "train_only", "val"].
            accelerator (str): Execution accelerator ("cuda" or "cpu").
            seed_value (int): RNG seed.
            val_epoch_freq (int): Frequency of running validation.
            distributed (Dict[str, bool]): Distributed training config dictionary.
            cuda (Dict[str, bool]): CUDA training config dictionary.
            env_variables (Dict[str, Any]): Environment variables to set.
            optim (Dict[str, Any]): Optimizer config dictionary.
            optim_overrides (List[Dict[str, Any]]): Overrides to apply to the optimizer.
            meters (Dict[str, Any]): Meters configuration dictionary.
            loss (Dict[str, Any]): Loss functions configuration dictionary.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store conf objects
        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss

        # Distributed & CUDA configs
        distributed = DistributedConf(**(distributed or {}))
        cuda = CudaConf(**(cuda or {}))
        self.where = 0.0

        # Infer distributed backend if not set
        self._infer_distributed_backend_if_none(distributed, accelerator)

        # Setup device based on accelerator type
        self._setup_device(accelerator)

        # Setup distributed backend
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Ensure logging directory exists
        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        # Set random seeds
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed must be initialized before calling the trainer."

        # Setup components (model, meters, losses, etc.) except optimizer
        self._setup_components()
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        # Track total time elapsed
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        # Handle checkpoint resume
        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            # Copy the resume-from checkpoint to the checkpoint folder (rank 0 only)
            if self.distributed_rank == 0 and not os.path.exists(dst):
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()

    def _setup_timers(self):
        """
        Initializes timers for training. Helps in tracking total elapsed
        time and checkpoint-based elapsed time.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        # Estimated epoch times are stored in this dictionary.
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _get_meters(self, phase_filters=None) -> Dict[str, Any]:
        """
        Retrieves all meters, optionally filtered by phase.

        Args:
            phase_filters (List[str]): If provided, only returns meters for these phases.

        Returns:
            Dict[str, Any]: A dictionary mapping "phase_key/meter_name" -> meter object.
        """
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf: DistributedConf, accelerator: str) -> None:
        """
        Infers distributed backend if none is specified. Defaults to 'nccl'
        for CUDA and 'gloo' for CPU.

        Args:
            distributed_conf (DistributedConf): The distributed configuration.
            accelerator (str): The accelerator type, e.g. 'cuda' or 'cpu'.
        """
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf: Dict[str, Any]) -> None:
        """
        Sets environment variables from a dictionary of key-value pairs.

        Args:
            env_variables_conf (Dict[str, Any]): Environment variables to set.
        """
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf: CudaConf, distributed_conf: DistributedConf) -> None:
        """
        Sets up Torch's distributed backend (if any) and configures CUDA
        performance options.

        Args:
            cuda_conf (CudaConf): CUDA-specific configuration.
            distributed_conf (DistributedConf): Distributed training configuration.
        """
        # Set up CUDNN flags
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        # Initialize distributed training
        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator: str) -> None:
        """
        Sets up the device on which computations will run (CPU or GPU).

        Args:
            accelerator (str): 'cuda' or 'cpu'.
        """
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf: DistributedConf, accelerator: str) -> None:
        """
        Wraps model in DistributedDataParallel (DDP) if distributed is enabled.

        Args:
            distributed_conf (DistributedConf): The distributed config object.
            accelerator (str): 'cuda' or 'cpu'.
        """
        assert isinstance(self.model, torch.nn.Module)
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
        )

        # Optionally enable communication hooks for grad compression
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks
            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")

            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self) -> None:
        """
        Moves the model (and potentially other components) to the chosen device.
        """
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None) -> None:
        """
        Saves a checkpoint at the specified epoch. Only rank 0 saves checkpoints.

        Args:
            epoch (int): The epoch number for which we save the checkpoint.
            checkpoint_names (List[str]): List of checkpoint file name prefixes.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        # Exclude parameters that match skip_saving_parameters patterns
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP: Only rank 0 writes checkpoints
        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
        """
        Saves a checkpoint atomically by first writing to a temp file,
        then moving it over the original file. Helps avoid corruption
        if the job is killed.

        Args:
            checkpoint (Dict[str, Any]): The checkpoint data dictionary.
            checkpoint_path (str): Destination path for the checkpoint file.
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        # Once the save completes, move the file to the final path
        if g_pathmgr.exists(checkpoint_path):
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success

    def load_checkpoint(self) -> None:
        """
        Loads a checkpoint from the specified save directory. If no checkpoint is found,
        model initialization logic is applied instead.
        """
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self) -> None:
        """
        Initializes model state before any checkpointing. Ensures that any skipped
        parameters are indeed frozen and that initialization code is applied if needed.
        """
        # Check that unsaved parameters are frozen
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )

        # Check that unsaved parameters are valid for re-initialization
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self) -> None:
        """
        Calls the user-specified initializer for model weights, if provided.
        """
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str) -> None:
        """
        Loads from an existing checkpoint and resumes optimizer, epoch, loss,
        and meter states accordingly.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        logging.info(f"Resuming training from {ckpt_path}")
        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_meter_values = checkpoint.get("best_meter_values", {})

        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def is_intermediate_val_epoch(self, epoch: int) -> bool:
        """
        Checks if we should run validation at the given epoch.

        Args:
            epoch (int): Current epoch.

        Returns:
            bool: True if we need to run validation on this epoch, False otherwise.
        """
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):
        """
        Runs the forward pass for a single batch, computes and logs losses.

        Args:
            batch (BatchedVideoDatapoint): Input data batch.
            model (nn.Module): The model used for forward pass.
            phase (str): The phase ('train' or 'val').

        Returns:
            tuple: (dict of losses, batch size, dict of extra losses)
        """
        outputs = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)

        key = batch.dict_key
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"
        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss might be a dict containing multiple sub-losses
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        # Periodically log the loss
        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(loss_log_str, loss, self.steps[phase])

        self.steps[phase] += 1

        # Update meters if any
        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )

        return {loss_str: loss}, batch_size, step_losses

    def run(self) -> None:
        """
        Main entry point to run training/validation loops based on the specified mode.
        """
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            # If resuming, possibly run a val for the last epoch
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self) -> None:
        """
        Instantiates dataset loaders for training and validation from the data config.
        """
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_train(self) -> None:
        """
        Executes the main training loop for the configured number of epochs.
        Saves checkpoints and occasionally runs validation, if configured.
        """
        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            # Logger call only on rank 0
            self.logger.log_dict(outs, self.epoch)

            # Log stats to disk
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Save checkpoint before validation
            self.save_checkpoint(self.epoch + 1)

            del dataloader
            gc.collect()

            # Possibly run val
            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()

            # Update best meter values
            if self.distributed_rank == 0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1

        # The loop increments epoch but val logic might run outside
        self.epoch -= 1

    def run_val(self) -> None:
        """
        Executes the validation loop.
        """
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        # Log validation metrics
        self.logger.log_dict(outs, self.epoch)

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, phase: str) -> Dict[str, float]:
        """
        Validation for one epoch. Measures losses and logs stats.

        Args:
            val_loader: DataLoader for validation.
            phase (str): Typically "val".

        Returns:
            Dict[str, float]: Validation stats dictionary.
        """
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)
        curr_phases = [phase]
        curr_models = [self.model]

        # Prepare meter keys for losses
        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        # Create dictionary of AverageMeters for each loss
        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        # Switch all models in this phase to eval
        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(val_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            batch = batch.to(self.device, non_blocking=True)

            # Forward pass
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for p, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            p,
                        )
                        # There's only one loss key in loss_dict
                        assert len(loss_dict) == 1
                        loss_key, loss_val = list(loss_dict.items())[0]
                        loss_mts[loss_key].update(loss_val.item(), batch_size)

                        # Handle extra losses
                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item(), batch_size)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                # Log progress meters
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )

            if data_iter % 10 == 0:
                dist.barrier()

        # Time estimates
        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)

        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        # Prepare final output dictionary
        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)
        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        # Trainer state
        for p in curr_phases:
            out_dict.update(self._get_trainer_state(p))

        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")
        return out_dict

    def _get_trainer_state(self, phase: str) -> Dict[str, Any]:
        """
        Returns a dictionary of some trainer state (epoch, steps, etc.).

        Args:
            phase (str): The phase (TRAIN/VAL).

        Returns:
            Dict[str, Any]: The trainer state dictionary.
        """
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Trains over a single epoch.

        Args:
            train_loader: DataLoader for the training dataset.

        Returns:
            Dict[str, float]: A dictionary of training stats (loss, etc.).
        """
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        # Prepare the list of loss meters
        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # Switch model to train mode
        self.model.train()
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            batch = batch.to(self.device, non_blocking=True)

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON

                # Update schedulers only if we have not finished training
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since training is at {self.where} of [0,1]."
                    )

                # Log scheduler parameters occasionally
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                # Gradient clipping
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                # Gradient logging
                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                # Log meters occasionally
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            except FloatingPointError as e:
                # If the loss is NaN/Inf, raise an error
                raise e

        # Estimate epoch time
        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])
        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    def _log_sync_data_times(self, phase: str, data_times: List[float]) -> None:
        """
        Synchronizes data loading times across ranks and logs the max.

        Args:
            phase (str): "train" or "val".
            data_times (List[float]): Per-iteration data loading times.
        """
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

    def _run_step(
        self,
        batch: BatchedVideoDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ) -> None:
        """
        Runs a single training step: forward, backward, and meter updates.

        Args:
            batch (BatchedVideoDatapoint): Input batch of data.
            phase (str): "train" or "val".
            loss_mts (Dict[str, AverageMeter]): Meters for the main loss values.
            extra_loss_mts (Dict[str, AverageMeter]): Meters for auxiliary losses.
            raise_on_error (bool): Whether to raise an exception on floating point errors.

        Raises:
            FloatingPointError: If the loss is non-finite (NaN or Inf).
        """
        # Zero out gradients
        self.optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            loss_dict, batch_size, extra_losses = self._step(
                batch,
                self.model,
                phase,
            )

        assert len(loss_dict) == 1
        loss_key, loss = list(loss_dict.items())[0]

        if not math.isfinite(loss.item()):
            error_msg = f"Loss is {loss.item()}, attempting to stop training"
            logging.error(error_msg)
            if raise_on_error:
                raise FloatingPointError(error_msg)
            else:
                return

        # Backprop
        self.scaler.scale(loss).backward()
        # Update main loss meter
        loss_mts[loss_key].update(loss.item(), batch_size)
        # Update extra losses meter
        for extra_loss_key, extra_loss in extra_losses.items():
            if extra_loss_key not in extra_loss_mts:
                extra_loss_mts[extra_loss_key] = AverageMeter(extra_loss_key, self.device, ":.2e")
            extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

    def _log_meters_and_save_best_ckpts(self, phases: List[str]) -> Dict[str, float]:
        """
        Synchronizes meters, logs them, and saves checkpoints for any improved metrics.

        Args:
            phases (List[str]): List of phases to process (e.g., [Phase.TRAIN]).

        Returns:
            Dict[str, float]: A dictionary of meter values after logging.
        """
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []

        # Sync meters and find improved values
        for key, meter in self._get_meters(phases).items():
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                # Update best meter values
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

                    # If we track best_meters, save a checkpoint
                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        # Save best checkpoint if we found improved metrics
        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase: str) -> None:
        """
        Logs time metrics (elapsed time and estimated time remaining).

        Args:
            phase (str): "train" or "val".
        """
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        # We guarantee a final val run if we haven't ended on the same interval
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # If we are currently in VAL, subtract one from the estimate
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _reset_meters(self, phases: List[str]) -> None:
        """
        Resets meters after an epoch or phase completes.

        Args:
            phases (List[str]): List of phases to reset the meters for.
        """
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys: Optional[List[str]], phase: str) -> None:
        """
        Validates that the set of dataset keys for validation matches the
        meter keys and loss keys if they are specified.

        Args:
            val_keys (List[str]): Dataset keys used for validation.
            phase (str): The phase, e.g. 'val'.

        Raises:
            AssertionError: If there's a mismatch in the dataset keys vs meters/losses keys.
        """
        if val_keys is not None:
            # Check duplicates
            assert len(val_keys) == len(
                set(val_keys)
            ), f"Duplicate keys in val datasets, keys: {val_keys}"

            # Check meter keys
            if self.meters_conf is not None and phase in self.meters_conf:
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            # Check loss keys
            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys())
                assert all(k in loss_keys for k in val_keys), (
                    f"Keys in val datasets do not match the keys in losses."
                    f"\nMissing in losses: {set(val_keys) - loss_keys}"
                    f"\nMissing in val datasets: {loss_keys - set(val_keys)}"
                )

    def _setup_components(self) -> None:
        """
        Instantiates main trainer components: model, loss, meters, logger, etc.
        """
        # Collect dataset keys for validation, if any
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Check correctness of val dataset keys
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        # Setup logger
        self.logger = Logger(self.logging_conf)

        # Instantiate model
        self.model = instantiate(self.model_conf, _convert_="all")
        print_model_summary(self.model)

        # Instantiate loss
        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        # Instantiate meters
        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        # AMP GradScaler
        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        # Gradient clipping/logging
        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _construct_optimizers(self) -> None:
        """
        Constructs the optimizer(s) based on the provided configuration.
        """
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
        )

    def _log_loss_detailed_and_return_core_loss(
        self, loss: Dict[str, torch.Tensor], loss_str: str, step: int
    ) -> torch.Tensor:
        """
        Logs sub-losses included in 'loss' dict, then returns the core loss.

        Args:
            loss (Dict[str, torch.Tensor]): Dict containing sub-losses, including CORE_LOSS_KEY.
            loss_str (str): Base log string for losses.
            step (int): Current training step.

        Returns:
            torch.Tensor: The core loss tensor.
        """
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss


def print_model_summary(model: nn.Module, log_dir: str = "") -> None:
    """
    Prints a summary of a model, including trainable and non-trainable parameters.

    Args:
        model (nn.Module): The PyTorch model to summarize.
        log_dir (str): Optional directory to write the model summary to a file.
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )
    logging.info("==" * 10)

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer with suffixes such as K (thousand), M (million),
    B (billion), T (trillion).

    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)
        '1.2 K'
        >>> get_human_readable_count(2e6)
        '2.0 M'
        >>> get_human_readable_count(3e9)
        '3.0 B'
        >>> get_human_readable_count(4e14)
        '400 T'
        >>> get_human_readable_count(5e15)
        '5,000 T'

    Args:
        number (int): The integer to abbreviate.

    Returns:
        str: The abbreviated number with appropriate suffix.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # do not abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"