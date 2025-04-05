import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import submitit
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers


os.environ["HYDRA_FULL_ERROR"] = "1"


def add_pythonpath_to_sys_path() -> None:
    """
    Adds the paths specified in the PYTHONPATH environment variable
    to sys.path for module resolution.
    """
    python_path = os.environ.get("PYTHONPATH", "")
    if not python_path:
        return
    sys.path = python_path.split(":") + sys.path


def format_exception_trace(e: Exception, limit: int = 20) -> str:
    """
    Formats the traceback of the given exception into a string.

    Args:
        e (Exception): The exception to format.
        limit (int): Limit the number of stack trace entries.

    Returns:
        str: The formatted traceback.
    """
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


def single_proc_run(local_rank: int, main_port: int, cfg, world_size: int) -> None:
    """
    Runs training on a single GPU process.

    Args:
        local_rank (int): The local rank index.
        main_port (int): The main port for distributed training.
        cfg: The configuration object.
        world_size (int): The total number of processes in the world.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int) -> None:
    """
    Handles single-node training (potentially multi-GPU).

    Args:
        cfg: The configuration object.
        main_port (int): The main port for distributed training.
    """
    assert cfg.launcher.num_nodes == 1, "single_node_runner can only run on 1 node."
    num_proc = cfg.launcher.gpus_per_node

    # CUDA runtime does not support `fork`
    torch.multiprocessing.set_start_method("spawn")

    if num_proc == 1:
        # Directly call single_proc_run so we can easily set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        # Using "spawn" to start the processes
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


class SubmititRunner(submitit.helpers.Checkpointable):
    """
    A callable passed to Submitit to launch distributed training jobs.
    It is used for multi-node or SLURM-based launches.

    Attributes:
        port (int): Main port for distributed training.
        cfg: Configuration object.
        has_setup (bool): Whether the job info setup has been done or not.
    """

    def __init__(self, port: int, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False
        self.job_info = {}

    def __call__(self):
        """
        Main entry point for Submitit. Sets up the job environment and runs
        the trainer. Any exceptions are logged before being re-raised.
        """
        job_env = submitit.JobEnvironment()
        self.setup_job_info(job_env.job_id, job_env.global_rank)

        try:
            self.run_trainer()
        except Exception as e:
            message = format_exception_trace(e)
            logging.error(message)
            raise e

    def run_trainer(self) -> None:
        """
        Runs the trainer with the configured environment variables
        for distributed training.
        """
        job_env = submitit.JobEnvironment()

        # Re-add PYTHONPATH for Hydra job environment
        add_pythonpath_to_sys_path()

        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        register_omegaconf_resolvers()

        # Create a resolved configuration for the trainer
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        trainer.run()

    def setup_job_info(self, job_id: str, rank: int) -> None:
        """
        Set up job metadata including cluster, experiment directory, etc.

        Args:
            job_id (str): The SLURM or cluster job ID.
            rank (int): The global rank.
        """
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None),
            "experiment_log_dir": self.cfg.launcher.experiment_log_dir,
        }
        self.has_setup = True


def resolve_and_save_configs(cfg, config_path: str, config_resolved_path: str) -> None:
    """
    Saves both the user-facing and resolved Hydra config YAML files
    to the specified paths.

    Args:
        cfg: The original Hydra configuration object.
        config_path (str): Where to save the config (user-facing).
        config_resolved_path (str): Where to save the fully resolved config.
    """
    # Write original config
    with g_pathmgr.open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Create a resolved version of the config
    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(config_resolved_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))


def submit_to_cluster(cfg, submitit_conf, main_port) -> None:
    """
    Submits the job to a SLURM cluster using Submitit.

    Args:
        cfg: The configuration object.
        submitit_conf: The Submitit-related configuration (partition, account, etc.).
        main_port (int): The main port for distributed training.
    """
    submitit_dir = os.path.join(cfg.launcher.experiment_log_dir, "submitit_logs")
    executor = submitit.AutoExecutor(folder=submitit_dir)

    # Build Submitit job parameters
    job_params = {
        "timeout_min": 60 * submitit_conf.timeout_hour,
        "name": submitit_conf.get("name", None) or cfg.config,
        "slurm_partition": submitit_conf.partition,
        "gpus_per_node": cfg.launcher.gpus_per_node,
        "tasks_per_node": cfg.launcher.gpus_per_node,
        "cpus_per_task": submitit_conf.cpus_per_task,
        "nodes": cfg.launcher.num_nodes,
        "slurm_additional_parameters": {
            "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
        },
    }

    # Node inclusion
    if "include_nodes" in submitit_conf:
        assert len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes, (
            "Not enough nodes specified in include_nodes "
            "to match cfg.launcher.num_nodes"
        )
        job_params["slurm_additional_parameters"]["nodelist"] = " ".join(
            submitit_conf["include_nodes"]
        )

    # Account, QoS, memory constraints, etc.
    if submitit_conf.account is not None:
        job_params["slurm_additional_parameters"]["account"] = submitit_conf.account
    if submitit_conf.qos is not None:
        job_params["slurm_additional_parameters"]["qos"] = submitit_conf.qos
    if submitit_conf.get("mem_gb", None) is not None:
        job_params["mem_gb"] = submitit_conf.mem_gb
    elif submitit_conf.get("mem", None) is not None:
        job_params["slurm_mem"] = submitit_conf.mem

    if submitit_conf.get("constraints", None) is not None:
        job_params["slurm_constraint"] = submitit_conf.constraints

    if submitit_conf.get("comment", None) is not None:
        job_params["slurm_comment"] = submitit_conf.comment

    # srun arguments (e.g., cpu-bind)
    if submitit_conf.get("srun_args", None) is not None:
        job_params["slurm_srun_args"] = []
        if submitit_conf.srun_args.get("cpu_bind", None) is not None:
            job_params["slurm_srun_args"].extend(
                ["--cpu-bind", submitit_conf.srun_args.cpu_bind]
            )

    # Print job configuration for debugging
    print("###################### SLURM Config ######################")
    print(job_params)
    print("##########################################################")

    # Update executor settings
    executor.update_parameters(**job_params)

    # Prepare the runner and submit the job
    runner = SubmititRunner(port=main_port, cfg=cfg)
    job = executor.submit(runner)
    print(f"Submitit Job ID: {job.job_id}")

    # Setup job info for local logging (rank=0)
    runner.setup_job_info(job_id=job.job_id, rank=0)


def main(args) -> None:
    """
    Main entry point for the script. Parses configuration, handles local or
    cluster-based job submission, and triggers the training run.

    Args:
        args: Command-line arguments, typically from argparse.
    """
    # Load the Hydra config
    cfg = compose(config_name=args.config)

    # Set default experiment_log_dir if not provided
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )

    # Print user-facing config
    print("###################### Train App Config ###################")
    print(OmegaConf.to_yaml(cfg))
    print("###########################################################")

    # Ensure PYTHONPATH is set for Hydra job environment
    add_pythonpath_to_sys_path()

    # Create experiment directory and save configs
    makedir(cfg.launcher.experiment_log_dir)
    config_path = os.path.join(cfg.launcher.experiment_log_dir, "config.yaml")
    resolved_path = os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml")
    resolve_and_save_configs(cfg, config_path, resolved_path)

    # Resolve some key configuration parameters based on CLI args
    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config."
    submitit_dir = cfg.launcher.experiment_log_dir

    # Override config from CLI if provided
    if args.num_gpus is not None:
        cfg.launcher.gpus_per_node = args.num_gpus
    if args.num_nodes is not None:
        cfg.launcher.num_nodes = args.num_nodes

    if args.use_cluster is not None:
        submitit_conf.use_cluster = args.use_cluster

    if submitit_conf.use_cluster:
        # Set additional parameters from CLI
        if args.partition is not None:
            submitit_conf.partition = args.partition
        if args.account is not None:
            submitit_conf.account = args.account
        if args.qos is not None:
            submitit_conf.qos = args.qos

        # Pick a random port in the specified range
        main_port = random.randint(submitit_conf.port_range[0], submitit_conf.port_range[1])

        # Submit to SLURM cluster
        submit_to_cluster(cfg, submitit_conf, main_port)
    else:
        # Local (single-node) run
        cfg.launcher.num_nodes = 1
        main_port = random.randint(submitit_conf.port_range[0], submitit_conf.port_range[1])
        single_node_runner(cfg, main_port)


if __name__ == "__main__":
    # initialize_config_module("sam2", version_base="1.2")
    config_path = "/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/sam2/configs" 
    # i.e. the directory that directly contains "oxford_pets"

    initialize_config_dir(config_dir=config_path, version_base="1.2")
    cfg = compose(config_name="oxford_pets/sam2_pets_finetune_no_checkpoint.yaml")
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="oxford_pets/sam2_pets_finetune_no_checkpoint.yaml")
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="0 to run locally, 1 to run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QoS")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs per node (overrides config.launcher.gpus_per_node)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Number of nodes (overrides config.launcher.num_nodes)",
    )

    args = parser.parse_args()
    from hydra.core.global_hydra import GlobalHydra
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=config_path, version_base="1.2")
    cfg = compose(config_name=args.config)
    # Convert --use-cluster integer to boolean if provided
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

    # Register OmegaConf resolvers and run
    register_omegaconf_resolvers()
    main(args)


"""
CUDA_VISIBLE_DEVICES="" python training/train.py -c configs/oxford_pets/sam2_pets_finetune.yaml --use-cluster 0 --num-gpus 1
"""