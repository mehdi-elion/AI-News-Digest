import datetime
from pathlib import Path

import torch
from loguru import logger
from rich import print


def create_run_folder(path_to_artifact: str = "data/08_reporting/artifacts/") -> Path:
    """Create folders to store run-related artifacts.

    Parameters
    ----------
    path_to_artifact : str, optional
        Path to the folder where run-specific subfolders will be created,
        by default "data/08_reporting/artifacts/"

    Returns
    -------
    run_path : Path
        Path to the folder where run artifacts will be saved

    """
    # create artifacts folder if doesn't exist
    artifacts_path = Path(path_to_artifact)
    artifacts_path.mkdir(exist_ok=True)

    # create new subfolder folder for current run
    dt_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = Path(artifacts_path / dt_now)
    run_path.mkdir(exist_ok=True)
    run_path = Path(run_path / "run_artifacts")
    run_path.mkdir(exist_ok=True)

    return run_path


def check_gpu_availability() -> str:
    """Check gpu availability and return subsequent torch device.

    Returns
    -------
    device : str
        Torch device identified as available

    """
    # check gpu availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"Is cuda available ? --> {cuda_available}")
    if cuda_available:
        print(torch.cuda.get_device_properties(0))

    # choose device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Chose following device: '{device}'")
    return device
