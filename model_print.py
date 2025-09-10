import argparse

import mlflow
import numpy as np
import torch
from torch.optim import *
from torchinfo import summary
from brevitas import config as cf

cf.IGNORE_MISSING_KEYS = True

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import FWL, RSAT, AEE, NEE, AE
from models.model import (
    FireNet,
    FireNet_short,
    FireFlowNet,
)
from models.model import (
    LIFFireNet,
    LIFFireNet_short,
    LIFFireFlowNet,
)
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization, vis_activity


def test(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    
    # model initialization and settings 
    model = eval(config["model"]["name"])(config["model"]).to(device)
    model.eval()

    # data loader
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # print the summary of the model we are going to evaluate
    with torch.no_grad():
        printed_summary = False 
        for inputs in dataloader:

            if not printed_summary:
                event_voxel = inputs["event_voxel"].to(device)
                event_cnt = inputs["event_cnt"].to(device)

                summary(
                    model,
                    input_data=[event_voxel, event_cnt],
                    device=device,
                    col_names=["input_size", "output_size", "num_params"],
                )
                printed_summary = True
                    
            break
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_MVSEC.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))