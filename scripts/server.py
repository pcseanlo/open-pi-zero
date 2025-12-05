import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import json_numpy

"""
Launcher for all experiments.

"""

import logging
import math
import os
import random
import sys

import hydra
import numpy as np
import pretty_errors
import torch
from omegaconf import OmegaConf, open_dict

import traceback
from typing import Dict, Any

# dummy
print(pretty_errors.__version__)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # figure out the current gpu
    multi_gpu = torch.cuda.device_count() > 1 or cfg.get("n_nodes", 1) > 1
    if multi_gpu:
        from torch.distributed import destroy_process_group, init_process_group

        def ddp_setup():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            init_process_group(backend="nccl")

        ddp_setup()
        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = 0
    with open_dict(cfg):
        cfg.gpu_id = gpu_id
        cfg.multi_gpu = multi_gpu

    # seeding
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)

    app = FastAPI()

    @app.post("/reset")
    def reset(payload: Dict[str, Any]):
        try:
            # Reset internal states if any (currently stateless)
            return JSONResponse({"status": "reset successful"})
        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": "Reset failed"}, status_code=400)
    
    @app.post("/set_task")
    def set_task(payload: Dict[str, Any]):
        try:
            # Set task description or other task-specific info if needed
            return JSONResponse({"status": "task set successful"})
        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": "Set task failed"}, status_code=400)
    
    @app.post("/act")
    def act(payload: Dict[str, Any]):
        try:
            actions = agent.get_actions(payload)
            return JSONResponse(content={"action": actions.tolist()})
        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": "Action computation failed"}, status_code=400)
    
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
    

    if multi_gpu:
        destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config/server"),
    config_name="bridge.yaml",
)  # defaults
def main(cfg: OmegaConf):
    _main(cfg)


if __name__ == "__main__":
    main()
