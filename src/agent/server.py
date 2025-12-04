"""
Main Server agent. Only for Simpler for now.

"""

import logging
import os

import hydra
import imageio
import numpy as np
import simpler_env
import torch

from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)

import json_numpy



class ServerAgent:
    def __init__(self, cfg):
        # model
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
        self.model = PiZeroInference(cfg, use_ddp=False)
        self.load_checkpoint(cfg.checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if cfg.get(
            "use_torch_compile", True
        ):  # model being compiled in the first batch which takes some time
            self.model = torch.compile(
                self.model,
                mode="default",  # "reduce-overhead", max-autotune(-no-cudagraphs)
                # backend="inductor", # default: inductor; cudagraphs
            )
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        self.model.eval()
        log.info(f"Using cuda device: {self.device} dtype: {self.dtype}")
        log_allocated_gpu_memory(log, "loading model")
        self.act_steps = (
            cfg.act_steps
        )  # e.g., run first two steps of predicted four steps

        # env specifics
        self.env_adapter = hydra.utils.instantiate(cfg.env.adapter)

    def get_actions(self, payload):
        """
        Roughly following simpler_env/simple_inference_visual_matching_prepackaged_envs.py

        Assume no obs history for now
        """
        env_adapter = self.env_adapter
       

        env_adapter.reset()
        instruction = payload["instruction"]
        
        log.info(
            f"Request received with Instruction: {instruction}"
        )

        payload["image"] = np.array(json_numpy.loads(payload["image"]))
        payload["proprio"] = np.array(json_numpy.loads(payload["proprio"]))
        # Bridge: {'scene_name': 'bridge_table_1_v1', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': '', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png', 'rgb_overlay_cameras': ['3rd_view_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'episode_model_ids': ['bridge_carrot_generated_modified', 'bridge_plate_objaverse_larger'], 'episode_model_scales': [1.0, 1.0], 'episode_source_obj_name': 'bridge_carrot_generated_modified', 'episode_target_obj_name': 'bridge_plate_objaverse_larger', 'episode_source_obj_init_pose_wrt_robot_base': Pose([0.381995, 0.104536, 0.0175282], [-0.706719, 0.0305475, -0.0305745, -0.706173]), 'episode_target_obj_init_pose_wrt_robot_base': Pose([0.232, -0.047, -0.000468373], [2.00041e-10, -5.10387e-07, -1.6915e-06, -1]), 'episode_id': 5}
        # Fractal: {'scene_name': 'google_pick_coke_can_1_v4', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': 'recolor_tabletop_visual_matching_1', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png', 'rgb_overlay_cameras': ['overhead_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'model_id': 'opened_coke_can', 'model_scale': 1.0, 'distractor_model_ids': None, 'distractor_model_scales': None, 'obj_init_pose_wrt_robot_base': Pose([0.587925, -0.0238302, 0.840576], [0.707052, -0.0081018, -0.01162, -0.70702]), 'orientation': 'laid_vertically'} Instruction: pick coke can Max episode length: 80
        # infer action chunk
        inputs = self.env_adapter.preprocess(None, None, payload["instruction"], image=payload["image"], proprio=payload["proprio"])
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.model.build_causal_mask_and_position_ids(
                inputs["attention_mask"], dtype=self.dtype
            )
        )
        image_text_proprio_mask, action_mask = (
            self.model.split_full_mask_into_submasks(causal_mask)
        )
        inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "image_text_proprio_mask": image_text_proprio_mask,
            "action_mask": action_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": inputs["proprios"].to(self.dtype),
        }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # using bf16 shows ~0.001 difference in action inferred when using vs. not using kv cache (infer_action_naive, needs to pass in full causal_mask instead), if starting from the same initial noise. no difference when using float32
        # https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
        with torch.inference_mode():
            actions = self.model(**inputs)
            # actions_naive = self.model.infer_action_naive(**inputs_naive)
            # print(torch.allclose(actions, actions_naive))
        env_actions = self.env_adapter.postprocess(actions[0].float().cpu().numpy())

        return env_actions[: self.act_steps]

    @log_execution_time(log)
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")
