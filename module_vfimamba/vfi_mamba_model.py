import gc

import torch
import tqdm
import comfy.model_management as model_management
from comfy.utils import ProgressBar

from . import config as cfg
from .padder import InputPadder
from .Trainer_finetune import Model


def run_vfi_mamba(
    src_video: torch.Tensor,
    model_name: str,
    model_path: str,
    vfi_scale=2,
    scale: float = 0.0,
):
    print(f"{model_name=}")
    print(f"{model_path=}")
    TTA = False
    cfg.MODEL_CONFIG["LOGNAME"] = model_name
    if model_name == "VFIMamba":
        TTA = True
        cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(
            F=32, depth=[2, 2, 2, 3, 3]
        )
    else:
        cfg.MODEL_CONFIG = cfg.MODEL_CONFIG_DEFAULT.copy()
    print()
    model = Model(-1, cfg.MODEL_CONFIG)
    model.load_model(model_path=model_path)
    model.eval()
    model.device()

    device = torch.device("cuda")

    src_video = src_video.permute(0, 3, 1, 2).contiguous().cpu()

    out_frames: list[torch.Tensor] = []

    num_frames = src_video.shape[0]
    pbar = ProgressBar(num_frames)
    for i in tqdm.tqdm(range(src_video.shape[0] - 1), "Generating frames"):
        frame_a_cpu = src_video[i].unsqueeze(dim=0)
        frame_b_cpu = src_video[i + 1].unsqueeze(dim=0)
        frame_a = frame_a_cpu.to(device)
        frame_b = frame_b_cpu.to(device)

        padder = InputPadder(frame_a.shape, divisor=32)
        I0_, I2_ = padder.pad(frame_a, frame_b)

        out_frames.append(frame_a_cpu)

        for j in range(1, vfi_scale):
            timestep = j / vfi_scale
            out = model.inference(I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=scale, timestep=timestep)
            mid_frame = padder.unpad(out).cpu()
            out_frames.append(mid_frame)

        pbar.update_absolute(i, num_frames)

    out_frames.append(src_video[-1].unsqueeze(0))

    out_tensor = torch.cat(out_frames, dim=0).permute(0, 2, 3, 1)
    print(f"{out_tensor.shape=}")
    frame_a = frame_b = I0_ = I2_ = out = None
    padder = None
    del out_frames
    del src_video
    del model
    gc.collect()
    model_management.soft_empty_cache()
    return out_tensor
