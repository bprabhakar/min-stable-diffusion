import argparse
import random
import os
import torch
import cv2
import numpy as np
from io import BytesIO
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def image_array_to_bytes(np_image: np.ndarray) -> bytes:
    """Converts RGB(A) np array to bytes.
    Always converts to RGBA.
    """
    # change channels to make it compatible with opencv
    if len(np_image.shape) == 3 and np_image.shape[2] == 4:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGRA)
    else:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGRA)
    # encode to bytes
    data = cv2.imencode(".jpg", np_image)[1]
    image_bytes = data.tobytes()
    return image_bytes


def bytes_to_image_array(img_bytes, gray: bool = False) -> np.array:
    """Converts bytes to np array.
    Always converts to RGB, unless gray is True.
    """
    np_array = np.frombuffer(img_bytes, np.uint8)
    if gray:
        np_image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    else:
        np_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if len(np_image.shape) == 3 and np_image.shape[-1] == 4:  # RGBA
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGRA2RGB)
        elif len(np_image.shape) == 3 and np_image.shape[-1] == 3:  # RGB
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        elif len(np_image.shape) == 2:  # gray
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
    return


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_args():
    opt = argparse.Namespace()
    opt.ddim_steps = 50
    opt.ddim_eta = 0.0
    opt.n_iter = 1
    opt.H = 512
    opt.W = 512
    opt.C = 4
    opt.f = 8
    opt.n_samples = 1
    opt.scale = 7.5
    opt.config = "configs/stable-diffusion/v1-inference.yaml"
    opt.ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
    opt.seed = 42
    opt.precision = "autocast"
    return opt


@app.on_event("startup")
async def startup_event():
    global opt, config, model, sampler, device
    opt = load_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)
    sampler = PLMSSampler(model)
    # sampler = DDIMSampler(model)


def _txt2img(prompt):
    start_code = None
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                prompts = [prompt]
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(
                        1 * [""]
                    )
                c = model.get_learned_conditioning(prompts)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(
                    S=opt.ddim_steps,
                    conditioning=c,
                    batch_size=opt.n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    x_T=start_code,
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )
                x_samples_ddim = (
                    x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                )
                x_checked_image_torch = torch.from_numpy(
                    x_samples_ddim
                ).permute(0, 3, 1, 2)
                x_sample = 255.0 * rearrange(
                    x_checked_image_torch[0].cpu().numpy(), "c h w -> h w c"
                )
                img = Image.fromarray(x_sample.astype(np.uint8))
    image_array = np.array(img)
    return image_array


@app.post("/text2img")
async def txt2img(prompt: str = Form()):
    image_array = _txt2img(prompt)
    # convert to file object
    image_file = BytesIO(image_array_to_bytes(image_array))
    return StreamingResponse(image_file, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app)
