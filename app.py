from flask import Flask, request
from flask_cors import CORS

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.utils import get_network
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# Load diffusion model
model_args = model_and_diffusion_defaults()
# MODEL_FLAGS=
# "--attention_resolutions 32,16,8
# --class_cond False
# --diffusion_steps 1000
# --dropout 0.1
# --image_size 256
# --learn_sigma True --noise_schedule linear
# --num_channels 256 --num_head_channels 64
# --num_res_blocks 2 --resblock_updown True
# --use_fp16 True --use_scale_shift_norm True"

model_args = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    num_heads=4,
    num_heads_upsample=-1,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    channel_mult="",
    dropout=0.1,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_fp16=False,
    use_new_attention_order=False,
    learn_sigma=True,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="ddim20",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
)
diff_model, diffusion = create_model_and_diffusion(**model_args)
diff_model.load_state_dict(
    torch.load("models/256x256_diffusion_uncond.pt", map_location="cpu")
)
diff_model.eval()

# Load classification model
model = get_network("resnet50")
state_dict = torch.load("models/imagenet_adm.pth", map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict)
model.eval()


@app.route("/", methods=["POST"])
def index():
    file = request.files["file"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = Image.fromarray(img.astype("uint8"), "RGB")

    return f"Prob: {detect(img)}"

def detect(img):
    print("Start detection")
    dire = compute_dir(img)
    print(dire.size())

    trans = transforms.Compose(
        (
            transforms.Resize(256),
            transforms.CenterCrop(224),
        )
    )
    dire = trans(dire)
    dire = TF.normalize(
        dire, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    in_tens = dire.unsqueeze(0)

    with torch.no_grad():
        prob = model(in_tens).sigmoid().item()
    return prob

def compute_dir(img):
    image_size = 256
    real_step = 0
    use_ddim = False
    clip_denoised = True

    reverse_fn = diffusion.ddim_reverse_sample_loop

    # Convert image pixel value from [0, 255] to [-1, 1]
    arr = np.array(img)
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = transforms.ToTensor()(arr)
    arr = reshape_image(arr, image_size)

    model_kwargs = {}
    latent = reverse_fn(
        diff_model,
        (1, 3, image_size, image_size),
        noise=arr,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        real_step=real_step,
    )
    print("Finish latent")
    # sample_fn = (
    #     diffusion.p_sample_loop
    #     if not use_ddim else diffusion.ddim_sample_loop
    # )
    sample_fn = diffusion.ddim_sample_loop
    recons = sample_fn(
        diff_model,
        (1, 3, image_size, image_size),
        noise=latent,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        real_step=real_step,
    )
    print("Finish recons")

    dire = torch.abs(arr - recons)
    # recons = ((recons + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # recons = recons.permute(0, 2, 3, 1)
    # recons = recons.contiguous()

    # imgs = ((arr + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # imgs = imgs.permute(0, 2, 3, 1)
    # imgs = imgs.contiguous()

    dire = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)
    # dire = dire.permute(0, 2, 3, 1)
    # dire = dire.contiguous()
    torch.save(dire, "dire_tensor.pt");

    return dire[0]


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(
            imgs, size=(image_size, image_size), mode="bicubic"
        )
    return imgs


if __name__ == "__main__":
    app.run(port=5000)
