import cv2
import numpy as np
import torch
import einops
import random
import os
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# 初始化模型和DDIM采样器
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_any4.5_openpose.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# 定义Canny边缘检测器
apply_openpose = OpenposeDetector()

# 读取输入图片
input_image_path = 'test_imgs/human.png'  # 替换为你的图片路径
input_image = cv2.imread(input_image_path)

# 设置参数
prompt ='man raise hand'#'man stand'#'man at kitchen'#'a thinking man '# "a beautiful lady gently at a table"  # 替换为你的提示
a_prompt = 'best quality, extremely detailed'  # 替换为附加提示
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'  # 替换为负面提示
num_samples = 2  # 生成图像的数量
image_resolution = 512  # 图像分辨率
ddim_steps = 20  # DDIM步骤
guess_mode = False  # 猜测模式
strength = 1.0  # 控制强度
scale = 9.0  # 指导比例
seed = -1  # 随机种子
eta = 0.0  # DDIM的eta参数
detect_resolution = 512

# 处理函数
def process(input_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(cv2.imread(input_image_path))
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

# 调用处理函数并保存结果
#generated_images = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)

# 保存生成的图像
#for i, image in enumerate(generated_images):
    cv2.imwrite(f'generated_image_{i}.png', image)
    # 定义保存图像的文件夹路径
output_folder = 'a-openpose-human-manraisehand-3'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 调用处理函数
generated_images = process(input_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)

# 保存生成的图像到指定文件夹
for i, image in enumerate(generated_images):
    # 构建文件保存路径
    file_path = os.path.join(output_folder, f'generated_image_{i}.png')
    # 保存图像
    cv2.imwrite(file_path, image)