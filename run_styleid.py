# run_styleid.py
"""
python run_styleid.py --cnt ./data_vis/cnt --sty ./data_vis/sty

"""
import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
from pathlib import Path

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle

feat_maps = []

def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T,
                'timestep':_,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps


def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = './data/cnt')
    parser.add_argument('--sty', default = './data/sty')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--precomputed', type=str, default="/temp/hanmo/style_output/StyleID/precomputed_feats", help='save path for precomputed feature')  # './precomputed_feats'
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    return parser.parse_args()



def load_or_invert_feature(
    img_path,        # 图像路径
    feat_path_root,  # 预存路径根目录
    feat_suffix,     # 保存名后缀（'_sty.pkl' 或 '_cnt.pkl'）
    model, sampler,  # 模型与采样器
    uc,              # 无条件条件（unconditional_conditioning）
    ddim_inversion_steps,
    time_idx_dict,
    save_feature_timesteps,
    start_step,
    feat_maps,       # 全局特征映射表
    device,
    save_func,       # 保存特征回调函数（用于DDIM采样）
):
    """
    通用化的内容/风格特征提取逻辑。
    1️⃣ 若特征已存在 -> 直接加载
    2️⃣ 若不存在 -> 进行DDIM反演提取特征并保存
    返回:
        feat, z_enc, feat_name, cache_hit
    """
    base_name = Path(img_path).stem
    feat_name = os.path.join(feat_path_root, base_name + feat_suffix)
    cache_hit = False

    # ---------------------------
    # 💾 Step 1: 尝试加载已有特征
    # ---------------------------
    if len(feat_path_root) > 0 and os.path.isfile(feat_name):
        print(f"✅ Precomputed feature loading: {feat_name}")
        with open(feat_name, 'rb') as h:
            feat = pickle.load(h)
            z_enc = torch.clone(feat[0]['z_enc'])
        cache_hit = True

    # ---------------------------
    # 🚧 Step 2: 若不存在则执行反演提取
    # ---------------------------
    else:
        print(f"🚧 Feature not found — building new: {feat_name}")
        init_img = load_img(img_path).to(device)
        init_img = model.get_first_stage_encoding(model.encode_first_stage(init_img))
        z_enc, _ = sampler.encode_ddim(
            init_img.clone(),
            num_steps=ddim_inversion_steps,
            unconditional_conditioning=uc,
            end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
            callback_ddim_timesteps=save_feature_timesteps,
            img_callback=save_func,
        )
        feat = copy.deepcopy(feat_maps)
        z_enc = feat[0]['z_enc']

        # 🧾 自动保存新特征（启用缓存时）
        if len(feat_path_root) > 0:
            os.makedirs(feat_path_root, exist_ok=True)
            with open(feat_name, 'wb') as h:
                pickle.dump(feat, h)
            print(f"💾 Saved new feature cache: {feat_name}")

    return feat, z_enc, feat_name, cache_hit


# def data_loader(sty_img_list, style_base_dir, cnt_img_list, cnt_base_dir):
#     for sty_name in sty_img_list:
#         for cnt_name in cnt_img_list:
#             sty_path = os.path.join(style_base_dir, sty_name)
#             cnt_path = os.path.join(cnt_base_dir, cnt_name)
#             yield sty_name, sty_path, cnt_name, cnt_path
            
            
def main():
    # ===========================
    # 🎯 1. 参数解析与基础设置
    # ===========================
    opt = parse_args()
    feat_path_root = opt.precomputed

    seed_everything(22)
    os.makedirs(opt.output_dir, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    # ===========================
    # ⚙️ 2. 模型加载与推理初始化
    # ===========================
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 

    # 构建时间步索引映射
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict, time_idx_dict = {}, {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    # ===========================
    # 🧩 3. 特征缓存结构初始化
    # ===========================
    global feat_maps
    feat_maps = [{'config': {'gamma':opt.gamma, 'T':opt.T}} for _ in range(50)]

    # ---------------------------
    # 🔁 内部辅助函数：特征保存回调
    # ---------------------------
    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    # ===========================
    # 🧠 4. 图像加载与特征提取阶段
    # ===========================
    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))

    begin = time.time()

    # 遍历所有风格图片
    for sty_name in sty_img_list:
        for cnt_name in cnt_img_list:
            sty_path = os.path.join(opt.sty, sty_name)
            cnt_path = os.path.join(opt.cnt, cnt_name)
            output_name = f"{Path(cnt_name).stem}@{Path(sty_name).stem}.png"
            output_path = os.path.join(opt.output_dir, output_name)
            
            # 🖼️ Step 4.1~4.2: 加载或反演风格特征
            sty_feat, sty_z_enc, sty_feat_name, cache_hit = load_or_invert_feature(
                img_path=sty_path,
                feat_path_root=feat_path_root,
                feat_suffix='_sty.pkl',
                model=model,
                sampler=sampler,
                uc=uc,
                ddim_inversion_steps=ddim_inversion_steps,
                time_idx_dict=time_idx_dict,
                save_feature_timesteps=save_feature_timesteps,
                start_step=start_step,
                feat_maps=feat_maps,
                device=device,
                save_func=ddim_sampler_callback
            )

            # 🖼️ Step 4.3~4.4: 加载或反演内容特征
            cnt_feat, cnt_z_enc, cnt_feat_name, cache_hit = load_or_invert_feature(
                img_path=cnt_path,
                feat_path_root=feat_path_root,
                feat_suffix='_cnt.pkl',
                model=model,
                sampler=sampler,
                uc=uc,
                ddim_inversion_steps=ddim_inversion_steps,
                time_idx_dict=time_idx_dict,
                save_feature_timesteps=save_feature_timesteps,
                start_step=start_step,
                feat_maps=feat_maps,
                device=device,
                save_func=ddim_sampler_callback
            )               
            
            # 🎨 5. 特征融合与风格生成阶段
            with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
                # 5.1 特征归一化融合（AdaIN）
                adain_z_enc = cnt_z_enc if opt.without_init_adain else adain(cnt_z_enc, sty_z_enc)
                
                # 5.2 注意力特征注入融合
                feat_maps = None if opt.without_attn_injection else feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)

                # 5.3 执行风格化采样（反向扩散生成）
                samples_ddim, _intermediates = sampler.sample(
                    S=ddim_steps,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    x_T=adain_z_enc,
                    injected_features=feat_maps,
                    start_step=start_step,
                )

                # 💾 6. 解码与结果保存阶段
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                img.save(output_path)
                print(f"image saved to {output_path}")
                
    # ===========================
    # ✅ 7. 全流程结束
    # ===========================
    print(f"Total end: {time.time() - begin:.2f}s")

if __name__ == "__main__":
    main()
