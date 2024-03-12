"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse

from diffusers import DiffusionPipeline, DDIMScheduler
import torch
from accelerate.utils import set_seed
import os
import yaml

import ptp_utils

import numpy as np

from ptp_utils import AttentionStore,AttentionValueStore

from collections import defaultdict

from PIL import Image

class BreakASceneInference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True)
        # parser.add_argument(
        #     "--prompt", type=str, default="a photo of <asset0> at the beach"
        # )
        parser.add_argument(
            "--prompt_list", type=str, nargs="*",default=[]
        )
        parser.add_argument(
            "--prompt_file_list", type=str, nargs="*",default=[]
        )
        parser.add_argument(
            "--force_forward", type=bool, default=False
        )
        parser.add_argument(
            "--overwrite", action="store_true"
        )

        # parser.add_argument(
        #     "--project_token", type=str, default=None
        # )

        parser.add_argument("--output_path", type=str, default="outputs/result.jpg")
        parser.add_argument("--seed_list", type=int,nargs="*", default=[0])
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()

    def _load_pipeline(self):
        # print(self.args.model_path)
        # print(os.path.isdir(self.args.model_path),os.path.exists(self.args.model_path))
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.to(self.args.device)
        self.unet=self.pipeline.unet
        self.tokenizer=self.pipeline.tokenizer

    @torch.no_grad()
    def infer_and_save(self):
        def get_prommpt_list():
            prompts=break_a_scene_inference.args.prompt_list
            prompt_set=set(prompts)
            prompt_dict=defaultdict(set)
            prompt_dict["temp_list"]=prompt_set
            # 打开 YAML 文件
            for prompt_file in self.args.prompt_file_list:
                with open(prompt_file, "r") as file:
                    # 加载 YAML 文件内容并解析为 Python 对象
                    data = yaml.load(file, Loader=yaml.FullLoader)
                    for cate,pmps in data.items():
                        prompt_dict[cate].update(pmps)
            return prompt_dict

        prompt_dict=get_prommpt_list()
        print(prompt_dict)
            
        seeds=break_a_scene_inference.args.seed_list
        # self.controller = AttentionValueStore()
        self.controller = AttentionValueStore()
        self.register_attention_control(self.controller)

        for seed in seeds:
            set_seed(seed)
            for cate,prompt_set in prompt_dict.items():
                prompt_list=list(prompt_set)

                prompt_forward=[]
                for prompt in prompt_list:
                    if break_a_scene_inference.args.overwrite or not os.path.exists(os.path.join(self.args.output_path,cate,prompt,f"{seed}.jpg")):
                        prompt_forward.append(prompt)
                if len(prompt_forward)==0 and not self.args.force_forward:
                    continue
                # self.controller.cur_step = 0
                # self.controller.attention_store = {}
                # self.controller.global_value = {}
                self.controller.reset()
                images = self.pipeline(prompt_forward).images
                print(f"seed:{seed},prompt:{prompt_forward}")
                for idx,(prompt,image) in enumerate(zip(prompt_forward,images)):
                    
                    save_root=os.path.join(self.args.output_path,cate,prompt)
                    os.makedirs(save_root,exist_ok=True)
                    image.save(os.path.join(save_root,f"{seed}.jpg"))
                    # self.perform_full_inference(
                    #     path=os.path.join(
                    #         save_root, f"{seed}_atten.jpg"
                    #     ),
                    #     instance_prompt=prompt
                    # )
                    full_agg_attn = self.aggregate_attention(
                        res=16, from_where=("up", "down"), is_cross=True,
                        select=idx,train_batch_size=len(prompt_forward)
                    )
                    full_agg_value = self.aggregate_value(
                        res=16, from_where=("up", "down"), is_cross=True,
                        select=idx,train_batch_size=len(prompt_forward)
                    )
                    print("\n",prompt,full_agg_attn.shape)
                    def get_min_mean_max(atten_value):
                        return atten_value.min().item(),atten_value.mean().item(),atten_value.max().item()
                    tokens = self.tokenizer.encode(prompt)
                    for i in range(len(tokens)):
                        attn_value=full_agg_attn[...,i]
                        min_mean_max=get_min_mean_max(attn_value)
                        if i==0:
                            print("startofext",min_mean_max)
                        elif i<=len(prompt.split(" ")):
                            print(self.tokenizer.decode(int(tokens[i])),min_mean_max)
                        elif i==len(prompt.split(" "))+1:
                            print("endoftext",min_mean_max)
                        else:
                            break
                    self.save_cross_attention_vis(
                        prompt,
                        attention_maps=full_agg_attn.detach().cpu(),
                        path=os.path.join(
                            save_root, f"{seed}_atten.jpg"
                        ),
                    )
                # self.controller.cur_step = 0
                # self.controller.attention_store = {}
                # self.controller.global_value = {}
                self.controller.reset()
    
    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(
        self, res: int, from_where, is_cross: bool, select: int,train_batch_size
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                # print(item.shape,"Sxc")
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        train_batch_size, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out
    
    # def aggregate_attention_and_value(
    #     self, res: int, from_where, is_cross: bool, select: int,train_batch_size
    # ):
    #     out = []
    #     out_value=[]
    #     attention_maps = self.get_average_attention()
    #     value_maps = self.get_average_value()
    #     num_pixels = res**2
    #     for location in from_where:
    #         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
    #             # print(item.shape,"Sxc")
    #             if item.shape[1] == num_pixels:
    #                 cross_maps = item.reshape(
    #                     train_batch_size, -1, res, res, item.shape[-1]
    #                 )[select]
    #                 out.append(cross_maps)
    #                 out_value.append(value_maps)
    #     out = torch.cat(out, dim=0)
    #     out = out.sum(0) / out.shape[0]
    #     return out,out_value

    def get_average_value(self):
        # print(self.controller.cur_step,"sxc debug")
        average_value = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.global_value[key]
            ]
            for key in self.controller.global_value
        }
        return average_value
    
    @torch.no_grad()
    def perform_full_inference(self, path, instance_prompt,guidance_scale=7.5):
        self.unet.eval()
        self.text_encoder.eval()

        latents = torch.randn((1, 4, 64, 64), device=self.accelerator.device)
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.accelerator.device)
        input_ids = self.tokenizer(
            [instance_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        cond_embeddings = self.text_encoder(input_ids)[0]
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        images = self.vae.decode(latents.to(self.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.unet.train()
        if self.args.train_text_encoder:
            self.text_encoder.train()

        Image.fromarray(images[0]).save(path)
    
    @torch.no_grad()
    def perform_full_inference_from_list(self, path_list, guidance_scale=7.5):
        self.unet.eval()
        self.text_encoder.eval()

        latents = torch.randn((1, 4, 64, 64), device=self.accelerator.device)
        uncond_input = self.tokenizer(
            [""]*len(self.args.instance_prompt_list),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.accelerator.device)
        input_ids = self.tokenizer(
            self.args.instance_prompt_list,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        cond_embeddings = self.text_encoder(input_ids)[0]
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        images = self.vae.decode(latents.to(self.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.unet.train()
        if self.args.train_text_encoder:
            self.text_encoder.train()

        for i,image in enumerate(images):
            Image.fromarray(images[i]).save(path_list[i])

    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, attention_maps, path):
        tokens = self.tokenizer.encode(prompt)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(
                image, self.tokenizer.decode(int(tokens[i]))
            )
            images.append(image)
        vis = ptp_utils.view_images(np.stack(images, axis=0))
        vis.save(path)



class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        kwargs={
            "v":value
        }
        # one line change
       
        self.controller(attention_probs, is_cross, self.place_in_unet,**kwargs)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


if __name__ == "__main__":
    break_a_scene_inference = BreakASceneInference()
    break_a_scene_inference.infer_and_save()
