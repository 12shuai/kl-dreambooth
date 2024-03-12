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


from collections import defaultdict


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
        # if break_a_scene_inference.args.project_token:
        #     # asset_token,projection_token,projection_token2=break_a_scene_inference.args.project_token.split(" ")
        #     token_list=break_a_scene_inference.args.project_token.split("&")
        #     asset_token,projection_token_list=token_list[0],token_list[1:]
        #     token_embeds = self.pipeline.text_encoder.get_input_embeddings().weight.data
        #     curr_token_ids_list=[]
        #     asset_token_ids = self.pipeline.tokenizer.encode(
        #         asset_token, add_special_tokens=False
        #     )

        #     for projection_token in projection_token_list:
        #         curr_token_ids = self.pipeline.tokenizer.encode(
        #             projection_token, add_special_tokens=False
        #         )
        #         curr_token_ids_list.append(curr_token_ids)

        #     projection_vector_list= [token_embeds[id].sum(0)
        #                             for id in curr_token_ids_list]
            
        #     asset_vector=token_embeds[
        #         asset_token_ids[0]
        #     ]
        #     # print(projection_vector.shape,asset_vector.shape)
        #     # norm_projection= torch.norm(projection_vector)

        #     # normalized_projection_vector = projection_vector / norm_projection
        #     # token_embeds[asset_token_ids[0]]=torch.dot(asset_vector,normalized_projection_vector)*normalized_projection_vector 
        #     # token_embeds[asset_token_ids[0]]=asset_vector- torch.dot(asset_vector,normalized_projection_vector)*normalized_projection_vector
        #     token_embeds[asset_token_ids[0]]=torch.stack(projection_vector_list).sum(0)

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
                images = self.pipeline(prompt_forward).images
                print(f"seed:{seed},prompt:{prompt_forward}")
                for prompt,image in zip(prompt_forward,images):
                    save_root=os.path.join(self.args.output_path,cate,prompt)
                    os.makedirs(save_root,exist_ok=True)
                    image.save(os.path.join(save_root,f"{seed}.jpg"))
        
            

if __name__ == "__main__":
    break_a_scene_inference = BreakASceneInference()
    break_a_scene_inference.infer_and_save()
