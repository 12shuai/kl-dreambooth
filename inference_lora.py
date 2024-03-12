import argparse
from diffusers.utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE, logging
LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
from diffusers import DiffusionPipeline, DDIMScheduler,StableDiffusionPipeline



from diffusers.models.modeling_utils import _get_model_file
import torch
from accelerate.utils import set_seed
import os
import yaml


from collections import defaultdict

from transformers import AutoTokenizer, PretrainedConfig

from my_lora_processor import EvalMyLoRACrossAttnProcessor
def hack_load_attn_procs(self, pretrained_model_name_or_path_or_dict, **kwargs):
    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)

    weight_name = kwargs.pop("weight_name", LORA_WEIGHT_NAME)

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        model_file = _get_model_file(
            pretrained_model_name_or_path_or_dict,
            weights_name=weight_name,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
        )
        state_dict = torch.load(model_file, map_location="cpu")
    else:
        state_dict = pretrained_model_name_or_path_or_dict

    # fill attn processors
    attn_processors = {}

    is_lora = all("lora" in k for k in state_dict.keys())

    if is_lora:
        lora_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
            lora_grouped_dict[attn_processor_key][sub_key] = value

        for key, value_dict in lora_grouped_dict.items():
            rank = value_dict["to_k_lora.down.weight"].shape[0]
            cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
            hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

            attn_processors[key] = EvalMyLoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
            )
            attn_processors[key].load_state_dict(value_dict)

    else:
        raise ValueError(f"{model_file} does not seem to be in the correct format expected by LoRA training.")

    # set correct dtype & device
    attn_processors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in attn_processors.items()}

    # set layers
    self.set_attn_processor(attn_processors)




def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str=None
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

class BreakASceneInference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True)
        # parser.add_argument("--pretrained_path", type=str, required=True)
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
        parser.add_argument("--output_path", type=str, default="outputs/result.jpg")
        parser.add_argument("--seed_list", type=int,nargs="*", default=[0])
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()

    def _load_pipeline(self):
        # text_encoder_cls = import_model_class_from_model_name_or_path(
        #     self.args.model_path
        # )
        # text_encoder = text_encoder_cls.from_pretrained(
        #     self.args.model_path,
        #     subfolder="text_encoder",
        #     # revision=self.args.revision,
        # )
        self.pipeline = DiffusionPipeline.from_pretrained(
            # self.args.pretrained_path,
            self.args.model_path,
            # text_encoder=text_encoder,
            # torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.unet.load_attn_procs=hack_load_attn_procs.__get__(self.pipeline.unet)
        self.pipeline.unet.load_attn_procs(self.args.model_path)
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
        for seed in seeds:
            set_seed(seed)
            for cate,prompt_set in prompt_dict.items():
                prompt_list=list(prompt_set)

                prompt_forward=[]
                for prompt in prompt_list:
                    if not os.path.exists(os.path.join(self.args.output_path,cate,prompt,f"{seed}.jpg")):
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
