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
import shutil
import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional
import random

from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
from torch.utils.data import Dataset
import numpy as np

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version,randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import ptp_utils
from ptp_utils import AttentionStore
from diffusers.models.cross_attention import CrossAttention


import accelerate

accelerate.utils.set_seed(42)

check_min_version("0.12.0")

logger = get_logger(__name__)



def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/CV_teamz/pretrained/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--dataset_length",
        type=int,
        default=0,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--no_prior_preservation",
        action="store_false",
        help="Flag to add prior preservation loss.",
        dest="with_prior_preservation"
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    # parser.add_argument(
    #     "--save_interpolation",
    #     default=,
    #     help=(
    #         "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
    #         " cropped. The images will be resized to the resolution first before cropping."
    #     ),
    # )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
        dest="train_text_encoder"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--phase1_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the first phase.",
    )
    parser.add_argument(
        "--phase2_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the second phase.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=5e-4,
        help="The LR for the Textual Inversion steps.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )


    parser.add_argument(
        "--use_ppo",
        action="store_true",
        help="The beta1 parameter for the Adam optimizer.",
    )

    parser.add_argument(
        "--ppo_episilon",
        type=float,
        default=1e-8,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--use_approximate",
        action="store_true",
        help="The beta1 parameter for the Adam optimizer.",
    )
    

    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="one_minus",
        choices=["one_minus","logexp","logsigmoid","max_one_minus"],
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    
    parser.add_argument("--lambda_attention", type=float, default=0)
    parser.add_argument("--img_log_steps", type=int, default=200)
    parser.add_argument("--beta_dpo", type=float, default=1000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument("--num_of_assets", type=int, default=1)
    # parser.add_argument("--initializer_tokens", type=str, nargs="+", default=[])
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="sbs",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--train_token",
        action="store_true",
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--exchange_interval",
        type=int,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--proj_name",
        type=str,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--apply_masked_loss",
        action="store_true",
        help="Use masked loss instead of standard epsilon prediciton loss",
        dest="apply_masked_loss"
    )
    parser.add_argument(
        "--log_checkpoints",
        action="store_true",
        help="Indicator to log intermediate model checkpoints",
    )
    parser.add_argument(
        "--instance_class",
        type=str,
        required=True,
        help="Indicator to log intermediate model checkpoints",
    )

    parser.add_argument(
        "--gen_num",
        type=int,
        required=True,
        help="Indicator to log intermediate model checkpoints",
    )

    parser.add_argument(
        "--gen_data_dir",
        type=str,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--use_prior_data",
        action="store_true",
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--prior_num",
        type=int,
        default=1,
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--diff_2_weights",
        type=float,
        default=1,
        help="A token to use as a placeholder for the concept.",
    )
    

    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # assert len(args.initializer_tokens) == 0 or len(args.initializer_tokens) == args.num_of_assets
    args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps

    assert not args.apply_masked_loss
    assert args.lambda_attention==0
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.with_prior_preservation:
    #     if args.class_data_dir is None:
    #         raise ValueError("You must specify a data directory for class images.")
    #     if args.class_prompt is None:
    #         raise ValueError("You must specify prompt for class images.")
    # else:
    #     # logger is not available yet
    #     if args.class_data_dir is not None:
    #         warnings.warn(
    #             "You need not use --class_data_dir without --with_prior_preservation."
    #         )
    #     if args.class_prompt is not None:
    #         warnings.warn(
    #             "You need not use --class_prompt without --with_prior_preservation."
    #         )

    return args



# class DreamBoothDataset(Dataset):
#     """
#     A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
#     It pre-processes the images and the tokenizes prompts.
#     """

#     def __init__(
#         self,
#         instance_prompt,
#         instance_data_root,
#         # placeholder_tokens,
#         tokenizer,
#         class_data_root=None,
#         class_prompt=None,
#         size=512,
#         center_crop=False,
#         flip_p=1,
#         length=0
#     ):
#         self.size = size
#         self.center_crop = center_crop
#         self.tokenizer = tokenizer
#         self.flip_p = flip_p

#         # print(size)
#         self.image_transforms = transforms.Compose(
#             [
#                 transforms.Resize(
#                     size, interpolation=transforms.InterpolationMode.BILINEAR
#                 ),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#         )
#         self.mask_transforms = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#             ]
#         )

#         self.instance_data_root = Path(instance_data_root)
#         if not self.instance_data_root.exists():
#             raise ValueError(
#                 f"Instance {self.instance_data_root} images root doesn't exists."
#             )

#         # self.placeholder_tokens = placeholder_tokens

#         instance_img_path = [os.path.join(instance_data_root, i) for i in os.listdir(instance_data_root)]
#         self.instance_image = [self.image_transforms(Image.open(_path)) for _path in instance_img_path]

#         # print(len(self.instance_image),self.instance_image[0].shape,"sxc")

#         # self.instance_masks = []
#         # for i in range(num_of_assets):
#         #     instance_mask_path = os.path.join(instance_data_root, f"mask{i}.png")
#         #     curr_mask = Image.open(instance_mask_path)
#         #     curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
#         #     self.instance_masks.append(curr_mask)
#         # self.instance_masks = torch.cat(self.instance_masks)

#         self._length = len(self.instance_image) if length==0 else length

#         self.num_instance_images=self._length
#         if isinstance(instance_prompt,str):
#             self.instance_prompt=[instance_prompt]*self._length

#         assert len(self.instance_prompt)==self._length

#         if class_data_root is not None:
#             self.class_data_root = Path(class_data_root)
#             self.class_data_root.mkdir(parents=True, exist_ok=True)
#             self.class_images_path = list(self.class_data_root.iterdir())
#             self.num_class_images = len(self.class_images_path)
#             self._length = max(self.num_class_images, self._length)
#             self.class_prompt = class_prompt
#             self.class_images_prompts=[self.class_prompt]*self.num_class_images
#         else:
#             self.class_data_root = None

#     def __len__(self):
#         return self._length

#     def __getitem__(self, index):
#         example = {}
#         # num_of_tokens = random.randrange(1, len(self.placeholder_tokens) + 1)
#         # tokens_ids_to_use = random.sample(
#         #     range(len(self.placeholder_tokens)), k=num_of_tokens
#         # )
#         # tokens_to_use = [self.placeholder_tokens[tkn_i] for tkn_i in tokens_ids_to_use]
#         # prompt = "a photo of " + " and ".join(tokens_to_use)

#         example["instance_images"] = self.instance_image[index%self.num_instance_images]
#         # example["instance_images"] = self.instance_image[0]
#         # example["instance_images"] =self.image_transforms(example["instance_images"])
#         # example["token_ids"] = torch.tensor(tokens_ids_to_use)
#         example["instance_prompt"]= self.instance_prompt[index%self.num_instance_images]
#         if random.random() > self.flip_p:
#             example["instance_images"] = TF.hflip(example["instance_images"])
#             # example["instance_masks"] = TF.hflip(example["instance_masks"])

#         example["instance_prompt_ids"] = self.tokenizer(
#             example["instance_prompt"],
#             truncation=True,
#             padding="max_length",
#             max_length=self.tokenizer.model_max_length,
#             return_tensors="pt",
#         ).input_ids

#         if self.class_data_root:
#             class_image = Image.open(
#                 self.class_images_path[index % self.num_class_images]
#             )
#             if not class_image.mode == "RGB":
#                 class_image = class_image.convert("RGB")
#             example["class_images"] = self.image_transforms(class_image)
#             example["class_prompt_ids"] = self.tokenizer(
#                 self.class_images_prompts[index % self.num_class_images],
#                 truncation=True,
#                 padding="max_length",
#                 max_length=self.tokenizer.model_max_length,
#                 return_tensors="pt",
#             ).input_ids

#         return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    instance_prompts=[example["instance_prompt"] for example in examples]
    # masks = [example["instance_masks"] for example in examples]
    # token_ids = [example["token_ids"] for example in examples]

    if with_prior_preservation:
        input_ids = [example["class_prompt_ids"] for example in examples] + input_ids
        pixel_values = [example["class_images"] for example in examples] + pixel_values

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    # masks = torch.stack(masks)
    # token_ids = torch.stack(token_ids)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_prompt":instance_prompts
        # "instance_masks": masks,
        # "token_ids": token_ids,
    }
    return batch





class SelfPlayDreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_prompt_list,
        instance_data_root,
        # placeholder_tokens,
        tokenizer,
        class_prompt=None,
        gen_data_root=None,
        size=512,
        center_crop=False,
        flip_p=1,
        length=0,
        prior_data_path=""
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p

        self.class_prompt=class_prompt
        # print(size)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        # self.placeholder_tokens = placeholder_tokens

        instance_img_path = [os.path.join(instance_data_root, i) for i in os.listdir(instance_data_root)]
        self.instance_image = [self.image_transforms(Image.open(_path)) for _path in instance_img_path]

        # self.prior_image=defaultdict(list)
        self.prior_data_path=prior_data_path
        self.use_prior_data=False
        

        # print(len(self.instance_image),self.instance_image[0].shape,"sxc")

        # self.instance_masks = []
        # for i in range(num_of_assets):
        #     instance_mask_path = os.path.join(instance_data_root, f"mask{i}.png")
        #     curr_mask = Image.open(instance_mask_path)
        #     curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
        #     self.instance_masks.append(curr_mask)
        # self.instance_masks = torch.cat(self.instance_masks)

        self._length = len(self.instance_image)

        self.num_instance_images=self._length
        if isinstance(instance_prompt_list,str):
            self.instance_prompt_list=[instance_prompt_list]*self._length

        assert isinstance(instance_prompt_list,list)
        if len(instance_prompt_list)<self._length:
            self.instance_prompt_list.extend([instance_prompt_list[-1]]*(self._length-len(instance_prompt_list)))
        elif len(instance_prompt_list)>self._length:
            self.instance_prompt_list=instance_prompt_list[:self._length]
        else:
            self.instance_prompt_list=instance_prompt_list
        assert len(self.instance_prompt_list)==self._length
        if self.prior_data_path and os.path.exists(self.prior_data_path):
            # print(self.prior_data_path)
            # exit(0)
            self.use_prior_data=True
            for prior_prompt in os.listdir(prior_data_path):
                prior_image_path_list=[os.path.join(self.prior_data_path,prior_prompt,_path) for _path in os.listdir(os.path.join(self.prior_data_path,prior_prompt))]
                # self.prior_image[prior_prompt].extend([self.image_transforms(Image.open(_path)) for _path in prior_image_path_list])
                self.instance_image.extend([self.image_transforms(Image.open(_path)) for _path in prior_image_path_list])
                self.instance_prompt_list.extend([prior_prompt]*len(prior_image_path_list))
        
        self._length = len(self.instance_image)
        self.num_instance_images=self._length


        assert gen_data_root is not None and os.path.exists(gen_data_root),gen_data_root
        total_num_class_images=0
        self.class_images_path_list,self.class_images_prompts,self.class_images_path_dict,self.num_class_images_dict,self.class_images_cur_index=[],[],defaultdict(list),defaultdict(int),defaultdict(int)
        seen=set()
        for instance_prompt in self.instance_prompt_list:
            if instance_prompt in seen:
                continue
            seen.add(instance_prompt)
            class_images_path = list(os.listdir(os.path.join(gen_data_root,instance_prompt)))
            class_images_path=[os.path.join(os.path.join(gen_data_root,instance_prompt),i) for i in class_images_path]
            num_class_images = len(class_images_path)
            assert num_class_images >0
            total_num_class_images+=num_class_images
            self.class_images_path_dict[instance_prompt].extend(class_images_path)
            self.num_class_images_dict[instance_prompt]+=num_class_images
            self.class_images_cur_index[instance_prompt]=0
            # self.class_images_path_list.extend(class_images_path)
            # self.class_images_prompts.extend([instance_prompt]*num_class_images)
        
        self.total_num_class_images=total_num_class_images
        # assert total_num_class_images==self._length,f"{total_num_class_images} {self._length}"
        self._length = max(total_num_class_images, self._length)

        
        # if gen_data_root is not None:
        #     self.class_data_root = Path(gen_data_root)
        #     self.class_data_root.mkdir(parents=True, exist_ok=True)
        #     self.class_images_path = list(self.class_data_root.iterdir())
        #     self.num_class_images = len(self.class_images_path)
        #     self._length = max(self.num_class_images, self._length)
        #     self.class_prompt = class_prompt
        #     self.class_images_prompts=[self.class_prompt]*self.num_class_images
        # else:
        #     self.class_data_root = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # num_of_tokens = random.randrange(1, len(self.placeholder_tokens) + 1)
        # tokens_ids_to_use = random.sample(
        #     range(len(self.placeholder_tokens)), k=num_of_tokens
        # )
        # tokens_to_use = [self.placeholder_tokens[tkn_i] for tkn_i in tokens_ids_to_use]
        # prompt = "a photo of " + " and ".join(tokens_to_use)

        example["instance_images"] = self.instance_image[index%self.num_instance_images]
        # example["instance_images"] = self.instance_image[0]
        # example["instance_images"] =self.image_transforms(example["instance_images"])
        # example["token_ids"] = torch.tensor(tokens_ids_to_use)
        example["instance_prompt"]= self.instance_prompt_list[index%self.num_instance_images]
        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            # example["instance_masks"] = TF.hflip(example["instance_masks"])

        example["instance_prompt_ids"] = self.tokenizer(
            example["instance_prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        instance_prompt=example["instance_prompt"]
        cls_index=self.class_images_cur_index[instance_prompt]% self.num_class_images_dict[instance_prompt]
        # print("Sxc1",self.class_images_cur_index[instance_prompt],self.num_class_images_dict[instance_prompt])
        class_image = Image.open(
            self.class_images_path_dict[instance_prompt][cls_index]

        )
        self.class_images_cur_index[instance_prompt]+=1
        # print("sxc2",self.class_images_cur_index[instance_prompt],self.num_class_images_dict[instance_prompt])
        if not class_image.mode == "RGB":
            class_image = class_image.convert("RGB")
        example["class_images"] = self.image_transforms(class_image)
        if self.class_prompt is not None:
            example["class_prompt"]=self.class_prompt
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        else:
            example["class_prompt"]=example["instance_prompt"]
            example["class_prompt_ids"]=example["instance_prompt_ids"]
        example["cur_index"]=cls_index
        # print("Sxc3",example["cur_index"])
        # if self.prior_image:
        # print("sd",example["instance_prompt"])
        return example


def self_play_collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    instance_prompts=[example["instance_prompt"] for example in examples]
    # masks = [example["instance_masks"] for example in examples]
    # token_ids = [example["token_ids"] for example in examples]
    input_ids = [example["class_prompt_ids"] for example in examples] + input_ids
    pixel_values = [example["class_images"] for example in examples] + pixel_values
    instance_prompts=[example["class_prompt"] for example in examples]+instance_prompts
    cur_index=[example["cur_index"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    # masks = torch.stack(masks)
    # token_ids = torch.stack(token_ids)

    # print("hahasxc",cur_index)
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_prompt":instance_prompts,
        "class_index":cur_index
        # "instance_masks": masks,
        # "token_ids": token_ids,
    }
    return batch

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class SelfPlayPromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt_list, num_samples):
        self.prompt_list = [prompt for prompt in prompt_list for _ in range(num_samples)]
        self.num_samples = len(self.prompt_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt_list[index%len(self)]
        example["index"] = index
        return example

def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

import random

TEMPLATES=[
    "a photo of a {}",
    # "a {}"
]
def gen_prompt_list(placeholder,cls_name,data_num):
    res=[]
    
    for _ in range(data_num):
        index=random.randint(0,len(TEMPLATES)-1)
        template=TEMPLATES[index]
        res.append(template.format(f"{placeholder} {cls_name}"))

    return res



class SpatialDreambooth:
    def __init__(self):
        self.args = parse_args()
        self.generator=torch.Generator('cuda').manual_seed(self.args.seed)
        self.main()

    def generate_prior_data(self,path,prompt_list,gen_num):


        torch_dtype = (
            torch.float16
            if self.accelerator.device.type == "cuda"
            else torch.float32
        )
        if self.args.prior_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif self.args.prior_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif self.args.prior_generation_precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = DiffusionPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=self.args.revision,
        )
        pipeline.set_progress_bar_config(disable=True)

        for prompt in prompt_list:
            sample_dataset = PromptDataset(prompt, gen_num)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=self.args.sample_batch_size
            )

            sample_dataloader = self.accelerator.prepare(sample_dataloader)
            pipeline.to(self.accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not self.accelerator.is_local_main_process,
            ):

                images = pipeline(example["prompt"]).images

                for i, (image,_prompt) in enumerate(zip(images,example["prompt"])):
                    if not os.path.exists(os.path.join(path,_prompt)):
                        os.makedirs(os.path.join(path,_prompt),exist_ok=True)
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = os.path.join(
                        path,
                        _prompt,
                        f"{example['index'][i]}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def main(self):
        self.args.output_dir=os.path.join(self.args.output_dir,self.args.proj_name)
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)



        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            logging_dir=logging_dir,
        )

        if (
            self.args.train_text_encoder
            and self.args.gradient_accumulation_steps > 1
            and self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.args.pretrained_model_name_or_path, self.args.revision
        )

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )##fixed_small
        

        self.cur_text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )
        self.last_text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
        )
        self.cur_unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )
        self.last_unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )

        # Load the tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, revision=self.args.revision, use_fast=False
            )
        elif self.args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.args.revision,
                use_fast=False,
            )

        # Add assets tokens to tokenizer
        self.args.num_of_assets=1
        self.placeholder_tokens = self.args.placeholder_token
        self.args.instance_prompt = gen_prompt_list(self.args.placeholder_token,self.args.instance_class,len(os.listdir(self.args.instance_data_dir)))

        self.args.extend_instance_prompt=[prompt for prompt in self.args.instance_prompt]
        print(self.args.instance_prompt,"sxc")
        
        if self.args.use_prior_data:
            prior_data_path=os.path.join(self.args.gen_data_dir,self.args.proj_name,"prior_data")
            self.prior_prompt_list=[f"a photo of a {self.args.instance_class}"]
            self.generate_prior_data(prior_data_path,self.prior_prompt_list,self.args.prior_num)
            self.args.extend_instance_prompt.extend(self.prior_prompt_list*self.args.prior_num)
        # [
        #     self.args.placeholder_token.replace(">", f"{idx}>")
        #     for idx in range(self.args.num_of_assets)
        # ]
        if self.args.train_token:
            num_added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)
            assert num_added_tokens == self.args.num_of_assets
            self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(
                self.placeholder_tokens
            )
            self.cur_text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.last_text_encoder.resize_token_embeddings(len(self.tokenizer))

            token_embeds = self.cur_text_encoder.get_input_embeddings().weight.data
            token_embeds[-self.args.num_of_assets :] = token_embeds[
                -3 * self.args.num_of_assets : -2 * self.args.num_of_assets
            ]

            token_embeds = self.last_text_encoder.get_input_embeddings().weight.data
            token_embeds[-self.args.num_of_assets :] = token_embeds[
                -3 * self.args.num_of_assets : -2 * self.args.num_of_assets
            ]

        # Set validation scheduler for logging
        self.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.validation_scheduler.set_timesteps(50)

        # We start by only optimizing the embeddings
        self.vae.requires_grad_(False)
        self.cur_unet.requires_grad_(False)
        self.last_unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        self.last_text_encoder.requires_grad_(False)
        if not self.args.train_token:
            self.cur_text_encoder.requires_grad_(False)
            
        else:
            self.cur_text_encoder.text_model.encoder.requires_grad_(False)
            self.cur_text_encoder.text_model.final_layer_norm.requires_grad_(False)
            self.cur_text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.cur_unet.enable_xformers_memory_efficient_attention()
                self.last_unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.args.gradient_checkpointing:
            self.cur_unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.cur_text_encoder.gradient_checkpointing_enable()

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # We start by only optimizing the embeddings
        optimizer=None
        lr_scheduler=None
        # assert self.args.train_token
        if self.args.train_token:
            params_to_optimize = self.cur_text_encoder.get_input_embeddings().parameters()
            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.args.initial_learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.adam_weight_decay,
                eps=self.args.adam_epsilon,
            )
            lr_scheduler = get_scheduler(
                self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.args.lr_warmup_steps
                * self.args.gradient_accumulation_steps,
                num_training_steps=self.args.max_train_steps
                * self.args.gradient_accumulation_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )
        assert (self.args.train_token and self.args.phase1_train_steps>0) or (not self.args.train_token and self.args.phase1_train_steps==0)
        
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(os.listdir(self.args.instance_data_dir)) / (self.args.gradient_accumulation_steps* self.args.train_batch_size)
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        if optimizer is not None:
            (
                self.cur_unet,
                self.cur_text_encoder,
                optimizer,
                lr_scheduler,
            ) = self.accelerator.prepare(
                self.cur_unet, self.cur_text_encoder, optimizer,  lr_scheduler
            )
        else:
            (
                self.cur_unet,
                self.cur_text_encoder,
            ) = self.accelerator.prepare(
                self.cur_unet,
                self.cur_text_encoder
            )


        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        self.last_unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.last_text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.cur_unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.cur_unet).dtype}. {low_precision_error_string}"
            )

        if (
            self.args.train_text_encoder
            and self.accelerator.unwrap_model(self.cur_text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.cur_text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(os.listdir(self.args.instance_data_dir)) / (self.args.gradient_accumulation_steps* self.args.train_batch_size)
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        if self.accelerator.is_main_process:
            import copy
            # args=copy.deepcopy(self.args)
            # args.instance_prompt =self.args.instance_prompt[0]
            # print(vars(args))
            # self.accelerator.init_trackers("dreambooth", config=vars(self.args))
            self.accelerator.init_trackers("dreambooth", config={"haha":3})

        # Train
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        # logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")##200
        logger.info(
            f"  Instantaneous batch size per device = {self.args.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * self.args.gradient_accumulation_steps
                )

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        # keep original embeddings as reference
        if self.args.train_token:
            orig_embeds_params = (
                self.accelerator.unwrap_model(self.cur_text_encoder)
                .get_input_embeddings()
                .weight.data.clone()
            )

        # Create attention controller
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)

        for epoch in range(first_epoch, self.args.num_train_epochs):
            if (epoch-first_epoch) % self.args.exchange_interval==0:
            # if (epoch-first_epoch)==0:
                train_dataloader=self.exchange(epoch)
                train_dataloader= self.accelerator.prepare(
                    train_dataloader
                )
                print(f"exchange in {epoch}/{self.args.num_train_epochs}")
                # exit(0)

            self.cur_unet.train()
            self.last_unet.train()
            
            if self.args.train_text_encoder:
                self.cur_text_encoder.train()
                self.last_text_encoder.train()
                
            for step, batch in enumerate(train_dataloader):
                if self.args.phase1_train_steps == global_step:
                    self.cur_unet.requires_grad_(True)
                    if self.args.train_text_encoder:
                        self.cur_text_encoder.requires_grad_(True)
                    unet_params = self.cur_unet.parameters()

                    if self.args.train_token:
                        params_to_optimize = (
                            itertools.chain(unet_params, self.cur_text_encoder.parameters())
                            if self.args.train_text_encoder
                            else itertools.chain(unet_params, self.cur_text_encoder.get_input_embeddings().parameters())
                        )
                    else:
                        params_to_optimize = (
                            itertools.chain(unet_params, self.cur_text_encoder.parameters())
                            if self.args.train_text_encoder
                            else unet_params
                        )
                    
                    del optimizer
                    optimizer = optimizer_class(
                        params_to_optimize,
                        lr=self.args.learning_rate,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        weight_decay=self.args.adam_weight_decay,
                        eps=self.args.adam_epsilon,
                    )
                    del lr_scheduler
                    lr_scheduler = get_scheduler(
                        self.args.lr_scheduler,
                        optimizer=optimizer,
                        num_warmup_steps=self.args.lr_warmup_steps
                        * self.args.gradient_accumulation_steps,
                        num_training_steps=self.args.max_train_steps
                        * self.args.gradient_accumulation_steps,
                        num_cycles=self.args.lr_num_cycles,
                        power=self.args.lr_power,
                    )
                    optimizer, lr_scheduler = self.accelerator.prepare(
                        optimizer, lr_scheduler
                    )

                logs = {}

                # Skip steps until we reach the resumed step
                if (
                    self.args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.cur_unet):
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["pixel_values"].to(dtype=self.weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    _,cur_latents=latents.chunk(2)
                    # last_noise,cur_noise = torch.randn_like(last_latents),torch.randn_like(cur_latents)
                    # noise=torch.cat([last_noise,cur_noise])

                    noise=torch.randn_like(latents)
                    last_noise,cur_noise=noise.chunk(2) 

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz//2,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()
                    double_timesteps=torch.cat([timesteps,timesteps],dim=0)

                                        # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, double_timesteps
                    )
                    
                    _,cur_noisy_latents=noisy_latents.chunk(2)

                    # Get the text embedding for conditioning
                    last_token_ids,cur_token_ids=batch["input_ids"].chunk(2)
                    last_encoder_hidden_states = self.last_text_encoder(last_token_ids)[0]
                    cur_encoder_hidden_states = self.cur_text_encoder(cur_token_ids)[0]
                    

                    
                    if self.args.use_approximate:
                        shape=cur_latents.shape
                        
                        last_model_output,last_sample=self.last_model_inference_from_scratch(shape,self.noise_scheduler.timesteps[0],last_encoder_hidden_states)
                        # last_latents=self.last_forward(batch["instance_prompt"].chunk(2)[0])
                        # print(self.noise_scheduler.timesteps)
                        last_latents=self.last_forward(last_model_output,last_sample,(self.noise_scheduler.timesteps[0]))
                        last_noisy_latents = self.noise_scheduler.add_noise(
                            last_latents, last_noise, timesteps
                        )
                        # exit(0)
                    else:
                        last_latents=latents.chunk(2)[0]
                        last_noisy_latents=noisy_latents.chunk(2)[0]
                    # last_timesteps,cur_timesteps=timesteps.chunk(2)
                    # last_noise,cur_noise=noise.chunk(2)
                    

                    
                    # Predict the noise residual
                    # cur_model_pred_cur = self.cur_unet(
                    #     cur_noisy_latents, timesteps, cur_encoder_hidden_states
                    # ).sample
                    # last_model_pred_cur = self.last_unet(
                    #     cur_noisy_latents, timesteps, last_encoder_hidden_states
                    # ).sample.detach()

                    # cur_model_pred_last = self.cur_unet(
                    #     last_noisy_latents, timesteps, cur_encoder_hidden_states
                    # ).sample
                    # last_model_pred_last = self.last_unet(
                    #     last_noisy_latents, timesteps, last_encoder_hidden_states
                    # ).sample.detach()

                    
                        
                    cat_noisy_latents=torch.cat([last_noisy_latents,cur_noisy_latents])
                    cat_cur_encoder_hidden_states=torch.cat([cur_encoder_hidden_states,cur_encoder_hidden_states])
                    cat_last_encoder_hidden_states=torch.cat([last_encoder_hidden_states,last_encoder_hidden_states])
                    
                    cur_model_pred_total = self.cur_unet(
                        cat_noisy_latents, double_timesteps, cat_cur_encoder_hidden_states
                    ).sample
                    last_model_pred_total = self.last_unet(
                        cat_noisy_latents, double_timesteps, cat_last_encoder_hidden_states
                    ).sample.detach()

                    
                    cur_model_pred_last,cur_model_pred_cur= cur_model_pred_total.chunk(2)
                    last_model_pred_last,last_model_pred_cur= last_model_pred_total.chunk(2)

                    
                        


                    def get_ht_pow_2(timesteps):
                        # self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
                        # timesteps = timesteps.to(original_samples.device)
                        res=[]
                        for t in timesteps:
                            alphas_cumprod=self.noise_scheduler.alphas_cumprod
                            var_t=compute_variance(t)
                            # var_t=0
                            # print(alphas_cumprod,timesteps)
                            alphas_cumprod_prev=alphas_cumprod[t-1] if t>0 else self.noise_scheduler.one
                            sqrt_one_minus_alpha_prod_prev = (1-alphas_cumprod_prev) ** 0.5
                            first_term=(1-alphas_cumprod_prev-var_t) ** 0.5
                            sqrt_alpha_prod_prev_div_alpha_prod = (alphas_cumprod_prev/alphas_cumprod[t]) ** 0.5
                            res.append((first_term-sqrt_alpha_prod_prev_div_alpha_prod*sqrt_one_minus_alpha_prod_prev).pow(2))
                        return torch.FloatTensor(res).to(timesteps.device)
                        # alphas_cumprod=self.noise_scheduler.alphas_cumprod
                        # # print(alphas_cumprod,timesteps)
                        # sqrt_one_minus_alpha_prod_prev = (1-alphas_cumprod[timesteps-1]) ** 0.5
                        # sqrt_alpha_prod_prev_div_alpha_prod = (alphas_cumprod[timesteps-1]/alphas_cumprod[timesteps]) ** 0.5
                        # return (sqrt_one_minus_alpha_prod_prev-sqrt_alpha_prod_prev_div_alpha_prod*sqrt_one_minus_alpha_prod_prev).pow(2)
                    def compute_variance(t):
                        if self.noise_scheduler.variance_type == "fixed_small_log":
                            variance = self.noise_scheduler._get_variance(t, predicted_variance=None)**2
                        else:
                            variance = self.noise_scheduler._get_variance(t, predicted_variance=None)
                        return variance

                    def get_beta_t(timesteps):
                        assert self.noise_scheduler.variance_type not in ["learned", "learned_range"]
                        res=[]
                        for t in timesteps:
                            # variance=compute_variance(t)
                            variance=1
                            # res.append(variance+0.0005)
                            res.append(variance+1e-8)
                            
                        variance=torch.FloatTensor(res).to(timesteps.device)
                        # print(variance,"sxc_variance")
                        return self.args.beta_dpo/(2*variance)
                    
                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        last_target,cur_target = last_noise,cur_noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        last_target = self.noise_scheduler.get_velocity(
                            last_latents, last_noise, timesteps
                        )
                        cur_target = self.noise_scheduler.get_velocity(
                            cur_latents, cur_noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )


                    cur_model_pred_cur_losses = (cur_model_pred_cur - cur_target).pow(2).mean(dim=[1,2,3])
                    last_model_pred_cur_losses = (last_model_pred_cur - cur_target).pow(2).mean(dim=[1,2,3])

                    # print("sxc1",cur_model_pred_cur_losses,last_model_pred_cur_losses)
                    cur_model_pred_last_losses = (cur_model_pred_last - last_target).pow(2).mean(dim=[1,2,3])
                    last_model_pred_last_losses = (last_model_pred_last - last_target).pow(2).mean(dim=[1,2,3])

                    # print("sxc2",cur_model_pred_last_losses,last_model_pred_last_losses)
                    diff_loss_1=cur_model_pred_cur_losses-last_model_pred_cur_losses 
                    diff_loss_2=cur_model_pred_last_losses-last_model_pred_last_losses 
                    # total_loss=diff_loss_1-diff_loss_2*0.1
                    total_loss=diff_loss_1-diff_loss_2*self.args.diff_2_weights
                    # print(total_loss,"sxc")
                    # total_loss=diff_loss_1
                    # print("sxc3",diff_loss,total_loss)
                    ord_total_loss=total_loss.mean().detach().item()
                    if self.args.use_ppo:
                        ppo_episilon=torch.Tensor([self.args.ppo_episilon]).to(cur_model_pred_last_losses)
                        ppo_cur_model_pred_cur_losses=torch.clamp(cur_model_pred_cur_losses,
                                                        last_model_pred_cur_losses*torch.exp(-ppo_episilon),
                                                        last_model_pred_cur_losses *torch.exp(ppo_episilon))
                        diff1_use_ppo=torch.logical_or(cur_model_pred_cur_losses<last_model_pred_cur_losses*torch.exp(-ppo_episilon),cur_model_pred_cur_losses>last_model_pred_cur_losses *torch.exp(ppo_episilon)).sum()
                        # ppo_cur_model_pred_cur_losses= cur_model_pred_cur_losses
                        ppo_cur_model_pred_last_losses=torch.clamp(cur_model_pred_last_losses,
                                                        last_model_pred_last_losses*torch.exp(-ppo_episilon),
                                                        last_model_pred_last_losses *torch.exp(ppo_episilon))       

                        diff2_use_ppo=torch.logical_or(cur_model_pred_last_losses<last_model_pred_last_losses*torch.exp(-ppo_episilon),cur_model_pred_last_losses>last_model_pred_last_losses *torch.exp(ppo_episilon)).sum()
                        ppo_diff_loss_1=ppo_cur_model_pred_cur_losses-last_model_pred_cur_losses 
                        ppo_diff_loss_2=ppo_cur_model_pred_last_losses-last_model_pred_last_losses  
                        ppo_total_loss=ppo_diff_loss_1-ppo_diff_loss_2*self.args.diff_2_weights

                        # total_loss=torch.min(total_loss,ppo_total_loss) ##sxc??????????????
                        # total_loss=ppo_total_loss
                        # print(total_loss,ppo_total_loss,"sxc")
                        _ppo_total_loss=ppo_total_loss.mean().detach().item()
                        total_loss_use_ppo=(total_loss<ppo_total_loss).sum()
                        total_loss=torch.max(total_loss,ppo_total_loss) 
                        
                        # print(total_loss)
                    scale_term = -1 * get_beta_t(timesteps)*get_ht_pow_2(timesteps)
                    # scale_term = get_beta_t(timesteps)*get_ht_pow_2(timesteps)
                    # print("sxc1000",scale_term)
                    # inside_term = scale_term * total_loss *100000
                    # inside_term = -1000 * diff_loss_1
                    inside_term = -1000 * total_loss ##越大最后的损失越小。对于ppo越小越好，也就是total_loss越大越好，对于ppo
                    # inside_term = -100000000 * get_ht_pow_2(timesteps) * total_loss
                    # print("inside_term:",scale_term,inside_term)
                    # print("sxc4",get_beta_t(timesteps),get_ht(timesteps))
                    # print("sxc5",scale_term,inside_term)
                    # implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    if self.args.loss_type=="one_minus":
                        loss = (1-inside_term).mean()
                    elif self.args.loss_type=="logexp":
                        loss = (torch.log(1+torch.exp(-inside_term))).mean()
                    elif self.args.loss_type=="max_one_minus":
                        loss = torch.max(1-inside_term,torch.zeros_like(inside_term)).mean()
                    elif self.args.loss_type=="logsigmoid":
                        loss=-1*F.logsigmoid(inside_term).mean()
                    else:
                        raise NotImplementedError("sxc")        

                    # print("losshh",loss)
                    self.accelerator.backward(loss)
                    # loss.backward()
                    # print(self.cur_unet.grad)
                    # No need to keep the attention store
                    self.controller.attention_store = {}
                    self.controller.cur_step = 0

                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                self.cur_unet.parameters(), self.cur_text_encoder.parameters()
                            )
                            if self.args.train_text_encoder
                            else self.cur_unet.parameters()
                        )
                        # for param in self.cur_unet.parameters():
                        #     if param.grad is not None:
                        #         print("haha",param.grad)
                        #         exit(0)
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, self.args.max_grad_norm
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

                    # if global_step < self.args.phase1_train_steps:
                        # Let's make sure we don't update any embedding weights besides the newly added token
                    if self.args.train_token:
                        with torch.no_grad():
                            self.accelerator.unwrap_model(
                                self.cur_text_encoder
                            ).get_input_embeddings().weight[
                                : -self.args.num_of_assets
                            ] = orig_embeds_params[
                                : -self.args.num_of_assets
                            ]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(
                                self.args.output_dir, f"checkpoint-{global_step}"
                            )
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                    if (
                        self.args.log_checkpoints
                        and global_step % self.args.img_log_steps == 0
                        and global_step > self.args.phase1_train_steps-1 ##sxc modify
                    ):
                        ckpts_path = os.path.join(
                            self.args.output_dir, "checkpoints", f"{global_step:05}"
                        )
                        os.makedirs(ckpts_path, exist_ok=True)
                        self.save_pipeline(ckpts_path)

                        img_logs_path = os.path.join(self.args.output_dir, "img_logs")
                        os.makedirs(img_logs_path, exist_ok=True)

                        # if self.args.lambda_attention != 0:
                        #     self.controller.cur_step = 1
                        #     last_sentence = batch["input_ids"][curr_cond_batch_idx]
                        #     last_sentence = last_sentence[
                        #         (last_sentence != 0)
                        #         & (last_sentence != 49406)
                        #         & (last_sentence != 49407)
                        #     ]
                        #     last_sentence = self.tokenizer.decode(last_sentence)
                        #     self.save_cross_attention_vis(
                        #         last_sentence,
                        #         attention_maps=agg_attn.detach().cpu(),
                        #         path=os.path.join(
                        #             img_logs_path, f"{global_step:05}_step_attn.jpg"
                        #         ),
                            # )
                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                        self.perform_full_inference(
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_full_pred.jpg"
                            )
                        )
                        full_agg_attn = self.aggregate_attention(
                            res=16, from_where=("up", "down"), is_cross=True, select=0
                        )
                        self.save_cross_attention_vis(
                            self.args.instance_prompt[0],
                            attention_maps=full_agg_attn.detach().cpu(),
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_full_attn.jpg"
                            ),
                        )
                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                logs["loss"] = loss.detach().item()
                logs["cur_model_pred_cur_losses"]=cur_model_pred_cur_losses.mean().detach().item()
                logs["scale_term"]=scale_term.mean().detach().item()
                # logs["timesteps"]=timesteps.detach().item()
                logs["beta_t"]=get_beta_t(timesteps).mean().detach().item()
                logs["ht_pow_2"]=get_ht_pow_2(timesteps).mean().detach().item()
                logs["diff_loss_1"]=diff_loss_1.mean().detach().item()
                logs["diff_loss_2"]=diff_loss_2.mean().detach().item()
                logs["ord_total_loss"]=ord_total_loss
                if self.args.use_ppo:
                    logs["ppo_diff_loss_1"]=ppo_diff_loss_1.mean().detach().item()
                    logs["ppo_diff_loss_2"]=ppo_diff_loss_2.mean().detach().item()
                    logs["ppo_total_loss"]=_ppo_total_loss
                    logs["diff1_use_ppo"]=diff1_use_ppo
                    logs["diff2_use_ppo"]=diff2_use_ppo
                    logs["total_loss_use_ppo"]=total_loss_use_ppo
                logs["total_loss"]=total_loss.mean().detach().item()
                logs["cur_model_pred_last_losses"]=cur_model_pred_last_losses.mean().detach().item()
                logs["last_model_pred_cur_losses"]=last_model_pred_cur_losses.mean().detach().item()
                logs["last_model_pred_last_losses"]=last_model_pred_last_losses.mean().detach().item()
                
                logs["inside_term"] = inside_term.mean().detach().item()
                # logs["normalize_loss"]=total_loss
                # logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    print(global_step,"done")
                    break
        
        print("haha,last done!!!!!!!!!!!!!!1",global_step,epoch,first_epoch,self.args.max_train_steps)
        self.save_pipeline(self.args.output_dir)

        self.accelerator.end_training()


    def last_model_inference_from_scratch(self,shape,timestep,prompt_embeds,latents=None):
        # shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents is None:
            latents = randn_tensor(shape, generator=self.generator, device=self.accelerator.device, dtype=self.weight_dtype)
        else:
            latents = latents.to(self.accelerator.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma

        # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latents= self.noise_scheduler.scale_model_input(latents, timestep)

        # predict the noise residual
        noise_pred = self.last_unet(latents, timestep, encoder_hidden_states=prompt_embeds).sample

        # # perform guidance
        # if do_classifier_free_guidance:
        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        return noise_pred,latents

    def last_forward(self,model_output,sample,timestep):
        t = timestep
        if model_output.shape[1] == sample.shape[1] * 2 and self.noise_scheduler.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[t - 1] if t > 0 else self.noise_scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.noise_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.noise_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.noise_scheduler.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        return pred_original_sample

    
    def save_pipeline(self, path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.cur_unet),
                text_encoder=self.accelerator.unwrap_model(self.cur_text_encoder),
                tokenizer=self.tokenizer,
                revision=self.args.revision,
            )
            pipeline.save_pretrained(path)

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.cur_unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.cur_unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.cur_unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.cur_unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.cur_unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.cur_unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count
    
    def exchange(self,epoch):
        self._copy_model_weights()
        return self._make_new_dataloader(epoch)

    def _copy_model_weights(self):
        self.last_unet.load_state_dict(self.cur_unet.state_dict())
        # self.last_unet=self.accelerator.prepare(self.last_unet)
        self.last_unet.requires_grad_(False)
        if self.args.train_text_encoder or self.args.train_token:
            self.last_text_encoder.load_state_dict(self.cur_text_encoder.state_dict())
            # self.last_text_encoder=self.accelerator.prepare(self.last_text_encoder)
            self.last_text_encoder.requires_grad_(False)
        
        self.last_unet.train()
        self.last_text_encoder.train()
        
    def _make_new_dataloader(self,epoch):
        
        def generate_data(epoch_num,prompt_list,gen_num=1):
            gen_images_dir = os.path.join(self.args.gen_data_dir,self.args.proj_name,str(epoch_num))
            if os.path.exists(gen_images_dir):
                shutil.rmtree(gen_images_dir)
            os.makedirs(os.path.join(self.args.gen_data_dir,self.args.proj_name,str(epoch_num)),exist_ok=True)
            torch_dtype = (
                torch.float16
                if self.accelerator.device.type == "cuda"
                else torch.float32
            )
            if self.args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif self.args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif self.args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16

            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.last_unet,
                text_encoder=self.last_text_encoder,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=self.args.revision,
            )
            # self.last_unet.eval()
            # self.last_text_encoder.eval()

            with torch.no_grad():
                pipeline.set_progress_bar_config(disable=True)
                num_new_images = gen_num
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = SelfPlayPromptDataset(prompt_list, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(
                    sample_dataset, batch_size=self.args.sample_batch_size
                )

                sample_dataloader = self.accelerator.prepare(sample_dataloader)
                pipeline.to(self.accelerator.device)
                for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=not self.accelerator.is_local_main_process,
                ):
                    images = pipeline(example["prompt"]).images

                    for i, (image,prompt) in enumerate(zip(images,example["prompt"])):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        img_root=os.path.join(gen_images_dir,prompt)
                        os.makedirs(img_root,exist_ok=True)
                        image_filename = (
                            os.path.join(img_root,f"{hash_image}.jpg")
                        )
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(gen_images_dir," is created")
                self.last_unet.train()
                self.last_text_encoder.train()
        
        generate_data(epoch,self.args.instance_prompt if not self.args.use_prior_data else self.args.extend_instance_prompt,self.args.gen_num)

        train_dataset = SelfPlayDreamBoothDataset(
            instance_prompt_list=self.args.instance_prompt,
            instance_data_root=self.args.instance_data_dir,
            # placeholder_tokens=self.placeholder_tokens,
            gen_data_root=os.path.join(self.args.gen_data_dir,self.args.proj_name,str(epoch)),
            tokenizer=self.tokenizer,
            class_prompt=None if len(self.args.class_prompt)==0 else self.args.class_prompt,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
            length=self.args.dataset_length,
            prior_data_path="" if not self.args.use_prior_data else os.path.join(self.args.gen_data_dir,self.args.proj_name,"prior_data")
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: self_play_collate_fn(
                examples
            ),
            num_workers=self.args.dataloader_num_workers,
        )
        return train_dataloader

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
        self, res: int, from_where: List[str], is_cross: bool, select: int
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        self.args.train_batch_size, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    @torch.no_grad()
    def perform_full_inference(self, path, guidance_scale=7.5):
        self.cur_unet.eval()
        self.cur_text_encoder.eval()

        latents = torch.randn((1, 4, 64, 64), device=self.accelerator.device)
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.accelerator.device)
        input_ids = self.tokenizer(
            [self.args.instance_prompt[0]],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        cond_embeddings = self.cur_text_encoder(input_ids)[0]
        uncond_embeddings = self.cur_text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.cur_unet(
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

        self.cur_unet.train()
        if self.args.train_text_encoder:
            self.cur_text_encoder.train()

        Image.fromarray(images[0]).save(path)

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
        attn: CrossAttention,
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

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


if __name__ == "__main__":
    SpatialDreambooth()