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
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional
import random

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



def calculate_steer_loss(token_embedding,
                         input_ids,
                         batch_placeholder_token_id_list,
                         stop_ids,
                         special_ids,
                         batch_positive_ids_list,
                        #  with_prior_preservation,
                         temperature=0.07):
    """L_steer"""
    # compute input embeddings
    # print(
    #     input_ids,
    #     batch_placeholder_token_id_list,
    #     stop_ids,
    #     special_ids,
    #     batch_positive_ids_list
    # )
    batch_loss=0
    cnt=0
    inputs_embeds = token_embedding(input_ids)  # (bs, 77, 768)
    for b_idx,(placeholder_token_id,positive_ids) in enumerate(zip(batch_placeholder_token_id_list,batch_positive_ids_list)):
        with torch.no_grad(
        ):  # no gradients from positive and negative embeds, only from <R>
            positive_embeds = token_embedding(positive_ids)
            stop_mask = torch.isin(
                input_ids[b_idx],
                # torch.tensor(stop_ids + special_ids +
                #             [placeholder_token_id]).cuda())  # (bs, 77)
                torch.tensor(stop_ids +special_ids +
                            [placeholder_token_id]+[torch.tensor(0).cuda()]).cuda())  # (bs, 77)
                # torch.tensor(
                #             [placeholder_token_id]+[torch.tensor(0).cuda()]).cuda())  # (bs, 77)
                # torch.tensor(stop_ids +special_ids +
                #             placeholder_token_id_list.tolist()+[torch.tensor(0).cuda()]).cuda())  # (bs, 77)
            # print("neg_stop_mask",stop_mask)
            # exit(0)
            negative_embds = inputs_embeds[b_idx][~stop_mask]  # (num_stop_tokens, 768) 3
            # print("negative_embds:",negative_embds.shape[0])
            # remove bos and eos in positive embeddings
            stop_mask = torch.isin(positive_ids,
                                torch.tensor(special_ids).cuda()).cuda()  # (bs, 77)
            zero_mask=torch.isin(positive_ids,
                                torch.tensor([0]).cuda()).cuda()
            stop_mask=stop_mask | zero_mask
            positive_embeds = positive_embeds[
                ~stop_mask]  # (num_positive_tokens, 768), where num_positive_tokens = num_positives * bs
            # print("pos_stop_mask",stop_mask)
            # stack positives and negatives as a pn_block
            # print("positive_embeds:",positive_embeds.shape[0])

            
            pn_embeds = torch.cat([positive_embeds, negative_embds], dim=0)
            pn_embeds_normalized = F.normalize(
                pn_embeds, p=2,
                dim=1)  # (num_positive_tokens+num_negative_tokens, 768)

        # compute relation embeds <R>
        relation_mask = (input_ids[b_idx] == placeholder_token_id)  # (77)
        # print("placeholder_token_id:",placeholder_token_id)
        # print("relation_mask:",relation_mask)
        relation_embeds = inputs_embeds[b_idx][relation_mask]  # (1, 1024)
        relation_embeds_normalized = F.normalize(relation_embeds, p=2, dim=1)
        # print("relation_embeds:",relation_embeds.shape)
        # compute Multi-Instance InfoNCE loss
        logits = torch.einsum('nc,mc->nm',
                            [relation_embeds_normalized, pn_embeds_normalized
                            ])  # (1, num_positive_tokens+num_negative_tokens)

        logits /= temperature
        nominator = torch.logsumexp(logits[:, :positive_embeds.shape[0]], dim=1)
        denominator = torch.logsumexp(logits, dim=1)
        # print("denominator ,nominator",denominator,nominator)
        # loss_list.append()
        cnt+=1
        batch_loss+=torch.mean(denominator - nominator)
        # batch_loss_list.append(loss_list)
    batch_loss=batch_loss/cnt
    # exit(0)
    return batch_loss



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
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument("--num_positives", type=int,default=4)
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo at the beach",
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
        "--null_loss_weight",
        type=float,
        default=0,
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
    parser.add_argument("--beta_dpo", type=float, default=1000, help="The beta DPO temperature controlling strength of KL penalty")
    # parser.add_argument(
    #     "--save_interpolation",
    #     default=,
    #     help=(
    #         "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
    #         " cropped. The images will be resized to the resolution first before cropping."
    #     ),
    # )

    parser.add_argument(
        "--scaled_cosine_alpha",
        type=float,
        default=0.5,
        help="The skewness (alpha) of the Importance Sampling Function",
    )
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
        "--importance_sampling",
        action="store_true",
        help="A folder containing the training data of instance images.",
    )

    parser.add_argument(
        "--with_kl",
        action="store_true",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--kl_loss_weight",
        type=float,
        default=0,
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # parser.add_argument(
    #     "--kl_batch",
    #     type=int,
    #     default=10,
    #     help=(
    #         "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
    #         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    #     ),
    # )
    parser.add_argument("--steer_loss_weight",type=float,default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default="0.07",
        help="Temperature parameter for L_steer",
    )

    parser.add_argument(
        "--norm_loss_weight",
        type=float,
        default=0,
        help="Temperature parameter for L_steer",
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
        "--relation_words_list",
        type=str,
        nargs="*",
        # action="append",
        default=[],
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
        "--dataset_length",
        type=int,
        default=0,
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--not_use_instance_class_suffix",
        action="store_false",
        dest="use_instance_class_suffix",
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--instance_class",
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

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args


def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_prompt,
        instance_data_root,
        tokenizer,
        placeholder,
        class_prompt=None,
        size=512,
        center_crop=False,
        flip_p=1,
        length=0,
        relation_words_list=[],
        num_positives=1
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p
        self.placeholder=placeholder
        self.num_positives=num_positives

        self.relation_words_list= relation_words_list
        assert isinstance(self.relation_words_list,list)
        
        # print(size)
        print("sxc:relation_words_list and placeholder:",relation_words_list,placeholder)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (size,size), interpolation=transforms.InterpolationMode.BILINEAR
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

        # print(len(self.instance_image),self.instance_image[0].shape,"sxc")

        # self.instance_masks = []
        # for i in range(num_of_assets):
        #     instance_mask_path = os.path.join(instance_data_root, f"mask{i}.png")
        #     curr_mask = Image.open(instance_mask_path)
        #     curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
        #     self.instance_masks.append(curr_mask)
        # self.instance_masks = torch.cat(self.instance_masks)

        self._length = len(self.instance_image) if length==0 else length

        self.num_instance_images=self._length
        if isinstance(instance_prompt,str):
            self.instance_prompt=[instance_prompt]*self._length

        assert len(self.instance_prompt)==self._length


        self.class_prompt = class_prompt
        # if class_data_root is not None:
        #     self.class_data_root = Path(class_data_root)
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
        example["instance_prompt"]= self.instance_prompt[index%self.num_instance_images]
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

        example["class_prompt_ids"] = self.tokenizer(
            self.class_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        example["null_prompt_ids"] = self.tokenizer(
            "",
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if len(self.relation_words_list) > 0:
            # print(self.relation_words_list,self.placeholder_tokens)
            # example["token_ids"]=[]
            example["positive_ids"]=[]
            example["positive_prompt"] = []
            relation_words=self.relation_words_list
            positive_words = random.sample(
                relation_words, k=min(self.num_positives,len(self.relation_words_list)))
            positive_words_string = " ".join(positive_words)
            example["token_ids"] = torch.tensor(
                self.tokenizer.encode(self.placeholder)[1:-1])
            # print(self.tokenizer.decode(self.tokenizer.encode(self.placeholder)[1:-1]))
            # exit(0)
            
            # print(self.tokenizer.encode(self.placeholder),example["token_ids"],"sxc_hh")
            # exit(0)
            # example["token_ids"].append(self.tokenizer(
            #     positive_words_string,
            #     padding="max_length",
            #     truncation=True,
            #     max_length=self.tokenizer.model_max_length,
            #     return_tensors="pt",
            # ).input_ids[0])
            example["positive_ids"]=(self.tokenizer(
                positive_words_string,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0])
            example["positive_prompt"]=(positive_words_string)

        return example


def collate_fn(examples, use_postive_ids=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    instance_prompts=[example["instance_prompt"] for example in examples]
    # masks = [example["instance_masks"] for example in examples]
    # token_ids = [example["token_ids"] for example in examples]

    class_ids = [example["class_prompt_ids"] for example in examples]
    null_prompt_ids = [example["null_prompt_ids"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()


    input_ids = torch.cat(input_ids, dim=0)
    class_ids = torch.cat(class_ids, dim=0)
    null_prompt_ids=torch.cat(null_prompt_ids,dim=0)
    # masks = torch.stack(masks)
    # token_ids = torch.stack(token_ids)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_prompt":instance_prompts,
        "class_ids":class_ids,
        "null_ids":null_prompt_ids
        # "instance_masks": masks,
        # "token_ids": token_ids,
    }
    if use_postive_ids:
        token_ids=[example["token_ids"] for example in examples]
        # token_ids = torch.stack(token_ids)

        positive_ids=[example["positive_ids"] for example in examples]
        positive_prompt=[example["positive_prompt"] for example in examples]
        # print("sxc2:token_ids:",token_ids,positive_ids,positive_prompt)
        # exit(0)
        batch.update({
            "token_ids":token_ids,
            "positive_ids":positive_ids,
            "positive_prompt":positive_prompt
        })
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


class SpatialDreambooth:
    def __init__(self):
        self.args = parse_args()
        self.generator=torch.Generator('cuda').manual_seed(self.args.seed)
        self.main()

    def main(self):
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

        # Generate class images if prior preservation is enabled.
        if self.args.with_prior_preservation:
            class_images_dir = Path(self.args.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.args.num_class_images:
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

                num_new_images = self.args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
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

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir
                            / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
        )
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )

        self.ref_text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )

        self.ref_text_encoder.requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )
        self.ref_unet=UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )
        self.ref_unet.requires_grad_(False)
        

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
        self.args.instance_prompt = f"a photo of {self.placeholder_tokens} {self.args.instance_class}" if self.args.use_instance_class_suffix else f"a photo of {self.placeholder_tokens}"
        print("sxc:instance_prompt:",self.args.instance_prompt)
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
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            # if len(self.args.initializer_tokens) > 0:
            #     # Use initializer tokens
            #     token_embeds = self.text_encoder.get_input_embeddings().weight.data
            #     for tkn_idx, initializer_token in enumerate(self.args.initializer_tokens):
            #         curr_token_ids = self.tokenizer.encode(
            #             initializer_token, add_special_tokens=False
            #         )
            #         # assert (len(curr_token_ids)) == 1
            #         token_embeds[self.placeholder_token_ids[tkn_idx]] = token_embeds[
            #             curr_token_ids[0]
            #         ]
            # else:
                # Initialize new tokens randomly
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
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
        self.unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        if self.args.train_token:
            self.text_encoder.requires_grad_(False)
        else:
            self.text_encoder.text_model.encoder.requires_grad_(False)
            self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
            self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

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

        assert (self.args.steer_loss_weight>0 and len(self.args.relation_words_list)>0) or (self.args.steer_loss_weight==0 and len(self.args.relation_words_list)==0)
        # We start by only optimizing the embeddings
        optimizer=None
        lr_scheduler=None
        if self.args.train_token:
            params_to_optimize = self.text_encoder.get_input_embeddings().parameters()
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
        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_prompt=self.args.instance_prompt,
            instance_data_root=self.args.instance_data_dir,
            class_prompt=self.args.class_prompt,
            placeholder=self.args.placeholder_token,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
            length=self.args.dataset_length,
            relation_words_list=self.args.relation_words_list if self.args.steer_loss_weight>0 else [],
            num_positives=self.args.num_positives
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(
                examples, self.args.steer_loss_weight>0 and len(self.args.relation_words_list)>0
            ),
            num_workers=self.args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True


        
        if optimizer is not None:
            
            (
                self.unet,
                self.text_encoder,
                optimizer,
                train_dataloader,
                lr_scheduler,
                self.ref_text_encoder,
                self.ref_unet
            ) = self.accelerator.prepare(
                self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler,self.ref_text_encoder,self.ref_unet
            )
        else:
            (
                self.unet,
                self.text_encoder,
                train_dataloader,
                self.ref_text_encoder,
                self.ref_unet,
            ) = self.accelerator.prepare(
                self.unet, self.text_encoder, train_dataloader,self.ref_text_encoder,self.ref_unet
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
        # self.ref_unet.to(self.accelerator.device, dtype=self.weight_dtype)
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if (
            self.args.train_text_encoder
            and self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        # if len(self.args.initializer_tokens) > 0:
        #     # Only for logging
        #     self.args.initializer_tokens = ", ".join(self.args.initializer_tokens)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth", config={})

        # Train
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
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
                self.accelerator.unwrap_model(self.text_encoder)
                .get_input_embeddings()
                .weight.data.clone()
            )

        # Create attention controller
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)

        if self.args.importance_sampling:
            print("Using Relation-Focal Importance Sampling")
            list_of_candidates = [
                x for x in range(self.noise_scheduler.config.num_train_timesteps)
            ]
            prob_dist = [
                importance_sampling_fn(x,
                                    self.noise_scheduler.config.num_train_timesteps,
                                    self.args.scaled_cosine_alpha)
                for x in list_of_candidates
            ]
            prob_sum = 0
            # normalize the prob_list so that sum of prob is 1
            for i in prob_dist:
                prob_sum += i
            prob_dist = [x / prob_sum for x in prob_dist]

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                if self.args.phase1_train_steps == global_step:
                    self.unet.requires_grad_(True)
                    if self.args.train_text_encoder:
                        self.text_encoder.requires_grad_(True)
                    unet_params = self.unet.parameters()

                    if self.args.train_token:
                        base_params=itertools.chain(unet_params, self.text_encoder.get_input_embeddings().parameters())
                    else:
                        base_params=unet_params
                    params_to_optimize = (
                        itertools.chain(unet_params, self.text_encoder.parameters())
                        if self.args.train_text_encoder
                        else base_params
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

                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["pixel_values"].to(dtype=self.weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    null_timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    null_timesteps = null_timesteps.long()
                    # null_timesteps = timesteps

                    null_noise = torch.randn_like(latents)
                    # null_noise = noise
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    null_noisy_latents = self.noise_scheduler.add_noise(
                        latents, null_noise, null_timesteps
                    )

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    null_encoder_hidden_states = self.text_encoder(batch["null_ids"])[0]

                    assert self.args.with_kl
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    cls_encoder_hidden_states = self.text_encoder(batch["class_ids"])[0]
                    cls_kl_encoder_hidden_states = self.ref_text_encoder(batch["class_ids"])[0]

                    null_ref_encoder_hidden_states = self.ref_text_encoder(batch["null_ids"])[0]

                    if self.args.importance_sampling:
                        kl_ref_timesteps = np.random.choice(
                            list_of_candidates,
                            size=bsz,
                            replace=True,
                            p=prob_dist)
                        kl_ref_timesteps = torch.tensor(kl_ref_timesteps,device=latents.device)
                    else:
                        kl_ref_timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        )
                    kl_ref_timesteps = kl_ref_timesteps.long()
                    shape=latents.shape
                    # shape[0]=self.args.kl_batch
                    
                    kl_ref_model_output,kl_ref_sample=self._model_inference_from_scratch(shape,self.noise_scheduler.timesteps[0],cls_kl_encoder_hidden_states,self.ref_unet)
                    # last_latents=self.last_forward(batch["instance_prompt"].chunk(2)[0])
                    # print(self.noise_scheduler.timesteps)
                    kl_ref_initial_latents=self._forward(kl_ref_model_output,kl_ref_sample,(self.noise_scheduler.timesteps[0]))

                    kl_ref_noise=torch.randn_like(kl_ref_initial_latents)
                    kl_ref_noisy_latents = self.noise_scheduler.add_noise(
                        kl_ref_initial_latents, kl_ref_noise, kl_ref_timesteps
                    )

                    # self_model_pred=self.unet(
                    #     ref_noisy_latents, ref_timesteps, cls_encoder_hidden_states
                    # ).sample
                    ord_noisy_latents,ord_timesteps,ord_encoder_hidden_states=noisy_latents,timesteps,encoder_hidden_states
                    noisy_latents=torch.cat([noisy_latents,kl_ref_noisy_latents,null_noisy_latents])
                    timesteps=torch.cat([timesteps,kl_ref_timesteps,null_timesteps])
                    encoder_hidden_states=torch.cat([encoder_hidden_states,cls_encoder_hidden_states,null_encoder_hidden_states])
                    # ord_timesteps,kl_ref_timesteps,null_timesteps=timesteps.chunk(3)
                    # Predict the noise residual
                    model_pred = self.unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample
                    kl_ref_model_pred, null_ref_model_pred= self.ref_unet(
                        torch.cat([kl_ref_noisy_latents,null_noisy_latents]), 
                        torch.cat([kl_ref_timesteps, null_timesteps]), 
                        torch.cat([cls_kl_encoder_hidden_states,null_ref_encoder_hidden_states])
                    ).sample.chunk(2)

                    model_pred,kl_model_pred,null_model_pred=model_pred.chunk(3)
                    
                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target,kl_ref_target,null_target = noise,kl_ref_noise,null_noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        
                        target,kl_ref_target,null_target = self.noise_scheduler.get_velocity(
                            latents, noise, ord_timesteps
                        ),
                        self.noise_scheduler.get_velocity(
                            kl_ref_initial_latents, kl_ref_noise, kl_ref_timesteps
                        ),
                        self.noise_scheduler.get_velocity(
                            latents, null_noise, null_timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                    # Compute prior loss
                    kl_loss = F.mse_loss(
                        kl_model_pred.float(),
                        # kl_ref_model_pred.float(),
                        kl_ref_target.float(),
                        reduction="mean",
                    )
                    # null_loss=F.mse_loss(
                    #     null_model_pred.float(),
                    #     # kl_ref_model_pred.float(),
                    #     null_target.float(),
                    #     reduction="mean", 
                    # )

                    # null_loss=F.mse_loss(
                    #     model_pred.float(),
                    #     null_model_pred.float().detach(),
                    #     # kl_ref_model_pred.float(),
                    #     # null_target.float(),
                    #     reduction="mean", 
                    # )
                    null_loss=F.mse_loss(
                        # model_pred.float(),
                        null_ref_model_pred.float().detach(),
                        null_model_pred.float(),
                        # kl_ref_model_pred.float(),
                        # null_target.float(),
                        reduction="mean", 
                    )

                    logs["null_loss"]=null_loss.detach().item()

                    weighted_steer_loss=0

                    if self.args.steer_loss_weight!=0:
                        stop_words= ["a", "the"]
                        expanded_stop_words = stop_words
                        
                        placeholder_token_ids_list=batch["token_ids"]
                        # print("placeholder_token_ids_list:",placeholder_token_ids_list)
                        stop_ids = self.tokenizer(
                            " ".join(expanded_stop_words),
                            truncation=False,
                            return_tensors="pt",
                        ).input_ids[0].detach().tolist()
                        # print("stop_id:",stop_ids)

                        special_ids = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
                        # print("special_id:",special_ids)
                        # print("positive_prompt:",batch["positive_prompt"])
                        # print("input_ids:",batch["input_ids"])
                        batch_steer_loss= calculate_steer_loss(
                            self.text_encoder.get_input_embeddings(),
                            batch["input_ids"],
                            placeholder_token_ids_list,
                            stop_ids,
                            special_ids,
                            batch["positive_ids"],
                            temperature=self.args.temperature)

                        logs["steer_loss"]=batch_steer_loss.detach().item()
                        weighted_steer_loss = self.args.steer_loss_weight * batch_steer_loss
                    # Add the prior loss to the instance loss.

                    norm_loss=0
                    if self.args.norm_loss_weight>0:
                        assert len(self.args.instance_class)>0
                        token_embeds = self.text_encoder.get_input_embeddings()
                        instance_class_ids=torch.tensor(self.tokenizer.encode(
                            self.args.instance_class, add_special_tokens=False
                        ),device=self.accelerator.device)
                        placeholder_ids=torch.tensor(self.tokenizer.encode(
                            self.args.placeholder_token, add_special_tokens=False
                        ),device=self.accelerator.device)
                        # print(instance_class_ids,placeholder_ids)
                        placeholder_embedds,instance_class_embedds=token_embeds(placeholder_ids),token_embeds(instance_class_ids)
                        # print(placeholder_embedds,instance_class_embedds)
                        norm_loss=(placeholder_embedds.norm()-instance_class_embedds.norm().detach())**2
                        logs["norm_loss"]=norm_loss
                        # print(norm_loss,norm_loss.requires_grad,placeholder_embedds.requires_grad,placeholder_embedds.norm(),instance_class_embedds.norm())
                        # exit(0)


                    loss = loss + self.args.kl_loss_weight * kl_loss+weighted_steer_loss+null_loss*self.args.null_loss_weight+self.args.norm_loss_weight*norm_loss


                    # Attention loss
                    # print(self.unet.requires_grad)

                    self.accelerator.backward(loss)

                    # No need to keep the attention store
                    self.controller.attention_store = {}
                    self.controller.cur_step = 0

                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                self.unet.parameters(), self.text_encoder.parameters()
                            )
                            if self.args.train_text_encoder
                            else self.unet.parameters()
                        )
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
                                self.text_encoder
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
                            self.args.instance_prompt,
                            attention_maps=full_agg_attn.detach().cpu(),
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_full_attn.jpg"
                            ),
                        )
                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                logs["loss"] = loss.detach().item()
                logs["kl_loss"] = kl_loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break

        self.save_pipeline(self.args.output_dir)

        self.accelerator.end_training()

    
    def _model_inference_from_scratch(self,shape,timestep,prompt_embeds,model,latents=None):
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
        noise_pred = model(latents, timestep, encoder_hidden_states=prompt_embeds).sample

        # # perform guidance
        # if do_classifier_free_guidance:
        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        return noise_pred,latents

    def _infer_from_T_to_t(self,pred_original_sample,sample,timestep_t,timestep_T):
        def compute_alpha_bar(t):
            return self.noise_scheduler.alphas_cumprod[t]
        def compute_variance(t):
            if self.noise_scheduler.variance_type == "fixed_small_log":
                variance = self.noise_scheduler._get_variance(t, predicted_variance=None)**2
            else:
                variance = self.noise_scheduler._get_variance(t, predicted_variance=None)
            return variance
        alpha_bar_t,alpha_bar_T=compute_alpha_bar(timestep_t),compute_alpha_bar(timestep_T)
        alpha_bar_T_div_t=alpha_bar_t/alpha_bar_T
        variance_T,variance_t=compute_variance(timestep_T),compute_variance(timestep_t)
        variance_T_div_t=variance_T-alpha_bar_T_div_t*variance_t
        variance=variance_T_div_t*variance_t/variance_T

        mu=(alpha_bar_T_div_t*variance_t/variance_T)*sample+(alpha_bar_t*variance_T_div_t/variance_T)*pred_original_sample
            
        epsilon=torch.randn_like(pred_original_sample)

        return  mu+epsilon*variance

    

    def _forward(self,model_output,sample,timestep):
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
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                revision=self.args.revision,
            )
            pipeline.save_pretrained(path)

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
            [self.args.instance_prompt],
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
