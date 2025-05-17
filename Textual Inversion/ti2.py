import argparse
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
# Removed Hub utilities (load_or_create_model_card, populate_model_card)
from diffusers.utils.import_utils import is_xformers_available

# Manual WandB import if needed
# if is_wandb_available():
# import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

check_min_version("0.33.0.dev0") # Or your desired diffusers version

logger = logging.getLogger(__name__) # Standard Python logger

# Removed save_model_card as it's Hub-specific

def log_validation(text_encoder, tokenizer, unet, vae, args, device, weight_dtype, epoch):
    """
    Logs validation images.
    Args:
        text_encoder: The text encoder model.
        tokenizer: The tokenizer.
        unet: The UNet model.
        vae: The VAE model.
        args: The command line arguments.
        device: The device to run on (e.g., 'cuda', 'cpu').
        weight_dtype: The dtype for model weights (e.g., torch.float16).
        epoch: The current epoch number.
    Returns:
        A list of PIL.Image.Image objects.
    """
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # Create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder, # No unwrap needed
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference
    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    images = []
    autocast_device_type = device.type # 'cuda' or 'cpu' or 'mps'
    if autocast_device_type == 'mps': # MPS has limited autocast support
        autocast_ctx = nullcontext()
    else:
        # Only use weight_dtype for autocast if mixed precision is enabled and on CUDA
        autocast_dtype = weight_dtype if args.mixed_precision != "no" and autocast_device_type == 'cuda' else torch.float32
        autocast_ctx = torch.autocast(device_type=autocast_device_type, dtype=autocast_dtype, enabled=(args.mixed_precision != "no"))


    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    # Manual logging to TensorBoard/WandB would go here if desired
    # Example:
    # if args.report_to == "tensorboard" and writer is not None: # Assuming 'writer' is your SummaryWriter
    #     np_images = np.stack([np.asarray(img) for img in images])
    #     writer.add_images("validation", np_images, epoch, dataformats="NHWC")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images


def save_progress(text_encoder, placeholder_token_ids, args, save_path, safe_serialization=True):
    """
    Saves the learned embeddings.
    Args:
        text_encoder: The text encoder model.
        placeholder_token_ids: List of IDs for the placeholder tokens.
        args: Command line arguments.
        save_path: Path to save the embeddings.
        safe_serialization: Whether to use safetensors.
    """
    logger.info(f"Saving embeddings to {save_path}")
    # Ensure placeholder_token_ids is not empty before trying to access min/max
    if not placeholder_token_ids:
        logger.error("placeholder_token_ids is empty. Cannot save progress.")
        return

    learned_embeds = (
        text_encoder # No unwrap needed
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] # Slicing the embeddings
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    try:
        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, save_path)
    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")


def set_manual_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a Textual Inversion training script.")
    parser.add_argument(
        "--save_steps", type=int, default=1200, help="Save learned_embeds.bin every X updates steps."
    )
    parser.add_argument(
        "--save_as_full_pipeline", action="store_true", help="Save the complete stable diffusion pipeline."
    )
    parser.add_argument(
        "--num_vectors", type=int, default=1, help="How many textual inversion vectors to learn."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier."
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Revision of pretrained model identifier."
    )
    parser.add_argument(
        "--variant", type=str, default=None, help="Variant of the model files (e.g., fp16)."
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path."
    )
    parser.add_argument(
        "--train_data_dir", type=str, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token", type=str, required=True, help="A token to use as a placeholder for the concept."
    )
    parser.add_argument(
        "--initializer_token", type=str, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", choices=["object", "style"], help="Choose between 'object' and 'style'.")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir", type=str, default="text-inversion-model", help="The output directory for models and checkpoints."
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution", type=int, default=512, help="Input image resolution."
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument(
        "--max_train_steps", type=int, default=3600, help="Total number of training steps. Overrides num_train_epochs."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients."
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5.0e-04, help="Initial learning rate."
    )
    parser.add_argument(
        "--scale_lr", action="store_true", default=True, help="Scale LR by grad_acc_steps * batch_size."
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type."
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of warmup steps for the LR scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine_with_restarts scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0, help="Number of subprocesses for data loading."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="AdamW epsilon.")
    # Hub related arguments are parsed but their functionality is removed.
    parser.add_argument("--push_to_hub", action="store_true", help="[INFO] Push to Hub functionality is removed in this version.")
    parser.add_argument("--hub_token", type=str, default=None, help="[INFO] Hub token is not used in this version.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="[INFO] Hub model ID is not used in this version.")
    parser.add_argument(
        "--logging_dir", type=str, default="logs", help="TensorBoard log directory (manual setup required)."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision training."
    )
    parser.add_argument(
        "--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs."
    )
    parser.add_argument(
        "--report_to", type=str, default="tensorboard", help="Reporting integration (manual setup required)."
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="Prompt for validation."
    )
    parser.add_argument(
        "--num_validation_images", type=int, default=4, help="Number of validation images to generate."
    )
    parser.add_argument(
        "--validation_steps", type=int, default=100, help="Run validation every X steps."
    )
    parser.add_argument(
        "--validation_epochs", type=int, default=None, help="Deprecated. Use validation_steps."
    )
    parser.add_argument(
        "--checkpointing_steps", type=int, default=500, help="Save a checkpoint every X steps."
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Resume from a checkpoint."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization", action="store_true", help="Save embeddings in .bin instead of .safetensors."
    )

    args = parser.parse_args()

    if args.train_data_dir is None:
        raise ValueError("You must specify a --train_data_dir.")

    if args.push_to_hub:
        logger.warning("--push_to_hub is set, but Hub uploading functionality has been removed in this script version.")


    return args


imagenet_templates_small = ["{}"] # Simplified templates
imagenet_style_templates_small = ["{}"] # Simplified templates


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if os.path.isfile(os.path.join(self.data_root, file_path))]
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_root}. Please check the directory and image extensions.")


        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        try:
            image = Image.open(self.image_paths[i % self.num_images])
        except FileNotFoundError:
            logger.error(f"Image not found: {self.image_paths[i % self.num_images]}")
            # Return a dummy example or raise an error
            # For now, let's try to load the next one or raise error if persistent
            if self.num_images > 0:
                 return self.__getitem__((i + 1) % self._length) # Risky, could lead to infinite loop if all fail
            else:
                 raise RuntimeError("No valid images could be loaded from the dataset.")


        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32) # Normalize to [-1, 1]

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Running with arguments: {args}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Handle mixed precision
    use_amp = False
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        use_amp = True if device.type == 'cuda' else False # AMP for CUDA only
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        use_amp = True if device.type == 'cuda' and torch.cuda.is_bf16_supported() else False
        if use_amp is False and args.mixed_precision == "bf16":
            logger.warning("BF16 specified but not supported or not on CUDA. Falling back to FP32.")
            args.mixed_precision = "no" # Update args to reflect actual precision
    else:
        weight_dtype = torch.float32

    if use_amp and device.type != 'cuda':
        logger.warning(f"Mixed precision ({args.mixed_precision}) is enabled but device is {device.type}. AMP will be disabled.")
        use_amp = False
        weight_dtype = torch.float32


    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Set verbosity for transformers and diffusers
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Set seed
    if args.seed is not None:
        set_manual_seed(args.seed)
        logger.info(f"Set seed for reproducibility: {args.seed}")

    # Create output directory
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    else:
        raise ValueError("Either --tokenizer_name or --pretrained_model_name_or_path must be provided.")


    # Load models
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
    except Exception as e:
        logger.error(f"Error loading pretrained models: {e}")
        raise

    # Add placeholder tokens
    placeholder_tokens = [args.placeholder_token]
    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be >= 1, but is {args.num_vectors}")
    for i in range(1, args.num_vectors):
        placeholder_tokens.append(f"{args.placeholder_token}_{i}")

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        logger.warning(
            f"The tokenizer already contained some of the placeholder tokens. Number of actually new tokens: {num_added_tokens}."
            " This might lead to unexpected behavior if the tokens were already in use."
            " Consider using truly unique placeholder tokens."
        )
        # It's not necessarily an error to continue if some tokens existed,
        # but the user should be aware. If strict newness is required:
        # raise ValueError(
        #     f"The tokenizer already contains some of the tokens {placeholder_tokens}. Please pass different placeholder tokens."
        # )


    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer_token must be a single token.")
    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            if 0 <= token_id < len(token_embeds) and 0 <= initializer_token_id < len(token_embeds):
                 token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            else:
                logger.error(f"Token ID out of bounds during initialization. Token ID: {token_id}, Initializer ID: {initializer_token_id}, Embeds length: {len(token_embeds)}")


    # Freeze VAE and UNet, and parts of Text Encoder
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing() # UNet is not trained, but if it were, this would be relevant.

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            try:
                unet.enable_xformers_memory_efficient_attention()
                if hasattr(vae, 'enable_xformers_memory_efficient_attention'): # Some VAEs might support it
                    vae.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers memory efficient attention: {e}")
        else:
            logger.warning("xformers is not available. Memory efficient attention disabled.")

    if args.allow_tf32 and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizer
    if args.scale_lr:
        effective_learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size
    else:
        effective_learning_rate = args.learning_rate
    
    logger.info(f"Effective learning rate: {effective_learning_rate}")

    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(), # Only optimize embeddings
        lr=effective_learning_rate, # Use the potentially scaled LR
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoader
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
        pin_memory=True if device.type == 'cuda' else False # Pin memory for faster CUDA transfers
    )

    if args.validation_epochs is not None:
        warnings.warn(
            "FutureWarning: --validation_epochs is deprecated. Use --validation_steps.",
            FutureWarning,
            stacklevel=2,
        )
        # Assuming average steps per epoch for conversion if dataset length is known
        if len(train_dataset) > 0 and args.gradient_accumulation_steps > 0 :
             steps_per_epoch = math.ceil(len(train_dataset) / (args.train_batch_size * args.gradient_accumulation_steps))
             args.validation_steps = args.validation_epochs * steps_per_epoch
             logger.info(f"Converted validation_epochs to validation_steps: {args.validation_steps}")


    # Scheduler and training steps calculation
    overrode_max_train_steps = False
    if len(train_dataloader) == 0:
        raise ValueError("Train dataloader is empty. Check your dataset and batch size.")
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None or args.max_train_steps <= 0 : # Handle if not provided or invalid
        if args.num_train_epochs <= 0:
            raise ValueError("Either --max_train_steps or --num_train_epochs (positive) must be specified.")
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        logger.info(f"max_train_steps not provided, calculating from num_train_epochs: {args.max_train_steps}")


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, # No scaling by num_processes
        num_training_steps=args.max_train_steps, # No scaling by num_processes
        num_cycles=args.lr_num_cycles,
    )

    # Move models to device
    text_encoder.to(device)
    # VAE and UNet are used for inference, can be cast to weight_dtype if using mixed precision on CUDA
    vae_dtype = weight_dtype if use_amp and device.type == 'cuda' else torch.float32
    unet_dtype = weight_dtype if use_amp and device.type == 'cuda' else torch.float32
    vae.to(device, dtype=vae_dtype)
    unet.to(device, dtype=unet_dtype)


    if overrode_max_train_steps: # Recalculate epochs if max_train_steps was derived
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Training details logging
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        checkpoint_to_load = None
        if args.resume_from_checkpoint == "latest":
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                checkpoint_to_load = os.path.join(args.output_dir, dirs[-1], "training_state.bin")
        else:
            checkpoint_to_load = os.path.join(args.resume_from_checkpoint, "training_state.bin") # Assuming full path to dir

        if checkpoint_to_load and os.path.isfile(checkpoint_to_load):
            logger.info(f"Resuming from checkpoint {checkpoint_to_load}")
            try:
                checkpoint = torch.load(checkpoint_to_load, map_location=device)
                text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                if use_amp and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                
                global_step = checkpoint.get("global_step", 0)
                initial_global_step = global_step
                # first_epoch = checkpoint.get("epoch", 0) + 1 # Resume from next epoch
                # More robust epoch calculation based on global_step and num_update_steps_per_epoch
                if num_update_steps_per_epoch > 0:
                     first_epoch = global_step // num_update_steps_per_epoch
                else: # Should not happen if dataloader is not empty
                     first_epoch = checkpoint.get("epoch", 0)

                logger.info(f"  Resumed global_step = {global_step}, starting from epoch = {first_epoch}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Starting a new training run.")
                args.resume_from_checkpoint = None # Reset to avoid retry
                initial_global_step = 0
                global_step = 0
                first_epoch = 0
        else:
            logger.warning(
                f"Checkpoint '{args.resume_from_checkpoint}' not found or invalid. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0


    progress_bar = tqdm(
        range(initial_global_step, args.max_train_steps), # Start from initial_global_step
        desc="Steps",
        disable=False, # Always show progress bar
    )

    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone().to(device) # Ensure it's on the correct device

    # Training loop
    text_encoder.train() # Set text_encoder to train mode
    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss = 0.0
        num_batches_in_epoch = 0
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            try:
                pixel_values = batch["pixel_values"].to(device=device) # VAE input dtype handled by VAE
                input_ids = batch["input_ids"].to(device=device)
            except Exception as e:
                logger.error(f"Error moving batch to device or accessing batch items: {e}. Skipping batch.")
                continue


            # Autocast for mixed precision
            autocast_dtype_train = weight_dtype if use_amp else torch.float32
            with torch.autocast(device_type=device.type, dtype=autocast_dtype_train, enabled=use_amp):
                # VAE and UNet are in eval mode and use no_grad for their parts.
                # Their dtypes (vae_dtype, unet_dtype) are set when moved to device.
                with torch.no_grad(): # VAE encoding is correctly done with no_grad
                    # Ensure pixel_values are compatible with vae_dtype if vae is in lower precision
                    latents = vae.encode(pixel_values.to(vae_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Text encoder forward pass (trainable part)
                encoder_hidden_states = text_encoder(input_ids)[0] # Output is float32 by default from CLIP
                
                # Predict noise with UNet (UNet input should match unet_dtype)
                # Ensure encoder_hidden_states is cast if unet_dtype is lower precision
                # REMOVED `with torch.no_grad():` from here
                model_pred = unet(noisy_latents.to(unet_dtype), timesteps, encoder_hidden_states.to(unet_dtype)).sample

                # Calculate loss (typically in float32 for stability)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Accumulate gradients
            actual_loss = loss / args.gradient_accumulation_steps
            
            if use_amp:
                scaler.scale(actual_loss).backward()
            else:
                actual_loss.backward()
            
            epoch_loss += loss.detach().item()
            num_batches_in_epoch +=1

            # Optimizer step
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                if use_amp:
                    scaler.unscale_(optimizer) # Unscale before clipping, if any
                    # Optional: torch.nn.utils.clip_grad_norm_(text_encoder.get_input_embeddings().parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Optional: torch.nn.utils.clip_grad_norm_(text_encoder.get_input_embeddings().parameters(), 1.0)
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Reset non-placeholder token embeddings
                with torch.no_grad():
                    current_embeds = text_encoder.get_input_embeddings().weight
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool, device=current_embeds.device)
                    if placeholder_token_ids: # Ensure not empty
                        min_id = min(placeholder_token_ids)
                        max_id = max(placeholder_token_ids)
                        # Check bounds carefully
                        if 0 <= min_id < len(tokenizer) and 0 <= max_id < len(tokenizer) and min_id <= max_id:
                             index_no_updates[min_id : max_id + 1] = False
                             current_embeds[index_no_updates] = orig_embeds_params[index_no_updates]
                        else:
                             logger.warning(f"Invalid placeholder token ID range ({min_id}-{max_id}) for tokenizer size {len(tokenizer)}. Skipping embedding reset.")


                # Corresponds to accelerator.sync_gradients block
                progress_bar.update(1)
                global_step += 1

                # Logging
                current_lr = lr_scheduler.get_last_lr()[0]
                logs = {"loss": loss.detach().item(), "lr": current_lr} # Log instantaneous loss before accumulation division
                progress_bar.set_postfix(**logs)
                # Manual logging (e.g., to console or file)
                if global_step % 100 == 0: # Log every 100 steps
                    logger.info(f"Epoch {epoch}, Step {global_step}: Loss: {logs['loss']:.4f}, LR: {logs['lr']:.6e}")


                # Save learned embeddings
                if global_step % args.save_steps == 0:
                    save_embed_filename = (
                        f"learned_embeds-steps-{global_step}.bin"
                        if args.no_safe_serialization
                        else f"learned_embeds-steps-{global_step}.safetensors"
                    )
                    save_embed_path = os.path.join(args.output_dir, save_embed_filename)
                    save_progress(
                        text_encoder, placeholder_token_ids, args, save_embed_path,
                        safe_serialization=not args.no_safe_serialization
                    )

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Manage total checkpoints
                    if args.checkpoints_total_limit is not None:
                        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            for ckpt_to_remove_name in checkpoints[:num_to_remove]:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt_to_remove_name))
                                logger.info(f"Removed old checkpoint: {ckpt_to_remove_name}")
                    
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'text_encoder_state_dict': text_encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if use_amp else None,
                        'args': args # Save args for easier resumption
                    }, os.path.join(checkpoint_dir, "training_state.bin"))
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

                # Validation
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    logger.info(f"Running validation at step {global_step}")
                    # Ensure models are in eval mode for validation if they were changed
                    text_encoder.eval()
                    vae.eval()
                    unet.eval()
                    
                    validation_images = log_validation(
                        text_encoder, tokenizer, unet, vae, args, device, weight_dtype, epoch # Use the overall weight_dtype for pipeline
                    )
                    # Save validation images (optional)
                    for i, img in enumerate(validation_images):
                        img.save(os.path.join(args.output_dir, f"validation_step_{global_step}_img_{i}.png"))
                    
                    text_encoder.train() # Return text_encoder to train mode
                    # VAE and UNET are not trained, so they can stay in eval or be set explicitly if needed by other parts.

            if global_step >= args.max_train_steps:
                break
        
        avg_epoch_loss = epoch_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else 0.0
        logger.info(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.4f}")

        if global_step >= args.max_train_steps:
            logger.info("Reached max_train_steps. Stopping training.")
            break
    
    progress_bar.close()

    # Save final model and embeddings
    logger.info("Saving final model and embeddings.")
    if args.save_as_full_pipeline:
        # Ensure models are on CPU for saving pipeline, or handle device mapping in from_pretrained/save_pretrained
        text_encoder_cpu = text_encoder.to('cpu')
        vae_cpu = vae.to('cpu') # vae might have been cast to weight_dtype
        unet_cpu = unet.to('cpu') # unet might have been cast to weight_dtype

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, # Base pipeline structure
            text_encoder=text_encoder_cpu,
            vae=AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant).to('cpu'), # Fresh VAE on CPU
            unet=UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant).to('cpu'), # Fresh UNet on CPU
            tokenizer=tokenizer,
            # torch_dtype=torch.float32, # Save pipeline in fp32 for broader compatibility
        )
        try:
            pipeline.save_pretrained(args.output_dir)
            logger.info(f"Saved full pipeline to {args.output_dir}")
        except Exception as e:
            logger.error(f"Error saving full pipeline: {e}")
        # Move models back to original device if needed for further operations (though training is done)
        text_encoder.to(device)
        vae.to(device)
        unet.to(device)


    final_embed_filename = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
    final_embed_path = os.path.join(args.output_dir, final_embed_filename)
    save_progress(
        text_encoder, placeholder_token_ids, args, final_embed_path,
        safe_serialization=not args.no_safe_serialization
    )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
