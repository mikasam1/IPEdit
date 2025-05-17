import torchvision
torchvision.disable_beta_transforms_warning()

import argparse
import torch
import numpy as np
from PIL import Image
from ptpcore import (
    LocalBlend,
    AttentionReplace,
    AttentionRefine,
    AttentionReweight,
    EmptyControl,
    AttentionStore,
    get_equalizer,
    run_and_display,
    show_cross_attention,
    show_self_attention_comp,
)
import ptpcore

def parse_args():
    parser = argparse.ArgumentParser(description="ptpedit argparser")
    parser.add_argument(
        "--prompts", nargs='+', required=True,
        help="List of text prompts to generate images."
    )
    parser.add_argument(
        "--mode", choices=['baseline', 'replace', 'refine', 'reweight'], default='baseline',
        help="Attention control mode to apply."
    )
    parser.add_argument(
        "--blend_words", nargs='*', default=None,
        help="Words or word groups for LocalBlend masking (only used in replace/refine)."
    )
    parser.add_argument(
        "--cross_steps", type=float, default=1.0,
        help="Fraction of diffusion steps to apply cross-attention replacement/refinement."
    )
    parser.add_argument(
        "--self_steps", type=float, default=0.5,
        help="Fraction of diffusion steps to apply self-attention replacement."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed to generate."
    )
    parser.add_argument(
        "--eq_words", nargs='*', default=None,
        help="words to Re-weight (equalizing)"
    )
    parser.add_argument(
        "--equalizer_vals", nargs='*', type=float, default=None,
        help="Values to reweight selected words (only for reweight mode)."
    )
    parser.add_argument(
        "--embeds", type=str, default=None,
        help="path to textual inversion .safetensors"
)
    return parser.parse_args()


def build_controller(args):
    prompts = args.prompts
    # must be same as NUM_DIFFUSION_STEPS
    num_steps = 50
    # pass prompts to ptpcore.show_* functions
    ptpcore.prompts = prompts
    # pass embeds to ptpcore
    if args.embeds:
        ptpcore.ldm_stable.load_textual_inversion(args.embeds)

    if args.mode == 'baseline':
        return EmptyControl()
    
    base_controller = None
    local_blend = None
    if args.blend_words:
        local_blend = LocalBlend(prompts, args.blend_words)
        print(args.blend_words)

    if args.mode in ['replace', 'refine']:
        if args.mode == 'replace':
            print("replacing...")
            base_controller = AttentionReplace(
                prompts, num_steps,
                cross_replace_steps=args.cross_steps,
                self_replace_steps=args.self_steps,
                local_blend=local_blend
            )
        else:
            print("refining...")
            base_controller = AttentionRefine(
                prompts, num_steps,
                cross_replace_steps=args.cross_steps,
                self_replace_steps=args.self_steps,
                local_blend=local_blend
            )
        if args.equalizer_vals is not None and args.eq_words is not None:

            print("args.equalizer_vals:", args.equalizer_vals)
            print("args.eq_words:", args.eq_words)

            print("add reweighting...")
            equalizer = get_equalizer(prompts[1], args.eq_words, tuple(args.equalizer_vals))
            return AttentionReweight(
                prompts, num_steps,
                cross_replace_steps=args.cross_steps,
                self_replace_steps=args.self_steps,
                equalizer=equalizer,
                local_blend=local_blend,
                controller=base_controller
            )

        return base_controller
    
    if args.mode == 'reweight':

        print("args.equalizer_vals:", args.equalizer_vals)
        print("args.eq_words:", args.eq_words)

        if args.equalizer_vals is None or args.eq_words is None:
            raise ValueError("Equalizer values required for reweight mode.")

        print("reweighting...")
        equalizer = get_equalizer(prompts[1], args.eq_words, tuple(args.equalizer_vals))
        return AttentionReweight(
            prompts, num_steps,
            cross_replace_steps=args.cross_steps,
            self_replace_steps=args.self_steps,
            equalizer=equalizer,
            local_blend=local_blend,
        )


def save_images(images,prompts: list):
    for idx,img in enumerate(images):
        if (idx == 0):
            filename = f"{prompts[idx]}_src.png"
        elif (idx == 1):
            filename = f"{prompts[idx]}_dst.png"
        # Convert numpy arrays to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.save(filename)
        print(f"Saved image: {filename}")


def main():
    args = parse_args()
    print("mode:", args.mode)
    controller = build_controller(args)

    if args.seed is not None:
        generator = torch.Generator().manual_seed(args.seed)
    else:
        generator = None
    # Generate and display images
    images, latent = run_and_display(
        args.prompts,
        controller,
        run_baseline=(args.mode=='baseline'),
        generator=generator
    )
    # Save outputs for terminal environments
    save_images(images, args.prompts)

    # If controller collects attention (Editing modes), visualize
    if isinstance(controller, AttentionStore):
        print("Saving cross-attention maps...")
        show_cross_attention(controller, res=16, from_where=['down', 'mid', 'up'], select=0, output_path = args.prompts[0])
        show_cross_attention(controller, res=16, from_where=['down', 'mid', 'up'], select=1, output_path = args.prompts[1])
        print("Saving self-attention maps...")
        show_self_attention_comp(controller, res=16, from_where=['down', 'mid', 'up'], select=0, output_path = args.prompts[0])
        show_self_attention_comp(controller, res=16, from_where=['down', 'mid', 'up'], select=1, output_path = args.prompts[1])
    else:
        print("Attention maps not available for baseline mode.")


if __name__ == '__main__':
    main()
