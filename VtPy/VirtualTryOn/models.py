import sys
import os
import argparse
# idm_vton_path = os.path.abspath(
#     os.path.join(
#         os.path.dirname(__file__), "../configs/densepose_rcnn_R_50_FPN_s1x.yaml"
#     )
# )
# print(idm_vton_path)
# exit()
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# IDM_VTON 폴더 경로 추가
idm_vton_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# /nstm
sys.path.append(idm_vton_path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
# /VirtualTryOn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./detectron")))

import shutil
import os
import argparse
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
# src __init__ done
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List
import torch
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


class IDMVTON():
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # download ckpt
        folder_path = '/ckpt/densepose/model_final_162be9.pkl'

        if not os.path.exists(folder_path):
            repo_id = "yisol/IDM-VTON"
            # save_path = "/ckpt"
            save_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "ckpt")
            )

            # Download checkpoints to run the vton model
            hf_hub_download(repo_id=repo_id, filename="densepose/model_final_162be9.pkl", local_dir=save_path)
            hf_hub_download(repo_id=repo_id, filename="humanparsing/parsing_atr.onnx", local_dir=save_path)
            hf_hub_download(repo_id=repo_id, filename="humanparsing/parsing_lip.onnx", local_dir=save_path)
            hf_hub_download(repo_id=repo_id, filename="openpose/ckpts/body_pose_model.pth", local_dir=save_path)
        else:
            print("Checkpoints already exist.")
        print()
        # ----------------------------------------------------------

        base_path = 'yisol/IDM-VTON'
        example_path = os.path.join(os.path.dirname(__file__), 'example')

        self.unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        self.unet.requires_grad_(False)
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        self.vae = AutoencoderKL.from_pretrained(base_path,
                                            subfolder="vae",
                                            torch_dtype=torch.float16,
        )

        self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )

        #

        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)

        self.UNet_Encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
        )
        self.pipe.unet_encoder = self.UNet_Encoder

    def start_tryon(self, human_img_path, garm_img_path, garment_des, is_checked=True, is_checked_crop=False, denoise_steps=30, seed=42, output_path="../output_images/output.png"):
        # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 현재 스크립트 파일 기준으로 경로 계산
        script_directory = os.path.dirname(os.path.abspath(__file__))
        example_human_dir = os.path.join(script_directory, './example', 'human')
        example_cloth_dir = os.path.join(script_directory, './example', 'cloth')

        # 최종 대상 경로
        human_img_dest = os.path.join(example_human_dir, os.path.basename(human_img_path))
        garm_img_dest = os.path.join(example_cloth_dir, os.path.basename(garm_img_path))

        # 디렉토리 생성
        os.makedirs(example_human_dir, exist_ok=True)
        os.makedirs(example_cloth_dir, exist_ok=True)

        # 파일 복사
        shutil.copy2(human_img_path, human_img_dest)
        shutil.copy2(garm_img_path, garm_img_dest)

        # ----------------------------------------------------------------

        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)

        garm_img = Image.open(garm_img_path).convert("RGB").resize((768, 1024))
        human_img_orig = Image.open(human_img_path).convert("RGB")

        if is_checked_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024))
        else:
            human_img = human_img_orig.resize((768, 1024))

        if is_checked:
            keypoints = self.openpose_model(human_img.resize((384, 512)))
            model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            mask = pil_to_binary_mask(human_img.resize((768, 1024)))

        mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        yaml_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../configs/densepose_rcnn_R_50_FPN_s1x.yaml"
            )
        )
        pkl_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../ckpt/densepose/model_final_162be9.pkl"
            )
        )
        args = apply_net.create_argument_parser().parse_args(('show', yaml_path, pkl_path, 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    prompt = "model is wearing " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    with torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = self.pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )

                        prompt = "a photo of " + garment_des
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                        if not isinstance(prompt, List):
                            prompt = [prompt] * 1
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * 1
                        with torch.inference_mode():
                            (
                                prompt_embeds_c,
                                _,
                                _,
                                _,
                            ) = self.pipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )

                        pose_img = self.tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
                        garm_tensor = self.tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
                        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                        images = self.pipe(
                            prompt_embeds=prompt_embeds.to(device, torch.float16),
                            negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=pose_img.to(device, torch.float16),
                            text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                            cloth=garm_tensor.to(device, torch.float16),
                            mask_image=mask,
                            image=human_img,
                            height=1024,
                            width=768,
                            ip_adapter_image=garm_img.resize((768, 1024)),
                            guidance_scale=2.0,
                        )[0]

        if is_checked_crop:
            out_img = images[0].resize(crop_size)
            human_img_orig.paste(out_img, (int(left), int(top)))
            result_img = human_img_orig
        else:
            result_img = images[0]

        result_img.save(output_path, "PNG")
        print(f"합성된 이미지를 {output_path}에 저장했습니다.")
        # return result_img, mask_gray
        return result_img


def main():
    parser = argparse.ArgumentParser(description="Run the try-on pipeline")
    parser.add_argument('--human_img_path', type=str, required=True, help="Path to the human image")
    parser.add_argument('--garm_img_path', type=str, required=True, help="Path to the garment image")
    parser.add_argument('--garment_des', type=str, required=True, help="Description of the garment")
    parser.add_argument('--is_checked', type=bool, default=True, help="Use human parsing and openpose")
    parser.add_argument('--is_checked_crop', type=bool, default=False, help="Crop the human image")
    parser.add_argument('--denoise_steps', type=int, default=30, help="Number of denoise steps")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output_path', type=str, default='../output.png', help="Path to save the output image")

    args = parser.parse_args()

    # 현재 스크립트 파일 기준으로 경로 계산
    script_directory = os.path.dirname(os.path.abspath(__file__))
    example_human_dir = os.path.join(script_directory, './example', 'human')
    example_cloth_dir = os.path.join(script_directory, './example', 'cloth')

    # 최종 대상 경로
    human_img_dest = os.path.join(example_human_dir, os.path.basename(args.human_img_path))
    garm_img_dest = os.path.join(example_cloth_dir, os.path.basename(args.garm_img_path))

    # 디렉토리 생성
    os.makedirs(example_human_dir, exist_ok=True)
    os.makedirs(example_cloth_dir, exist_ok=True)

    # 파일 복사
    shutil.copy2(args.human_img_path, human_img_dest)
    shutil.copy2(args.garm_img_path, garm_img_dest)
    # Update paths for start_tryon
    
    start_tryon(human_img_dest, garm_img_dest, args.garment_des, args.is_checked, args.is_checked_crop, args.denoise_steps, args.seed, args.output_path)

if __name__ == "__main__":
    main()
