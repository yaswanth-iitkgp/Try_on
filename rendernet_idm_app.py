import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import gradio as gr
import requests
import json
import base64
from PIL import Image
import io
import time
import os
from termcolor import colored

def encode_image_to_base64(image, max_size=1024):
    """Convert PIL Image to base64 string with size limit"""
    # Resize image if it's too large
    if image.size[0] > max_size or image.size[1] > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Resized image to {new_size}")
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()

def upload_image_to_rendernet(image):
    """Upload image to RenderNet and get asset ID"""
    try:
        url = "https://api.rendernet.ai/pub/v1/assets/upload"
        
        # Get image dimensions
        width, height = image.size
        
        # Request upload URL
        payload = {
            "size": {
                "height": height,
                "width": width
            }
        }
        headers = {
            "X-API-KEY": "5EZ_dJagNaZyserBEN73Z9Y_XyciXFT5Htts3OeBonA",
            "Content-Type": "application/json"
        }
        
        print("Requesting upload URL...")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        upload_data = response.json()
        upload_url = upload_data['data']['upload_url']
        asset_id = upload_data['data']['asset']['id']
        print(f"Got asset ID: {asset_id}")
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Upload image to the provided URL
        print("Uploading image...")
        upload_response = requests.put(upload_url, data=img_byte_arr)
        upload_response.raise_for_status()
        
        return asset_id
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise

def get_available_controlnets():
    """Get list of available controlnets from the API"""
    try:
        url = "https://api.rendernet.ai/pub/v1/controlnets"
        headers = {
            "X-API-KEY": "5EZ_dJagNaZyserBEN73Z9Y_XyciXFT5Htts3OeBonA"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            controlnets = response.json()
            print("Available controlnets:", json.dumps(controlnets, indent=2))
            
            # Save controlnets response for reference
            with open('available_controlnets.json', 'w') as f:
                json.dump(controlnets, f, indent=2)
            
            return controlnets['data']
        else:
            print(f"Error getting controlnets: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def setup_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def log_info(message):
    """Print info message in cyan"""
    print(colored(message, 'cyan'))

def log_success(message):
    """Print success message in green"""
    print(colored(message, 'green'))

def log_error(message):
    """Print error message in red"""
    print(colored(message, 'red'))

def generate_fashion_image(input_image, prompt):
    """Generate fashion image using RenderNet API"""
    try:
        output_dir = setup_output_directory()
        
        log_info(f"Received input image: {type(input_image)}")
        log_info(f"Original image size: {input_image.size}")
        log_info(f"Received prompt: {prompt}")
        
        url = "https://api.rendernet.ai/pub/v2/assets/upload"
        generation_url = "https://api.rendernet.ai/pub/v1/generations"
        
        # Try to upload the reference image first
        reference_image_id = None
        if input_image is not None:
            try:
                log_info("Requesting upload URL...")
                # Request upload URL
                payload = {
                    "size": {
                        "height": 512,
                        "width": 512
                    }
                }
                headers = {
                    "X-API-KEY": "5EZ_dJagNaZyserBEN73Z9Y_XyciXFT5Htts3OeBonA",
                    "Content-Type": "application/json"
                }
                
                print("Requesting upload URL...")
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                upload_data = response.json()
                upload_url = upload_data['data']['upload_url']
                reference_image_id = upload_data['data']['id']
                
                # Convert and resize image to bytes
                img_byte_arr = io.BytesIO()
                resized_img = input_image.resize((512, 512), Image.Resampling.LANCZOS)
                resized_img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Upload image
                log_info("Uploading reference image...")
                upload_response = requests.put(upload_url, data=img_byte_arr)
                upload_response.raise_for_status()
                log_success(f"Successfully uploaded reference image with ID: {reference_image_id}")
                
                log_info("Waiting for image processing...")
                time.sleep(20)
                
            except Exception as e:
                log_error(f"Failed to upload reference image: {str(e)}")
                reference_image_id = None
        
        # Prepare generation payload
        payload = [{
            "aspect_ratio": "1:1",
            "batch_size": 1,
            "cfg_scale": 7,
            "model": "JuggernautXL",
            "prompt": {
                "negative": "nsfw, deformed, extra limbs, bad anatomy, deformed pupils, text, worst quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, blurry, low resolution",
                "positive": f"masterpiece, high quality, detailed, professional photograph, {prompt}"
            },
            "quality": "Plus",
            "sampler": "DPM++ 2M Karras",
            "seed": 1234,
            "steps": 20,
            "style": "Bokeh"
        }]
        
        # Add reference image if available
        if reference_image_id:
            payload[0]["reference_image_id"] = reference_image_id
        
        headers = {
            "X-API-KEY": "5EZ_dJagNaZyserBEN73Z9Y_XyciXFT5Htts3OeBonA",
            "Content-Type": "application/json"
        }
        
        log_info("Sending generation request...")
        response = requests.post(generation_url, json=payload, headers=headers)
        response.raise_for_status()
        
        init_result = response.json()
        
        # Save response for debugging
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        response_filename = os.path.join(output_dir, f'generation_response_{timestamp}.json')
        with open(response_filename, 'w') as f:
            json.dump(init_result, f, indent=2)
        
        credits_remaining = init_result['data'].get('credits_remaining', 'unknown')
        generation_id = init_result['data']['generation_id']
        status = init_result['data']['media'][0]['status']
        
        status_message = (
            f"Generation {status}!\n"
            f"Generation ID: {generation_id}\n"
            f"Credits remaining: {credits_remaining}\n"
            f"Please check the 'Check Generation Status' tab in a few moments."
        )
        
        log_success(status_message)
        return status_message
        
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        log_error(error_msg)
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def check_generation_status(generation_id):
    """Check status of a specific generation using its ID"""
    try:
        output_dir = setup_output_directory()
        
        log_info("\n=== Checking Generation Status ===")
        log_info(f"Generation ID: {generation_id}")
        
        url = f"https://api.rendernet.ai/pub/v1/generations/{generation_id}"
        headers = {
            "X-API-KEY": "5EZ_dJagNaZyserBEN73Z9Y_XyciXFT5Htts3OeBonA"
        }
        
        log_info(f"Making API request to: {url}")
        response = requests.get(url, headers=headers)
        log_info(f"API Response status: {response.status_code}")
        
        response.raise_for_status()
        result = response.json()
        
        # Save response for debugging
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        debug_filename = os.path.join(output_dir, f'status_check_{generation_id}_{timestamp}.json')
        with open(debug_filename, 'w') as f:
            json.dump(result, f, indent=4)
        log_info(f"Saved status response to {debug_filename}")
            
        if 'data' in result and 'media' in result['data'] and len(result['data']['media']) > 0:
            media = result['data']['media'][0]
            status = media.get('status', 'unknown')
            log_info(f"Generation status: {status}")
            
            if status == 'success' and 'url' in media:
                log_success(f"Success! Downloading image from URL: {media['url']}")
                image_response = requests.get(media['url'])
                image_data = image_response.content
                
                # Save the downloaded image
                output_filename = os.path.join(output_dir, f'generation_{generation_id}_{timestamp}.png')
                with open(output_filename, 'wb') as f:
                    f.write(image_data)
                log_success(f"Saved generated image to {output_filename}")
                
                output_image = Image.open(io.BytesIO(image_data))
                log_success(f"Successfully created PIL Image: {output_image.size}")
                return output_image
                
            elif status in ['pending', 'initiated']:
                status_message = f"Status: {status}\nPlease wait a moment and check again..."
                log_info(status_message)
                return gr.update(value=None, label=status_message)
                
            elif status == 'failed':
                error_msg = result.get('err', {}).get('message', 'Unknown error')
                log_error(f"Generation failed: {error_msg}")
                return gr.update(value=None, label=f"Generation failed: {error_msg}")
            else:
                log_info(f"Status is {status}, generation not yet complete")
                return gr.update(value=None, label=f"Status: {status}")
        else:
            error_msg = "Error: Invalid response format or no media found"
            log_error(error_msg)
            return gr.update(value=None, label=error_msg)
            
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        log_error(error_msg)
        import traceback
        traceback.print_exc()
        return gr.update(value=None, label=f"Error: {str(e)}")


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

def start_tryon(dict,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict["background"].convert("RGB")    
    
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
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))


    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        # Extract the images
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
                    ) = pipe.encode_prompt(
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
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )



                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig
    else:
        return images[0]

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# AI Fashion Assistant")
        
        with gr.Tabs():
            # First Tab - RenderNet Generation
            with gr.Tab("Generate Fashion Image"):
                with gr.Row():
                    input_image = gr.Image(label="Upload Reference Image", type="pil")
                    prompt = gr.Textbox(label="Enter your prompt", placeholder="e.g., a professional model wearing a blue dress")
                
                generate_btn = gr.Button("Generate")
                status_text = gr.Textbox(label="Status", interactive=False)
                
                generate_btn.click(
                    fn=generate_fashion_image,
                    inputs=[input_image, prompt],
                    outputs=status_text
                )
            
            # Second Tab - Check RenderNet Status
            with gr.Tab("Check Generation Status"):
                with gr.Row():
                    with gr.Column():
                        generation_id = gr.Textbox(label="Enter Generation ID", placeholder="e.g., gen_pIv4zLi6Nz")
                        check_btn = gr.Button("Check Status")
                    
                    with gr.Column():
                        status_output = gr.Image(label="Waiting for result...")
                
                check_btn.click(
                    fn=check_generation_status,
                    inputs=generation_id,
                    outputs=status_output
                )

            # Third Tab - IDM Virtual Try-on
            with gr.Tab("Virtual Try-on"):
                gr.Markdown("## IDM-VTON 👕👔👚")
                gr.Markdown("Virtual Try-on with your image and garment image.")
                with gr.Row():
                    with gr.Column():
                        imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)
                        with gr.Row():
                            is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)",value=True)
                        with gr.Row():
                            is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=False)

                    with gr.Column():
                        garm_img = gr.Image(label="Garment", sources='upload', type="pil")
                        with gr.Row(elem_id="prompt-container"):
                            with gr.Row():
                                prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
                    with gr.Column():
                        image_out = gr.Image(label="Output", elem_id="output-img", show_share_button=False)

                with gr.Column():
                    try_button = gr.Button(value="Try-on")
                    with gr.Accordion(label="Advanced Settings", open=False):
                        with gr.Row():
                            denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                            seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)

                try_button.click(
                    fn=start_tryon, 
                    inputs=[imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed], 
                    outputs=image_out
                )

    demo.launch(share=True)

if __name__ == "__main__":
    main()

