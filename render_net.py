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

# Create Gradio Interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# AI Fashion Model Generator")
        
        with gr.Tabs():
            with gr.Tab("Generate New Image"):
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
        
        gr.Markdown("""
        ## How to use:
        ### Generate New Image:
        1. Upload a reference image
        2. Enter your desired prompt describing the fashion model
        3. Click Generate
        
        ### Check Generation Status:
        1. Enter a generation ID
        2. Click Check Status to see the result
        
        Note: Make sure you have a valid RenderNet API key to use this service.
        """)
    
    demo.launch(share=True)

if __name__ == "__main__":
    main()