import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import necessary libraries
from openai import OpenAI, AzureOpenAI
from google import genai
from google.genai import types
from google.oauth2 import service_account  # Required for Vertex AI authentication

# ==========================================
# 1. Configuration Section
# ==========================================
from PIL import Image
import io
import numpy as np
import base64

def image_to_base64(image_source, format="JPEG", quality=85):
    """
    Converts an image source (numpy.array, file path, or PIL.Image object)
    into a Base64 encoded string.

    Args:
        image_source: Source of the image. Can be:
                      - numpy.array: NumPy array (e.g., from OpenCV or PIL.Image)
                      - str: Path to an image file (e.g., "path/to/image.jpg")
                      - PIL.Image.Image: A PIL Image object
        format (str): Target image format for encoding, e.g., "JPEG", "PNG", "WEBP".
                      Defaults to "JPEG".
        quality (int): Only valid if format is "JPEG" or "WEBP", representing 
                       compression quality (1-95). Defaults to 85.

    Returns:
        str: Base64 encoded image string.
    
    Raises:
        ValueError: If the image source type is not supported.
        FileNotFoundError: If the image file path is invalid.
    """
    
    img = None
    
    # --- 1. Standardize all inputs into a PIL.Image object ---
    if isinstance(image_source, np.ndarray):
        # Handle numpy.array
        # Ensure the data type is uint8 and handle color channel conversions
        if image_source.dtype != np.uint8:
            image_source = image_source.astype(np.uint8)
            
        # Basic channel check: handle standard 3 or 4 channel images
        if image_source.ndim == 3 and image_source.shape[2] == 3:
            # Standard RGB case. Note: If read via cv2.imread, user must 
            # convert BGR to RGB before calling this function.
            img = Image.fromarray(image_source)
        elif image_source.ndim == 3 and image_source.shape[2] == 4:
            # RGBA Image
            img = Image.fromarray(image_source, 'RGBA')
        else:
            # Handle other dimensions (e.g., Grayscale (H, W))
            img = Image.fromarray(image_source)

    elif isinstance(image_source, str):
        # Handle image file path
        try:
            img = Image.open(image_source)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_source}")
        except Exception as e:
            raise ValueError(f"Unable to open image file {image_source}: {e}")

    elif isinstance(image_source, Image.Image):
        # Handle existing PIL.Image object
        img = image_source
        
    else:
        # Unsupported types
        raise ValueError(
            f"Unsupported image source type: {type(image_source)}. "
            "Please provide numpy.ndarray, file path (str), or PIL.Image.Image object."
        )

    # --- 2. Save PIL.Image object to memory buffer and perform Base64 encoding ---
    buffer = io.BytesIO()
    
    # Select save parameters based on format
    save_params = {'format': format}
    if format.upper() in ["JPEG", "WEBP"]:
        save_params['quality'] = quality
    
    try:
        img.save(buffer, **save_params)
    except KeyError:
        raise ValueError(f"Unsupported image format: {format}. Try 'JPEG', 'PNG', 'WEBP'.")
    except Exception as e:
        raise ValueError(f"Failed to save image to buffer: {e}")

    # 3. Retrieve byte content and encode to Base64
    base64_encoded_data = base64.b64encode(buffer.getvalue())
    base64_string = base64_encoded_data.decode('utf-8')
    
    return base64_string

# Load API configurations from JSON
import json
with open('api_config.json', 'r', encoding='utf-8') as f:
    API_CONFIGS = json.load(f)

# ==========================================
# 2. Client Initialization (Factory)
# ==========================================
def init_client(provider, model):
    """
    Initializes the Client based on the provider, including 
    custom logic for Google Service Account authentication.
    """
    config = API_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Provider {provider} not found in configuration.")

    client = None

    # --- Google Vertex AI Initialization (Custom implementation) ---
    if provider == "google":
        # 1. Set environment variables
        os.environ['GOOGLE_CLOUD_PROJECT'] = 'decision-agent-gemini'
        os.environ['GOOGLE_CLOUD_LOCATION'] = 'us-central1'
        os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
        
        # 2. Load credentials
        # Check if service account file exists to prevent path errors
        if not os.path.exists(config["service_account_file"]):
            raise FileNotFoundError(f"Key file not found: {config['service_account_file']}")

        gemini_creds = service_account.Credentials.from_service_account_file(
            filename=config["service_account_file"], 
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # 3. Create Google GenAI Client
        client = genai.Client(credentials=gemini_creds)

    # --- Azure OpenAI Initialization ---
    elif provider == "azure":
        client = AzureOpenAI(
            azure_endpoint=config["endpoint"],
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_deployment=model
        )

    # --- Standard OpenAI Compatible Interfaces (Moonshot, DeepSeek, etc.) ---
    else:
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url")
        )
    
    return client, model

# ==========================================
# 3. Unified Inference Function (Unified Infer)
# ==========================================
def unified_infer(prompt, client, model, provider, temperature=0.8, max_tokens=2048, retries=5):
    """
    Standardized inference wrapper to handle both Google and OpenAI-style interfaces.
    """
    for attempt in range(retries):
        try:
            # >>> Google Vertex AI Branch <<<
            if provider == "google":
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                # Return standardized text output
                return response.text

            # >>> OpenAI / Azure / Compatible Branch <<<
            else:
                if isinstance(prompt, str):
                    # Handle text-only prompt
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                elif isinstance(prompt, dict):
                    # Handle multi-modal prompt (Image + Text)
                    multi_modal_prompt = [
                        {"type": "image", "image": image_to_base64(prompt["image"])},
                        {"type": "text", "text": prompt["text"]},
                    ]
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": multi_modal_prompt}],
                    )
                    
                # Special handling for DeepSeek reasoning content
                if provider=="deepseek":
                    message = response.choices[0].message
                    reasoning = getattr(message, 'reasoning_content', "") or ""
                    content = message.content or ""
                    # Simple concatenation of reasoning and content
                    full_text = reasoning + "\n" + content
                else:
                    full_text = response.choices[0].message.content

                if full_text.strip():
                    return str(full_text)
                else:
                    print(response.choices[0])
                    raise ValueError("Empty response from model.")
                    
        except Exception as e:
            print(f"[{provider}] Error: {e}. Retrying... ({attempt + 1}/{retries})")
            time.sleep(2 ** attempt)  # Exponential backoff

    print(f"[{provider}] All retries failed.")
    return f"[{provider}] No reply after {retries} retries"

# ==========================================
# 4. Batch Processing Function (Batch Infer)
# ==========================================
def batch_infer(prompts, active_masks, provider, model_name, max_workers, **kwargs):
    """
    Executes multiple inference tasks in parallel using a thread pool.
    """
    # 1. Initialize Client (Perform only once per batch)
    try:
        client, model_name = init_client(provider, model_name)
        print(f"Loaded Provider: {provider} | Model: {model_name}")
    except Exception as e:
        print(f"Client Init Failed: {e}")
        # If initialization fails, return error messages for all prompts
        return [f"Init Error: {e}"] * len(prompts)

    original_length = len(prompts)
    results = ["no action"] * original_length
    
    # 2. Filter active tasks based on masks
    active_jobs = []
    for i, (prompt, is_active) in enumerate(zip(prompts, active_masks)):
        if is_active:
            active_jobs.append((i, prompt))

    print(f'Starting batch_infer with {max_workers} threads. Tasks: {len(active_jobs)}/{original_length}.')

    # 3. Parallel execution via Thread Pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit active jobs
        future_to_idx = {
            executor.submit(
                unified_infer, 
                prompt, client, model_name, provider, 
                **kwargs # Pass-through parameters like temperature
            ): i
            for i, prompt in active_jobs
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Job {idx} crashed: {e}")
                results[idx] = f"Crash: {e}"

    print('Batch infer finished.')
    print(results[0:2])
    return results

# ==========================================
# 5. Example Usage
# ==========================================
if __name__ == "__main__":
    # Mock data
    prompts = ["Hello, who are you?", "1+1=?", "Pass this one."]
    masks = [True, True, False]

    # Run execution
    # Ensure api_config.json and service account files are in place
    final_res = batch_infer(
        prompts=prompts, 
        active_masks=masks, 
        provider="google", 
        model_name="gemini-2.5-pro",
        max_workers=1,
    )

    for res in final_res:
        print("-" * 20)
        print(res)