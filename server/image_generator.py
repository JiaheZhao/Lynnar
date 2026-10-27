# image_generator.py
# -------------------
# Image generation service supporting multiple APIs:
# 1. DALL-E (OpenAI) - Recommended for prototyping
# 2. Stable Diffusion (via Replicate or Hugging Face)
# 3. Third-party Midjourney APIs (if available)

import os
import openai
import requests
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import logging

# Try to import replicate (optional)
try:
    import replicate
    HAS_REPLICATE = True
except ImportError:
    HAS_REPLICATE = False

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))
logging.basicConfig(level=logging.INFO)

# Configuration
DEFAULT_PROVIDER = os.getenv("IMAGE_GEN_PROVIDER", "dalle")  # "dalle", "replicate", "midjourney_api"
DEFAULT_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2.5-flare")

# DALL-E 3 sizes and quality names, mapped onto the current image model.
SIZE_MAP = {
    "1792x1024": "1536x1024",
    "1024x1792": "1024x1536",
}
QUALITY_MAP = {
    "standard": "medium",
    "hd": "high",
}
VALID_QUALITIES = {"low", "medium", "high", "xhigh", "max", "auto"}

# Fast model recommendations for Replicate
FAST_REPLICATE_MODELS = {
    "flux-schnell": "black-forest-labs/flux-schnell",  # 2-5 seconds, very fast
    "flux-dev": "black-forest-labs/flux-dev",  # 5-10 seconds, high quality
    "sdxl": "stability-ai/sdxl",  # 5-15 seconds, standard
}

def generate_with_dalle(prompt: str, size: str = "1024x1024", quality: str = "medium") -> bytes:
    """
    Generate an image with the OpenAI image model.
    Requires: OPENAI_API_KEY in environment.
    """
    try:
        import time
        start_time = time.time()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment")

        size = SIZE_MAP.get(size, size)
        quality = QUALITY_MAP.get(quality, quality)
        if quality not in VALID_QUALITIES:
            quality = "medium"

        logging.info(f"Generating image with {DEFAULT_IMAGE_MODEL}: {prompt[:50]}...")
        response = openai.images.generate(
            model=DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
            timeout=60.0
        )
        
        generation_time = time.time() - start_time
        logging.info(f"Image generation took {generation_time:.2f} seconds")

        image = response.data[0]
        if getattr(image, "b64_json", None):
            return base64.b64decode(image.b64_json)
        if getattr(image, "url", None):
            img_response = requests.get(image.url, timeout=30)
            img_response.raise_for_status()
            return img_response.content
        raise ValueError("Image response did not include a file")
        
    except Exception as e:
        logging.error(f"DALL-E generation failed: {e}")
        raise

def generate_with_replicate(prompt: str, model: str = "stability-ai/sdxl") -> bytes:
    """
    Generate image using Replicate API (supports Stable Diffusion, Midjourney-like models).
    Requires: REPLICATE_API_TOKEN in environment
    
    Typical generation time: 5-15 seconds (varies by model)
    Faster models available: flux/schnell (2-5 seconds)
    """
    if not HAS_REPLICATE:
        raise ImportError("Replicate module not installed. Install with: pip install replicate")
    
    try:
        import time
        start_time = time.time()
        logging.info(f"Generating image with Replicate ({model}): {prompt[:50]}...")
        
        output = replicate.run(
            model,
            input={"prompt": prompt}
        )
        
        # Replicate returns a URL or list of URLs
        if isinstance(output, list):
            image_url = output[0]
        else:
            image_url = output
        
        # Download the image with timeout
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        total_time = time.time() - start_time
        logging.info(f"Total Replicate process took {total_time:.2f} seconds")
        
        return img_response.content
        
    except Exception as e:
        logging.error(f"Replicate generation failed: {e}")
        raise

def generate_with_midjourney_api(prompt: str, api_key: str = None) -> bytes:
    """
    Generate image using third-party Midjourney API service.
    This is a placeholder - you'll need to integrate with a specific service.
    
    Example services:
    - Midjourney API (midjourneyapi.com)
    - Imagine API
    - Other third-party services
    
    Requires: MIDJOURNEY_API_KEY in environment
    """
    try:
        api_key = api_key or os.getenv("MIDJOURNEY_API_KEY")
        if not api_key:
            raise ValueError("MIDJOURNEY_API_KEY not found in environment")
        
        # Example API call structure (adjust based on your chosen service)
        api_url = os.getenv("MIDJOURNEY_API_URL", "https://api.midjourneyapi.com/v2/imagine")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "aspect_ratio": "1:1",  # Adjust as needed
            "mode": "fast"  # or "relax"
        }
        
        logging.info(f"Generating image with Midjourney API: {prompt[:50]}...")
        
        # Submit generation request
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("task_id")
        
        # Poll for completion (adjust based on API)
        import time
        max_wait = 300  # 5 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            status_response = requests.get(
                f"{api_url}/{task_id}",
                headers=headers
            )
            status_response.raise_for_status()
            status_data = status_response.json()
            
            if status_data.get("status") == "completed":
                image_url = status_data.get("image_url")
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                return img_response.content
            elif status_data.get("status") == "failed":
                raise Exception(f"Generation failed: {status_data.get('error')}")
            
            time.sleep(5)
            wait_time += 5
        
        raise TimeoutError("Image generation timed out")
        
    except Exception as e:
        logging.error(f"Midjourney API generation failed: {e}")
        raise

def generate_image(prompt: str, provider: str = None, **kwargs) -> bytes:
    """
    Main function to generate images using the specified provider.
    
    Args:
        prompt: The text prompt for image generation
        provider: "dalle", "replicate", or "midjourney_api"
        **kwargs: Additional parameters (size, quality, model, etc.)
    
    Returns:
        Image bytes (PNG format)
    """
    provider = provider or DEFAULT_PROVIDER
    
    try:
        if provider == "dalle":
            size = kwargs.get("size", "1024x1024")
            quality = kwargs.get("quality", "standard")
            return generate_with_dalle(prompt, size, quality)
        
        elif provider == "replicate":
            model = kwargs.get("model", "stability-ai/sdxl")
            return generate_with_replicate(prompt, model)
        
        elif provider == "midjourney_api":
            return generate_with_midjourney_api(prompt, kwargs.get("api_key"))
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    except Exception as e:
        logging.error(f"Image generation failed: {e}")
        raise

def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def resize_image(image_bytes: bytes, max_width: int = 2048, max_height: int = 2048) -> bytes:
    """Resize image if it exceeds maximum dimensions."""
    try:
        img = Image.open(BytesIO(image_bytes))
        width, height = img.size
        
        if width > max_width or height > max_height:
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            output = BytesIO()
            img.save(output, format='PNG')
            return output.getvalue()
        
        return image_bytes
    except Exception as e:
        logging.warning(f"Could not resize image: {e}")
        return image_bytes


