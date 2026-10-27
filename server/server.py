from flask import Flask, request, jsonify
import tempfile
import os
import json
import logging

from dotenv import load_dotenv

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))

import openai
from layout_predict import predict_layout_from_data, save_grids_as_images
from chat_bot import chat_with_model, chat_with_prompt_refinement
from image_generator import generate_image, image_to_base64, resize_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def call_regular_gpt4o_mini(user_message, conversation_history, temperature=0.7):
    """Call regular GPT-4o-mini for general conversation."""
    try:
        # Prepare messages
        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=4000
        )
        
        ai_response = response.choices[0].message.content
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response}
        ]
        
        return {
            "response": ai_response,
            "conversation_history": updated_history,
            "model_used": "gpt-4o-mini",
            "is_question": False
        }
        
    except Exception as e:
        return {
            "response": "I'm sorry, I encountered an error. Please try again.",
            "conversation_history": conversation_history,
            "error": str(e),
            "model_used": "gpt-4o-mini",
            "is_question": False
        }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    primer_path = os.path.join(BASE_DIR, "primer.jsonl")
    field_map = {
        "title": "title_heat",
        "location": "location_heat",
        "time": "time_heat",
        "host": "host_organization_heat",
        "purpose": "call_to_action_purpose_heat",
        "other/descriptions": "text_descriptions_details_heat"
    }
    selected_fields = [field_map.get(f, f) for f in data.get("selectedOptions", [])]
    user_data = json.dumps(data)
    grids = predict_layout_from_data(
        user_data=user_data,
        primer_path=primer_path,
        selected_fields=selected_fields
    )
    frame_w = data["frame"]["width"]
    frame_h = data["frame"]["height"]
    with tempfile.TemporaryDirectory() as tmpdir:
        filepaths = save_grids_as_images(grids, tmpdir, frame_w, frame_h)
        # For demo: return all images as base64
        import base64
        results = {}
        for field, path in filepaths.items():
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                results[field] = b64
        return jsonify(results)

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for conversation with fine-tuned model or regular GPT-4o-mini."""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data['message']
    conversation_history = data.get('conversation_history', [])
    model = data.get('model', None)  # Will use default from chat_bot if None
    temperature = data.get('temperature', 0.7)
    use_fine_tuned = data.get('use_fine_tuned', True)  # Default to fine-tuned model
    enhance_with_gpt4o = data.get('enhance_with_gpt4o', True)  # Default to enhancing with GPT-4o-mini
    
    try:
        if use_fine_tuned:
            # Use fine-tuned model for prompt refinement
            result = chat_with_model(
                user_message=user_message,
                conversation_history=conversation_history,
                model=model,
                temp=temperature,
                enhance_with_gpt4o=enhance_with_gpt4o
            )
        else:
            # Use regular GPT-4o-mini for general conversation
            result = call_regular_gpt4o_mini(
                user_message=user_message,
                conversation_history=conversation_history,
                temperature=temperature
            )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Chat failed: {str(e)}",
            "response": "I'm sorry, I encountered an error. Please try again."
        }), 500

@app.route('/chat/reset', methods=['POST'])
def reset_chat():
    """Reset conversation history."""
    return jsonify({
        "message": "Conversation history reset",
        "conversation_history": []
    })

@app.route('/chat/refine', methods=['POST'])
def chat_refine():
    """Multi-turn conversation with prompt refinement."""
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required"}), 400
    
    initial_prompt = data['prompt']
    conversation_history = data.get('conversation_history', [])
    model = data.get('model', None)
    temperature = data.get('temperature', 0.7)
    
    try:
        result = chat_with_prompt_refinement(
            initial_prompt=initial_prompt,
            conversation_history=conversation_history,
            model=model,
            temp=temperature
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Chat refinement failed: {str(e)}",
            "response": "I'm sorry, I encountered an error. Please try again."
        }), 500

@app.route('/generate-image', methods=['POST'])
def generate_image_endpoint():
    """
    Generate an image from a prompt using the configured image generation provider.
    
    Request body:
    {
        "prompt": "your prompt text",
        "provider": "dalle" | "replicate" | "midjourney_api" (optional),
        "size": "1024x1024" (for DALL-E, optional),
        "quality": "standard" | "hd" (for DALL-E, optional) - "standard" is faster,
        "model": "stability-ai/sdxl" (for Replicate, optional)
    }
    
    Returns:
    {
        "success": true,
        "image_base64": "base64 encoded image",
        "provider": "provider used",
        "generation_time": 3.5  # seconds
    }
    
    Typical generation times:
    - DALL-E 3 (standard): 3-5 seconds
    - DALL-E 3 (HD): 8-12 seconds (slower!)
    - Replicate (fast models): 2-5 seconds
    - Replicate (standard): 5-15 seconds
    """
    import time
    start_time = time.time()
    
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required"}), 400
    
    prompt = data['prompt']
    provider = data.get('provider', None)
    size = data.get('size', '1024x1024')
    quality = data.get('quality', 'standard')  # Default to "standard" for speed
    model = data.get('model', None)
    
    try:
        # Generate image
        kwargs = {}
        if provider == "dalle":
            kwargs['size'] = size
            kwargs['quality'] = quality  # Use "standard" for faster generation
        elif provider == "replicate" and model:
            kwargs['model'] = model
        
        image_bytes = generate_image(prompt, provider=provider, **kwargs)
        
        # Resize if needed (Figma has size limits)
        image_bytes = resize_image(image_bytes, max_width=2048, max_height=2048)
        
        # Convert to base64
        image_base64 = image_to_base64(image_bytes)
        
        generation_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "image_base64": image_base64,
            "provider": provider or "default",
            "generation_time": round(generation_time, 2)
        })
        
    except Exception as e:
        generation_time = time.time() - start_time
        logging.error(f"Image generation failed after {generation_time:.2f}s: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to generate image. Please check your API keys and try again.",
            "generation_time": round(generation_time, 2)
        }), 500

if __name__ == '__main__':
    app.run(port=5000)