# chat_bot.py – conversation feature for fine-tuned 4o-mini model
# -----------------------------------------------------
# Usage:
#   from chat_bot import chat_with_model
#   response = chat_with_model(user_message, conversation_history)

import os, sys, json, logging
import openai
from openai import APITimeoutError, APIError
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import time

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))

# You can replace this with your specific fine-tuned model ID
DEFAULT_CHAT_MODEL = "ft:gpt-4o-mini-2024-07-18:sia-project-1:aug3-testconv:C0QJ9rOE"

# --- Helper functions ---
def call_chat_model_once(
    user_message: str, 
    conversation_history: List[Dict[str, str]], 
    model: str, 
    temp: float = 0.7
) -> str:
    """Single OpenAI call for chat conversation."""
    system_msg = """You are a prompt-refinement assistant. Ask ONE concise question with no more than four short options. After the user answers, reply ONLY with 'Updated prompt:' followed by the improved prompt, and nothing else. Incorporate the answer into the original prompt, using proper punctuation and commas. Never ask which image model, Midjourney version, or --v parameter to use. Ask only about the subject, audience, mood, setting, color, or typography."""
    
    messages = [{"role": "system", "content": system_msg}] + conversation_history + [
        {"role": "user", "content": user_message}
    ]
    
    logging.info(f"→ calling chat model…")
    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=4000,
        timeout=60.0
    )
    logging.info("← chat reply received")
    return resp.choices[0].message.content

def enhance_prompt_with_gpt4o(
    original_prompt: str,
    user_answer: str,
    ai_question: str,
    current_updated_prompt: str = None,
    temp: float = 0.7
) -> str:
    """Fix grammar and make minimal changes to the updated prompt using GPT-4o."""
    system_msg = """You are a prompt refinement specialist. Understand the question-answer context and update prompts semantically, not by simple text combination. Keep original structure intact and ensure Midjourney compatibility. Use proper punctuation and commas to separate multiple elements. Return ONLY the corrected prompt."""
    
    # If we have the current updated prompt, include it for evaluation
    if current_updated_prompt:
        user_msg = f"""Original: "{original_prompt}"
Question: "{ai_question}"
Answer: "{user_answer}"
Current: "{current_updated_prompt}"

Evaluate semantic accuracy. Fix if needed."""
    else:
        user_msg = f"""Original: "{original_prompt}"
Question: "{ai_question}"
Answer: "{user_answer}"

Create semantically accurate prompt."""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    logging.info(f"→ calling GPT-4o for semantic refinement…")
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temp,
        max_tokens=4000,
        timeout=60.0
    )
    logging.info("← GPT-4o semantic refinement received")
    return resp.choices[0].message.content

def extract_updated_prompt(ai_response: str) -> Optional[str]:
    """Extract the updated prompt from AI response if it starts with 'Updated prompt:'"""
    if ai_response.strip().startswith("Updated prompt:"):
        return ai_response.replace("Updated prompt:", "").strip()
    return None

def is_question_response(ai_response: str) -> bool:
    """Check if the AI response is asking a question (not an updated prompt)"""
    response = ai_response.strip()
    return not response.startswith("Updated prompt:") and "?" in response

def call_chat_model_with_retry(
    user_message: str, 
    conversation_history: List[Dict[str, str]], 
    model: str, 
    temp: float = 0.7,
    max_retries: int = 3, 
    retry_wait: float = 2.0
) -> str:
    """Repeat-call until successful or retries exhausted."""
    for attempt in range(1, max_retries + 1):
        try:
            response = call_chat_model_once(user_message, conversation_history, model, temp)
            return response
        except (APITimeoutError, APIError) as e:
            logging.warning(f"⏳ Chat API error ({e.__class__.__name__}); retry {attempt}/{max_retries}")
            if attempt < max_retries:
                time.sleep(retry_wait)
                continue
            else:
                logging.error(f"❌ Chat failed after {max_retries} retries")
                raise

def chat_with_model(
    user_message: str,
    conversation_history: List[Dict[str, str]] = None,
    model: str = None,
    temp: float = 0.7,
    max_retries: int = 3,
    retry_wait: float = 2.0,
    enhance_with_gpt4o: bool = True
) -> Dict[str, Any]:
    """
    Chat with the fine-tuned model with multi-turn prompt refinement.
    
    Args:
        user_message: The user's message
        conversation_history: List of previous messages in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        model: Model ID to use
        temp: Temperature for response generation
        max_retries: Maximum retry attempts
        retry_wait: Wait time between retries
        enhance_with_gpt4o: Whether to enhance updated prompts with GPT-4o-mini
    
    Returns:
        Dict with response, updated conversation history, and next prompt if available
    """
    if conversation_history is None:
        conversation_history = []
    
    # Use default model if None is provided
    if model is None:
        model = DEFAULT_CHAT_MODEL
    
    try:
        response = call_chat_model_with_retry(
            user_message, 
            conversation_history, 
            model, 
            temp, 
            max_retries, 
            retry_wait
        )
        
        # Check if this is an updated prompt response
        updated_prompt = extract_updated_prompt(response)
        is_question = is_question_response(response)
        
        # If we got an updated prompt and enhancement is enabled, enhance it with GPT-4o-mini
        if updated_prompt and enhance_with_gpt4o:
            try:
                # Get the last question from conversation history
                last_question = ""
                for msg in reversed(conversation_history):
                    if msg["role"] == "assistant" and "?" in msg["content"]:
                        last_question = msg["content"]
                        break
                
                # Get the original prompt (first user message)
                original_prompt = ""
                for msg in conversation_history:
                    if msg["role"] == "user":
                        original_prompt = msg["content"]
                        break
                
                # Enhance the prompt with GPT-4o
                enhanced_prompt = enhance_prompt_with_gpt4o(
                    original_prompt=original_prompt,
                    user_answer=user_message,
                    ai_question=last_question,
                    current_updated_prompt=updated_prompt,
                    temp=temp
                )
                
                # Update the response with the enhanced prompt
                response = f"Updated prompt: {enhanced_prompt}"
                updated_prompt = enhanced_prompt
                
                logging.info(f"Enhanced prompt with GPT-4o")
                
            except Exception as e:
                logging.warning(f"Failed to enhance prompt with GPT-4o: {e}")
                # Continue with the original updated prompt if enhancement fails
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ]
        
        result = {
            "response": response,
            "conversation_history": updated_history,
            "model_used": model,
            "is_question": is_question
        }
        
        # If we got an updated prompt, add it to the result
        if updated_prompt:
            result["updated_prompt"] = updated_prompt
        
        return result
        
    except Exception as e:
        logging.error(f"Chat failed: {e}")
        return {
            "response": "I'm sorry, I encountered an error. Please try again.",
            "conversation_history": conversation_history,
            "error": str(e),
            "model_used": model,
            "is_question": False
        }

def format_conversation_history(messages: List[Dict[str, str]]) -> str:
    """Format conversation history for display or logging."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role.upper()}: {content}")
    return "\n".join(formatted)

def chat_with_prompt_refinement(
    initial_prompt: str,
    conversation_history: List[Dict[str, str]] = None,
    model: str = None,
    temp: float = 0.7,
    max_retries: int = 3,
    retry_wait: float = 2.0
) -> Dict[str, Any]:
    """
    Multi-turn conversation with prompt refinement.
    
    This function handles the flow:
    1. User sends initial prompt
    2. AI asks a question with options
    3. User responds
    4. AI generates updated prompt
    5. Updated prompt becomes next user input
    6. Repeat until user is satisfied
    
    Args:
        initial_prompt: The user's initial prompt
        conversation_history: Previous conversation history
        model: Model ID to use
        temp: Temperature for response generation
        max_retries: Maximum retry attempts
        retry_wait: Wait time between retries
    
    Returns:
        Dict with conversation flow and final refined prompt
    """
    if conversation_history is None:
        conversation_history = []
    
    current_prompt = initial_prompt
    conversation_flow = []
    
    # First turn: Send initial prompt
    result = chat_with_model(
        user_message=current_prompt,
        conversation_history=conversation_history,
        model=model,
        temp=temp,
        max_retries=max_retries,
        retry_wait=retry_wait
    )
    
    conversation_flow.append({
        "turn": 1,
        "user_input": current_prompt,
        "ai_response": result["response"],
        "is_question": result.get("is_question", True)
    })
    
    # If AI asked a question, we need user input to continue
    if result.get("is_question", True):
        return {
            "status": "waiting_for_user_response",
            "conversation_flow": conversation_flow,
            "conversation_history": result["conversation_history"],
            "current_prompt": current_prompt,
            "model_used": result["model_used"]
        }
    
    # If AI provided an updated prompt, we can continue the conversation
    if "updated_prompt" in result:
        current_prompt = result["updated_prompt"]
        conversation_flow.append({
            "turn": 2,
            "user_input": current_prompt,
            "ai_response": result["response"],
            "is_question": result.get("is_question", False)
        })
    
    return {
        "status": "completed",
        "conversation_flow": conversation_flow,
        "conversation_history": result["conversation_history"],
        "final_prompt": current_prompt,
        "model_used": result["model_used"]
    }

# --- CLI interface for testing ---
def main():
    """Simple CLI for testing the chat functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chat with fine-tuned model")
    parser.add_argument("--message", required=True, help="User message to send")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, help="Model to use")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--history", help="Path to conversation history JSON file")
    
    args = parser.parse_args()
    
    # Load conversation history if provided
    conversation_history = []
    if args.history and os.path.exists(args.history):
        with open(args.history, 'r') as f:
            conversation_history = json.load(f)
    
    # Chat with model
    result = chat_with_model(
        args.message,
        conversation_history,
        args.model,
        args.temp
    )
    
    print(f"\n🤖 Response: {result['response']}")
    
    # Save updated history
    if args.history:
        with open(args.history, 'w') as f:
            json.dump(result['conversation_history'], f, indent=2)
        print(f"💾 Conversation history saved to {args.history}")

if __name__ == "__main__":
    main() 