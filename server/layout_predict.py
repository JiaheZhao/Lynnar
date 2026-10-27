# layout_predict.py – production-ready refactor
# -----------------------------------------------------
# Usage (as CLI or import):
#   python layout_predict.py --prompt prompt.txt --output_dir ./outputs
#   # or import and call predict_layout_from_data(user_data, ...)

import os, sys, json, argparse, pathlib, re, time, logging
import numpy as np
import matplotlib.pyplot as plt
import openai
from openai import APITimeoutError, APIError
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import math
from scipy.ndimage import zoom

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))

EXPECTED = [
    "title_heat", "location_heat", "time_heat",
    "host_organization_heat", "call_to_action_purpose_heat",
    "text_descriptions_details_heat",
]

DEFAULT_MODEL = "ft:gpt-4o-mini-2024-07-18:sia-project-1:jun24-test:Bm0QDLkZ"

# --- Helper functions ---
def to_grid(val) -> np.ndarray:
    """Convert a value to a 21x12 numpy grid."""
    if isinstance(val, (list, tuple)):
        nums = [float(x) for x in val]
    else:
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", str(val))]
    if len(nums) < 252:
        logging.info(f"  · {len(nums)} floats → padded to 252")
        nums += [0.0] * (252 - len(nums))
    elif len(nums) > 252:
        logging.info(f"  · {len(nums)} floats → truncated to 252")
        nums = nums[:252]
    return np.asarray(nums, dtype=float).reshape(21, 12)

def load_primer(path: str) -> List[Dict[str, str]]:
    """Load primer messages from a .jsonl file (always required)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Primer file is required and was not found.")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                break
        else:
            raise ValueError("Primer file is empty.")
    msgs = row.get("messages", [])
    ua = [m for m in msgs if m["role"] in ("user", "assistant")]
    if len(ua) >= 2 and ua[0]["role"] == "user" and ua[1]["role"] == "assistant":
        logging.info("✓ primer loaded")
        return ua[:2]
    raise ValueError("Primer has no usable user/assistant pair.")

def call_model_once(user_prompt: str, primer_msgs: List[Dict[str, str]], model: str, temp: float, field: Optional[str]=None) -> Dict[str, Any]:
    """Single OpenAI call, may raise exceptions."""
    system_msg = f"""
    <IMAGE_HEAT> Predict {'ONE' if field else 'six'} 12×21 heat-maps.
    Return JSON ONLY{f' with key: {field}' if field else ' with exactly these keys:'}
    {' '.join(EXPECTED) if not field else ''}
    Each value = 252 floats separated by a single space.
    """.strip()
    messages = [{"role": "system", "content": system_msg}] + primer_msgs + [
        {"role": "user", "content": user_prompt if not field else f"FIELD {field}\n{user_prompt}"}
    ]
    logging.info(f"→ calling model…{' ('+field+')' if field else ''}")
    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        response_format={"type": "json_object"},
        max_tokens=10000,
        timeout=60.0
    )
    logging.info("← reply received")
    raw_content = resp.choices[0].message.content
    logging.debug(f"Raw model output: {raw_content}")
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        raise

def call_model_with_retry(user_prompt: str, primer_msgs: List[Dict[str, str]], model: str, temp: float, field: str, max_retries: int, retry_wait: float) -> Any:
    """Repeat-call until non-zero output or retries exhausted."""
    for attempt in range(1, max_retries + 1):
        try:
            data = call_model_once(user_prompt, primer_msgs, model, temp, field=field)
        except (APITimeoutError, APIError) as e:
            logging.warning(f"⏳  {field}: API error ({e.__class__.__name__}); retry {attempt}/{max_retries}")
            time.sleep(retry_wait)
            continue
        if field in data:
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", str(data[field]))] \
                   if not isinstance(data[field], (list, tuple)) else data[field]
            if any(abs(n) > 1e-6 for n in nums):
                return data[field]
            logging.info(f"🔁  {field}: all zeros; retry {attempt}/{max_retries}")
        else:
            logging.info(f"🔁  {field}: key omitted; retry {attempt}/{max_retries}")
        time.sleep(retry_wait)
    logging.warning(f"⚠️  {field}: still zero/omitted after {max_retries} tries → zero-fill")
    return "0.0 " * 252

def predict_layout_from_data(
    user_data: str,
    primer_path: str,
    model: str = DEFAULT_MODEL,
    temp: float = 1.0,
    max_retries: int = 10,
    retry_wait: float = 2.0,
    selected_fields: list = None
) -> Dict[str, np.ndarray]:
    """
    Main entry: Given user_data (prompt string), returns dict of {field: grid}.
    Only uses fields in selected_fields (from user), not all EXPECTED.
    """
    primer_msgs = load_primer(primer_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if selected_fields is None:
        # Try to extract from user_data if it's a JSON string
        try:
            user_json = json.loads(user_data)
            selected_fields = user_json.get("selectedOptions", EXPECTED)
            # Map UI options to field names if needed
            field_map = {
                "title": "title_heat",
                "location": "location_heat",
                "time": "time_heat",
                "host": "host_organization_heat",
                "purpose": "call_to_action_purpose_heat",
                "other/descriptions": "text_descriptions_details_heat"
            }
            selected_fields = [field_map.get(f, f) for f in selected_fields]
        except Exception:
            selected_fields = EXPECTED
    # Only call for selected fields
    reply_raw = call_model_once(user_data, primer_msgs, model, temp, field=None if len(selected_fields) > 1 else selected_fields[0])
    reply = {}
    for fld in selected_fields:
        if fld in reply_raw:
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", str(reply_raw[fld]))]
            if nums and any(abs(n) > 1e-6 for n in nums):
                reply[fld] = reply_raw[fld]
                continue
        # fall back to retry loop if missing or zero
        reply[fld] = call_model_with_retry(user_data, primer_msgs, model, temp, fld, max_retries, retry_wait)
    # Convert to grids
    grids = {k: to_grid(v) for k, v in reply.items()}
    return grids

FIELD_COLORMAPS = {
    "title_heat": "viridis",
    "location_heat": "plasma",
    "time_heat": "inferno",
    "host_organization_heat": "magma",
    "call_to_action_purpose_heat": "cividis",
    "text_descriptions_details_heat": "cubehelix"
}

def save_grids_as_images(grids: Dict[str, np.ndarray], output_dir: str, frame_width: int, frame_height: int) -> Dict[str, str]:
    """Save each grid as an image file, upscaled to the frame size with pixelated effect and subtle noise. Returns dict of {field: filepath}."""
    os.makedirs(output_dir, exist_ok=True)
    filepaths = {}
    for k, grid in grids.items():
        # Nearest-neighbor upscaling for pixelated effect
        zoom_y = frame_height / grid.shape[0]
        zoom_x = frame_width / grid.shape[1]
        upscaled = zoom(grid, (zoom_y, zoom_x), order=0)  # nearest-neighbor (pixelated)
        # Add subtle noise
        noise = np.random.normal(loc=0.0, scale=0.03, size=upscaled.shape)
        upscaled_noisy = np.clip(upscaled + noise, 0, 1)
        path = os.path.join(output_dir, f"{k}.png")
        cmap = FIELD_COLORMAPS.get(k, "viridis")
        plt.imsave(path, upscaled_noisy, vmin=0, vmax=1, cmap=cmap, origin="upper")
        filepaths[k] = path
    return filepaths

def parse_plugin_log_to_heatmaps(log_json: dict, grid_shape=(21, 12), selected_fields=None) -> dict:
    """
    Given a log dict from the Figma plugin (with frame, children, selectedOptions),
    return a dict of {field: 252-float list} for each selected field, representing occupancy heatmaps.
    """
    frame = log_json["frame"]
    children = log_json.get("children", [])
    selected_options = log_json.get("selectedOptions", [])
    H, W = grid_shape
    frame_w, frame_h = frame["width"], frame["height"]
    # Helper to map absolute (x, y, w, h) to grid indices
    def rect_to_grid(x, y, w, h):
        grid = np.zeros((H, W), dtype=float)
        x0 = int(W * x / frame_w)
        y0 = int(H * y / frame_h)
        x1 = int(W * (x + w) / frame_w)
        y1 = int(H * (y + h) / frame_h)
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] = 1.0
        return grid
    # Base: frame is 100% occupancy
    frame_grid = np.ones((H, W), dtype=float)
    # Occupancy: sum all children
    occ_grid = np.zeros((H, W), dtype=float)
    for child in children:
        occ_grid += rect_to_grid(child["x"], child["y"], child["width"], child["height"])
    occ_grid = np.clip(occ_grid, 0, 1)
    # For each selected field, use occupancy or frame grid as a placeholder
    if selected_fields is None:
        selected_fields = [
            "title_heat", "location_heat", "time_heat",
            "host_organization_heat", "call_to_action_purpose_heat",
            "text_descriptions_details_heat",
        ]
    result = {}
    for field in selected_fields:
        # Example: use occupancy for all selected fields, can be customized per field
        if field in selected_options:
            result[field] = occ_grid.flatten().tolist()
        else:
            result[field] = [0.0] * (H * W)
    return result

def extract_json_from_log_line(line: str) -> dict:
    """Extracts the JSON object from a log line with a prefix."""
    start = line.find('{')
    end = line.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in line")
    json_str = line[start:end+1]
    return json.loads(json_str)

# --- CLI interface ---
def main():
    parser = argparse.ArgumentParser(description="Predict layout heatmaps from prompt.")
    parser.add_argument("--prompt", required=True, help="Prompt file (text or JSON from Figma plugin)")
    parser.add_argument("--primer", required=True, help="Primer .jsonl file (always required)")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save output images")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_retries", type=int, default=10, help="Max retries per field")
    parser.add_argument("--retry_wait", type=float, default=2.0, help="Seconds to wait between retries")
    args = parser.parse_args()

    if not os.path.exists(args.prompt):
        logging.error(f"Prompt file not found: {args.prompt}")
        sys.exit(1)
    with open(args.prompt, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        try:
            user_json = json.loads(raw)
            user_data = raw
        except Exception:
            # Try to extract JSON from log line
            user_json = extract_json_from_log_line(raw)
            user_data = json.dumps(user_json)
    # Try to extract selected fields from user_json
    try:
        selected_fields = user_json.get("selectedOptions", None)
        field_map = {
            "title": "title_heat",
            "location": "location_heat",
            "time": "time_heat",
            "host": "host_organization_heat",
            "purpose": "call_to_action_purpose_heat",
            "other/descriptions": "text_descriptions_details_heat"
        }
        if selected_fields:
            selected_fields = [field_map.get(f, f) for f in selected_fields]
    except Exception:
        selected_fields = None
    try:
        grids = predict_layout_from_data(
            user_data=user_data,
            primer_path=args.primer,
            model=args.model,
            temp=args.temp,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
            selected_fields=selected_fields
        )
        frame_w = user_json["frame"]["width"]
        frame_h = user_json["frame"]["height"]
        filepaths = save_grids_as_images(grids, args.output_dir, frame_w, frame_h)
        logging.info(f"Saved images: {filepaths}")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
