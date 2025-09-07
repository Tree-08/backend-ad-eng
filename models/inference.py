# backend/models/inference.py
import os, json, time, random, pathlib
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "hf").lower()  # "hf" or "gemini"
USE_HF = AI_PROVIDER == "hf"

HF_TEXT_MODEL = os.getenv("HF_TEXT_MODEL", "google/flan-t5-small")
HF_EN_HI_MODEL = os.getenv("HF_EN_HI_MODEL", "Helsinki-NLP/opus-mt-en-hi")
HF_HI_EN_MODEL = os.getenv("HF_HI_EN_MODEL", "Helsinki-NLP/opus-mt-hi-en")

# Optional: where images would be placed if you later wire a local generator
STATIC_DIR = pathlib.Path(__file__).resolve().parents[1] / "static"
STATIC_DIR.mkdir(exist_ok=True)

_text_pipe = None
_en_hi_pipe = None
_hi_en_pipe = None

def _lazy_text_pipe():
    global _text_pipe
    if _text_pipe is None:
        from transformers import pipeline
        # flan-t5 uses the "text2text-generation" task
        _text_pipe = pipeline(
            "text2text-generation",
            model=HF_TEXT_MODEL,
            device_map="auto" if os.getenv("HF_DEVICE_MAP") else None
        )
    return _text_pipe

def _lazy_translator(src: str, tgt: str):
    global _en_hi_pipe, _hi_en_pipe
    from transformers import MarianMTModel, MarianTokenizer, pipeline

    if src == "en" and tgt == "hi":
        if _en_hi_pipe is None:
            tok = MarianTokenizer.from_pretrained(HF_EN_HI_MODEL)
            mdl = MarianMTModel.from_pretrained(HF_EN_HI_MODEL)
            _en_hi_pipe = pipeline("translation", model=mdl, tokenizer=tok)
        return _en_hi_pipe

    if src == "hi" and tgt == "en":
        if _hi_en_pipe is None:
            tok = MarianTokenizer.from_pretrained(HF_HI_EN_MODEL)
            mdl = MarianMTModel.from_pretrained(HF_HI_EN_MODEL)
            _hi_en_pipe = pipeline("translation", model=mdl, tokenizer=tok)
        return _hi_en_pipe

    # fallback: identity
    return lambda x: [{"translation_text": x}]

# -------- TEXT GENERATION (ad copy) --------
def generate_copy_gpt(brand: Dict, brief: Dict, n: int = 4) -> List[Dict]:
    """
    HF replacement for previous GPT function.
    Returns a list of {headline, primary_text, tags}.
    """
    if AI_PROVIDER == "gemini":
        # Delegate to Gemini provider
        from .gemini import generate_copy as _g_copy
        return _g_copy(brand, brief, n=n)
    if not USE_HF:
        raise RuntimeError("HF disabled (set AI_PROVIDER=hf or AI_PROVIDER=gemini)")

    tone = ", ".join(brand.get("tone", []))
    palette = ", ".join(brand.get("palette", []))
    banned = ", ".join(brand.get("banned_phrases", [])) or "none"

    # We prompt flan-t5 to output JSON; we’ll still validate/repair below
    prompt = f"""
You are BrandCopyGen.
Brand: {brand.get('name','')}
Tone: {tone}
Palette: {palette}
Banned: {banned}

Brief:
- Product: {brief['product']}
- Audience: {brief['audience']}
- Value props: {", ".join(brief['value_props'])}
- CTA: {brief['cta']}
- Channels: {", ".join(brief.get('channels', []))}

Task: Create {n} diverse ad variants. Return ONLY a JSON array of objects:
[{{"headline":"<=40 chars","primary_text":"<=120 chars","tags":["2-4 words"]}}, ...]
Keep tone; avoid ALL CAPS; ≤1 emoji if tone includes "playful".
""".strip()

    pipe = _lazy_text_pipe()
    out = pipe(prompt, max_new_tokens=256, temperature=0.7)[0]["generated_text"]

    # Try to parse JSON; repair if needed
    try:
        data = json.loads(out)
        if not isinstance(data, list):
            raise ValueError("not a list")
    except Exception:
        # fallback: simple templated variants
        base = f"{brief['value_props'][0]} • {brief['cta']}"
        seeds = ["Level up your day", f"{brief['product']}: pure boost",
                 "Energy that lasts", "Zero sugar. All power."]
        return [{"headline": h[:40], "primary_text": base[:120], "tags": ["fallback"]} for h in seeds[:n]]

    cleaned = []
    for item in data[:n]:
        cleaned.append({
            "headline": str(item.get("headline", ""))[:40],
            "primary_text": str(item.get("primary_text", ""))[:120],
            "tags": item.get("tags", [])[:4]
        })
    # Ensure non-empty fallback
    if not cleaned:
        base = f"{brief['value_props'][0]} • {brief['cta']}"
        cleaned = [{"headline": "Energy that lasts", "primary_text": base, "tags": ["fallback"]}]
    return cleaned

# -------- LOCALIZATION / “TRANSCREATION” --------
def transcreate_copy_gpt(brand: Dict, brief: Dict, copy: Dict, region: str) -> Dict:
    """
    HF replacement for previous GPT transcreation.
    For demo: IN↔US both English, but we show currency/idiom tweak.
    If you want Hindi, we translate EN->HI with MarianMT.
    """
    if AI_PROVIDER == "gemini":
        from .gemini import transcreate_copy as _g_trans
        return _g_trans(brand, brief, copy, region)
    tone = ", ".join(brand.get("tone", []))
    headline = copy["headline"]
    body = copy["primary_text"]

    # Simple regional tweaks first
    if region == "IN":
        # Optionally translate to Hindi with MarianMT (comment out if you want EN)
        to_hi = _lazy_translator("en", "hi")
        try:
            h_hi = to_hi(headline)[0]["translation_text"]
            b_hi = to_hi(body)[0]["translation_text"]
            return {"headline": h_hi[:40], "primary_text": b_hi[:120], "notes": "MarianMT en→hi"}
        except Exception:
            return {"headline": headline[:40], "primary_text": f"₹ {body}"[:120], "notes": "fallback ₹"}

    if region == "US":
        # Keep English, ensure $
        return {"headline": headline[:40], "primary_text": f"$ {body}"[:120], "notes": "US currency"}

    # Default: return as-is
    return {"headline": headline[:40], "primary_text": body[:120], "notes": "no-op"}

# -------- IMAGES (placeholder for now) --------
_pipe = None

def _lazy_sd():
    global _pipe
    if _pipe is None:
        # Lazy import heavy deps so Gemini-only setups can run without them
        try:
            from diffusers import StableDiffusionPipeline
            import torch
        except Exception as e:
            raise RuntimeError("Stable Diffusion dependencies not installed") from e
        model_id = "runwayml/stable-diffusion-v1-5"
        _pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        _pipe = _pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return _pipe

def generate_image_gpt(brand: Dict, brief: Dict, copy: Dict, prompt_info: Dict | None = None) -> str:
    if AI_PROVIDER == "gemini":
        from .gemini import generate_image as _g_img
        return _g_img(brand, brief, copy, prompt_info=prompt_info)
    # HF/local fallback (Stable Diffusion)
    pipe = _lazy_sd()
    if prompt_info and isinstance(prompt_info, dict):
        prompt = prompt_info.get("prompt") or ""
    else:
        prompt = f"""
Ad image for {brief['product']} targeting {brief.get('audience','')}.
Style: {", ".join(brand.get('tone', []))}, modern, minimal.
Include slogan: {copy['headline']}
Brand colors: {", ".join(brand.get('palette', []))}
"""
    image = pipe(prompt).images[0]

    fname = f"C{int(time.time()*1000)}{random.randint(100,999)}.png"
    out_path = STATIC_DIR / fname
    image.save(out_path)
    return f"/static/{fname}"
