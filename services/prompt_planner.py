from __future__ import annotations

import os, json
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

PROHIBITED = {
    "gun", "revolver", "weapon", "explosive", "drugs",
    "thief", "robber", "without licence", "without license",
}


def _brief_to_dict(brief) -> Dict:
    if hasattr(brief, "model_dump"):
        return brief.model_dump()
    if hasattr(brief, "dict"):
        return brief.dict()
    return dict(brief)


def ensure_safe_5ps(brief) -> None:
    br = _brief_to_dict(brief)
    text = " ".join(str(v).lower() for k, v in br.items() if k in ["product", "price", "place", "promotion", "people"])
    if any(term in text for term in PROHIBITED):
        raise ValueError("Provided 5Ps contain prohibited/illegal content. Please provide a safe, legal product.")


def build_image_prompt_llm(brand: Dict, brief, copy: Dict, region: str | None) -> Dict:
    """
    Use Gemini text model to turn the 5 P's + brand tone/palette into a JSON image prompt.
    Returns a dict with keys: prompt, negative, style, aspect_ratio, safety.
    """
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)

    br = _brief_to_dict(brief)
    enable_loc = bool(br.get("enable_localization", False))
    reg = (region or (br.get("regions", ["US"]) or ["US"])[0]).upper()

    system_instruction = (
        "You are a creative director. Generate an image prompt based on the 5Ps and brand cues. "
        "Return ONLY JSON: {\"prompt\":\"...\", \"negative\":\"...\", \"style\":\"...\", "
        "\"aspect_ratio\":\"1:1|4:5|3:4|16:9\", \"safety\":\"...\"}. No prose or markdown."
    )

    payload = {
        "brand": {
            "name": brand.get("name", ""),
            "palette": brand.get("palette", []),
            "tone": brand.get("tone", []),
        },
        "fiveps": {k: br.get(k, "") for k in ["product", "price", "place", "promotion", "people"]},
        "copy": {"headline": copy.get("headline", ""), "primary_text": copy.get("primary_text", "")},
        "region": reg,
        "localization_enabled": enable_loc,
    }

    model_name = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
    gen_cfg = genai.GenerationConfig(response_mime_type="application/json", max_output_tokens=512, temperature=0.4)
    resp = model.generate_content(json.dumps(payload), generation_config=gen_cfg)
    txt = getattr(resp, "text", "") or "{}"

    try:
        data = json.loads(txt.strip().strip("`"))
    except Exception:
        data = {}

    # Build a conservative fallback if JSON parsing fails
    palette = ", ".join(brand.get("palette", []))
    tone = ", ".join(brand.get("tone", [])) or "modern, clean"
    backdrop = "neutral studio backdrop"
    if enable_loc and reg == "IN":
        backdrop = "café table in Mumbai at golden hour"
    elif enable_loc and reg == "US":
        backdrop = "modern suburban kitchen countertop at dusk"

    default_prompt = (
        f"Photorealistic product shot of '{br.get('product','Product')}', centered subject. "
        f"Background: {backdrop}. Tone: {tone}. Color accents: {palette}. "
        f"Leave clean space in top-right for logo; avoid text overlays."
    )

    return {
        "prompt": data.get("prompt") or default_prompt,
        "negative": data.get("negative") or "no watermarks, no text blocks, no brand logos",
        "style": data.get("style") or "photorealistic product photography",
        "aspect_ratio": data.get("aspect_ratio") or "1:1",
        "safety": data.get("safety") or "standard",
        "region": reg,
    }

