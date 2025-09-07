"""
Gemini provider: text + image using google-generativeai SDK.

Environment variables:
- GEMINI_API_KEY or GOOGLE_API_KEY
- GEMINI_TEXT_MODEL (default: gemini-1.5-flash)
- GEMINI_IMAGE_MODEL (default: gemini-2.5-flash-image-preview)
"""
from __future__ import annotations

import os, json, time, random, pathlib, base64
from typing import List, Dict, Optional, Any
import google.generativeai as genai

# Static directory for saving generated images
STATIC_DIR = pathlib.Path(__file__).resolve().parents[1] / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ---- Text (google-generativeai) ----
def _gemini_text_model():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    name = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name=name)


def generate_copy(brand: Dict, brief: Dict, n: int = 4) -> List[Dict]:
    tone = ", ".join(brand.get("tone", []))
    palette = ", ".join(brand.get("palette", []))
    banned = ", ".join(brand.get("banned_phrases", [])) or "none"

    prompt = f"""
You are BrandCopyGen.
Brand: {brand.get('name','')}
Tone: {tone}
Palette: {palette}
Strictly avoid phrases: {banned}

Brief:
- Product: {brief.get('product','')}
- Audience: {brief.get('audience','')}
- Value props: {', '.join(brief.get('value_props', []))}
- CTA: {brief.get('cta','')}
- Channels: {', '.join(brief.get('channels', []))}

Task: Create {n} diverse ad variants. Return ONLY JSON array of objects:
[
  {{"headline":"<=40 chars","primary_text":"<=120 chars","tags":["2-4 words"]}},
  ...
]
Keep tone, avoid ALL CAPS, ≤1 emoji if tone includes "playful".
""".strip()

    mdl = _gemini_text_model()
    resp = mdl.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.7,
            "max_output_tokens": 512,
        },
    )

    txt = resp.text if hasattr(resp, "text") else (getattr(resp, "candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""))

    try:
        data = json.loads(txt)
        if not isinstance(data, list):
            raise ValueError("not a list")
    except Exception:
        base = f"{brief.get('value_props', [''])[0]} • {brief.get('cta','')}"
        seeds = ["Level up your day", f"{brief.get('product','')}: pure boost", "Energy that lasts", "Zero sugar. All power."]
        return [{"headline": h[:40], "primary_text": base[:120], "tags": ["fallback"]} for h in seeds[:n]]

    cleaned = []
    for item in data[:n]:
        cleaned.append({
            "headline": str(item.get("headline", ""))[:40],
            "primary_text": str(item.get("primary_text", ""))[:120],
            "tags": list(item.get("tags", []))[:4],
        })
    if not cleaned:
        base = f"{brief.get('value_props', [''])[0]} • {brief.get('cta','')}"
        cleaned = [{"headline": "Energy that lasts", "primary_text": base, "tags": ["fallback"]}]
    return cleaned


def transcreate_copy(brand: Dict, brief: Dict, copy: Dict, region: str) -> Dict:
    """
    Ask Gemini to adapt the copy to the region (currency idioms, dialect, or language).
    Return JSON with {headline, primary_text}.
    """
    tone = ", ".join(brand.get("tone", []))
    headline = copy.get("headline", "")
    body = copy.get("primary_text", "")

    prompt = f"""
You are a marketing transcreation expert. Adapt the following ad copy for region {region}.
- Keep the brand tone: {tone}
- If region is IN, localize to Hindi when appropriate and use ₹; if US, use $ and US idioms.
- Keep headline ≤ 40 chars and primary_text ≤ 120 chars.

Input Copy (JSON): {{"headline": {json.dumps(headline)}, "primary_text": {json.dumps(body)}}}

Return ONLY a JSON object: {{"headline": "...", "primary_text": "..."}}
""".strip()

    mdl = _gemini_text_model()
    resp = mdl.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json", "temperature": 0.6, "max_output_tokens": 256},
    )
    txt = resp.text if hasattr(resp, "text") else ""
    try:
        data = json.loads(txt)
        return {
            "headline": str(data.get("headline", headline))[:40],
            "primary_text": str(data.get("primary_text", body))[:120],
            "notes": "gemini"
        }
    except Exception:
        # Lightweight currency tweak fallback
        if region == "IN":
            return {"headline": headline[:40], "primary_text": f"₹ {body}"[:120], "notes": "fallback ₹"}
        if region == "US":
            return {"headline": headline[:40], "primary_text": f"$ {body}"[:120], "notes": "fallback $"}
        return {"headline": headline[:40], "primary_text": body[:120], "notes": "fallback"}


def _gemini_image_model():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set for images")
    genai.configure(api_key=api_key)
    name = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image-preview")
    return genai.GenerativeModel(model_name=name)


def generate_image(brand: Dict, brief: Dict, copy: Dict, prompt_info: Dict | None = None) -> str:
    if prompt_info and isinstance(prompt_info, dict):
        prompt = prompt_info.get("prompt") or ""
    else:
        prompt = (
            f"Ad image for {brief.get('product','')} targeting {brief.get('audience','')}. "
            f"Style: {', '.join(brand.get('tone', []))}, modern, minimal. "
            f"Include slogan: {copy.get('headline','')}. "
            f"Brand colors: {', '.join(brand.get('palette', []))}."
        )

    mdl = _gemini_image_model()
    resp = mdl.generate_content(prompt)
    # Extract first inline image part
    try:
        cand = resp.candidates[0]
        parts = getattr(cand.content, "parts", [])
        image_part = next(p for p in parts if getattr(getattr(p, "inline_data", None), "mime_type", "").startswith("image/"))
        data = image_part.inline_data.data
        if isinstance(data, str):
            try:
                data = base64.b64decode(data)
            except Exception:
                data = data.encode("latin1")
    except Exception as e:
        raise RuntimeError(f"Gemini image response parsing failed: {e}")

    fname = f"C{int(time.time()*1000)}{random.randint(100,999)}.png"
    out_path = STATIC_DIR / fname
    with open(out_path, "wb") as f:
        f.write(data)
    return f"/static/{fname}"
