from __future__ import annotations

import random
from typing import Dict, Optional


def _currency_for_region(region: str) -> str:
    return "₹" if region == "IN" else "$"


def _region_backdrop(region: str) -> str:
    if region == "IN":
        return "café table in Mumbai at golden hour"
    if region == "US":
        return "modern suburban kitchen countertop at dusk"
    return "neutral studio backdrop"


def _aspect_for_channels(channels):
    # Basic heuristic: default 1:1; could map per-channel later
    return "1:1"


def build_image_prompt(brand: Dict, brief: Dict, copy: Dict, region: Optional[str]) -> Dict:
    """
    Build a deterministic, template-driven prompt from 5 P's + brand context.
    Returns a dict with prompt metadata: {prompt, negative_prompt, aspect_ratio, seed}.
    """
    product = brief.get("product", "Product")
    price = brief.get("price", "")
    place = brief.get("place", "")
    promotion = brief.get("promotion", "")
    people = brief.get("people", "")
    channels = brief.get("channels", ["Instagram"]) or ["Instagram"]
    enable_loc = bool(brief.get("enable_localization", False))

    palette = ", ".join(brand.get("palette", []))
    tone = ", ".join(brand.get("tone", [])) or "modern, clean"

    # Region selection if localization enabled
    region = (region or (brief.get("regions", ["US"]) or ["US"])[0]).upper()
    currency = _currency_for_region(region) if enable_loc else ""
    backdrop = _region_backdrop(region) if enable_loc else "neutral studio backdrop"

    headline = copy.get("headline", "")

    # Visual policy: small, subtle price/promo; leave space for logo; avoid text blocks
    price_part = f"Subtle small price tag: {currency}{price}. " if price and currency else ""
    promo_part = f"Show promotion hint: {promotion}. " if promotion else ""
    place_part = f"Background: {backdrop}. " if backdrop else ""
    people_part = f"Audience: {people}. " if people else ""

    style_rules = (
        "photorealistic product photography, centered subject, shallow depth of field, "
        "soft rim lighting, slight vignette, high contrast. "
    )
    brand_rules = (
        f"Color palette accents: {palette}. "
        f"Tone: {tone}. "
        "Leave clean space in top-right for logo; do not render logos. "
    )

    prompt = (
        f"Photorealistic product shot of '{product}' — show primary subject clearly. "
        f"Include slogan text concept only for composition reference: {headline}. Do not add text overlays. "
        f"{place_part}{price_part}{promo_part}{people_part}{brand_rules}{style_rules}"
    ).strip()

    negative_prompt = (
        "no watermarks, no text blocks, no brand logos, no extra limbs, "
        "no distorted anatomy, no deformed hands, no duplicated objects"
    )

    aspect_ratio = _aspect_for_channels(channels)
    seed = random.randint(1, 10_000_000)

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "region": region,
    }

