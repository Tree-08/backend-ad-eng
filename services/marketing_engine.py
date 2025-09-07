import os
import io
import json
import time
import random
import uuid
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError
from google.api_core.exceptions import ResourceExhausted, NotFound
from dotenv import load_dotenv
from urllib.parse import urlparse
import hashlib

# This class encapsulates the core logic for generating ad creatives.
class MarketingEngine:
    """
    Handles the two-step process of generating a marketing brief from 5Ps
    and then creating an image from that brief.
    """
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # API key handling (support both GOOGLE_API_KEY and GEMINI_API_KEY)
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Error: GOOGLE_API_KEY or GEMINI_API_KEY must be set in environment or .env file.")
        genai.configure(api_key=self.api_key)

        # Model Configuration (allow override via env)
        self.text_model_name = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-flash-latest")
        self.image_model_name = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image-preview")

        # System instructions for the text model
        self.system_instructions = """
        You are a creative director. Your task is to generate an image generation prompt based on the provided product marketing mix (5Ps).
        Output ONLY a valid JSON object with the following fields:
        {"prompt":"...", "negative":"...", "style":"...", "aspect_ratio":"1:1|4:5|3:4|16:9", "safety":"..."}
        Do not include any other text, prose, comments, or markdown formatting like ```json.
        """

        # In‑memory feedback memory keyed by 5Ps signature
        self._feedback_memory: dict[str, list[str]] = {}

    def _call_api_with_retry(self, api_call_func, api_name="API", max_retries=5, initial_delay=10):
        """Wrapper to handle API calls with exponential backoff for retries."""
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                return api_call_func()
            except ResourceExhausted as e:
                retries += 1
                if retries >= max_retries:
                    raise
                jitter = random.uniform(0, 5)
                print(f"Quota exceeded for {api_name}. Retrying in {delay + jitter:.2f}s...")
                time.sleep(delay + jitter)
                delay *= 2
            except Exception as e:
                print(f"An unexpected error occurred with {api_name}: {e}")
                raise
        raise RuntimeError(f"API call for {api_name} failed after {max_retries} retries.")


    def _key_from_5ps(self, fiveps_data: dict) -> str:
        # Stable hash of the five P's to group a session
        payload = json.dumps({k: fiveps_data.get(k, "") for k in ["product","price","place","promotion","people"]}, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _generate_brief_from_5ps(self, fiveps_data, current_feedback: str | None = None):
        """Step 1: Calls the text model to get a JSON creative brief."""
        print("Step 1: Generating creative brief...")
        text_model = genai.GenerativeModel(
            model_name=self.text_model_name,
            system_instruction=self.system_instructions
        )
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=800,
            temperature=0.4,
        )

        # Pull feedback history for this 5Ps set and include in the payload
        key = self._key_from_5ps(fiveps_data)
        history = self._feedback_memory.get(key, [])
        payload = {
            "fiveps": fiveps_data,
            "feedback_history": history,
            "current_feedback": (current_feedback or "").strip(),
            # Guidance to incorporate contact details and feedback explicitly
            "instructions": "Incorporate feedback and contact details if provided; render small, readable contact text (e.g., phone, email) in a clean area. Keep composition balanced; avoid large text blocks."
        }

        def api_call():
            return text_model.generate_content(
                json.dumps(payload),
                generation_config=generation_config
            )

        response = self._call_api_with_retry(api_call, "Text Model")
        
        raw_text = response.text
        print(f"Raw JSON output from text model:\n{raw_text}")
        
        clean_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        brief = json.loads(clean_text)
        
        prompt = brief.get("prompt")
        if not prompt:
            raise ValueError(f"No 'prompt' field found in the generated brief: {brief}")
            
        return prompt

    def _generate_image_from_prompt(self, prompt):
        """Step 2: Calls the image model to generate an image from the prompt."""
        print("\nStep 2: Generating image from prompt...")
        print(f"Prompt: {prompt}")
        
        image_model = genai.GenerativeModel(self.image_model_name)

        def api_call():
            return image_model.generate_content(prompt)

        img_resp = self._call_api_with_retry(api_call, "Image Model")

        try:
            image_part = next(p for p in img_resp.candidates[0].content.parts if p.inline_data.mime_type.startswith("image/"))
            return image_part.inline_data.data
        except (StopIteration, IndexError, AttributeError):
            raise RuntimeError(f"Could not extract image data from API response. Full response:\n{img_resp}")

    def _extract_inline_image(self, resp):
        image_part = next(p for p in resp.candidates[0].content.parts if getattr(getattr(p, "inline_data", None), "mime_type", "").startswith("image/"))
        return image_part.inline_data.data

    def _generate_image_from_prompt_with_reference(self, prompt: str, ref_image_path: str):
        """Generate a new image using a reference image + text prompt.
        Falls back to text-only generation if the model doesn't emit an image with reference input.
        """
        image_model = genai.GenerativeModel(self.image_model_name)

        # 1) Try file-upload + reference conditioned generation (if supported by the model)
        try:
            uploaded = genai.upload_file(path=ref_image_path)

            def api_call_ref():
                return image_model.generate_content([uploaded, prompt])

            img_resp = self._call_api_with_retry(api_call_ref, "Image Model (with reference)")
            try:
                return self._extract_inline_image(img_resp)
            except (StopIteration, IndexError, AttributeError):
                # Fall through to text-only if no inline image was returned
                pass
        except Exception:
            # If upload/reference flow isn't supported, try text-only prompt
            pass

        # 2) Fallback: text-only with explicit style retention instruction
        def api_call_text():
            return image_model.generate_content(f"{prompt}. Keep visual style consistent with the previously selected reference image: similar palette, lighting, and composition cues.")

        img_resp2 = self._call_api_with_retry(api_call_text, "Image Model (text-only fallback)")
        try:
            return self._extract_inline_image(img_resp2)
        except (StopIteration, IndexError, AttributeError):
            raise RuntimeError(f"Could not extract image data from API response. The model returned no image parts. Full response:\n{img_resp2}")

    def _variant_prompts(self, base_prompt: str, feedback: str | None = None):
        """
        Create related-but-different variants by nudging composition/style.
        Keeps the core prompt intact and adds lightweight directives.
        """
        variants = [
            "studio macro shot, centered subject, soft rim lighting, condensation details",
            "lifestyle in-situ scene, shallow depth of field, warm golden hour light",
            "top-down flat lay composition on textured surface, high contrast",
            "dynamic action shot with subtle motion blur, cool lighting",
        ]
        out = []
        fb = (feedback or "").strip()
        fb_str = f" Incorporate feedback: {fb}." if fb else ""
        for v in variants:
            out.append(f"{base_prompt} — {v}.{fb_str}")
        return out

    def generate_creative(self, fiveps_data):
        """
        Executes the full workflow: 5Ps -> Brief -> Image.
        Saves the image to a file and returns its path.
        """
        # Ensure output directory exists
        output_dir = "backend/static/generated_images"
        os.makedirs(output_dir, exist_ok=True)

        try:
            prompt = self._generate_brief_from_5ps(fiveps_data)
            # Build 4 related variants (initially no feedback)
            prompts = self._variant_prompts(prompt, feedback=None)

            web_paths = []
            for p in prompts:
                img_bytes = self._generate_image_from_prompt(p)
                filename = f"{uuid.uuid4()}.png"
                filepath = os.path.join(output_dir, filename)
                self._save_png_white_bg(img_bytes, filepath)
                web_paths.append(f"/static/generated_images/{filename}")

            print(f"\n✅ Success! Generated {len(web_paths)} images under {output_dir}")
            # Return the list; keep first for backward compatibility at the route layer
            return web_paths

        except (ValueError, RuntimeError, json.JSONDecodeError, UnidentifiedImageError) as e:
            print(f"An error occurred during creative generation: {e}")
            return None

    def _resolve_local_static_path(self, selected_image_url: str) -> str:
        """Map a /static/... url (or full http URL) to a local filesystem path."""
        path = selected_image_url
        if path.startswith("http://") or path.startswith("https://"):
            path = urlparse(path).path
        if not path.startswith("/static/"):
            raise ValueError("selected_image_url must point to /static path")
        rel = path[len("/static/"):]
        fs_path = os.path.join("backend", "static", rel)
        if not os.path.isfile(fs_path):
            raise FileNotFoundError(f"Selected image not found: {fs_path}")
        return fs_path

    def regenerate_from_selection(self, fiveps_data, selected_image_url: str, feedback: str | None = None):
        """
        Given the user's selected image and original 5Ps, generate 4 related variants
        that maintain the reference style while exploring new compositions.
        Returns a list of web paths.
        """
        output_dir = "backend/static/generated_images"
        os.makedirs(output_dir, exist_ok=True)

        # 1) Build base prompt again from the 5Ps
        # Log feedback into memory so it compounds across regenerations
        key = self._key_from_5ps(fiveps_data)
        if feedback:
            self._feedback_memory.setdefault(key, []).append(feedback)
        prompt = self._generate_brief_from_5ps(fiveps_data, current_feedback=feedback)
        prompts = self._variant_prompts(prompt, feedback=feedback)

        # 2) Load reference image from local static path
        fs_path = self._resolve_local_static_path(selected_image_url)

        # 3) Generate 4 variants conditioned on the reference image
        web_paths = []
        for p in prompts:
            enriched = f"{p} — maintain visual style and theme of the reference image"
            img_bytes = self._generate_image_from_prompt_with_reference(enriched, fs_path)
            filename = f"{uuid.uuid4()}.png"
            filepath = os.path.join(output_dir, filename)
            self._save_png_white_bg(img_bytes, filepath)
            web_paths.append(f"/static/generated_images/{filename}")

        print(f"\n✅ Success! Regenerated {len(web_paths)} images from selection under {output_dir}")
        return web_paths

    def _save_png_white_bg(self, img_bytes: bytes, filepath: str):
        """Save an image, compositing on a white background if it has transparency.
        This prevents black/dark fill around content in some viewers.
        """
        try:
            im = Image.open(io.BytesIO(img_bytes))
            if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                bg.save(filepath, format="PNG")
            else:
                im.convert("RGB").save(filepath, format="PNG")
        except Exception:
            # Fallback: write raw bytes; better to have a file than fail
            with open(filepath, "wb") as f:
                f.write(img_bytes)

    # -------- Social copy (title + caption) --------
    def generate_social_copy(self, platform: str, fiveps_data: dict, feedback: str | None = None) -> dict:
        """Generate platform-tailored title + caption using Gemini text model."""
        plat = (platform or "").lower()
        if plat not in {"instagram", "linkedin", "twitter", "youtube"}:
            plat = "instagram"

        limits = {
            "instagram": "Caption <= 2,000 chars, include 5-10 relevant hashtags at end.",
            "linkedin": "Caption <= 1,000 chars, professional tone, no more than 5 hashtags.",
            "twitter": "Tweet <= 280 chars, concise, 1-3 hashtags.",
            "youtube": "Title <= 70 chars; description <= 4,000 chars, first line compelling.",
        }[plat]

        text_model = genai.GenerativeModel(model_name=self.text_model_name)
        gen_cfg = genai.GenerationConfig(response_mime_type="application/json", max_output_tokens=600, temperature=0.6)

        key = self._key_from_5ps(fiveps_data)
        history = self._feedback_memory.get(key, [])
        payload = {
            "platform": plat,
            "fiveps": fiveps_data,
            "feedback_history": history,
            "current_feedback": (feedback or "").strip(),
            "constraints": limits,
        }
        prompt = (
            "You are a social media copywriter. Create platform-tailored copy from the 5Ps and feedback.\n"
            "Return ONLY JSON with keys: title, caption, hashtags (array). Keep it safe and brand-appropriate."
        )
        resp = text_model.generate_content([prompt, json.dumps(payload)], generation_config=gen_cfg)
        txt = getattr(resp, "text", "{}")
        try:
            data = json.loads(txt.strip().strip("`"))
        except Exception:
            data = {}
        def tidy_text(s: str) -> str:
            s = (s or "").strip()
            # Capitalize first letter of each sentence; ensure terminal punctuation
            import re
            parts = re.split(r"([.!?]\s+)", s)
            out = []
            for i in range(0, len(parts), 2):
                seg = parts[i].strip()
                if not seg:
                    continue
                seg = seg[:1].upper() + seg[1:]
                punct = parts[i+1] if i+1 < len(parts) else ". "
                if not re.search(r"[.!?]$", seg):
                    out.append(seg + punct.strip())
                else:
                    out.append(seg)
            txt = " ".join(out).strip()
            return txt

        raw_title = data.get("title") or f"Introducing {fiveps_data.get('product','')}"
        raw_caption = data.get("caption") or fiveps_data.get("promotion", "")
        title = tidy_text(raw_title)
        caption = tidy_text(raw_caption)
        hashtags = data.get("hashtags") or []
        # Minimal post-process for Twitter limit
        if plat == "twitter" and len(caption) > 275:
            caption = caption[:272] + "…"
        # Extract simple contact info from feedback if present
        contact_email = None
        website = None
        phone = None
        import re
        text_blob = " ".join([fiveps_data.get("promotion", ""), feedback or "", caption])
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text_blob)
        if m: contact_email = m.group(0)
        u = re.search(r"(https?://\S+|www\.[\w.-]+\.[A-Za-z]{2,})", text_blob)
        if u: website = u.group(0)
        p = re.search(r"(\+?\d[\d\s()\-]{6,}\d)", text_blob)
        if p: phone = p.group(0)

        # Guess a company/handle if possible
        company_name = None
        # naive: try to grab proper-noun-ish words from product
        prod = (fiveps_data.get("product") or "").strip()
        if prod:
            company_name = prod.split(" for ")[0].split(" – ")[0].split(" - ")[0].strip()
            # Title-case the guessed name for display
            company_name = " ".join([w.capitalize() for w in company_name.split()])
        if not company_name:
            company_name = "yourcompany"
        handle = "@" + re.sub(r"[^a-z0-9]", "", company_name.lower())[:15]

        return {
            "title": title,
            "caption": caption,
            "hashtags": hashtags,
            "company_name": company_name,
            "handle": handle,
            "contact_email": contact_email,
            "website": website,
            "phone": phone,
            "sponsored": True,
        }
