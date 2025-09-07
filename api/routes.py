from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from backend.services.marketing_engine import MarketingEngine
import os

# --- Pydantic Model for Request Validation ---
# This defines the expected structure of the incoming JSON payload.
# FastAPI will automatically validate the request against this model.
class FivePsRequest(BaseModel):
    product: str
    price: str
    place: str
    promotion: str
    people: str

class ReGenRequest(FivePsRequest):
    selected_image_url: str
    feedback: str | None = None

class CleanupRequest(BaseModel):
    image_urls: list[str] = []

class SocialCopyRequest(BaseModel):
    platform: str  # instagram | linkedin | twitter | youtube
    product: str
    price: str
    place: str
    promotion: str
    people: str
    feedback: str | None = None

# --- Router ---
router = APIRouter()

# --- Service Instantiation ---
# The core logic is instantiated once when the application starts.
try:
    engine = MarketingEngine()
except ValueError as e:
    # If the API key is missing, we should fail fast.
    print(f"FATAL ERROR: {e}")
    # A real production app might exit or have more robust config handling.
    # For now, we'll let it raise the error on startup.
    raise

# --- API Endpoint ---
@router.post("/api/v1/generate")
async def generate_ad_creative(payload: FivePsRequest, request: Request):
    """
    API endpoint to generate an ad creative from the 5Ps.
    """
    try:
        # Convert Pydantic model to a dictionary to pass to the engine
        fiveps_data = payload.model_dump()
        
        # The generate_creative method now returns a list of web paths
        result = engine.generate_creative(fiveps_data)
        
        if result:
            # Construct absolute URLs
            base = str(request.base_url).rstrip('/')
            if isinstance(result, list):
                image_urls = [f"{base}{p}" for p in result]
                # Backward compatibility: also include first as image_url
                return {
                    "message": "Creatives generated successfully!",
                    "image_urls": image_urls,
                    "image_url": image_urls[0] if image_urls else None
                }
            else:
                image_url = f"{base}{result}"
                return {
                    "message": "Creative generated successfully!",
                    "image_url": image_url
                }
        else:
            # If the engine returns None, it means an internal error occurred.
            raise HTTPException(status_code=500, detail="Failed to generate creative due to an internal engine error.")
            
    except Exception as e:
        print(f"An internal server error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# Optional: expose a tiny root on main app; router keeps endpoints minimal
@router.post("/api/v1/regenerate")
async def regenerate_from_selection(payload: ReGenRequest, request: Request):
    """Generate 4 new creatives based on a user-selected image + same 5Ps."""
    try:
        fiveps_data = payload.model_dump()
        selected_url = fiveps_data.pop("selected_image_url")
        feedback = fiveps_data.pop("feedback", None)
        result = engine.regenerate_from_selection(fiveps_data, selected_url, feedback=feedback)
        if result:
            base = str(request.base_url).rstrip('/')
            image_urls = [f"{base}{p}" for p in result]
            return {"message": "Regenerated creatives successfully!", "image_urls": image_urls}
        raise HTTPException(status_code=500, detail="Failed to regenerate creatives")
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Regenerate error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.post("/api/v1/cleanup")
async def cleanup_images(payload: CleanupRequest):
    """Delete generated images under /static/generated_images. Safe-guards path traversal."""
    deleted = 0
    base_dir = os.path.join("backend", "static", "generated_images")
    os.makedirs(base_dir, exist_ok=True)
    for url in payload.image_urls or []:
        try:
            # Accept absolute URLs or /static paths
            from urllib.parse import urlparse
            path = urlparse(url).path if (url.startswith("http://") or url.startswith("https://")) else url
            if not path.startswith("/static/generated_images/"):
                continue
            rel = path[len("/static/"):]
            fs_path = os.path.normpath(os.path.join("backend", "static", rel))
            # Ensure within base_dir
            if not fs_path.startswith(os.path.abspath(base_dir)) and not os.path.abspath(fs_path).startswith(os.path.abspath(base_dir)):
                continue
            if os.path.isfile(fs_path):
                os.remove(fs_path)
                deleted += 1
        except Exception:
            continue
    return {"deleted": deleted}

@router.post("/api/v1/social_copy")
async def social_copy(payload: SocialCopyRequest):
    try:
        fiveps = {
            "product": payload.product,
            "price": payload.price,
            "place": payload.place,
            "promotion": payload.promotion,
            "people": payload.people,
        }
        data = engine.generate_social_copy(payload.platform, fiveps, feedback=payload.feedback)
        return data
    except Exception as e:
        print(f"social_copy error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate social copy")
