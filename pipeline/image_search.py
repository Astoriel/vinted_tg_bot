import os
import httpx
from PIL import Image
from io import BytesIO
import torchvision.transforms as T


class ImageSearcher:
    """Fetches random images from Unsplash/Pexels based on keywords."""

    UNSPLASH_URL = "https://api.unsplash.com/photos/random"
    PEXELS_URL = "https://api.pexels.com/v1/search"

    def __init__(self):
        from config import UNSPLASH_ACCESS_KEY, PEXELS_API_KEY
        self.unsplash_key = UNSPLASH_ACCESS_KEY
        self.pexels_key = PEXELS_API_KEY
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.client = httpx.AsyncClient(timeout=10.0)

    async def search(self, words):
        """Search for image matching words. Returns (PIL Image, url)."""
        query = " ".join(words)

        # Try Unsplash
        if self.unsplash_key:
            try:
                resp = await self.client.get(
                    self.UNSPLASH_URL,
                    params={"query": query, "orientation": "squarish"},
                    headers={"Authorization": f"Client-ID {self.unsplash_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    image_url = data["urls"]["small"]
                    return await self._download(image_url), image_url
            except Exception:
                pass

        # Fallback to Pexels
        if self.pexels_key:
            try:
                resp = await self.client.get(
                    self.PEXELS_URL,
                    params={"query": query, "per_page": 1, "size": "small"},
                    headers={"Authorization": self.pexels_key},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("photos"):
                        image_url = data["photos"][0]["src"]["small"]
                        return await self._download(image_url), image_url
            except Exception:
                pass

        # Final fallback
        return self._generate_fallback(), None

    async def _download(self, url):
        resp = await self.client.get(url)
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def _generate_fallback(self):
        """Simple gray placeholder image."""
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def preprocess(self, pil_image):
        """PIL Image -> normalized tensor [1, 3, 224, 224]."""
        return self.transform(pil_image).unsqueeze(0)

    async def close(self):
        await self.client.aclose()
