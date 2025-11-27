import os
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import requests
import json
import base64

class ImageAnalyzer:
    """Analyzer for image files using PIL and Vision Models."""

    def __init__(self, vision_model: str = "llava"):
        self.vision_model = vision_model
        self.ollama_base_url = "http://localhost:11434/api/generate"

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image to extract metadata and description.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary containing metadata and description.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        result = {
            "filename": path.name,
            "path": str(path),
            "metadata": {},
            "description": None,
            "error": None
        }

        try:
            with Image.open(path) as img:
                result["metadata"] = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "info": {k: str(v) for k, v in img.info.items()}
                }
                
                # Attempt to get description from Vision Model
                description = self._get_image_description(path)
                if description:
                    result["description"] = description

        except Exception as e:
            result["error"] = str(e)

        return result

    def _get_image_description(self, image_path: Path) -> Optional[str]:
        """Get image description using Ollama vision model."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            payload = {
                "model": self.vision_model,
                "prompt": "Describe this image in detail.",
                "images": [encoded_string],
                "stream": False
            }

            response = requests.post(self.ollama_base_url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("response")
            else:
                return None
        except Exception:
            return None
