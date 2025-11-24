"""Vision tool using official Google Gemini API for image recognition"""

import google.generativeai as genai
from typing import Optional
import os
from pathlib import Path
from rag_system.core.config import get_config


class VisionTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.vision.enabled', True)
        self.api_key = self.config.get('tools.vision.api_key') or os.getenv('GEMINI_API_KEY')
        
        # Configure Gemini API
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # Use gemini-2.5-flash as the default model
        self.model_name = self.config.get('tools.vision.model', 'gemini-2.5-flash')
        self.max_tokens = self.config.get('tools.vision.max_tokens', 500)
    
    def describe_image(self, image_bytes: bytes, prompt: str = "Describe this image in detail. If it's a logo or emblem, identify the organization.") -> Optional[str]:
        """
        Describe an image using the official Google Gemini API.
        
        Args:
            image_bytes: Raw image bytes
            prompt: The prompt to send with the image
            
        Returns:
            Description of the image, or None if vision is disabled or fails
        """
        if not self.enabled:
            return None
        
        if not self.api_key:
            return None
        
        try:
            # Create the generative model
            model = genai.GenerativeModel(self.model_name)
            
            # Prepare image data using PIL Image format (Gemini expects PIL Image objects)
            from PIL import Image
            import io
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Generate content with prompt and image
            response = model.generate_content([prompt, image])
            
            # Return the text response
            return response.text
                
        except Exception as e:
            # Graceful degradation - return None on any error
            # Print error for debugging
            print(f"Vision API error: {e}")
            return None


# Singleton instance
_vision_tool = None

def get_vision_tool() -> VisionTool:
    """Get or create the vision tool singleton"""
    global _vision_tool
    if _vision_tool is None:
        _vision_tool = VisionTool()
    return _vision_tool
