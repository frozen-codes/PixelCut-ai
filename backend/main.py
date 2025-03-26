from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageEnhance
import io
import os
from dotenv import load_dotenv
import numpy as np
import logging
import cv2
from skimage import exposure
from rembg import remove

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rembg with error handling
try:
    from rembg import remove
    REMBG_AVAILABLE = True
    logger.info("rembg initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize rembg: {str(e)}")
    REMBG_AVAILABLE = False

load_dotenv()

app = FastAPI(title="PixelCut AI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def enhance_image_quality(image: Image.Image, params: dict) -> Image.Image:
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Apply denoising if requested
    if params.get('denoise', True):
        img_array = cv2.fastNlMeansDenoisingColored(
            img_array, None, 
            h=10,  # Luminance component weight
            hColor=10,  # Color components weight
            templateWindowSize=7,  # Size of template patch
            searchWindowSize=21  # Size of window for weighted average
        )
    
    # Apply sharpening if requested
    if params.get('sharpen', True):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_array = cv2.filter2D(img_array, -1, kernel)
    
    # Convert back to PIL Image for color/contrast adjustments
    enhanced_img = Image.fromarray(img_array)
    
    # Enhance colors if requested
    if params.get('enhance_colors', True):
        # Color enhancement
        color_enhancer = ImageEnhance.Color(enhanced_img)
        enhanced_img = color_enhancer.enhance(1.2)  # Increase color saturation by 20%
        
        # Contrast enhancement
        contrast_enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = contrast_enhancer.enhance(params.get('contrast', 1.1))
        
        # Brightness enhancement
        brightness_enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = brightness_enhancer.enhance(params.get('brightness', 1.1))
    
    return enhanced_img

@app.get("/")
async def root():
    return {"message": "Welcome to PixelCut AI API"}

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    try:
        if not REMBG_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Background removal service is currently unavailable"
            )
            
        # Read image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Remove background with alpha matting for better edge detection
        output_image = remove(
            input_image,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr, 
            media_type="image/png"
        )
    
    except Exception as e:
        logger.error(f"Error in remove_background: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    enhance_quality: bool = True,
    enhance_colors: bool = True,
    sharpen: bool = True,
    denoise: bool = True,
    brightness: float = 1.1,
    contrast: float = 1.1
):
    try:
        # Read image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Apply enhancements
        params = {
            'enhance_quality': enhance_quality,
            'enhance_colors': enhance_colors,
            'sharpen': sharpen,
            'denoise': denoise,
            'brightness': brightness,
            'contrast': contrast
        }
        
        enhanced_image = enhance_image_quality(input_image, params)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        enhanced_image.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr, 
            media_type="image/png"
        )
    
    except Exception as e:
        logger.error(f"Error in enhance_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply-filter/{filter_type}")
async def apply_filter(filter_type: str, file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array for processing
        img_array = np.array(input_image)
        
        if filter_type == "artistic":
            # Apply artistic enhancement using adaptive histogram equalization
            img_array = exposure.equalize_adapthist(img_array, clip_limit=0.03)
            img_array = (img_array * 255).astype(np.uint8)
            
            # Add slight color boost
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            hsv[..., 1] = hsv[..., 1] * 1.2  # Increase saturation
            img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown filter type: {filter_type}")
        
        # Convert back to PIL Image
        output_image = Image.fromarray(img_array)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr, 
            media_type="image/png"
        )
    
    except Exception as e:
        logger.error(f"Error in apply_filter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 