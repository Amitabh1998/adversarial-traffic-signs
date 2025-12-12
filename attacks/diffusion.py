"""
Diffusion-based adversarial attack generation.
Uses Stable Diffusion to create realistic weather/lighting perturbations.
"""

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default weather/lighting prompts for adversarial generation
DEFAULT_PROMPTS = [
    # Weather conditions
    "in heavy rain", "in dense fog", "in thick snow", "in a blizzard",
    "in a thunderstorm", "in freezing rain", "in sleet", "in hail",
    "in dust storm", "in sandstorm",
    
    # Lighting conditions
    "at night", "at night under street lights", "at sunset with low light",
    "at sunrise", "in bright sunlight", "in harsh midday sun with glare",
    "at dusk in dim lighting", "in deep shadow", "in backlighting (silhouetted)",
    "with lens flare",
    
    # Complex combinations
    "in fog at night", "in rain at dusk", "in fog and snow", "in fog and rain",
    "at night with heavy snow", "in rainstorm with wet reflections",
    "in rainstorm at dusk", "in snow at night", "in hazy sunlight",
    "with patches of snow and fog",
    
    # Surface/visibility conditions
    "covered in frost", "covered in ice", "dirty, muddy surface",
    "covered in graffiti", "faded, washed-out sign", "old, rusty sign",
    "partially covered by ice", "partially obscured by water droplets",
    "with extreme motion blur",
    
    # Additional challenging scenarios
    "almost invisible in snow at dusk", "illuminated only by car headlights",
    "overexposed in midday sun", "underexposed in deep shadow",
    "with strong glare and shadows", "in rain with puddles",
    "at night in fog and rain", "in heavy wind with rain",
    "covered in water spray", "in extreme weather conditions"
]


@dataclass
class DiffusionAttackConfig:
    """Configuration for diffusion-based attacks."""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    strength: float = 0.7
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    prompts: List[str] = field(default_factory=lambda: DEFAULT_PROMPTS)


class DiffusionAttacker:
    """
    Generate adversarial examples using Stable Diffusion.
    
    Creates realistic weather/lighting perturbations that maintain
    semantic content while degrading classifier performance.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_safety_checker: bool = False
    ):
        """
        Initialize the diffusion pipeline.
        
        Args:
            model_id: HuggingFace model ID for Stable Diffusion
            device: Device to run inference on
            dtype: Data type for model (float16 recommended for GPU)
            enable_safety_checker: Whether to enable NSFW filter
        """
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError:
            raise ImportError("Please install diffusers: pip install diffusers")
        
        logger.info(f"Loading Stable Diffusion pipeline: {model_id}")
        
        safety_checker = None if not enable_safety_checker else "default"
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=safety_checker
        ).to(device)
        
        # Disable progress bar for batch processing
        self.pipe.set_progress_bar_config(disable=True)
        
        self.device = device
        self.prompts = DEFAULT_PROMPTS
        
        logger.info("Stable Diffusion pipeline loaded successfully")
    
    def set_prompts(self, prompts: List[str]):
        """Set custom prompts for adversarial generation."""
        self.prompts = prompts
    
    def generate_single(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        strength: float = 0.7,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> Tuple[Image.Image, str]:
        """
        Generate a single adversarial image.
        
        Args:
            image: Input PIL Image
            prompt: Text prompt (random if None)
            strength: How much to transform the image (0-1)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
        
        Returns:
            Tuple of (adversarial image, used prompt)
        """
        if prompt is None:
            prompt = "a photo of a traffic sign " + random.choice(self.prompts)
        
        # Resize to 512x512 for Stable Diffusion
        image_resized = image.convert("RGB").resize((512, 512))
        
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=image_resized,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        
        return result.images[0], prompt
    
    def generate_dataset(
        self,
        image_paths: List[str],
        labels: List[int],
        output_dir: str,
        strength: float = 0.7,
        guidance_scale: float = 7.5,
        num_samples: Optional[int] = None
    ) -> List[Tuple[str, str, int, str]]:
        """
        Generate adversarial dataset from original images.
        
        Args:
            image_paths: List of paths to original images
            labels: List of labels for each image
            output_dir: Directory to save generated images
            strength: Diffusion strength parameter
            guidance_scale: Guidance scale parameter
            num_samples: Number of samples to generate (None for all)
        
        Returns:
            List of (orig_path, adv_path, label, prompt) tuples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample if requested
        if num_samples and num_samples < len(image_paths):
            indices = random.sample(range(len(image_paths)), num_samples)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        metadata = []
        
        logger.info(f"Generating {len(image_paths)} adversarial images...")
        
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                # Load and generate
                orig_image = Image.open(img_path).convert("RGB")
                adv_image, prompt = self.generate_single(
                    orig_image,
                    strength=strength,
                    guidance_scale=guidance_scale
                )
                
                # Save adversarial image
                class_dir = output_dir / f'{label:05d}'
                class_dir.mkdir(exist_ok=True)
                
                save_name = Path(img_path).stem + '_diff.png'
                save_path = class_dir / save_name
                adv_image.save(save_path)
                
                metadata.append((img_path, str(save_path), label, prompt))
                
                # Clear cache periodically
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        logger.info(f"Generated {len(metadata)} adversarial images")
        
        return metadata
    
    @staticmethod
    def load_gtsrb_for_diffusion(
        test_images_dir: str,
        test_csv_path: str
    ) -> Tuple[List[str], List[int]]:
        """
        Load GTSRB test images for diffusion attack generation.
        
        Args:
            test_images_dir: Path to test images directory
            test_csv_path: Path to test CSV with labels
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        try:
            df = pd.read_csv(test_csv_path, sep=';')
            
            for _, row in df.iterrows():
                img_name = row['Filename']
                class_id = row['ClassId']
                img_path = os.path.join(test_images_dir, img_name)
                
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(class_id)
            
            logger.info(f"Loaded {len(image_paths)} images from GTSRB test set")
            
        except Exception as e:
            logger.error(f"Error loading GTSRB: {e}")
        
        return image_paths, labels


def generate_diffusion_attacks(
    data_dir: str,
    output_dir: str,
    num_samples: int = 1000,
    strength: float = 0.7,
    guidance_scale: float = 7.5,
    device: str = "cuda"
) -> List[Tuple[str, str, int, str]]:
    """
    Convenience function to generate diffusion attacks on GTSRB.
    
    Args:
        data_dir: Path to GTSRB dataset
        output_dir: Output directory for adversarial images
        num_samples: Number of samples to generate
        strength: Diffusion strength
        guidance_scale: Guidance scale
        device: Computation device
    
    Returns:
        Metadata list for generated images
    """
    data_dir = Path(data_dir)
    
    # Load test images
    test_images_dir = data_dir / 'GTSRB_final_test_images'
    test_csv_path = data_dir / 'GTSRB_Final_Test_GT' / 'GT-final_test.csv'
    
    image_paths, labels = DiffusionAttacker.load_gtsrb_for_diffusion(
        str(test_images_dir),
        str(test_csv_path)
    )
    
    if not image_paths:
        raise ValueError("No images found in GTSRB dataset")
    
    # Initialize attacker and generate
    attacker = DiffusionAttacker(device=device)
    
    metadata = attacker.generate_dataset(
        image_paths=image_paths,
        labels=labels,
        output_dir=output_dir,
        strength=strength,
        guidance_scale=guidance_scale,
        num_samples=num_samples
    )
    
    return metadata


def save_metadata(metadata: List[Tuple], output_path: str):
    """Save metadata to CSV file."""
    df = pd.DataFrame(metadata, columns=['orig_path', 'adv_path', 'label', 'prompt'])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved metadata to {output_path}")


def load_metadata(metadata_path: str) -> List[Tuple[str, str, int, str]]:
    """Load metadata from CSV file."""
    df = pd.read_csv(metadata_path)
    return list(df.itertuples(index=False, name=None))
