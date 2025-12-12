#!/usr/bin/env python3
"""
Generate diffusion-based adversarial examples using Stable Diffusion.

Usage:
    python scripts/generate_diffusion_attacks.py --data_dir /path/to/GTSRB --output_dir diffused_images
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from attacks.diffusion import (
    DiffusionAttacker,
    generate_diffusion_attacks,
    save_metadata
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate diffusion adversarial examples')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to GTSRB dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for images')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--strength', type=float, default=0.7, help='Diffusion strength (0-1)')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("DIFFUSION-BASED ADVERSARIAL ATTACK GENERATION")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Strength: {args.strength}")
    print(f"Guidance scale: {args.guidance_scale}")
    print()
    
    # Generate attacks
    metadata = generate_diffusion_attacks(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        device=args.device
    )
    
    # Save metadata
    metadata_path = Path(args.output_dir) / 'metadata.csv'
    save_metadata(metadata, str(metadata_path))
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Generated {len(metadata)} adversarial images")
    print(f"Images saved to: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
