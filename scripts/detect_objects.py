#!/usr/bin/env python3
"""
Object Detection Script for Photo Derush

Extracts objects (faces, cats, bridges, etc.) from images using pretrained models.
Caches results to avoid re-computation.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Import the main object detection functions
from src.object_detection import get_objects_for_images, get_available_classes

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='Extract object detections from images')
    parser.add_argument('--images', '-i', nargs='+', help='Image file paths')
    parser.add_argument('--directory', '-d', help='Directory containing images')
    parser.add_argument('--cache-file', '-c', default='.cache/object_detections.joblib',
                       help='Cache file for detections')
    parser.add_argument('--confidence', '-t', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='cpu',
                       help='Device to run model on (cpu=CPU, cuda=NVIDIA GPU, mps=Apple Silicon GPU, auto=best available). Note: MPS may be slower for this model.')
    parser.add_argument('--max-size', type=int, default=600,
                       help='Maximum image dimension for processing (default: 600, smaller=faster)')
    parser.add_argument('--classes', nargs='+', choices=get_available_classes(),
                       help='Limit detection to specific classes (e.g., person car dog)')
    parser.add_argument('--interesting-only', action='store_true',
                       help='Detect only "interesting" classes (person, car, animal, etc.)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: auto, max 8)')
    parser.add_argument('--fast', action='store_true',
                       help='Enable all speed optimizations: smaller images, parallel processing, interesting classes only')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                       help='Number of images per batch per worker (default: 4). Larger batches reduce Python overhead but increase memory use.')

    args = parser.parse_args()

    setup_logging()

    # Handle --fast option (combines multiple optimizations)
    if args.fast:
        args.interesting_only = True
        args.max_size = min(args.max_size, 400)  # Even smaller for speed
        if args.workers is None:
            args.workers = min(os.cpu_count() or 4, 12)  # More workers for speed
        logging.info("Fast mode enabled: interesting classes only, smaller images, parallel processing")

    # Set up classes filter
    classes_filter = None
    if args.classes:
        # Convert class names to indices
        classes_filter = set()
        available_classes = get_available_classes()
        for class_name in args.classes:
            if class_name in available_classes:
                classes_filter.add(available_classes.index(class_name) + 1)  # +1 because COCO_CLASSES[0] is '__background__'
        logging.info(f"Limiting detection to classes: {args.classes}")
    elif args.interesting_only:
        # Get interesting classes from the main module
        all_classes = get_available_classes()
        interesting_classes = {
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'orange',
            'broccoli', 'carrot', 'pizza', 'cake', 'potted plant', 'dining table', 'book',
            'cell phone', 'sink', 'refrigerator'
        }
        classes_filter = set()
        for cls in all_classes:
            if cls in interesting_classes:
                classes_filter.add(all_classes.index(cls) + 1)  # +1 because COCO_CLASSES[0] is '__background__'
        logging.info(f"Using interesting classes only: {sorted(interesting_classes)}")

    # Get image paths
    image_paths = []
    if args.images:
        image_paths.extend(args.images)
    if args.directory:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_paths.extend(Path(args.directory).glob(f'**/{ext}'))

    if not image_paths:
        logging.error("No images specified. Use --images or --directory")
        sys.exit(1)

    logging.info(f"Found {len(image_paths)} images to process")
    if args.max_size != 800:
        logging.info(f"Resizing images to max dimension: {args.max_size}")

    # Process images using the main object detection module
    detections = get_objects_for_images(
        image_paths=image_paths,
        confidence_threshold=args.confidence,
        max_size=args.max_size,
        classes_filter=classes_filter
    )

    # Convert to the expected format for output (list of unique class names with confidence per image)
    output_detections = {}
    for basename, class_confidence_pairs in detections.items():
        # For compatibility with the output format, create detection entries with actual confidence scores
        output_detections[basename] = [
            {'class': cls, 'confidence': conf, 'bbox': [0, 0, 1, 1]} for cls, conf in class_confidence_pairs
        ]

    # Output results
    if args.output:
        output_data = {
            'detections': output_detections,
            'metadata': {
                'model': 'fasterrcnn_resnet50_fpn_v2',
                'confidence_threshold': args.confidence,
                'classes': get_available_classes(),
                'total_images': len(image_paths),
                'processed_images': len(detections),
                'max_size': args.max_size,
                'classes_filter': list(classes_filter) if classes_filter else None
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logging.info(f"Results saved to {args.output}")
    else:
        # Print summary
        total_unique_classes = sum(len(class_confidence_pairs) for class_confidence_pairs in detections.values())
        class_counts = {}
        for class_confidence_pairs in detections.values():
            for class_name, confidence in class_confidence_pairs:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"\nDetection Summary:")
        print(f"Images processed: {len(detections)}")
        print(f"Total unique object types: {total_unique_classes}")
        print(f"Classes detected: {len(class_counts)}")
        print("\nTop detected classes:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {class_name}: {count}")

        # Show confidence scores for detected objects
        print("\nDetected objects with confidence scores:")
        for basename, class_confidence_pairs in detections.items():
            if class_confidence_pairs:
                print(f"  {basename}:")
                for class_name, confidence in sorted(class_confidence_pairs, key=lambda x: x[1], reverse=True):
                    print(f"    {class_name}: {confidence:.3f}")


if __name__ == "__main__":
    main()