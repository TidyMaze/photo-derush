import os
import argparse


def list_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in exts]


def main():
    parser = argparse.ArgumentParser(description="Photo Derush CLI App")
    parser.add_argument('--version', action='version', version='Photo Derush 0.1')
    parser.add_argument('--list', action='store_true', help='List all images in the dataset directory')
    parser.add_argument('--dir', type=str, default='/Users/yannrolland/Pictures/photo-dataset', help='Image dataset directory')
    args = parser.parse_args()
    print("Welcome to Photo Derush CLI App!")
    if args.list:
        images = list_images(args.dir)
        print(f"Found {len(images)} images.")


if __name__ == "__main__":
    main()
