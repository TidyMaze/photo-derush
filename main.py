import os


def list_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in exts]


def list_extensions(directory):
    return sorted(set(os.path.splitext(f)[1].lower() for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f)) and '.' in f))


directory = '/Users/yannrolland/Pictures/photo-dataset'
print("Welcome to Photo Derush Script!")
images = list_images(directory)
print(f"Found {len(images)} images.")
for img in images:
    print(img)
exts = list_extensions(directory)
print(f"Extensions found: {', '.join(exts)}")
