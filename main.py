import os
from tkinter import Tk, Frame, Label, Scale, HORIZONTAL
from PIL import Image, ImageTk
import imagehash
import faiss
import numpy as np


def list_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in exts]


def list_extensions(directory):
    return sorted(set(os.path.splitext(f)[1].lower() for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f)) and '.' in f))


def is_image_extension(ext):
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def show_lightroom_ui(image_paths, directory, trashed_paths=None, trashed_dir=None):
    def update_thumbnails(n):
        for widget in frame.winfo_children():
            widget.destroy()
        thumbs.clear()
        # Display kept images
        for idx, img_name in enumerate(image_paths[:n]):
            img_path = os.path.join(directory, img_name)
            pil_img = Image.open(img_path)
            pil_img.thumbnail((180, 180))
            tk_img = ImageTk.PhotoImage(pil_img)
            thumbs.append(tk_img)
            lbl = Label(frame, image=tk_img, bg="#222")
            lbl.grid(row=idx//5, column=idx%5, padx=10, pady=10)
        # Display trashed images if provided
        if trashed_paths and trashed_dir:
            for idx, img_name in enumerate(trashed_paths):
                img_path = os.path.join(trashed_dir, img_name)
                pil_img = Image.open(img_path)
                pil_img.thumbnail((180, 180))
                tk_img = ImageTk.PhotoImage(pil_img)
                thumbs.append(tk_img)
                lbl = Label(frame, image=tk_img, bg="#800", text="TRASHED", compound="top", fg="#fff")
                lbl.grid(row=(n+idx)//5, column=(n+idx)%5, padx=10, pady=10)

    root = Tk()
    root.title("Photo Derush - Minimalist Lightroom UI")
    root.configure(bg="#222")
    frame = Frame(root, bg="#222")
    frame.pack(padx=20, pady=20)
    thumbs = []
    # Set a fixed window size
    root.geometry("1100x500")
    root.resizable(False, False)
    update_thumbnails(min(10, len(image_paths)))
    slider = Scale(root, from_=1, to=len(image_paths), orient=HORIZONTAL, bg="#222", fg="#fff", highlightthickness=0, troughcolor="#444", label="Number of images", font=("Arial", 12), command=lambda v: update_thumbnails(int(v)))
    slider.set(min(10, len(image_paths)))
    slider.pack(side="left", anchor="sw", padx=20, pady=20)
    root.mainloop()


def compute_dhash(image_path):
    img = Image.open(image_path)
    return imagehash.dhash(img)


def cluster_duplicates(image_paths, directory, hamming_thresh=5):
    hashes = []
    valid_paths = []
    for img_name in image_paths:
        img_path = os.path.join(directory, img_name)
        try:
            dh = compute_dhash(img_path)
            hashes.append(np.array([int(str(dh), 16)], dtype='uint64'))
            valid_paths.append(img_name)
        except Exception:
            continue
    if not hashes:
        return []
    hashes_np = np.array(hashes)
    index = faiss.IndexBinaryFlat(64)
    index.add(hashes_np)
    clusters = []
    visited = set()
    for i, h in enumerate(hashes_np):
        if i in visited:
            continue
        D, I = index.range_search(h, hamming_thresh)
        cluster = [valid_paths[j] for j in I if j not in visited]
        for j in I:
            visited.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters


def main_duplicate_detection():
    directory = '/Users/yannrolland/Pictures/photo-dataset'
    images = list_images(directory)
    clusters = cluster_duplicates(images, directory)
    print(f"Found {len(clusters)} duplicate clusters.")
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx+1}: {cluster}")


def duplicate_slayer(image_dir, trash_dir):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if not images:
        return [], []
    trashed = []
    kept = [os.path.join(image_dir, images[0])]
    for img in images[1:]:
        src = os.path.join(image_dir, img)
        dst = os.path.join(trash_dir, img)
        os.rename(src, dst)
        trashed.append(img)
    show_lightroom_ui([images[0]], image_dir, trashed_paths=trashed, trashed_dir=trash_dir)
    return kept, [os.path.join(trash_dir, t) for t in trashed]


if __name__ == "__main__":
    import sys
    # Only run main app if not running under pytest or any test file
    if not any(x in sys.argv[0] for x in ["pytest", "test_", "_test"]):
        directory = '/Users/yannrolland/Pictures/photo-dataset'
        print("Welcome to Photo Derush Script!")
        images = list_images(directory)
        print(f"Found {len(images)} images.")
        for img in images:
            print(img)
        exts = list_extensions(directory)
        print(f"Extensions found: {', '.join(exts)}")
        non_image_exts = [e for e in exts if not is_image_extension(e)]
        if non_image_exts:
            print(f"Warning: Non-image extensions detected: {', '.join(non_image_exts)}")
        if len(images) > 0:
            show_lightroom_ui(images[:10], directory)
        else:
            print("No images found.")
        main_duplicate_detection()
