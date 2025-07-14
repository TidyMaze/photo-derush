import os
from tkinter import Tk, Frame, Label, Scale, HORIZONTAL, Canvas, Scrollbar
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
    print(f"[Lightroom UI] Preparing to load {len(image_paths)} images from {directory}.")
    selected_idx = [None]
    def open_full_image(img_path):
        top = Tk()
        top.title("Full Image Viewer")
        top.configure(bg="#222")
        img = Image.open(img_path)
        screen_w = top.winfo_screenwidth()
        screen_h = top.winfo_screenheight()
        img.thumbnail((screen_w, screen_h))
        tk_img = ImageTk.PhotoImage(img, master=top)
        lbl = Label(top, image=tk_img, bg="#222")
        lbl.pack(fill="both", expand=True)
        top.geometry(f"{screen_w}x{screen_h}+0+0")
        top.attributes('-fullscreen', True)
        lbl.image = tk_img
        def close(event):
            top.destroy()
        lbl.bind("<Button-1>", close)
        top.mainloop()
    def update_thumbnails(n):
        for widget in frame.winfo_children():
            widget.destroy()
        thumbs.clear()
        # Cluster duplicates using dHash + FAISS
        hashes = []
        valid_paths = []
        for img_name in image_paths[:n]:
            print(f"[Lightroom UI] Processing image: {img_name}")
            img_path = os.path.join(directory, img_name)

            try:
                dh = compute_dhash(img_path)
                # Convert dHash to 8 bytes (uint8 array)
                hash_hex = int(str(dh), 16)
                hash_bytes = np.array([(hash_hex >> (8 * i)) & 0xFF for i in range(8)], dtype='uint8')[::-1]
                hashes.append(hash_bytes)
                valid_paths.append(img_name)
            except Exception:
                hashes.append(None)
                valid_paths.append(img_name)
        duplicate_indices = set()
        if hashes and all(h is not None for h in hashes):
            hashes_np = np.stack(hashes).astype('uint8')
            index = faiss.IndexBinaryFlat(64)
            index.add(hashes_np)
            for i, h in enumerate(hashes_np):
                res = index.range_search(h[np.newaxis, :], 5)
                lims, D, I = res
                if lims[1] - lims[0] > 1:
                    for j in I[lims[0]:lims[1]]:
                        if j != i:
                            duplicate_indices.add(j)
        # Display kept images
        for idx, img_name in enumerate(image_paths[:n]):
            img_path = os.path.join(directory, img_name)
            pil_img = Image.open(img_path)
            pil_img.thumbnail((180, 180))
            tk_img = ImageTk.PhotoImage(pil_img)
            thumbs.append(tk_img)
            try:
                img_hash = str(compute_dhash(img_path))
            except Exception:
                img_hash = "error"
            def on_click(event, i=idx):
                selected_idx[0] = i
                for w in frame.winfo_children():
                    w.config(highlightbackground="#FFD700" if w == event.widget else "#222", highlightthickness=2 if w == event.widget else 0)
            def on_double_click(event, i=idx, img_path=img_path):
                open_full_image(img_path)
            # Add duplicate logo if image is a duplicate
            logo = None
            if idx in duplicate_indices:
                logo = Label(frame, text="⧉", bg="#222", fg="#FF4444", font=("Arial", 24))
                logo.grid(row=idx//5, column=idx%5, padx=10, pady=(10,80), sticky="ne")
            lbl = Label(frame, image=tk_img, bg="#222", highlightthickness=0)
            lbl.grid(row=idx//5, column=idx%5, padx=10, pady=10)
            lbl.bind("<Button-1>", on_click)
            lbl.bind("<Double-Button-1>", on_double_click)
            hash_lbl = Label(frame, text=img_hash, bg="#222", fg="#aaa", font=("Arial", 8))
            hash_lbl.grid(row=idx//5, column=idx%5, padx=10, pady=(110,0))
        # Display trashed images if provided
        if trashed_paths and trashed_dir:
            for idx, img_name in enumerate(trashed_paths):
                img_path = os.path.join(trashed_dir, img_name)
                pil_img = Image.open(img_path)
                pil_img.thumbnail((180, 180))
                tk_img = ImageTk.PhotoImage(pil_img)
                thumbs.append(tk_img)
                try:
                    img_hash = str(compute_dhash(img_path))
                except Exception:
                    img_hash = "error"
                def on_click(event, i=n+idx):
                    selected_idx[0] = i
                    for w in frame.winfo_children():
                        w.config(highlightbackground="#FFD700" if w == event.widget else ("#800" if w.cget("bg") == "#800" else "#222"), highlightthickness=2 if w == event.widget else 0)
                def on_double_click(event, i=n+idx, img_path=img_path):
                    open_full_image(img_path)
                lbl = Label(frame, image=tk_img, bg="#800", text="TRASHED", compound="top", fg="#fff", highlightthickness=0)
                lbl.grid(row=(n+idx)//5, column=(n+idx)%5, padx=10, pady=10)
                lbl.bind("<Button-1>", on_click)
                lbl.bind("<Double-Button-1>", on_double_click)
                hash_lbl = Label(frame, text=img_hash, bg="#800", fg="#fff", font=("Arial", 8))
                hash_lbl.grid(row=(n+idx)//5, column=(n+idx)%5, padx=10, pady=(110,0))
    root = Tk()
    print("[Lightroom UI] Tkinter window created.")
    root.title("Photo Derush - Minimalist Lightroom UI")
    root.configure(bg="#222")
    # Add a scrolling panel for images
    canvas = Canvas(root, bg="#222")
    print("[Lightroom UI] Canvas created.")
    canvas.pack(side="top", fill="both", expand=True)
    scroll_y = Scrollbar(canvas, orient="vertical", command=canvas.yview)
    scroll_y.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scroll_y.set)
    # Frame for images inside canvas
    frame = Frame(canvas, bg="#222")
    frame_id = canvas.create_window((0,0), window=frame, anchor="nw")
    print("[Lightroom UI] Frame for images created.")
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    frame.bind("<Configure>", on_frame_configure)
    def _on_mousewheel(event):
        # macOS and Windows: event.delta, Linux: event.num
        if hasattr(event, 'delta'):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)
    canvas.bind_all("<Shift-MouseWheel>", _on_mousewheel)
    thumbs = []
    # Set a fixed window size
    root.geometry("1100x900")
    root.resizable(True, True)
    default_n = min(50, len(image_paths))
    print(f"[Lightroom UI] Loading thumbnails for {default_n} images.")
    update_thumbnails(default_n)
    slider = Scale(root, from_=1, to=min(50, len(image_paths)), orient=HORIZONTAL, bg="#222", fg="#fff", highlightthickness=0, troughcolor="#444", label="Number of images", font=("Arial", 12), command=lambda v: update_thumbnails(int(v)))
    slider.set(default_n)
    print("[Lightroom UI] Slider created.")
    toolbar = Frame(root, bg="#333")
    toolbar.place(x=0, rely=850, relwidth=1, height=50)
    selector_label = Label(toolbar, text="Number of images:", bg="#333", fg="#fff", font=("Arial", 12))
    selector_label.pack(side="left", padx=10, pady=5)
    slider.pack_forget()
    slider.pack(in_=toolbar, side="left", padx=10, pady=5)
    print("[Lightroom UI] Toolbar and slider packed.")
    root.mainloop()
    print("[Lightroom UI] Tkinter mainloop exited.")


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
    print(f"[Duplicate Slayer] Found {len(images)} images in {image_dir}.")
    if not images:
        print("[Duplicate Slayer] No images found. Exiting.")
        return [], []
    trashed = []
    kept = [os.path.join(image_dir, images[0])]
    for img in images[1:]:
        src = os.path.join(image_dir, img)
        dst = os.path.join(trash_dir, img)
        print(f"[Duplicate Slayer] Moving duplicate: {img} -> trash.")
        os.rename(src, dst)
        trashed.append(img)
    # Show up to 50 images (kept + trashed) in the UI
    all_images = [images[0]] + trashed
    print(f"[Duplicate Slayer] Loading {len(all_images[:50])} images in UI.")
    show_lightroom_ui(all_images[:50], image_dir, trashed_paths=trashed[:49], trashed_dir=trash_dir)
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
            show_lightroom_ui(images[:100], directory)
        else:
            print("No images found.")
        main_duplicate_detection()
