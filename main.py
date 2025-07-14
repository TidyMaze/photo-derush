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

def show_lightroom_ui(image_paths, directory, trashed_paths=None, trashed_dir=None):
    import threading
    print(f"[Lightroom UI] Preparing to load {len(image_paths)} images from {directory}.")
    selected_idx = [None]
    def update_thumbnails_threadsafe(n):
        import time
        start = time.time()
        print(f"[Lightroom UI] update_thumbnails_threadsafe: Start loading {n} images.")
        image_data = []
        hashes = []
        valid_paths = []
        for img_name in image_paths[:n]:
            print(f"[Lightroom UI] Processing image: {img_name}")
            img_path = os.path.join(directory, img_name)
            try:
                dh = compute_dhash(img_path)
                hash_hex = int(str(dh), 16)
                hash_bytes = np.array([(hash_hex >> (8 * i)) & 0xFF for i in range(8)], dtype='uint8')[::-1]
                hashes.append(hash_bytes)
                valid_paths.append(img_name)
                img = Image.open(img_path)
                img.thumbnail((150, 150))
                image_data.append((img, img_name))
            except Exception as e:
                print(f"[Lightroom UI] Error processing {img_name}: {e}")
                hashes.append(None)
                valid_paths.append(img_name)
        print(f"[Lightroom UI] Finished hashing. Time elapsed: {time.time() - start:.2f}s")
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
        print(f"[Lightroom UI] Duplicate clustering done. Time elapsed: {time.time() - start:.2f}s")
        def update_ui():
            for widget in frame.winfo_children():
                widget.destroy()
            thumbs.clear()
            for idx, (img, img_name) in enumerate(image_data):
                tk_img = ImageTk.PhotoImage(img, master=frame)
                cell_w, cell_h = 160, 160
                if idx in duplicate_indices:
                    border_color = "red"
                else:
                    border_color = "#444"
                lbl = Label(frame, image=tk_img, bg="#222", bd=2, relief="solid")
                lbl.image = tk_img
                lbl.grid(row=idx//5, column=idx%5, padx=5, pady=5)
                def on_click(event, i=idx, label=lbl):
                    selected_idx[0] = i
                    label.config(bg="#555")
                def on_double_click(event, img_path=os.path.join(directory, img_name)):
                    open_full_image(img_path)
                lbl.bind("<Button-1>", on_click)
                lbl.bind("<Double-Button-1>", on_double_click)
                thumbs.append(tk_img)
            frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
        root.after(0, update_ui)
    def update_thumbnails(n):
        import time
        start = time.time()
        print(f"[Lightroom UI] update_thumbnails: Start loading {n} images.")
        for widget in frame.winfo_children():
            widget.destroy()
        thumbs.clear()
        hashes = []
        valid_paths = []
        for img_name in image_paths[:n]:
            print(f"[Lightroom UI] Processing image: {img_name}")
            img_path = os.path.join(directory, img_name)
            try:
                dh = compute_dhash(img_path)
                hash_hex = int(str(dh), 16)
                hash_bytes = np.array([(hash_hex >> (8 * i)) & 0xFF for i in range(8)], dtype='uint8')[::-1]
                hashes.append(hash_bytes)
                valid_paths.append(img_name)
            except Exception as e:
                print(f"[Lightroom UI] Error processing {img_name}: {e}")
                hashes.append(None)
                valid_paths.append(img_name)
        print(f"[Lightroom UI] Finished hashing. Time elapsed: {time.time() - start:.2f}s")
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
        print(f"[Lightroom UI] Duplicate clustering done. Time elapsed: {time.time() - start:.2f}s")
        for img_name in valid_paths:
            img_path = os.path.join(directory, img_name)
            try:
                img = Image.open(img_path)
                img.thumbnail((150, 150))  # Always load as low-res thumbnail
                tk_img = ImageTk.PhotoImage(img, master=frame)
                idx = valid_paths.index(img_name)
                cell_w, cell_h = 160, 160
                x = (idx % 5) * cell_w + (cell_w - img.width) // 2
                y = (idx // 5) * cell_h + (cell_h - img.height) // 2
                if idx in duplicate_indices:
                    border_color = "red"
                else:
                    border_color = "#444"
                lbl = Label(frame, image=tk_img, bg="#222", bd=2, relief="solid")
                lbl.image = tk_img
                lbl.grid(row=idx//5, column=idx%5, padx=5, pady=5)
                def on_click(event, i=idx, label=lbl):
                    selected_idx[0] = i
                    label.config(bg="#555")
                def on_double_click(event, img_path=img_path):
                    open_full_image(img_path)
                lbl.bind("<Button-1>", on_click)
                lbl.bind("<Double-Button-1>", on_double_click)
                thumbs.append(tk_img)
            except Exception as e:
                print(f"[Lightroom UI] Error displaying {img_name}: {e}")
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
    root = Tk()
    print("[Lightroom UI] Tkinter window created.")
    root.title("Photo Derush - Minimalist Lightroom UI")
    root.configure(bg="#222")
    canvas = Canvas(root, bg="#222")
    print("[Lightroom UI] Canvas created.")
    canvas.pack(side="top", fill="both", expand=True)
    scroll_y = Scrollbar(canvas, orient="vertical", command=canvas.yview)
    scroll_y.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scroll_y.set)
    frame = Frame(canvas, bg="#222")
    frame_id = canvas.create_window((0,0), window=frame, anchor="nw")
    print("[Lightroom UI] Frame for images created.")
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    frame.bind("<Configure>", on_frame_configure)
    def _on_mousewheel(event):
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
    root.geometry("1100x900")
    root.resizable(True, True)
    default_n = min(50, len(image_paths))
    print(f"[Lightroom UI] Loading thumbnails for {default_n} images.")
    def on_slider_change(v):
        print(f"[Lightroom UI] Slider changed: {v}")
        update_thumbnails(int(v))
    slider = Scale(root, from_=1, to=min(50, len(image_paths)), orient=HORIZONTAL, bg="#222", fg="#fff", highlightthickness=0, troughcolor="#444", label="Number of images", font=("Arial", 12))
    slider.set(default_n)
    slider.config(command=on_slider_change)
    print("[Lightroom UI] Slider created.")
    toolbar = Frame(root, bg="#333")
    toolbar.place(x=0, rely=850, relwidth=1, height=50)
    selector_label = Label(toolbar, text="Number of images:", bg="#333", fg="#fff", font=("Arial", 12))
    selector_label.pack(side="left", padx=10, pady=5)
    slider.pack_forget()
    slider.pack(in_=toolbar, side="left", padx=10, pady=5)
    print("[Lightroom UI] Toolbar and slider packed.")
    # Load thumbnails in a separate thread
    def load_images_thread():
        update_thumbnails_threadsafe(default_n)
    threading.Thread(target=load_images_thread, daemon=True).start()
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
    all_images = [images[0]] + trashed
    print(f"[Duplicate Slayer] Loading {len(all_images[:50])} images in UI.")
    show_lightroom_ui(all_images[:50], image_dir, trashed_paths=trashed[:49], trashed_dir=trash_dir)
    return kept, [os.path.join(trash_dir, t) for t in trashed]

if __name__ == "__main__":
    import sys
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
            show_lightroom_ui(images[:50], directory)
        else:
            print("No images found.")
        main_duplicate_detection()
