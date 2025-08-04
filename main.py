import os
from tkinter import Tk, Frame, Label, Scale, HORIZONTAL, Canvas, Scrollbar
from PIL import Image, ImageTk
import imagehash
import faiss
import numpy as np
import cv2
from photo_derush.aesthetic import compute_nima_score

MAX_IMAGES = 200

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

def compute_blur_score(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()

def show_lightroom_ui(image_paths, directory, trashed_paths=None, trashed_dir=None):
    import threading
    print(f"[Lightroom UI] Preparing to load {len(image_paths)} images from {directory}.")
    selected_idx = [None]
    thumbs = []
    root = Tk()
    root.title("Photo Derush")
    canvas = Canvas(root, bg="#222")
    vscroll = Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    vscroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    frame = Frame(canvas, bg="#222")
    frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    frame.bind("<Configure>", on_frame_configure)
    def _on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
    # Bind mouse wheel events to both canvas and frame for proper focus
    for widget in (canvas, frame):
        widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", _on_mousewheel))
        widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))
        widget.bind_all("<Button-4>", _on_mousewheel)
        widget.bind_all("<Button-5>", _on_mousewheel)

    THUMB_SIZE = 160

    def get_images_per_row():
        width = canvas.winfo_width()
        if width < THUMB_SIZE:
            return 1
        return max(1, ((width // THUMB_SIZE) - 1))

    def relayout_grid():
        images_per_row = get_images_per_row()
        for pos, lbl in enumerate(image_labels):
            row = pos // images_per_row
            col = pos % images_per_row
            lbl.grid(row=row, column=col, padx=5, pady=5)
            top_labels[pos].grid(row=row, column=col, sticky="n", padx=5, pady=(0, 30))
            bottom_labels[pos].grid(row=row, column=col, sticky="s", padx=5, pady=(30, 0))
            blur_labels[pos].grid(row=row, column=col, sticky="e", padx=5, pady=(0, 0))
            aesthetic_labels[pos].grid(row=row, column=col, sticky="w", padx=5, pady=(0, 0))
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_resize(event):
        relayout_grid()

    canvas.bind('<Configure>', on_resize)
    # Add placeholder widgets for each image immediately
    placeholder_img = Image.new('RGB', (150, 150), color=(80, 80, 80))
    placeholder_tk_img = ImageTk.PhotoImage(placeholder_img, master=frame)
    image_labels = []
    top_labels = []
    bottom_labels = []
    blur_labels = []
    aesthetic_labels = []
    for pos, img_name in enumerate(image_paths[:MAX_IMAGES]):
        lbl = Label(frame, image=placeholder_tk_img, bg="#444", bd=4, relief="solid", highlightbackground="#444", highlightthickness=4)
        lbl.image = placeholder_tk_img
        lbl.grid(row=0, column=0)
        img_path = os.path.join(directory, img_name)
        try:
            date_str = str(os.path.getmtime(img_path))
        except Exception:
            date_str = "N/A"
        top_label = Label(frame, text="Loading...", bg="#222", fg="red", font=("Arial", 9, "bold"))
        top_label.grid(row=0, column=0, sticky="n")
        bottom_label = Label(frame, text=f"{img_name}\nDate: {date_str}", bg="#222", fg="white", font=("Arial", 9))
        bottom_label.grid(row=0, column=0, sticky="s")
        blur_label = Label(frame, text="Blur: ...", bg="#222", fg="yellow", font=("Arial", 9))
        blur_label.grid(row=0, column=0, sticky="e")
        aesthetic_label = Label(frame, text="Aesthetic: ...", bg="#222", fg="cyan", font=("Arial", 9))
        aesthetic_label.grid(row=0, column=0, sticky="w")
        image_labels.append(lbl)
        top_labels.append(top_label)
        bottom_labels.append(bottom_label)
        blur_labels.append(blur_label)
        aesthetic_labels.append(aesthetic_label)
    relayout_grid()
    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))
    def update_thumbnails(n, threaded=False):
        import time
        start = time.time()
        print(f"[Lightroom UI] update_thumbnails: Start loading {n} images.")
        image_data = [None] * n
        hashes = [None] * n
        valid_paths = [None] * n
        thumbnail_dir = os.path.join(directory, 'thumbnails')
        os.makedirs(thumbnail_dir, exist_ok=True)
        for idx, img_name in enumerate(image_paths[:n]):
            img_path = os.path.join(directory, img_name)
            thumb_path = os.path.join(thumbnail_dir, img_name)
            try:
                dh = compute_dhash(img_path)
                hash_hex = int(str(dh), 16)
                hash_bytes = np.array([(hash_hex >> (8 * i)) & 0xFF for i in range(8)], dtype='uint8')[::-1]
                hashes[idx] = hash_bytes
                valid_paths[idx] = img_name
                if os.path.exists(thumb_path):
                    img = Image.open(thumb_path)
                else:
                    img = Image.open(img_path)
                    img.thumbnail((150, 150))
                    img.save(thumb_path)
                image_data[idx] = (img, img_name)
                tk_img = ImageTk.PhotoImage(img, master=frame)
                image_labels[idx].config(image=tk_img)
                image_labels[idx].image = tk_img
                top_labels[idx].config(text=f"Hash: {''.join(f'{b:02x}' for b in hash_bytes)}")
                bottom_labels[idx].config(text=f"{img_name}\nDate: {str(os.path.getmtime(os.path.join(directory, img_name)))}")
                blur_score = compute_blur_score(img_path)
                blur_labels[idx].config(text=f"Blur: {blur_score:.1f}" if blur_score is not None else "Blur: N/A")
                aesthetic_score = compute_nima_score(img_path)
                aesthetic_labels[idx].config(text=f"Aesthetic: {aesthetic_score:.2f}" if aesthetic_score is not None else "Aesthetic: N/A")
            except Exception as e:
                print(f"[Lightroom UI] Error processing {img_name}: {e}")
                hashes[idx] = None
                valid_paths[idx] = img_name
                info_labels[idx].config(text="Error loading")
        print(f"[Lightroom UI] Finished hashing. Time elapsed: {time.time() - start:.2f}s")
        # Compute groups in a separate thread after all images are loaded
        def compute_and_update_groups():
            group_ids, group_cardinality, hash_map = compute_duplicate_groups(hashes)
            sorted_indices = sorted(
                range(len(valid_paths)),
                key=lambda i: (-group_cardinality.get(group_ids[i], 1 if group_ids[i] else 0), group_ids[i] if group_ids[i] else 9999, i)
            )
            for pos, idx in enumerate(sorted_indices):
                img, img_name = image_data[idx]
                tk_img = ImageTk.PhotoImage(img, master=frame)
                border_color = "red" if selected_idx[0] == idx else ("red" if group_ids[idx] else "#444")
                image_labels[pos].config(image=tk_img, bg=border_color, highlightbackground=border_color)
                image_labels[pos].image = tk_img
                top_text = ""
                if group_ids[idx]:
                    top_text += f"Group {group_ids[idx]}\n"
                top_text += f"Hash: {hash_map[idx]}"
                top_labels[pos].config(text=top_text)
                bottom_labels[pos].config(text=f"{img_name}\nDate: {str(os.path.getmtime(os.path.join(directory, img_name)))}")
                blur_score = compute_blur_score(os.path.join(directory, img_name))
                blur_labels[pos].config(text=f"Blur: {blur_score:.1f}" if blur_score is not None else "Blur: N/A")
                aesthetic_score = compute_nima_score(os.path.join(directory, img_name))
                aesthetic_labels[pos].config(text=f"Aesthetic: {aesthetic_score:.2f}" if aesthetic_score is not None else "Aesthetic: N/A")
                def on_click(event, i=idx, label=image_labels[pos]):
                    selected_idx[0] = i
                    print(f"[Lightroom UI] Image selected: {valid_paths[i]}")
                    for lbl in image_labels:
                        lbl.config(bg="#444", highlightbackground="#444")
                    label.config(bg="red", highlightbackground="red")
                def on_double_click(event, img_path=os.path.join(directory, img_name)):
                    open_full_image(img_path)
                image_labels[pos].bind("<Button-1>", on_click)
                image_labels[pos].bind("<Double-Button-1>", on_double_click)
            frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
        threading.Thread(target=compute_and_update_groups, daemon=True).start()
    threading.Thread(target=lambda: update_thumbnails(min(MAX_IMAGES, len(image_paths)), threaded=True), daemon=True).start()
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

def compute_duplicate_groups(hashes):
    print("[Duplicate Groups] Computing duplicate groups...")
    if not hashes or not all(h is not None for h in hashes):
        return {}, {}, {}
    hashes_np = np.stack(hashes).astype('uint8')
    index = faiss.IndexBinaryFlat(64)
    index.add(hashes_np)
    group_ids = {}
    group_cardinality = {}
    hash_map = {}
    current_group = 1
    clusters = []
    assigned = set()
    for i in range(len(hashes_np)):
        if i in assigned:
            continue
        res = index.range_search(hashes_np[i][np.newaxis, :], 5)
        lims, D, I = res
        cluster = set(j for j in I[lims[0]:lims[1]] if j != i)
        cluster.add(i)
        if len(cluster) > 1:
            for j in cluster:
                group_ids[j] = current_group
                assigned.add(j)
            group_cardinality[current_group] = len(cluster)
            clusters.append(list(cluster))
            current_group += 1
    for idx in range(len(hashes)):
        if idx not in group_ids:
            group_ids[idx] = None
    for idx, h in enumerate(hashes):
        if h is not None:
            hash_map[idx] = ''.join(f'{b:02x}' for b in h)
        else:
            hash_map[idx] = None
    return group_ids, group_cardinality, hash_map

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
    print(f"[Duplicate Slayer] Loading {len(all_images[:MAX_IMAGES])} images in UI.")
    show_lightroom_ui(all_images[:MAX_IMAGES], image_dir, trashed_paths=trashed[:MAX_IMAGES-1], trashed_dir=trash_dir)
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
            show_lightroom_ui(images[:MAX_IMAGES], directory)
        else:
            print("No images found.")
        main_duplicate_detection()
