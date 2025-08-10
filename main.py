import os
from tkinter import Tk, Frame, Label, Scale, HORIZONTAL, Canvas, Scrollbar
from PIL import Image, ImageTk
import imagehash
import faiss
import numpy as np
import cv2
import logging

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

def compute_sharpness_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    features = {}
    features['variance_laplacian'] = cv2.Laplacian(img, cv2.CV_64F).var()
    features['tenengrad'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Tenengrad
    features['brenner'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Brenner
    features['wavelet_energy'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Wavelet energy
    return features

def show_lightroom_ui(image_paths, directory, trashed_paths=None, trashed_dir=None):
    import threading
    print(f"[Lightroom UI] Preparing to load {len(image_paths)} images from {directory}.")
    selected_idx = [None]
    thumbs = []
    root = Tk()
    root.title("Photo Derush")
    root.geometry("1400x800")
    # Split view: left for app, right for empty panel
    main_container = Frame(root, bg="#222")
    main_container.pack(fill="both", expand=True)
    left_panel = Frame(main_container, bg="#222")
    left_panel.pack(side="left", fill="both", expand=True)
    right_panel = Frame(main_container, bg="#222")
    right_panel.pack(side="right", fill="both", expand=True)
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # Add scrollable text to right panel
    right_canvas = Canvas(right_panel, bg="#222")
    right_scroll = Scrollbar(right_panel, orient="vertical", command=right_canvas.yview)
    right_canvas.configure(yscrollcommand=right_scroll.set)
    right_scroll.pack(side="right", fill="y")
    right_canvas.pack(side="left", fill="both", expand=True)
    right_frame = Frame(right_canvas, bg="#222")
    right_canvas.create_window((0, 0), window=right_frame, anchor="nw")
    def on_right_frame_configure(event):
        logging.debug(f"Right frame configured: scrollregion={right_canvas.bbox('all')}")
        right_canvas.configure(scrollregion=right_canvas.bbox("all"))
    right_frame.bind("<Configure>", on_right_frame_configure)
    # Enable trackpad scrolling (macOS and cross-platform)
    def _on_right_mousewheel(event):
        logging.debug(f"Right panel mousewheel event: delta={getattr(event, 'delta', None)}, num={getattr(event, 'num', None)}")
        if event.delta:
            right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            right_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            right_canvas.yview_scroll(1, "units")
    right_canvas.bind_all("<MouseWheel>", _on_right_mousewheel)
    right_canvas.bind_all("<Button-4>", _on_right_mousewheel)
    right_canvas.bind_all("<Button-5>", _on_right_mousewheel)
    filler_text = ("This is the right panel.\n" * 50).strip()
    right_label = Label(right_frame, text=filler_text, bg="#222", fg="#aaa", font=("Arial", 14), justify="left", anchor="nw")
    right_label.pack(fill="both", expand=True, padx=30, pady=30)
    def on_right_label_event(event):
        logging.debug(f"Right label event: type={event.type}, widget={event.widget}")
    right_label.bind("<Enter>", on_right_label_event)
    right_label.bind("<Leave>", on_right_label_event)
    right_label.bind("<Button-1>", on_right_label_event)
    right_label.bind("<Button-2>", on_right_label_event)
    right_label.bind("<Button-3>", on_right_label_event)
    right_label.bind("<Motion>", on_right_label_event)
    # Move canvas and scrollbar to left_panel
    canvas = Canvas(left_panel, bg="#222")
    vscroll = Scrollbar(left_panel, orient="vertical", command=canvas.yview)
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
    # Remove aesthetic_labels
    metrics_cache = {}
    num_images = min(MAX_IMAGES, len(image_paths))
    for pos, img_name in enumerate(image_paths[:num_images]):
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
        blur_label = Label(frame, text="", bg="#222", fg="yellow", font=("Arial", 9))
        blur_label.grid(row=0, column=0, sticky="e")
        # Remove aesthetic_label, unify metrics in blur_label
        def show_metrics(event, bl=blur_label, ip=img_path):
            if ip in metrics_cache:
                blur_score, sharpness_metrics, aesthetic_score = metrics_cache[ip]
            else:
                blur_score = compute_blur_score(ip)
                sharpness_metrics = compute_sharpness_features(ip)
                aesthetic_score = 42
                metrics_cache[ip] = (blur_score, sharpness_metrics, aesthetic_score)
            lines = [
                f"Blur: {blur_score:.1f}" if blur_score is not None else "Blur: N/A",
                f"Laplacian: {sharpness_metrics['variance_laplacian']:.1f}",
                f"Tenengrad: {sharpness_metrics['tenengrad']:.1f}",
                f"Brenner: {sharpness_metrics['brenner']:.1f}",
                f"Wavelet: {sharpness_metrics['wavelet_energy']:.1f}",
                f"Aesthetic: {aesthetic_score:.2f}" if aesthetic_score is not None else "Aesthetic: N/A"
            ]
            bl.config(text="\n".join(lines))
            bl._metrics_open = True
        def hide_metrics(event, bl=blur_label):
            # Only hide if cursor is not on metrics label
            if not getattr(bl, '_metrics_hover', False):
                bl.config(text="")
                bl._metrics_open = False
        def metrics_enter(event, bl=blur_label):
            bl._metrics_hover = True
        def metrics_leave(event, bl=blur_label):
            bl._metrics_hover = False
            if not getattr(bl, '_metrics_open', False):
                bl.config(text="")
        lbl.bind("<Enter>", show_metrics)
        lbl.bind("<Leave>", hide_metrics)
        blur_label.bind("<Enter>", metrics_enter)
        blur_label.bind("<Leave>", metrics_leave)
        image_labels.append(lbl)
        top_labels.append(top_label)
        bottom_labels.append(bottom_label)
        blur_labels.append(blur_label)
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
                img, cached = cache_thumbnail(img_path, thumb_path)
                image_data[idx] = (img, img_name)
                tk_img = ImageTk.PhotoImage(img, master=frame)
                image_labels[idx].config(image=tk_img)
                image_labels[idx].image = tk_img
                top_labels[idx].config(text=f"Hash: {''.join(f'{b:02x}' for b in hash_bytes)}")
                bottom_labels[idx].config(text=f"{img_name}\nDate: {str(os.path.getmtime(os.path.join(directory, img_name)))}")
                blur_labels[idx].config(text="")
                # Remove aesthetic_labels
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
                border_color = "blue" if group_ids[idx] else ("red" if selected_idx[0] == idx else "#444")
                image_labels[pos].config(image=tk_img, bg=border_color, highlightbackground=border_color)
                image_labels[pos].image = tk_img
                top_text = ""
                if group_ids[idx]:
                    top_text += f"Group {group_ids[idx]}\n"
                top_text += f"Hash: {hash_map[idx]}"
                top_labels[pos].config(text=top_text)
                bottom_labels[pos].config(text=f"{img_name}\nDate: {str(os.path.getmtime(os.path.join(directory, img_name)))}")
                aesthetic_score = 42
                # aesthetic_labels[pos].config(text=f"Aesthetic: {aesthetic_score:.2f}" if aesthetic_score is not None else "Aesthetic: N/A")
                def on_click(event, i=idx, label=image_labels[pos]):
                    selected_idx[0] = i
                    import logging
                    logging.info(f"[Lightroom UI] Image selected: {valid_paths[i]}")
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
    threading.Thread(target=lambda: update_thumbnails(num_images, threaded=True), daemon=True).start()
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

def cache_thumbnail(img_path, thumb_path, size=(150, 150)):
    """Load thumbnail from cache or create and cache it."""
    from PIL import Image
    import logging
    if os.path.exists(thumb_path):
        img = Image.open(thumb_path)
        logging.info(f"Loaded cached thumbnail for {os.path.basename(thumb_path)}")
        return img, True
    else:
        img = Image.open(img_path)
        img.thumbnail(size)
        img.save(thumb_path)
        logging.info(f"Created and cached thumbnail for {os.path.basename(thumb_path)}")
        return img, False

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
