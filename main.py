import os
from tkinter import Tk, Frame, Label, Scale, HORIZONTAL
from PIL import Image, ImageTk


def list_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in exts]


def list_extensions(directory):
    return sorted(set(os.path.splitext(f)[1].lower() for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f)) and '.' in f))


def is_image_extension(ext):
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def show_lightroom_ui(image_paths, directory):
    def update_thumbnails(n):
        for widget in frame.winfo_children():
            widget.destroy()
        thumbs.clear()
        for idx, img_name in enumerate(image_paths[:n]):
            img_path = os.path.join(directory, img_name)
            pil_img = Image.open(img_path)
            pil_img.thumbnail((180, 180))
            tk_img = ImageTk.PhotoImage(pil_img)
            thumbs.append(tk_img)
            lbl = Label(frame, image=tk_img, bg="#222")
            lbl.grid(row=idx//5, column=idx%5, padx=10, pady=10)

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
