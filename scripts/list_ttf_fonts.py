import os

font_dirs = [
    "/System/Library/Fonts",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
]

for font_dir in font_dirs:
    if os.path.isdir(font_dir):
        for root, dirs, files in os.walk(font_dir):
            for file in files:
                if file.lower().endswith(".ttf"):
                    print(os.path.join(root, file))
