#!/usr/bin/env python3
"""Quick database verification script."""
import sqlite3
import sys
from pathlib import Path


def verify_database():
    """Verify database contents and EXIF data."""
    db_path = Path(__file__).parent.parent / "photoderush.db"

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    print(f"✅ Database found: {db_path}")
    print(f"   Size: {db_path.stat().st_size:,} bytes\n")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get counts
    projects = cur.execute('SELECT COUNT(*) FROM projects').fetchone()[0]
    images = cur.execute('SELECT COUNT(*) FROM images').fetchone()[0]
    exif = cur.execute('SELECT COUNT(*) FROM exif_data').fetchone()[0]
    labels = cur.execute('SELECT COUNT(*) FROM labels').fetchone()[0]
    tags = cur.execute('SELECT COUNT(*) FROM tags').fetchone()[0]

    print("📊 Database Contents:")
    print(f"   Projects: {projects}")
    print(f"   Images: {images}")
    print(f"   EXIF records: {exif}")
    print(f"   Labels: {labels}")
    print(f"   Tags: {tags}")
    print()

    if exif > 0:
        print("📸 Sample EXIF Data:")
        rows = cur.execute('''
            SELECT 
                i.filename,
                e.camera_make,
                e.camera_model,
                e.focal_length,
                e.aperture,
                e.iso,
                e.shutter_speed
            FROM exif_data e
            JOIN images i ON i.id = e.image_id
            LIMIT 5
        ''').fetchall()

        for filename, make, model, focal, aperture, iso, shutter in rows:
            display_name = filename[:45] + "..." if len(filename) > 45 else filename
            print(f"   {display_name}")
            if make or model:
                print(f"      Camera: {make or ''} {model or ''}")
            if focal or aperture or iso:
                settings = []
                if focal:
                    settings.append(f"{focal}mm")
                if aperture:
                    settings.append(f"f/{aperture}")
                if iso:
                    settings.append(f"ISO{iso}")
                if shutter:
                    settings.append(f"{shutter}s")
                print(f"      Settings: {' '.join(settings)}")
            print()
    else:
        print("⚠️  No EXIF data found!")

    conn.close()
    print("✅ Database verification complete!")

if __name__ == "__main__":
    verify_database()

