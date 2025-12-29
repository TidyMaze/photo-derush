"""Migrate legacy .ratings_tags.json files to SQLite database.

Usage:
    python scripts/migrate_json_to_db.py /path/to/photo/directory
    python scripts/migrate_json_to_db.py --scan-all  # Scan all known directories
"""
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import AsyncSessionLocal, init_db
from api.models import ExifData, Label, Project, Tag
from api.models import Image as ImageModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class JsonToDbMigrator:
    """Migrate .ratings_tags.json files to SQLite database."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def migrate_directory(self, directory: Path) -> dict[str, Any]:
        """Migrate a single photo directory to the database."""
        json_file = directory / ".ratings_tags.json"
        if not json_file.exists():
            logger.warning(f"No .ratings_tags.json found in {directory}")
            return {"status": "skipped", "reason": "no_json_file"}

        logger.info(f"Migrating {directory}...")

        # Load JSON data
        try:
            with open(json_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in {json_file}: {e}")
            return {"status": "error", "reason": "invalid_json"}

        # Create or get project
        project = await self._get_or_create_project(directory)

        # Migrate images, ratings, tags, labels
        stats = {
            "images": 0,
            "ratings": 0,
            "tags": 0,
            "labels": 0,
            "exif": 0,
            "skipped": 0,
        }

        for filename, metadata in data.items():
            if not isinstance(metadata, dict):
                stats["skipped"] += 1
                continue

            image = await self._get_or_create_image(project, directory, filename)
            if not image:
                stats["skipped"] += 1
                continue

            stats["images"] += 1

            # Migrate rating
            if "rating" in metadata and metadata["rating"] != 0:
                image.rating = metadata["rating"]
                stats["ratings"] += 1

            # Migrate tags
            if "tags" in metadata and metadata["tags"]:
                for tag_name in metadata["tags"]:
                    await self._add_tag(image, tag_name)
                    stats["tags"] += 1

            # Migrate label (keep/trash)
            if "label" in metadata and metadata["label"]:
                label_value = metadata["label"]
                source = metadata.get("label_source", "manual")
                confidence = metadata.get("confidence")
                await self._add_label(image, label_value, source, confidence)
                stats["labels"] += 1

            # Extract and save EXIF data
            file_path = directory / filename
            if await self._extract_and_save_exif(image, file_path):
                stats["exif"] += 1

        await self.session.commit()
        logger.info(f"✅ Migrated {directory.name}: {stats}")
        return {"status": "success", "stats": stats, "project_id": project.id}

    async def _get_or_create_project(self, directory: Path) -> Project:
        """Get existing project or create new one."""
        result = await self.session.execute(
            select(Project).where(Project.directory == str(directory))
        )
        project = result.scalar_one_or_none()

        if not project:
            project = Project(
                name=directory.name,
                directory=str(directory),
            )
            self.session.add(project)
            await self.session.flush()
            logger.info(f"Created project: {project.name}")

        return project

    async def _get_or_create_image(
        self, project: Project, directory: Path, filename: str
    ) -> ImageModel | None:
        """Get existing image or create new one."""
        file_path = directory / filename
        if not file_path.exists():
            logger.warning(f"Image file not found: {file_path}")
            return None

        result = await self.session.execute(
            select(ImageModel).where(
                ImageModel.project_id == project.id,
                ImageModel.filename == filename,
            )
        )
        image = result.scalar_one_or_none()

        if not image:
            image = ImageModel(
                project_id=project.id,
                filename=filename,
                file_path=str(file_path),
            )
            self.session.add(image)
            await self.session.flush()

        return image

    async def _add_tag(self, image: ImageModel, tag_name: str) -> None:
        """Add tag to image if not exists."""
        result = await self.session.execute(
            select(Tag).where(
                Tag.image_id == image.id,
                Tag.tag == tag_name,
            )
        )
        existing_tag = result.scalar_one_or_none()

        if not existing_tag:
            tag = Tag(image_id=image.id, tag=tag_name)
            self.session.add(tag)

    async def _extract_and_save_exif(self, image: ImageModel, file_path: Path) -> bool:
        """Extract EXIF data from image file and save to database."""
        try:
            with Image.open(file_path) as img:
                # Get EXIF data
                exif_data = img._getexif() if hasattr(img, '_getexif') and callable(img._getexif) else None
                if not exif_data or not isinstance(exif_data, dict):
                    return False

                # Extract relevant EXIF fields
                exif_dict = {}
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, str(tag))
                    exif_dict[tag_name] = value

                # Parse datetime
                datetime_original = None
                datetime_str = exif_dict.get('DateTimeOriginal') or exif_dict.get('DateTime')
                if datetime_str:
                    try:
                        datetime_original = datetime.strptime(str(datetime_str), '%Y:%m:%d %H:%M:%S')
                    except (ValueError, TypeError):
                        pass

                # Extract camera and lens info
                def get_rational(val):
                    """Convert EXIF rational to float."""
                    if isinstance(val, tuple) and len(val) == 2:
                        return float(val[0]) / float(val[1]) if val[1] != 0 else None
                    return float(val) if val else None

                focal_length = get_rational(exif_dict.get('FocalLength'))
                aperture = get_rational(exif_dict.get('FNumber'))

                # Shutter speed from ExposureTime
                shutter_speed_str = None
                exposure_time = exif_dict.get('ExposureTime')
                if exposure_time:
                    if isinstance(exposure_time, tuple) and len(exposure_time) == 2:
                        shutter_speed_str = f"{exposure_time[0]}/{exposure_time[1]}"
                    else:
                        shutter_speed_str = str(exposure_time)

                # Check if EXIF already exists
                result = await self.session.execute(
                    select(ExifData).where(ExifData.image_id == image.id)
                )
                existing_exif = result.scalar_one_or_none()

                if existing_exif:
                    # Update existing
                    existing_exif.camera_make = exif_dict.get('Make')
                    existing_exif.camera_model = exif_dict.get('Model')
                    existing_exif.lens_model = exif_dict.get('LensModel')
                    existing_exif.focal_length = focal_length
                    existing_exif.aperture = aperture
                    existing_exif.shutter_speed = shutter_speed_str
                    existing_exif.iso = exif_dict.get('ISOSpeedRatings')
                    existing_exif.datetime_original = datetime_original
                    existing_exif.width = img.width
                    existing_exif.height = img.height
                else:
                    # Create new
                    exif_record = ExifData(
                        image_id=image.id,
                        camera_make=exif_dict.get('Make'),
                        camera_model=exif_dict.get('Model'),
                        lens_model=exif_dict.get('LensModel'),
                        focal_length=focal_length,
                        aperture=aperture,
                        shutter_speed=shutter_speed_str,
                        iso=exif_dict.get('ISOSpeedRatings'),
                        datetime_original=datetime_original,
                        width=img.width,
                        height=img.height,
                    )
                    self.session.add(exif_record)

                return True

        except Exception as e:
            logger.debug(f"Failed to extract EXIF from {file_path}: {e}")
            return False

    async def _add_label(
        self, image: ImageModel, label_value: str, source: str, confidence: float | None
    ) -> None:
        """Add or update label for image."""
        result = await self.session.execute(
            select(Label).where(Label.image_id == image.id)
        )
        label = result.scalar_one_or_none()

        if label:
            label.label = label_value
            label.source = source
            label.confidence = confidence
            label.updated_at = datetime.now()
        else:
            label = Label(
                image_id=image.id,
                label=label_value,
                source=source,
                confidence=confidence,
            )
            self.session.add(label)


async def migrate_single_directory(directory_path: str) -> None:
    """Migrate a single directory."""
    directory = Path(directory_path).resolve()
    if not directory.is_dir():
        logger.error(f"Not a directory: {directory}")
        sys.exit(1)

    await init_db()

    async with AsyncSessionLocal() as session:
        migrator = JsonToDbMigrator(session)
        result = await migrator.migrate_directory(directory)

        if result["status"] == "success":
            logger.info("✅ Migration completed successfully!")
            logger.info(f"   Project ID: {result['project_id']}")
            logger.info(f"   Stats: {result['stats']}")
        else:
            logger.error(f"❌ Migration failed: {result.get('reason', 'unknown')}")


async def scan_and_migrate_all() -> None:
    """Scan home directory for .ratings_tags.json files and migrate all."""
    logger.info("Scanning for .ratings_tags.json files...")

    # Common photo directories to scan
    search_paths = [
        Path.home() / "Pictures",
        Path.home() / "Photos",
        Path.home() / "Documents",
    ]

    found_dirs = []
    for search_path in search_paths:
        if search_path.exists():
            for json_file in search_path.rglob(".ratings_tags.json"):
                found_dirs.append(json_file.parent)

    if not found_dirs:
        logger.warning("No .ratings_tags.json files found")
        return

    logger.info(f"Found {len(found_dirs)} directories with .ratings_tags.json")

    await init_db()

    async with AsyncSessionLocal() as session:
        migrator = JsonToDbMigrator(session)

        for directory in found_dirs:
            await migrator.migrate_directory(directory)


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_json_to_db.py <directory>")
        print("       python scripts/migrate_json_to_db.py --scan-all")
        sys.exit(1)

    if sys.argv[1] == "--scan-all":
        asyncio.run(scan_and_migrate_all())
    else:
        directory_path = sys.argv[1]
        asyncio.run(migrate_single_directory(directory_path))


if __name__ == "__main__":
    main()

