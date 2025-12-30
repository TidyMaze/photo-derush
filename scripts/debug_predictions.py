"""Debug script to check prediction query on actual database."""
import asyncio
from pathlib import Path

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import joinedload, sessionmaker

from api.models.image import Image, Label
from api.models.project import Project


async def check_predictions():
    """Check why predictions return 0 results."""
    # Connect to actual database
    import os
    db_path = Path(os.getenv("PHOTO_DERUSH_DB", "photoderush.db"))
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    db_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(db_url, echo=False)

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        print("\n=== CHECKING PREDICTION QUERY ===\n")

        # Get project
        result = await session.execute(select(Project))
        project = result.scalar_one_or_none()

        if not project:
            print("❌ No project found in database")
            return

        print(f"✓ Project: {project.name} (ID: {project.id})")
        print(f"  Directory: {project.directory}\n")

        # Check total images
        result = await session.execute(
            select(Image).where(Image.project_id == project.id)
        )
        total_images = result.scalars().all()
        print(f"📊 Total images: {len(total_images)}")

        # Check labeled images
        result = await session.execute(
            select(Image)
            .join(Label)
            .where(Image.project_id == project.id)
        )
        labeled = result.scalars().all()
        print(f"✓ Labeled images: {len(labeled)}")

        # Count by label type
        result = await session.execute(
            select(Label).where(Label.label == "keep")
        )
        keep_count = len(result.scalars().all())

        result = await session.execute(
            select(Label).where(Label.label == "trash")
        )
        trash_count = len(result.scalars().all())

        print(f"  - Keep: {keep_count}")
        print(f"  - Trash: {trash_count}")

        # Check unlabeled images (EXACT query from predict method)
        result = await session.execute(
            select(Image)
            .outerjoin(Label)
            .where(and_(Image.project_id == project.id, Label.id.is_(None)))
        )
        unlabeled = result.scalars().all()
        print(f"\n🔍 Unlabeled images (query from predict): {len(unlabeled)}")

        if unlabeled:
            print("\nFirst 5 unlabeled images:")
            for img in unlabeled[:5]:
                file_exists = Path(img.file_path).exists()
                status = "✓" if file_exists else "✗"
                print(f"  {status} {img.filename} - {img.file_path}")

        # Check if files exist
        existing_unlabeled = [img for img in unlabeled if Path(img.file_path).exists()]
        print(f"\n📁 Unlabeled images with existing files: {len(existing_unlabeled)}")

        # Detailed check - show relationship loading
        print("\n=== CHECKING IMAGE-LABEL RELATIONSHIPS ===\n")

        # Try with joinedload to see relationships
        result = await session.execute(
            select(Image)
            .where(Image.project_id == project.id)
            .options(joinedload(Image.label))
        )
        all_images = result.unique().scalars().all()

        labeled_count = sum(1 for img in all_images if img.label is not None)
        unlabeled_count = sum(1 for img in all_images if img.label is None)

        print("With joinedload:")
        print(f"  - Images with label: {labeled_count}")
        print(f"  - Images without label: {unlabeled_count}")

        # Show first few
        print("\nFirst 10 images:")
        for img in all_images[:10]:
            label_str = f"{img.label.label} ({img.label.source})" if img.label else "NO LABEL"
            file_exists = "✓" if Path(img.file_path).exists() else "✗"
            print(f"  [{file_exists}] {img.filename}: {label_str}")

        # Final analysis
        print("\n=== PREDICTION ANALYSIS ===\n")

        if unlabeled_count == 0:
            print("❌ ISSUE: All images are labeled!")
            print("   Prediction will return 0 results because there are no unlabeled images.")
            print("\n   Solutions:")
            print("   1. Delete some labels to create unlabeled images")
            print("   2. Add new images without labels")

        elif len(existing_unlabeled) == 0:
            print("❌ ISSUE: All unlabeled images have missing files!")
            print(f"   Found {unlabeled_count} unlabeled images, but none have existing files.")
            print("\n   Solutions:")
            print("   1. Check file paths in database")
            print("   2. Ensure images exist on disk")

        else:
            print(f"✓ READY: {len(existing_unlabeled)} unlabeled images with existing files")
            print("  Prediction should work if model is trained.")

            # Check model exists
            model_path = Path.home() / ".photo-derush-keep-trash-model.joblib"
            if model_path.exists():
                print(f"\n✓ Model exists: {model_path}")
            else:
                print(f"\n❌ Model not found: {model_path}")
                print("   Train the model first!")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(check_predictions())

