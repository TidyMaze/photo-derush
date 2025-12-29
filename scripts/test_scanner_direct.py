#!/usr/bin/env python3
"""Direct test of scanner service."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from api.database import AsyncSessionLocal, init_db
from api.models.project import Project
from api.services.scanner import ImageScanner


async def test_scanner():
    """Test the scanner service directly."""
    await init_db()

    async with AsyncSessionLocal() as session:
        # Get the first project
        result = await session.execute(select(Project))
        project = result.scalar_one_or_none()

        if not project:
            print("❌ No project found in database")
            return

        print(f"📁 Testing scanner on project: {project.name}")
        print(f"   Directory: {project.directory}")

        # Create scanner and scan
        scanner = ImageScanner(session)
        stats = await scanner.scan_project_directory(project)

        print("\n📊 Scan Results:")
        print(f"   Scanned: {stats['scanned']}")
        print(f"   Imported: {stats['imported']}")
        print(f"   Updated: {stats['updated']}")
        print(f"   Skipped: {stats['skipped']}")
        print(f"   EXIF extracted: {stats['exif_extracted']}")
        print(f"   Errors: {stats['errors']}")

        print("\n✅ Scanner test complete!")


if __name__ == "__main__":
    asyncio.run(test_scanner())

