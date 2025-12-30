"""Quick test of predictions with new threshold."""
import asyncio
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.services.ml import MLService


async def quick_test():
    db_path = Path(os.getenv("PHOTO_DERUSH_DB", "photoderush.db"))
    db_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        ml_service = MLService()
        result = await ml_service.predict(1, session)

        print(f"Success: {result['success']}")
        print(f"Predictions: {result.get('predictions_count', 0)}")

        if 'stats' in result:
            print("\nStats:")
            for k, v in result['stats'].items():
                print(f"  {k}: {v}")

        if result.get('predictions_count', 0) > 0:
            print(f"\n✓ SUCCESS - Got {result['predictions_count']} predictions!")
            print("\nFirst 3:")
            for p in result.get('predictions', [])[:3]:
                print(f"  Image {p['image_id']}: {p['label']} ({p['confidence']:.3f})")

    await engine.dispose()

asyncio.run(quick_test())

