"""Test predictions directly to find the issue."""
import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.services.ml import MLService

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_predictions():
    """Test prediction on actual database."""
    # Connect to database
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
        print("\n=== TESTING PREDICTIONS ===\n")

        # Create ML service
        ml_service = MLService()

        # Run predictions on project 1
        project_id = 1
        print(f"Running predictions on project {project_id}...")

        result = await ml_service.predict(project_id, session)

        print("\n=== PREDICTION RESULTS ===\n")
        print(f"Full result: {result}")
        print(f"\nSuccess: {result['success']}")
        print(f"Predictions count: {result.get('predictions_count', 0)}")
        print(f"Message: {result.get('message', 'N/A')}")

        if 'stats' in result:
            print("\nStatistics:")
            for key, value in result['stats'].items():
                print(f"  - {key}: {value}")

        if 'predictions' in result and result['predictions']:
            print("\nFirst 5 predictions:")
            for pred in result['predictions'][:5]:
                print(f"  - Image {pred['image_id']}: {pred['label']} (confidence: {pred['confidence']:.3f})")

        if result.get('predictions_count', 0) == 0:
            print("\n❌ ISSUE: 0 predictions generated")
            print("\nPossible causes:")
            print("1. All predictions have low confidence (<0.8)")
            print("2. Feature extraction is failing")
            print("3. Model prediction is failing")
            print("\nCheck the logs above for more details.")
        else:
            print(f"\n✓ SUCCESS: {result['predictions_count']} predictions generated")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_predictions())

