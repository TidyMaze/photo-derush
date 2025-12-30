#!/usr/bin/env python3
"""Direct test of predict method to see actual error."""
import asyncio
import os
import sys

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

async def test_predict():
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from api.config import settings
    from api.services.ml import ml_service

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as db:
        print("=" * 60)
        print("TESTING PREDICT METHOD")
        print("=" * 60)
        result = await ml_service.predict(1, db)
        print(f"Result: {result}")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_predict())

