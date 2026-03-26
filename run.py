#!/usr/bin/env python3
"""Entry point for the Instagram Analyzer application."""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log"),
    ],
)

logger = logging.getLogger(__name__)

# Ensure directories exist BEFORE importing app (which initializes database)
from config import config

logger.info("📁 Ensuring directories exist...")
config.ensure_directories()

# Now import app (this will initialize database with directories ready)
from app import app
from database import db
from migrations import run_migrations


def main():
    """Main entry point."""
    logger.info("🚀 Starting Instagram AI Analyzer")

    with app.app_context():
        # Create all tables first
        logger.info("📊 Creating database tables...")
        db.create_all()
        logger.info("✅ Database tables ready")

        # Run database migrations
        logger.info("📊 Running database migrations...")
        run_migrations()

        # Log configuration status
        if config.azure.is_configured():
            logger.info("✅ Azure OpenAI configured")
        else:
            logger.warning("⚠️ Azure OpenAI not configured - analysis disabled")

        if config.instagram.sessionid:
            logger.info("✅ Instagram session configured")
        else:
            logger.warning("⚠️ Instagram session not configured")

    # Start server
    logger.info(f"📍 Server running at http://{config.app.host}:{config.app.port}")
    logger.info(f"📍 UI available at http://{config.app.host}:{config.app.port}")
    logger.info(f"📍 API docs at http://{config.app.host}:{config.app.port}/api/health")

    app.run(
        host=config.app.host,
        port=config.app.port,
        debug=config.app.debug,
    )


if __name__ == "__main__":
    main()
