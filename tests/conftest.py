"""Pytest configuration and fixtures."""

import pytest

from app import app as flask_app
from database import db


def _rebind_default_engine(app):
    """Rebuild the default Flask-SQLAlchemy engine from current app config."""
    extension = app.extensions["sqlalchemy"]
    engines = extension._app_engines.setdefault(app, {})

    with app.app_context():
        db.session.remove()

    for engine in engines.values():
        engine.dispose()
    engines.clear()

    options = extension._engine_options.copy()
    options.update(app.config.get("SQLALCHEMY_ENGINE_OPTIONS", {}))
    options["url"] = app.config["SQLALCHEMY_DATABASE_URI"]

    echo = app.config.get("SQLALCHEMY_ECHO", False)
    options.setdefault("echo", echo)
    options.setdefault("echo_pool", echo)

    extension._make_metadata(None)
    extension._apply_driver_defaults(options, app)
    engines[None] = extension._make_engine(None, options, app)


@pytest.fixture
def app(tmp_path):
    """Create application for testing."""
    db_path = tmp_path / "test.sqlite"
    original_uri = flask_app.config["SQLALCHEMY_DATABASE_URI"]
    original_testing = flask_app.config.get("TESTING", False)

    flask_app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    flask_app.config["TESTING"] = True

    _rebind_default_engine(flask_app)

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()

    flask_app.config["SQLALCHEMY_DATABASE_URI"] = original_uri
    flask_app.config["TESTING"] = original_testing
    _rebind_default_engine(flask_app)


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create test CLI runner."""
    return app.test_cli_runner()
