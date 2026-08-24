from healthrisk.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CONFIG_DIR,
)


def test_project_directories():
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert MODELS_DIR.exists()
    assert REPORTS_DIR.exists()
    assert CONFIG_DIR.exists()