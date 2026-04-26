from pathlib import Path

def test_container_project_layout():
    assert Path("src").exists()
    assert Path("artifacts").exists()
    assert Path("requirements.txt").exists()