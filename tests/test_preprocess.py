from pathlib import Path


def test_real_data_not_baked_into_image():
    if Path.cwd() != Path("/app"):
        return

    assert not Path("data/raw").exists()
