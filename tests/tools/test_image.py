from PIL import Image
from math_agent.tools.image import inspect_image, encode_image_to_data_url


def test_inspect_image_returns_size_and_dpi(workdir):
    p = workdir / "t.png"
    Image.new("RGB", (300, 200), "white").save(p, dpi=(150, 150))
    info = inspect_image(p)
    assert info.width == 300 and info.height == 200
    assert info.dpi[0] == 150 or abs(info.dpi[0] - 150) < 1


def test_encode_image_to_data_url(workdir):
    p = workdir / "t.png"
    Image.new("RGB", (10, 10), "white").save(p)
    url = encode_image_to_data_url(p)
    assert url.startswith("data:image/png;base64,")
