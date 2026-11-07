from quaternion_unity._main import add  # noqa: PLC2701


def test_add():
    """Adding two number works as expected."""
    assert add(1, 1) == 2
