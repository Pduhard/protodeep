def _shape_error(expected, actual, critical=True):
    expected_shape = (None, *expected)
    actual_shape = (None, *actual)
    msg = f'ShapeError: input shape is incompatible with layer: ' \
        f'expected shape={expected_shape}, ' \
        f'found shape={actual_shape}'
    print(msg)
    if critical:
        quit()


def _ndim_error(expected, actual, critical=True):
    msg = f'DimError: input dim is incompatible with layer dim: ' \
        f'expected ndim={expected}, ' \
        f'found ndim={actual}'
    print(msg)
    if critical:
        quit()
