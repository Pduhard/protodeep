def _shape_error(expected, actual, critical=True):
    expected_shape = (None, *expected)
    actual_shape = (None, *actual)
    msg = f'ValueError: input shape is incompatible with layer model: ' \
        f'expected shape={expected_shape}, ' \
        f'found shape={actual_shape}'
    print(msg)
    if critical:
        quit()
