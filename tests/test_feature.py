from evolution import feature


def test_P_feature():
    result = feature.P_features('TEST', ph=11.5, max_len=5)
    assert isinstance(result, list)


def test_onehot_features():
    result = feature.onehot_features('TEST', max_len=5)
    assert isinstance(result, list)


def test_esm2_features():
    result = feature.esm2_features(sequences='TEST', max_len=5)
    assert isinstance(result, list)