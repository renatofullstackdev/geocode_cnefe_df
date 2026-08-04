def test_vocabulary_aliases_and_compatibility(vocabulary):
    assert vocabulary.alias_to_component["CJ"] == "CONJUNTO"
    assert vocabulary.alias_to_component["APT"] == "APARTAMENTO"
    assert vocabulary.compatibility_score("CASA", "LOTE") == 0.72
    assert vocabulary.compatibility_score("CASA", "APARTAMENTO") == 0.0


def test_vocabulary_fingerprint_is_valid(vocabulary):
    assert len(vocabulary.fingerprint) == 64
    int(vocabulary.fingerprint, 16)
