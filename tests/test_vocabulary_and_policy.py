def test_vocabulary_aliases_and_compatibility(vocabulary):
    assert vocabulary.alias_to_component["CJ"] == "CONJUNTO"
    assert vocabulary.alias_to_component["APT"] == "APARTAMENTO"
    assert vocabulary.compatibility_score("CASA", "LOTE") == 0.72
    assert vocabulary.compatibility_score("CASA", "APARTAMENTO") == 0.0


def test_configuration_fingerprints_are_stable(
    vocabulary,
    ranking_policy,
):
    for fingerprint in (
        vocabulary.fingerprint,
        ranking_policy.fingerprint,
    ):
        assert len(fingerprint) == 64
        int(fingerprint, 16)


def test_ranking_threshold_order(ranking_policy):
    assert ranking_policy.thresholds.candidate < ranking_policy.thresholds.strong_candidate
