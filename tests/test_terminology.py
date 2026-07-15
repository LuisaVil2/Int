"""TerminologyIndex: lookup por especialidad, incluyendo las especialidades nuevas."""


def test_loads_expected_specialties(terminology_index):
    specialties = {t.specialty for t in terminology_index.terms}
    for expected in ("emergency", "internal_medicine", "pediatrics", "oncology",
                     "psychiatry", "icu", "surgery", "cardiology", "general", "drug"):
        assert expected in specialties


def test_general_specialty_sees_all_specialties(terminology_index):
    # Fix de alcance: specialty="general" (el default de LiveEngine) ya NO filtra
    # por especialidad -- de lo contrario los términos de psychiatry/oncology/etc.
    # serían inalcanzables porque la GUI nunca pasa una especialidad distinta.
    text = "the patient has chest pain and schizophrenia and needs chemotherapy"
    hits = terminology_index.lookup(text, "general")
    joined = "\n".join(hits)
    assert "chest pain" in joined
    assert "schizophrenia" in joined
    assert "chemotherapy" in joined


def test_narrow_specialty_filters_out_other_specialties(terminology_index):
    text = "chest pain and schizophrenia"
    hits = terminology_index.lookup(text, "cardiology")
    joined = "\n".join(hits)
    assert "chest pain" in joined
    assert "schizophrenia" not in joined


def test_new_specialty_term_reachable_when_selected(terminology_index):
    hits = terminology_index.lookup("the patient may have sepsis", "emergency")
    assert any("sepsis" in h for h in hits)


def test_drug_terms_always_reachable(terminology_index):
    hits = terminology_index.lookup("taking Tylenol for pain", "psychiatry")
    assert any("Tylenol" in h for h in hits)


def test_unknown_term_not_hallucinated(terminology_index):
    hits = terminology_index.lookup("the xyzzy foobar syndrome", "general")
    assert hits == []
