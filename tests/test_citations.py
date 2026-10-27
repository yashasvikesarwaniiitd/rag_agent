from rag.domain.citations import check_answer, cited_numbers


def test_valid_citations_pass():
    r = check_answer("Recredential every 36 months [1]. Verify DEA [2, 3].", n_sources=3)
    assert r.ok and r.cited == {1, 2, 3}


def test_uncited_answer_fails():
    r = check_answer("Recredential every 36 months.", n_sources=3)
    assert not r.ok and "no citations" in r.reason


def test_out_of_range_citation_fails_closed():
    r = check_answer("Recredential every 36 months [1][7].", n_sources=3)
    assert not r.ok and r.invalid == {7}


def test_zero_is_not_a_valid_source():
    assert not check_answer("Something [0].", n_sources=3).ok


def test_not_found_is_an_honest_pass():
    for text in ["NOT_FOUND", "  not_found. ", "NOT_FOUND."]:
        r = check_answer(text, n_sources=3)
        assert r.ok and r.not_found


def test_not_found_inside_a_sentence_is_not_the_sentinel():
    r = check_answer("The answer is NOT_FOUND in source 1.", n_sources=3)
    assert not r.not_found and not r.ok


def test_empty_answer_fails():
    assert not check_answer("   ", n_sources=3).ok


def test_cited_numbers_dedupes_in_order():
    assert cited_numbers("a [2] b [1, 2] c [3]") == [2, 1, 3]
