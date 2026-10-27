from rag.domain.keyword import BM25, reciprocal_rank_fusion, stem, tokenize


def test_stem_joins_word_forms():
    assert stem("recredentialing") == stem("recredentialed") == stem("recredential")
    assert stem("policies") == "policy"


def test_short_words_and_numbers_are_not_stemmed():
    assert stem("dea") == "dea" and stem("2026") == "2026"


def test_tokenize_keeps_section_numbers_whole_and_drops_stopwords():
    assert tokenize("What does 42 CFR 438.214 require?") == ["42", "cfr", "438.214", "require"]


def test_bm25_ranks_the_rare_exact_term_first():
    idx = BM25(["a", "b", "c"], [
        "Medicare enrollment of providers and suppliers.",
        "Plans must recredential providers at least every 3 years.",
        "Enrollment providers enrollment revocation of providers.",
    ])
    assert idx.search("how often to recredential providers", k=3)[0][0] == "b"


def test_bm25_matches_across_word_forms():
    idx = BM25(["x"], ["Recredentialing occurs at least every 36 months."])
    assert [cid for cid, _ in idx.search("recredentialed", k=1)] == ["x"]


def test_bm25_returns_nothing_when_no_term_matches():
    assert BM25(["a"], ["licensure verification"]).search("capital of France", k=5) == []


def test_rrf_rewards_items_ranked_well_by_both_lists():
    fused = [cid for cid, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])]
    assert fused[0] == "b"  # ranks 2 and 1 beat a's ranks 1 and 3
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_surfaces_an_item_only_one_list_found():
    fused = [cid for cid, _ in reciprocal_rank_fusion([["v1", "v2"], ["k1"]])]
    assert "k1" in fused[:2]
