import pytest

from rag.adapters.embedders import HashingEmbedder
from rag.adapters.stores import MemoryStore
from rag.application.ingest import ingest
from rag.domain.models import Page

POLICY_PAGES = [
    Page("cms_manual", "cms_manual.pdf", 1,
         "Recredentialing. Network providers must be recredentialed at least every 36 months."),
    Page("cms_manual", "cms_manual.pdf", 2,
         "Primary source verification. The plan verifies a current, valid state license "
         "directly with the state licensing board."),
    Page("state_policy", "state_policy.pdf", 4,
         "DEA registration. Prescribers must hold a current DEA certificate where applicable."),
]


class ListLoader:
    """Loader fake: each 'path' is a key into a dict of pages."""

    def __init__(self, docs: dict[str, list[Page]]):
        self.docs = docs

    def load(self, path):
        return self.docs[str(path)]


@pytest.fixture
def indexed_store():
    embedder, store = HashingEmbedder(), MemoryStore()
    loader = ListLoader({"policies": POLICY_PAGES})
    ingest(["policies"], loader=loader, embedder=embedder, store=store)
    return embedder, store
