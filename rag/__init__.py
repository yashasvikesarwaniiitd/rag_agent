"""Credentialing RAG. Layout follows the dependency rule: `domain/` imports no adapter and no
SDK, so the whole pipeline is testable with a fake embedder, store and model."""
