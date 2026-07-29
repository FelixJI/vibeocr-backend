"""Backend route and served OpenAPI conformance with the formal Protocol v2."""

from scripts.check_runtime_protocol_conformance import (
    backend_app,
    backend_operations,
    check_conformance,
    formal_operations,
)


def test_backend_route_surface_and_operation_ids_match_formal_protocol() -> None:
    assert backend_operations() == formal_operations()
    check_conformance()


def test_backend_serves_the_committed_formal_openapi() -> None:
    app = backend_app()
    assert app.openapi()["info"]["title"] == "VibeOCR Local Runtime API"
    assert app.openapi()["openapi"] == "3.1.0"
