import httpx


async def test_export_route_rejects_unknown_canonical_table_schema(
    pdf_app, supervisor_token, tmp_path
):
    invalid_table = {
        "schema_version": 999,
        "table_id": "future",
        "row_count": 0,
        "column_count": 0,
        "coordinate_space": "unknown",
        "cells": [],
        "provenance": None,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=pdf_app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {supervisor_token}"},
    ) as client:
        response = await client.post(
            "/v2/export",
            json={
                "raw_text": "",
                "markdown_text": "",
                "html_text": "",
                "raw_blocks": [{"type": "table", "table": invalid_table}],
                "output_path": str(tmp_path / "invalid.html"),
                "format": "html",
            },
        )

    assert response.status_code == 400
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert not (tmp_path / "invalid.html").exists()
