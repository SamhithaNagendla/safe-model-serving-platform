def test_full_api_flow(api_client) -> None:
    assert api_client.get("/health").json() == {"status": "healthy", "version": "0.2.0"}
    assert api_client.put(
        "/routing",
        json={
            "champion": "v1",
            "challenger": "v2",
            "challenger_percent": 0,
            "shadow_enabled": True,
        },
    ).status_code == 200

    prediction = api_client.post(
        "/predict", json={"routing_key": "user-1", "features": [1, 2]}
    )
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["served_version"] == "v1"

    assert api_client.post(
        "/labels", json={"request_id": body["request_id"], "actual_label": 1}
    ).status_code == 200
    stored = api_client.get(f"/predictions/{body['request_id']}")
    assert stored.json()["actual_label"] == 1
    assert api_client.get("/metrics").json()["v1"]["labeled"] == 1
    assert api_client.post("/rollback").json()["challenger_percent"] == 0


def test_api_errors(api_client) -> None:
    bad = api_client.put(
        "/routing",
        json={
            "champion": "missing",
            "challenger": "v2",
            "challenger_percent": 10,
            "shadow_enabled": False,
        },
    )
    assert bad.status_code == 400
    assert api_client.post(
        "/labels", json={"request_id": "missing", "actual_label": 0}
    ).status_code == 404
    assert api_client.get("/predictions/missing").status_code == 404
