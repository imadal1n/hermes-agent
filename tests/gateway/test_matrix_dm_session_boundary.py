from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


BOUNDARY_NOW = datetime(2026, 7, 29, 6, 5, 0, tzinfo=timezone.utc)
BOUNDARY_START = datetime(2026, 7, 29, 4, 0, 0, tzinfo=timezone.utc).timestamp()


def _make_entry(key: str, session_id: str, *, platform: Platform) -> SessionEntry:
    now = datetime.now(timezone.utc)
    return SessionEntry(
        session_key=key,
        session_id=session_id,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        platform=platform,
        chat_type="dm",
    )


def _db_returning(rows: dict[str, dict[str, object]]) -> MagicMock:
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    db.find_latest_gateway_session_for_peer.return_value = None
    db.reopen_session.return_value = None
    db.create_session.return_value = None
    db.get_compression_tip.side_effect = lambda sid: sid
    return db


def _store_with_db(tmp_path, db: MagicMock) -> SessionStore:
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db
    store._loaded = True
    return store


def _matrix_dm_source() -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.invalid",
        chat_type="dm",
        user_id="@user:example.invalid",
    )


def _telegram_dm_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8494508720",
        chat_type="dm",
        user_id="8494508720",
    )


def test_matrix_dm_existing_route_before_boundary_creates_fresh_session(tmp_path) -> None:
    source = _matrix_dm_source()
    db = _db_returning({
        "sid_old": {
            "end_reason": None,
            "expiry_finalized": 0,
            "id": "sid_old",
            "started_at": datetime(2026, 7, 28, 5, 48, 0, tzinfo=timezone.utc).timestamp(),
        },
    })
    store = _store_with_db(tmp_path, db)
    key = store._generate_session_key(source)
    store._entries[key] = _make_entry(key, "sid_old", platform=Platform.MATRIX)

    with patch("gateway.session._now", return_value=BOUNDARY_NOW):
        result = store.get_or_create_session(source)

    assert result.session_id != "sid_old"
    assert store._entries[key].session_id == result.session_id
    db.reopen_session.assert_not_called()
    db.create_session.assert_called_once()
    call_kwargs = db.find_latest_gateway_session_for_peer.call_args.kwargs
    assert call_kwargs["started_after"] == BOUNDARY_START


def test_matrix_dm_missing_route_filters_previous_day_db_recovery(tmp_path) -> None:
    source = _matrix_dm_source()
    db = _db_returning({})

    def find_latest_gateway_session_for_peer(**kwargs):
        if kwargs.get("started_after") is None:
            return {
                "id": "sid_old",
                "started_at": datetime(2026, 7, 28, 5, 48, 0, tzinfo=timezone.utc).timestamp(),
            }
        return None

    db.find_latest_gateway_session_for_peer.side_effect = find_latest_gateway_session_for_peer
    store = _store_with_db(tmp_path, db)
    key = store._generate_session_key(source)

    with patch("gateway.session._now", return_value=BOUNDARY_NOW):
        result = store.get_or_create_session(source)

    assert result.session_id != "sid_old"
    assert store._entries[key].session_id == result.session_id
    db.reopen_session.assert_not_called()
    db.create_session.assert_called_once()
    call_kwargs = db.find_latest_gateway_session_for_peer.call_args.kwargs
    assert call_kwargs["started_after"] == BOUNDARY_START


def test_matrix_dm_same_day_missing_route_recovers_session(tmp_path) -> None:
    source = _matrix_dm_source()
    db = _db_returning({})

    def find_latest_gateway_session_for_peer(**kwargs):
        started_at = datetime(2026, 7, 29, 5, 30, 0, tzinfo=timezone.utc).timestamp()
        if kwargs["started_after"] <= started_at:
            return {"id": "sid_same_day", "started_at": started_at}
        return None

    db.find_latest_gateway_session_for_peer.side_effect = find_latest_gateway_session_for_peer
    store = _store_with_db(tmp_path, db)
    key = store._generate_session_key(source)

    with patch("gateway.session._now", return_value=BOUNDARY_NOW):
        result = store.get_or_create_session(source)

    assert result.session_id == "sid_same_day"
    assert store._entries[key].session_id == "sid_same_day"
    db.reopen_session.assert_called_once_with("sid_same_day")
    db.create_session.assert_not_called()


def test_non_matrix_dm_missing_route_recovery_has_no_daily_boundary(tmp_path) -> None:
    source = _telegram_dm_source()
    db = _db_returning({})
    db.find_latest_gateway_session_for_peer.return_value = {
        "id": "sid_telegram",
        "started_at": datetime(2026, 7, 28, 5, 48, 0, tzinfo=timezone.utc).timestamp(),
    }
    store = _store_with_db(tmp_path, db)
    key = store._generate_session_key(source)

    with patch("gateway.session._now", return_value=BOUNDARY_NOW):
        result = store.get_or_create_session(source)

    assert result.session_id == "sid_telegram"
    assert store._entries[key].session_id == "sid_telegram"
    db.reopen_session.assert_called_once_with("sid_telegram")
    db.create_session.assert_not_called()
    call_kwargs = db.find_latest_gateway_session_for_peer.call_args.kwargs
    assert call_kwargs["started_after"] is None
