"""The registry, and the one operation in it that changes what people see.

Promotion is the only transition here that is a decision rather than a
recording, so it is the one with rules. Those rules live in `schema.sql` as
constraints; these tests check the constraints hold, not merely that the
service is polite about them -- a rule enforced only in application code is one
that a migration or a psql session walks straight through.
"""

from __future__ import annotations

import sqlite3

import pytest

from services.api import db
from services.prediction import registry


@pytest.fixture
def conn(tmp_path):
    with db.session(tmp_path / "t.sqlite3") as c:
        yield c


def add(conn, name, *, measured=True, tx="NIC", rx="DOB", param="muf", sha=None):
    return registry.register(
        conn, name=name, param=param, tx=tx, rx=rx, origin="trained",
        framework="xgboost", loader="joblib", capability="slim",
        artifact=f"/models/{name}.joblib", sha256=sha or name,
        features=["x"], target_src="measured" if measured else "modelled",
    )


def test_reimporting_the_same_file_updates_rather_than_accumulates(conn):
    """The first thing anyone does when an import looks wrong is run it again."""
    first = add(conn, "xgb")
    second = add(conn, "xgb")
    assert first == second
    assert len(registry.models(conn)) == 1


def test_an_unbound_model_is_a_distinct_row_from_a_bound_one(conn):
    """NULL != NULL in a UNIQUE constraint, so identity uses COALESCE."""
    unbound = add(conn, "xgb", tx=None, rx=None)
    bound = add(conn, "xgb")
    assert unbound != bound
    assert len(registry.models(conn)) == 2

    # ...and re-importing the unbound one still updates it.
    assert add(conn, "xgb", tx=None, rx=None) == unbound
    assert len(registry.models(conn)) == 2


def test_a_modelled_target_can_never_be_promoted(conn):
    """The comparison-only rule. Legacy imports land here by default."""
    model = add(conn, "legacy-huber", measured=False)
    with pytest.raises(registry.RegistryError, match="modelled"):
        registry.activate(conn, model)
    assert registry.get(conn, model)["active"] == 0


def test_the_schema_refuses_a_modelled_promotion_even_without_the_service(conn):
    """The guarantee has to survive someone writing SQL directly."""
    model = add(conn, "legacy-huber", measured=False)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE model_registry SET active = 1 WHERE id = ?", (model,))


def test_an_unbound_model_cannot_be_promoted(conn):
    model = add(conn, "xgb", tx=None, rx=None)
    with pytest.raises(registry.RegistryError, match="unbound"):
        registry.activate(conn, model)


def test_activation_demotes_the_incumbent_in_one_step(conn):
    first, second = add(conn, "xgb-a"), add(conn, "xgb-b")
    registry.activate(conn, first, by="operator")

    result = registry.activate(conn, second, by="operator")
    assert result["deactivated"]["name"] == "xgb-a"
    assert registry.active(conn, "muf", "NIC", "DOB")["name"] == "xgb-b"
    assert registry.get(conn, first)["active"] == 0


def test_only_one_model_can_be_live_per_circuit_and_parameter(conn):
    first, second = add(conn, "xgb-a"), add(conn, "xgb-b")
    registry.activate(conn, first)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE model_registry SET active = 1 WHERE id = ?", (second,))


def test_a_second_circuit_has_its_own_live_model(conn):
    first = add(conn, "xgb-a")
    other = add(conn, "xgb-c", tx="SGO", rx="DOB", sha="c")
    registry.activate(conn, first)
    registry.activate(conn, other)
    assert registry.active(conn, "muf", "NIC", "DOB")["name"] == "xgb-a"
    assert registry.active(conn, "muf", "SGO", "DOB")["name"] == "xgb-c"


def test_rollback_is_reactivating_the_previous_row(conn):
    first, second = add(conn, "xgb-a"), add(conn, "xgb-b")
    registry.activate(conn, first)
    registry.activate(conn, second)
    registry.activate(conn, first)
    assert registry.active(conn, "muf", "NIC", "DOB")["name"] == "xgb-a"
    assert {m["name"] for m in registry.models(conn)} == {"xgb-a", "xgb-b"}


def test_retiring_leaves_the_row_and_its_history(conn):
    model = add(conn, "xgb-a")
    registry.activate(conn, model)
    registry.retire(conn, model)
    assert registry.active(conn, "muf", "NIC", "DOB") is None
    assert registry.get(conn, model) is not None


def test_state_reflects_the_lifecycle(conn):
    model = add(conn, "xgb-a")
    assert registry.get(conn, model)["state"] == "registered"

    registry.set_metrics(conn, model, {"mae": {"86400": 1.2}})
    assert registry.get(conn, model)["state"] == "scored"

    registry.activate(conn, model)
    assert registry.get(conn, model)["state"] == "active"

    legacy = add(conn, "legacy", measured=False, tx=None, rx=None)
    assert registry.get(conn, legacy)["state"] == "comparison"


def test_json_columns_come_back_as_objects(conn):
    model = registry.register(
        conn, name="m", param="muf", tx="NIC", rx="DOB", origin="legacy",
        framework="sklearn", loader="joblib", capability="slim",
        artifact="/m.sav", sha256="s", features=["a", "b"],
        feature_recipe={"lag": 288}, env={"sklearn": "1.4.2"},
        golden_input=[1.0, 2.0], golden_output=3.0, target_src="modelled",
    )
    row = registry.get(conn, model)
    assert row["features"] == ["a", "b"]
    assert row["feature_recipe"]["lag"] == 288
    assert row["env"]["sklearn"] == "1.4.2"
    assert row["golden_input"] == [1.0, 2.0]


def test_active_is_not_settable_at_registration(conn):
    with pytest.raises(registry.RegistryError, match="not registerable"):
        registry.register(conn, name="m", param="muf", active=1)
