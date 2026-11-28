import pytest

from saf.wrds import WrdsSession, is_transient


class OperationalError(Exception):
    pass


def test_is_transient():
    assert is_transient(OperationalError("server closed the connection"))
    assert not is_transient(OperationalError('relation "x" does not exist'))
    assert not is_transient(ValueError("bug"))


def test_session_retries_once_then_raises():
    calls = []

    class Conn:
        def raw_sql(self, q, params=None, date_cols=None):
            calls.append(q)
            raise OperationalError("connection reset")

    s = WrdsSession("u", reconnect_wait=0)
    s.conn = Conn()
    s.connect = lambda: setattr(s, "conn", Conn())
    with pytest.raises(OperationalError):
        s.sql("SELECT 1")
    assert len(calls) == 2
