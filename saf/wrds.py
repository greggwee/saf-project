"""WRDS connection without interactive prompts, plus a query helper that retries once.

Credentials: ``WRDS_PASSWORD`` from the environment (or a Modal secret), otherwise ``~/.pgpass``.
The wrds client only prompts for a username/password when the first attempt fails, and then hides
the real error; we print it (``verbose=True``) and replace the prompt with an exception.

Pin ``wrds==3.3.0``, ``pandas==2.2.3``, ``sqlalchemy>=2,<2.1``: newer pandas breaks ``raw_sql``.
"""
import logging
import os
import stat
import time
from pathlib import Path

PGPASS_HOST = "wrds-pgdata.wharton.upenn.edu"
PGPASS_PORT = 9737
RECONNECT_WAIT = 30          # seconds before the single reconnect + retry

_TRANSIENT = ("OperationalError", "InterfaceError", "DisconnectionError")


def pgpass_ok(user):
    """True when ~/.pgpass holds an entry for ``user`` on the WRDS host."""
    p = Path.home() / ".pgpass"
    if not p.exists():
        return False
    return any(line.startswith(f"{PGPASS_HOST}:{PGPASS_PORT}:") and f":{user}:" in line
               for line in p.read_text().splitlines())


def init_pgpass(user):
    """Create ~/.pgpass via the wrds client (asks for the password once, interactively)."""
    import wrds
    c = wrds.Connection(wrds_username=user)
    c.create_pgpass_file()
    c.close()
    p = Path.home() / ".pgpass"
    p.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"{p} created (chmod 600).")


def is_transient(exc):
    """Connection drop worth one retry; never a code bug or a missing table."""
    if "does not exist" in str(exc).lower():
        return False
    return type(exc).__name__ in _TRANSIENT or isinstance(exc, (ConnectionError, TimeoutError))


class WrdsSession:
    """One WRDS connection, reconnected once on a dropped connection.

    WRDS throttles parallel use: open a single session and run queries sequentially.
    """

    def __init__(self, user, log=None, reconnect_wait=RECONNECT_WAIT):
        if not user:
            raise ValueError("WRDS username required (--wrds-user or WRDS_USERNAME)")
        self.user = user
        self.conn = None
        self.log = log or logging.getLogger(__name__)
        self.reconnect_wait = reconnect_wait

    def connect(self):
        import wrds
        import wrds.sql

        def _no_prompt(_self):
            raise RuntimeError(
                "WRDS connection refused (cause in the err=... line above). Usual causes: wrong "
                "username/password, Duo push not approved in time, IP blocked.")

        wrds.sql.Connection._Connection__get_user_credentials = _no_prompt

        kw = {"wrds_username": self.user, "verbose": True}
        pwd = os.environ.get("WRDS_PASSWORD")
        if pwd:
            kw["wrds_password"] = pwd
        elif not pgpass_ok(self.user):
            raise RuntimeError(f"Neither WRDS_PASSWORD nor a ~/.pgpass entry for {self.user}. "
                               "Create one once with saf.wrds.init_pgpass(user).")
        self.close()
        self.log.info("connecting to WRDS as %s (%s)", self.user, "env password" if pwd else ".pgpass")
        self.conn = wrds.Connection(**kw)
        return self.conn

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def sql(self, query, params=None, date_cols=None, retries=1):
        """``raw_sql`` with up to ``retries`` reconnect + retry on a dropped connection only."""
        if self.conn is None:
            self.connect()
        for k in range(retries + 1):
            try:
                return self.conn.raw_sql(query, params=params, date_cols=date_cols)
            except Exception as e:
                if not is_transient(e) or k == retries:
                    raise
                self.log.warning("connection lost (%s) -> reconnect in %ss", str(e)[:120], self.reconnect_wait)
                time.sleep(self.reconnect_wait)
                self.connect()

    def describe(self, lib, table):
        """Column names and types of ``lib.table`` (DataFrame: name, nullable, type, ...)."""
        if self.conn is None:
            self.connect()
        return self.conn.describe_table(lib, table)
