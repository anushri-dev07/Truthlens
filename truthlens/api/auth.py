from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_LOGIN_WINDOW_SECONDS = 15 * 60
_LOGIN_MAX_FAILURES = 8
_login_failures: dict[str, list[float]] = {}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=64,
    )
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N,
        _SCRYPT_R,
        _SCRYPT_P,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_text, digest_text = encoded.split("$", maxsplit=5)
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text)
        expected = base64.urlsafe_b64decode(digest_text)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError):
        return False


def login_is_limited(identity: str) -> bool:
    now = time.monotonic()
    recent = [
        timestamp
        for timestamp in _login_failures.get(identity, [])
        if now - timestamp < _LOGIN_WINDOW_SECONDS
    ]
    if recent:
        _login_failures[identity] = recent
    else:
        _login_failures.pop(identity, None)
    return len(recent) >= _LOGIN_MAX_FAILURES


def record_login_failure(identity: str) -> None:
    now = time.monotonic()
    recent = [
        timestamp
        for timestamp in _login_failures.get(identity, [])
        if now - timestamp < _LOGIN_WINDOW_SECONDS
    ]
    _login_failures[identity] = recent[-(_LOGIN_MAX_FAILURES - 1) :] + [now]
    if len(_login_failures) > 10_000:
        for key in list(_login_failures)[:5_000]:
            _login_failures.pop(key, None)


def clear_login_failures(identity: str) -> None:
    _login_failures.pop(identity, None)