"""Operational email alerts (spend caps today; anything else later).

─────────────────────────────────────────────────────────────────────────────
TO ADD OR CHANGE WHO GETS ALERTED
─────────────────────────────────────────────────────────────────────────────
Either one works; the env var wins if it is set.

  1. Set ALERT_EMAILS on the host (Railway → Variables). Comma-separated:
         ALERT_EMAILS=rico@example.com, ops@example.com

  2. Or edit DEFAULT_ALERT_RECIPIENTS just below and redeploy.

No code changes are needed beyond that — every alert in the app routes through
send_alert() and uses this one list.
─────────────────────────────────────────────────────────────────────────────

Delivery is plain SMTP from the standard library, so there is no new dependency
and any provider works (Gmail with an app password, Fastmail, SES, Postmark…).
Configure via env:

    SMTP_HOST       e.g. smtp.gmail.com
    SMTP_PORT       default 587 (STARTTLS)
    SMTP_USER       usually the full sending address
    SMTP_PASSWORD   app password, NOT the account password
    ALERT_FROM      optional; defaults to SMTP_USER

If SMTP is not configured, send_alert() logs the alert at full detail and
returns False. It never raises: a spend alert must not be able to take down a
debate, and an unsendable email is strictly better than a 500.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

# Edit this list to change who gets alerted (or set ALERT_EMAILS — see above).
DEFAULT_ALERT_RECIPIENTS: list[str] = [
    "ricochandra128@gmail.com",
]

SMTP_TIMEOUT_SECONDS = 10


def recipients() -> list[str]:
    """Who to email. ALERT_EMAILS overrides DEFAULT_ALERT_RECIPIENTS entirely."""
    raw = os.getenv("ALERT_EMAILS", "")
    if raw.strip():
        parsed = [addr.strip() for addr in raw.split(",") if addr.strip()]
        if parsed:
            return parsed
    return list(DEFAULT_ALERT_RECIPIENTS)


def is_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD"))


def send_alert(subject: str, body: str) -> bool:
    """Email all recipients. Returns True if handed to the SMTP server.

    Never raises. Logs and returns False on any failure, including SMTP not
    being configured at all.
    """
    to = recipients()
    if not to:
        print(f"[alerts] no recipients configured; dropping alert: {subject}")
        return False

    if not is_configured():
        # Deliberately loud and complete: with no mail server, the host log is
        # the only place this alert exists.
        print(
            f"[alerts] SMTP not configured — alert NOT emailed.\n"
            f"         would have sent to: {', '.join(to)}\n"
            f"         subject: {subject}\n"
            f"         {body}"
        )
        return False

    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("ALERT_FROM", user)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg.set_content(body)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT_SECONDS,
                                  context=ssl.create_default_context()) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_SECONDS) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(user, password)
                s.send_message(msg)
        print(f"[alerts] sent {subject!r} to {', '.join(to)}")
        return True
    except Exception as e:
        print(f"[alerts] FAILED to send {subject!r} to {', '.join(to)}: {type(e).__name__}: {e}")
        print(f"[alerts] body was:\n{body}")
        return False
