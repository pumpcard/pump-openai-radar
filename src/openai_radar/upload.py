"""
Push onboarding CSVs to Pump's self-serve onboarding endpoint.

Same exchange as pump-aws-radar (``POST /api/v1/estimate/radar/urls``): the
token from ``openai-radar login`` (or ``--upload-token``) is traded for a
presigned S3 PUT URL, and each CSV is uploaded directly. The token carries no
company id — the backend pins the company and derives the S3 key server-side,
so the token can only write its own upload's prefix.

No Pump AWS credentials are involved on the client. The presigned URL already
carries everything the PUT needs. Costs go up as ``billing`` and token usage
as ``inventory``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from openai_radar import __version__

# A real User-Agent: Cloudflare in front of api.pump.co blocks the default
# urllib agent ("Python-urllib/..."), so the token exchange must identify itself.
_USER_AGENT = f"openai-radar/{__version__}"

# The backend mounts its router under API_V1_STR (default "/api/v1"). The exchange
# route is service/api/endpoints/estimate_radar.py :: exchange_token_for_url.
_URLS_PATH = "/api/v1/estimate/radar/urls"

# Roles the backend recognizes. Cost is billing; token usage is inventory.
_ROLES = ("billing", "inventory")

_HTTP_TIMEOUT_SECONDS = 60


class UploadError(RuntimeError):
    """Raised when the token exchange or the S3 PUT fails."""


def _exchange_token_for_url(api_base: str, token: str, role: str) -> str:
    """Exchange the upload token for a presigned PUT URL for *role*'s object."""
    url = api_base.rstrip("/") + _URLS_PATH
    body = json.dumps({"token": token, "role": role}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 401:
            raise UploadError(
                "Upload token was rejected (401). It may be wrong or expired — "
                "generate a fresh command in the Pump app."
            ) from e
        raise UploadError(f"Token exchange failed ({e.code}) for role '{role}': {detail}") from e
    except urllib.error.URLError as e:
        raise UploadError(f"Could not reach Pump at {url}: {e.reason}") from e

    upload_url = payload.get("upload_url")
    if not upload_url:
        raise UploadError(f"Exchange response for role '{role}' had no upload_url: {payload}")
    return upload_url


def _put_csv(upload_url: str, csv_path: str) -> None:
    """PUT the CSV at *csv_path* to the presigned *upload_url*.

    Content-Type must be text/csv: the presigned URL signs content-type, so a
    mismatched (or missing) type is rejected by S3 as a signature error.
    """
    with open(csv_path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(
        upload_url,
        data=data,
        method="PUT",
        # User-Agent isn't required by S3, but keeping both requests identical avoids surprises.
        headers={"Content-Type": "text/csv", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            if resp.status not in (200, 204):
                raise UploadError(f"S3 PUT of {csv_path} returned HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise UploadError(f"S3 PUT of {csv_path} failed ({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise UploadError(f"S3 PUT of {csv_path} could not connect: {e.reason}") from e


def upload_csvs(api_base: str, token: str, files: dict[str, str]) -> None:
    """Upload each role's CSV to Pump.

    :param api_base: Pump API base, e.g. https://api.pump.co (no trailing /api).
    :param token:    the --upload-token minted in the Pump app.
    :param files:    {role: local_csv_path}. ``billing`` is the cost CSV,
                     ``inventory`` is the usage CSV.
    """
    unknown = set(files) - set(_ROLES)
    if unknown:
        raise UploadError(f"Unknown upload role(s): {sorted(unknown)}. Expected {list(_ROLES)}.")

    for role in _ROLES:
        csv_path = files.get(role)
        if not csv_path:
            continue
        print(f"  • {role}: requesting upload URL …")
        upload_url = _exchange_token_for_url(api_base, token, role)
        print(f"  • {role}: uploading {csv_path} …")
        _put_csv(upload_url, csv_path)
        print(f"  ✓ {role} uploaded")
