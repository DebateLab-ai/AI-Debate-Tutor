# Full debate lifecycle — Python

Uses the `requests` library — `pip install requests`.

```python
import os
import requests

BASE = "https://api.debatelab.ai"
KEY = os.environ["DEBATELAB_KEY"]
HEADERS = {"X-API-Key": KEY, "Content-Type": "application/json"}

# Use a session so connections are reused and a long timeout is the default.
session = requests.Session()
session.headers.update(HEADERS)
TIMEOUT = 60  # seconds — AI turns take 5-15s, scoring 3-8s


def create_debate() -> dict:
    payload = {
        "motion": "THW ban single-use plastics globally",
        "starter": "user",
        "num_rounds": 2,
        "mode": "casual",
        "difficulty": "intermediate",
        "external_user_id": "student-123",
        "metadata": {
            "Debater": "Nguyen An",
            "Class": "Advanced Wednesdays",
        },
    }
    r = session.post(f"{BASE}/api/v1/debates", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def submit_turn(debate_id: str, content: str) -> dict:
    r = session.post(
        f"{BASE}/api/v1/debates/{debate_id}/turns",
        json={"content": content},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def finish_debate(debate_id: str) -> dict:
    r = session.post(f"{BASE}/api/v1/debates/{debate_id}/finish", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def download_pdf(debate_id: str, out_path: str) -> None:
    r = session.get(f"{BASE}/api/v1/debates/{debate_id}/report.pdf", timeout=TIMEOUT)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def main() -> None:
    debate = create_debate()
    debate_id = debate["id"]
    print(f"Created debate {debate_id}")

    turn = submit_turn(
        debate_id,
        "Plastic waste is choking our oceans. We see this with the "
        "Great Pacific Garbage Patch. The harm is irreversible and global.",
    )
    print(f"Round 1 done. Status: {turn['status']}, next: {turn['next_speaker']}")

    turn = submit_turn(
        debate_id,
        "Even granting the proposition framing, they still owe us a mechanism. "
        "None of their arguments engage with the displacement problem I raised.",
    )
    print(f"Round 2 done. Status: {turn['status']}")

    score = finish_debate(debate_id)
    print(f"Overall: {score['overall']} / 10")
    print(f"Weakness: {score['weakness_type']}")

    download_pdf(debate_id, "debate_report.pdf")
    print("PDF saved to debate_report.pdf")


if __name__ == "__main__":
    main()
```

## Handling errors

`requests.raise_for_status()` raises on any 4xx or 5xx. Wrap the calls you want to recover from:

```python
try:
    turn = submit_turn(debate_id, content)
except requests.HTTPError as e:
    if e.response.status_code == 502:
        # AI provider failed. Safe to retry once.
        turn = submit_turn(debate_id, content)
    elif e.response.status_code == 429:
        # Rate limited. Back off using Retry-After.
        import time
        time.sleep(int(e.response.headers.get("Retry-After", "5")))
        turn = submit_turn(debate_id, content)
    else:
        raise
```

## Listing a student's debates

```python
def list_debates(external_user_id: str | None = None, limit: int = 50) -> list[dict]:
    params: dict = {"limit": limit}
    if external_user_id is not None:
        params["external_user_id"] = external_user_id
    r = session.get(f"{BASE}/api/v1/debates", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


debates = list_debates(external_user_id="student-123")
for d in debates:
    print(f"{d['created_at']} — {d['motion']} — status={d['status']}")
```
