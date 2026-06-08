# Full debate lifecycle — curl

A complete student debate from create to PDF, in raw curl.

Set your key once:

```bash
export DEBATELAB_KEY="sk_live_..."
```

## 1. Create the debate

```bash
curl https://api.debatelab.ai/api/v1/debates \
  -H "X-API-Key: $DEBATELAB_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "motion": "THW ban single-use plastics globally",
    "starter": "user",
    "num_rounds": 2,
    "mode": "casual",
    "difficulty": "intermediate",
    "external_user_id": "student-123",
    "metadata": {
      "Debater": "Nguyen An",
      "Class": "Advanced Wednesdays"
    }
  }'
```

Capture the `id` from the response. We'll call it `$DEBATE_ID`:

```bash
export DEBATE_ID="8c2f..."
```

## 2. Submit the student's first speech

```bash
curl https://api.debatelab.ai/api/v1/debates/$DEBATE_ID/turns \
  -H "X-API-Key: $DEBATELAB_KEY" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d '{
    "content": "Plastic waste is choking our oceans. We see this with the Great Pacific Garbage Patch. The harm is irreversible and global."
  }'
```

Takes 5–15 seconds. The response includes both the student's saved message and the AI's reply, plus the new `next_speaker` and `status`.

## 3. Submit the student's rebuttal

```bash
curl https://api.debatelab.ai/api/v1/debates/$DEBATE_ID/turns \
  -H "X-API-Key: $DEBATELAB_KEY" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d '{
    "content": "Even granting the proposition framing, they still owe us a mechanism. None of their arguments engage with the displacement problem I raised."
  }'
```

When the response's `status` is `"completed"`, the debate is finished and ready to score.

## 4. Score the debate

```bash
curl -X POST https://api.debatelab.ai/api/v1/debates/$DEBATE_ID/finish \
  -H "X-API-Key: $DEBATELAB_KEY" \
  --max-time 30
```

Returns the score breakdown.

## 5. Download the PDF

```bash
curl https://api.debatelab.ai/api/v1/debates/$DEBATE_ID/report.pdf \
  -H "X-API-Key: $DEBATELAB_KEY" \
  --output debate_report.pdf
```

The PDF has two sections: feedback + next drill, and the full transcript.

## 6. (Optional) List the student's recent debates

```bash
curl "https://api.debatelab.ai/api/v1/debates?external_user_id=student-123&limit=20" \
  -H "X-API-Key: $DEBATELAB_KEY"
```
