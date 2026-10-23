-- Migration 004 — per-call id on the billing ledger
-- Apply in the Supabase SQL editor. Idempotent; safe to re-run.
--
-- request_id is one uuid per partner HTTP request. Every model call that
-- request made (speech, score, moderation) stores the same id, and api_usage
-- stores it too. GET /api/v1/usage/events groups on it.
--
-- Null on rows written before this migration. Those charges stay in the
-- month total from GET /api/v1/usage and are reported as
-- amount_without_call_id_usd. They do not appear as lines.
--
-- The app retries an insert without request_id if this column is missing,
-- so applying this after the code deploy does not drop charges. Per-call
-- lines stay empty until this has been applied.

alter table cost_events add column if not exists request_id uuid;
alter table api_usage   add column if not exists request_id uuid;

create index if not exists idx_cost_events_request on cost_events(tenant_id, request_id);
create index if not exists idx_api_usage_request   on api_usage(tenant_id, request_id);

comment on column cost_events.request_id is
    'Partner HTTP request that produced this model call. Null before migration 004.';
comment on column api_usage.request_id is
    'Same id as cost_events.request_id for this HTTP call. Null before migration 004.';
