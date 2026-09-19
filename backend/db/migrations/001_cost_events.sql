-- Migration 001 — billing ledger (cost_events)
-- Apply in the Supabase SQL editor. Idempotent; safe to re-run.
-- Adds the table that app/billing.py writes to. Until this is applied, cost
-- capture degrades to a logged warning and NO billing data is recorded.

-- cost_events: one row per billable vendor model call. This is the billing
-- ledger — SuperJuniors are invoiced at cost pass-through, so vendor_cost_usd
-- is literally what they owe for that line.
--
-- tenant_id is NULLABLE on purpose: the debatelab.ai website runs through the
-- same generation functions with no tenant, and we want those costs recorded
-- too (that is how we learn what the free site costs to run). A NULL tenant
-- means "our own website", never "unknown partner".
--
-- price_version references the immutable rate card in app/pricing.py. Never
-- recompute a historical row against a newer card.
create table if not exists cost_events (
    id               bigint generated always as identity primary key,
    tenant_id        uuid        references tenants(id) on delete set null,
    debate_id        uuid,
    message_id       uuid,
    event_type       text        not null,
    provider         text        not null,
    model            text        not null,
    input_tokens     integer     not null default 0,
    output_tokens    integer     not null default 0,
    cache_read_tokens  integer   not null default 0,
    cache_write_tokens integer   not null default 0,
    audio_seconds    numeric(10,2),
    vendor_cost_usd  numeric(12,8) not null default 0,
    price_version    text        not null,
    billable         boolean     not null default true,
    needs_repricing  boolean     not null default false,
    succeeded        boolean     not null default true,
    created_at       timestamptz not null default now()
);

-- ON DELETE SET NULL above (not CASCADE): deleting a tenant must not erase the
-- billing history for work already invoiced. Same reasoning as api_usage being
-- excluded from the retention sweep.
create index if not exists idx_cost_events_tenant_created on cost_events(tenant_id, created_at);
create index if not exists idx_cost_events_created_at     on cost_events(created_at);
create index if not exists idx_cost_events_debate_id      on cost_events(debate_id);
create index if not exists idx_cost_events_repricing      on cost_events(needs_repricing) where needs_repricing;

alter table cost_events enable row level security;
