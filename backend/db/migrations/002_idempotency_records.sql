-- Migration 002 — idempotency_records
-- Apply in the Supabase SQL editor. Idempotent; safe to re-run.
--
-- This table has been referenced by app/api_v1.py since the idempotency feature
-- shipped, but was never applied to the live database. _idempotency_cached()
-- swallows the missing-table error and returns None, so Idempotency-Key is
-- currently a SILENT NO-OP in production: a partner retry re-runs generation.
-- Under cost pass-through billing that double-charges for one speech.
--
-- Verify after applying: scripts/smoke_api_v1_lifecycle.py --with-ai
-- (the "idempotent turn" assertion fails until this exists).

-- idempotency_records: cached 2xx responses for POST /open and POST /turns.
-- Partners send Idempotency-Key to safely retry after timeouts. Only successful
-- responses are stored; failures roll back partial writes and are not cached.
create table if not exists idempotency_records (
    tenant_id        uuid        not null references tenants(id) on delete cascade,
    idempotency_key  text        not null,
    endpoint         text        not null,
    response_status  integer     not null,
    response_body    jsonb       not null,
    created_at       timestamptz not null default now(),
    primary key (tenant_id, idempotency_key)
);

create index if not exists idx_idempotency_created_at on idempotency_records(created_at);

alter table idempotency_records enable row level security;
