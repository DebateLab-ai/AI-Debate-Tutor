-- Migration 003 — per-tenant spend cap + alert de-duplication
-- Apply in the Supabase SQL editor. Idempotent; safe to re-run.
--
-- Adds:
--   * tenants.monthly_cost_limit_usd  — NULL means no cap (current behaviour)
--   * cost_alerts_sent                — one row per alert actually emailed
--   * tenant_mtd_cost_usd()           — server-side SUM over cost_events
--
-- The function exists because PostgREST rejects aggregate functions on this
-- project ("Use of aggregate functions is not allowed"), so without it the
-- backend would have to page every cost row into Python just to add them up.
-- app/spendcap.py falls back to exactly that if this function is missing, so
-- the feature degrades rather than breaks.

alter table tenants
    add column if not exists monthly_cost_limit_usd numeric(10,2);

comment on column tenants.monthly_cost_limit_usd is
    'Month-to-date vendor spend ceiling in USD. NULL = uncapped. Enforced in app/spendcap.py.';

-- One row per (tenant, month, threshold) that has ALREADY been emailed. The
-- primary key is the de-duplication: the insert fails on a repeat, which is how
-- we avoid emailing the same 80%-warning on every subsequent request.
create table if not exists cost_alerts_sent (
    tenant_id     uuid        not null references tenants(id) on delete cascade,
    period        text        not null,   -- 'YYYY-MM' in UTC
    threshold_pct integer     not null,   -- 50, 80, 100
    mtd_cost_usd  numeric(12,8),
    sent_at       timestamptz not null default now(),
    primary key (tenant_id, period, threshold_pct)
);

create index if not exists idx_cost_alerts_sent_at on cost_alerts_sent(sent_at);

alter table cost_alerts_sent enable row level security;

-- Month-to-date vendor cost for one tenant, in USD.
-- STABLE (not VOLATILE) so Postgres can optimise repeated calls in a statement.
create or replace function tenant_mtd_cost_usd(p_tenant_id uuid, p_since timestamptz)
returns numeric
language sql
stable
as $$
    select coalesce(sum(vendor_cost_usd), 0)::numeric
    from cost_events
    where tenant_id = p_tenant_id
      and created_at >= p_since;
$$;
