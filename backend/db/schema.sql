-- AI Debate Tutor — third-party API schema (Supabase / Postgres)
--
-- This mirrors the tables the backend expects at runtime:
--   * app/auth.py  reads  api_keys  (+ embedded tenants) on every authenticated request
--   * app/usage.py writes api_usage on every authenticated request
--
-- Apply via the Supabase SQL editor or `psql`. Safe to re-run (idempotent).
-- The backend connects with the service role key (see app/db.py), which bypasses
-- RLS; RLS is enabled below with no public policies so the anon/public key can
-- never read these tables (they hold API key hashes).

-- Required for gen_random_uuid() on older Postgres; no-op if already present.
create extension if not exists "pgcrypto";

-- tenants: an account/customer authorized to use the third-party API.
create table if not exists tenants (
    id         uuid primary key default gen_random_uuid(),
    name       text        not null,
    is_active  boolean     not null default true,
    created_at timestamptz not null default now()
);

-- api_keys: hashed API keys belonging to a tenant.
-- Only the SHA-256 hex digest is stored (auth.py hashes the raw key before lookup);
-- the raw key is shown to the tenant once at creation and never persisted.
create table if not exists api_keys (
    id           uuid primary key default gen_random_uuid(),
    tenant_id    uuid        not null references tenants(id) on delete cascade,
    key_hash     text        not null unique,
    is_active    boolean     not null default true,
    last_used_at timestamptz,
    created_at   timestamptz not null default now()
);

create index if not exists idx_api_keys_key_hash  on api_keys(key_hash);
create index if not exists idx_api_keys_tenant_id  on api_keys(tenant_id);

-- api_usage: one row per authenticated API call, for billing/limit accounting.
create table if not exists api_usage (
    id              bigint generated always as identity primary key,
    tenant_id       uuid        not null references tenants(id)  on delete cascade,
    api_key_id      uuid        not null references api_keys(id) on delete cascade,
    endpoint        text        not null,
    response_status integer     not null,
    latency_ms      integer,
    created_at      timestamptz not null default now()
);

create index if not exists idx_api_usage_tenant_id  on api_usage(tenant_id);
create index if not exists idx_api_usage_created_at on api_usage(created_at);

-- Lock the tables down: service role (used by the backend) bypasses RLS, so
-- enabling RLS with no policies blocks all access via the public/anon key.
alter table tenants   enable row level security;
alter table api_keys  enable row level security;
alter table api_usage enable row level security;
