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

-- debates: a single debate session owned by exactly one tenant.
-- Partners pass external_user_id to map debates back to their own student records
-- without us learning who the student is. metadata is a free-form JSONB bag the
-- partner controls (rendered into the PDF report header); we enforce a size cap
-- in application code, not in SQL.
create table if not exists debates (
    id               uuid primary key default gen_random_uuid(),
    tenant_id        uuid        not null references tenants(id) on delete cascade,
    external_user_id text,
    motion           text,
    starter          text        not null check (starter in ('user', 'assistant')),
    next_speaker     text        not null check (next_speaker in ('user', 'assistant')),
    mode             text        not null default 'casual' check (mode in ('casual', 'wsdc', 'ap')),
    difficulty       text        not null default 'intermediate' check (difficulty in ('beginner', 'intermediate', 'advanced')),
    num_rounds       integer     not null check (num_rounds between 1 and 10),
    current_round    integer     not null default 1,
    status           text        not null default 'active' check (status in ('active', 'completed')),
    metadata         jsonb       not null default '{}'::jsonb,
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);

create index if not exists idx_debates_tenant_id        on debates(tenant_id);
create index if not exists idx_debates_tenant_created   on debates(tenant_id, created_at desc);
create index if not exists idx_debates_external_user_id on debates(tenant_id, external_user_id);

-- messages: turns within a debate. tenant_id is denormalized from debates so
-- every query can scope by tenant directly (defense-in-depth against a missed
-- join condition).
create table if not exists messages (
    id         uuid primary key default gen_random_uuid(),
    tenant_id  uuid        not null references tenants(id) on delete cascade,
    debate_id  uuid        not null references debates(id) on delete cascade,
    round_no   integer     not null,
    speaker    text        not null check (speaker in ('user', 'assistant')),
    content    text        not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_messages_debate_id on messages(debate_id);
create index if not exists idx_messages_tenant_id on messages(tenant_id);

-- scores: at most one final score per debate. tenant_id denormalized for the same
-- reason as messages.
create table if not exists scores (
    debate_id                  uuid        primary key references debates(id) on delete cascade,
    tenant_id                  uuid        not null references tenants(id) on delete cascade,
    overall                    numeric(3,1) not null,
    content_structure          numeric(3,1) not null,
    engagement                 numeric(3,1) not null,
    strategy                   numeric(3,1) not null,
    feedback                   text         not null default '',
    content_structure_feedback text         not null default '',
    engagement_feedback        text         not null default '',
    strategy_feedback          text         not null default '',
    weakness_type              text         check (weakness_type in ('rebuttal', 'structure', 'weighing', 'evidence', 'strategy')),
    created_at                 timestamptz  not null default now()
);

create index if not exists idx_scores_tenant_id on scores(tenant_id);

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

-- Lock the tables down: service role (used by the backend) bypasses RLS, so
-- enabling RLS with no policies blocks all access via the public/anon key.
alter table tenants   enable row level security;
alter table api_keys  enable row level security;
alter table api_usage enable row level security;
alter table debates   enable row level security;
alter table messages  enable row level security;
alter table scores              enable row level security;
alter table idempotency_records enable row level security;
