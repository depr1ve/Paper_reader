-- ============================================================================
-- Supabase 数据库初始化 SQL
-- Paper Reader - RAG 论文问答系统
-- ============================================================================
-- 使用方式：在 Supabase Dashboard → SQL Editor 中粘贴执行
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. 启用扩展
-- ----------------------------------------------------------------------------

-- pgvector：向量存储 + 余弦相似度检索
create extension if not exists vector;

-- PGroonga：中日韩全文检索，BM25-like 相关性打分（pgroonga_score）
create extension if not exists pgroonga;

-- ----------------------------------------------------------------------------
-- 2. 创建 papers 表
-- ----------------------------------------------------------------------------
-- LangChain SupabaseVectorStore 写入时自动生成 id/content/metadata/embedding
-- 手动建表以确保字段类型和索引正确

create table if not exists papers (
    id          uuid primary key default gen_random_uuid(),
    content     text not null,
    metadata    jsonb not null default '{}'::jsonb,
    embedding   vector(512)      -- 与 BAAI/bge-small-zh-v1.5 维度一致
);

-- ----------------------------------------------------------------------------
-- 3. 索引
-- ----------------------------------------------------------------------------

-- HNSW 向量索引（Top-K 余弦相似度快速检索）
create index if not exists papers_embedding_idx
    on papers
    using hnsw (embedding vector_cosine_ops);

-- PGroonga 全文检索索引（BM25-like，TokenNgram 中文分词，替代 pg_trgm）
create index if not exists papers_content_pgroonga_idx
    on papers
    using pgroonga (content);

-- metadata 常用查询字段索引
create index if not exists papers_source_idx
    on papers ((metadata->>'paper_title'));

-- ----------------------------------------------------------------------------
-- 4. 基础向量检索函数（写入路径 / 纯向量检索用）
-- ----------------------------------------------------------------------------

create or replace function match_papers(
    query_embedding vector(512),
    match_count int default 4
) returns table (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
    return query
    select
        p.id,
        p.content,
        p.metadata,
        1 - (p.embedding <=> query_embedding) as similarity
    from papers p
    order by p.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- ----------------------------------------------------------------------------
-- 5. 混合检索函数（读取路径，Supabase 模式默认使用）
-- ----------------------------------------------------------------------------
-- Reciprocal Rank Fusion（RRF）：向量排名 + PGroonga BM25 排名 → 融合分数
-- 与 Chroma EnsembleRetriever 保持一致：权重 [0.5, 0.5]，c=60
--
-- 调用方式：supabase.rpc('hybrid_match_papers', {
--     query_embedding: <float[]>,
--     query_text: <string>,
--     match_count: <int>
-- })
--
-- 算法：
--   1. 向量检索取 top (match_count * 3)，记录向量排名 rank_vec
--   2. PGroonga BM25 全文检索取 top (match_count * 3)，记录文本排名 rank_txt
--      - 中文：TokenNgram 分词器自动切分
--      - 打分：pgroonga_score() 近似 BM25
--   3. 合并后计算 RRF = 0.5/(rank_vec+60) + 0.5/(rank_txt+60)
--   4. 按 RRF 降序返回 top match_count
--   5. 仅出现在单路的文档，另一路排名设为 pool_size+1（惩罚项）

-- 注意：返回列名从 id/content/metadata/similarity 改为了 out_id/out_content/out_meta/out_score
-- 如果之前建过旧版函数，需先 drop function if exists hybrid_match_papers(vector, text, integer);
create or replace function hybrid_match_papers(
    query_embedding vector(512),
    query_text text,
    match_count int default 4
) returns table (
    out_id      uuid,
    out_content text,
    out_meta    jsonb,
    out_score   float
) language plpgsql as $$
declare
    c constant int := 60;  -- RRF 平滑常量
    pool_size int;
begin
    pool_size := match_count * 3;

    return query
    with vec_top as (
        select p.id as doc_id, p.content, p.metadata,
               row_number() over (order by p.embedding <=> query_embedding) as rank_vec
        from papers p
        where p.embedding is not null
        order by p.embedding <=> query_embedding
        limit pool_size
    ),
    txt_top as (
        select p.id as doc_id,
               row_number() over (
                   order by pgroonga_score(p.tableoid, p.ctid) desc
               ) as rank_txt
        from papers p
        where p.content &@~ query_text
        order by pgroonga_score(p.tableoid, p.ctid) desc
        limit pool_size
    ),
    all_ids as (
        select doc_id from vec_top
        union
        select doc_id from txt_top
    )
    select
        p.id          as out_id,
        p.content     as out_content,
        p.metadata    as out_meta,
        (
            0.5 / (coalesce(v.rank_vec, pool_size + 1) + c) +
            0.5 / (coalesce(t.rank_txt, pool_size + 1) + c)
        )::float      as out_score
    from all_ids a
    join papers p on a.doc_id = p.id
    left join vec_top v on a.doc_id = v.doc_id
    left join txt_top t on a.doc_id = t.doc_id
    order by out_score desc
    limit match_count;
end;
$$;

-- ============================================================================
-- 6. 从旧版（pg_trgm）迁移（已有数据库执行）
-- ============================================================================
-- 如果你的 Supabase 项目之前使用 pg_trgm 版本的 hybrid_match_papers，
-- 按顺序执行以下 SQL 完成迁移：

-- 启用 PGroonga
-- create extension if not exists pgroonga;

-- 创建 PGroonga 索引（替换 pg_trgm GIN 索引）
-- create index if not exists papers_content_pgroonga_idx
--     on papers using pgroonga (content);

-- 执行上方最新的 hybrid_match_papers CREATE OR REPLACE 定义

-- 删除旧的 pg_trgm 索引（可选，释放磁盘空间）
-- drop index if exists papers_content_trgm_idx;

-- ============================================================================
-- 调试 / 维护 SQL（在 Supabase SQL Editor 中手动执行）
-- ============================================================================

-- 查看 papers 表结构
-- select column_name, data_type from information_schema.columns where table_name = 'papers';

-- 查看所有索引
-- select indexname, indexdef from pg_indexes where tablename = 'papers';

-- 查看文档数量
-- select count(*) from papers;

-- 按论文分组统计 chunk 数
-- select metadata->>'paper_title' as title, count(*) from papers group by 1;

-- 查看某个 chunk 的完整 metadata
-- select metadata from papers limit 1;

-- 清空所有文档（重新上传前）
-- truncate table papers;

-- 删除某篇论文的所有 chunk
-- delete from papers where metadata->>'paper_title' = '你的论文标题';
