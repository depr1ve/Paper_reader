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

-- pg_trgm：三元组文本相似度（混合检索的关键词匹配）
create extension if not exists pg_trgm;

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

-- pg_trgm 文本索引（LIKE / similarity() 加速）
create index if not exists papers_content_trgm_idx
    on papers
    using gin (content gin_trgm_ops);

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
-- 向量余弦相似度（50%）+ pg_trgm 文本相似度（50%）
-- 调用方式：supabase.rpc('hybrid_match_papers', {
--     query_embedding: <float[]>,
--     query_text: <string>,
--     match_count: <int>
-- })

create or replace function hybrid_match_papers(
    query_embedding vector(512),
    query_text text,
    match_count int default 4
) returns table (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
) language plpgsql as $$
begin
    return query
    select
        p.id,
        p.content,
        p.metadata,
        (0.5 * (1 - (p.embedding <=> query_embedding))
         + 0.5 * coalesce(similarity(p.content, query_text), 0)) as similarity
    from papers p
    where p.embedding is not null
      and p.content is not null
    order by similarity desc
    limit match_count;
end;
$$;

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
