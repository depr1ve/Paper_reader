# Split documents into chunks
import re
from collections import defaultdict
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 中文学术论文章节标题模式
# 注：PDF 提取时可能在字符间插入空格（如 "摘  要"），匹配前先做去空格归一化
SECTION_PATTERNS = [
    # 数字编号: "1. 引言", "3.2.1 实验设置", "1 绪论"
    re.compile(r'^\s*(\d+(?:\.\d+)*)\s+(.{2,60})$'),
    # 中文数字: "第一章 绪论", "第三节 实验设计"
    re.compile(r'^\s*(第[一二三四五六七八九十\d]+[章节部分])\s*(.{2,60})$'),
    # 常见章节关键词（匹配时忽略字符间空格）
    re.compile(r'^\s*(摘要|Abstract|引言|Introduction|绪论|Related\s*Work|相关工作'
               r'|方法|Method|Methodology|实验|Experiment|结果|Results?'
               r'|讨论|Discussion|结论|Conclusion|总结|Summary'
               r'|展望|Future\s*Work|致谢|Acknowledgement|参考文献|References?)\s*:?\s*(.{0,40})$',
               re.IGNORECASE),
    # 封面/前页材料
    re.compile(r'^\s*(独创性声明|学位论文原创性声明|学位论文版权使用授权|授权书'
               r'|Dissertation|Thesis|分类号|学校代码|学号|密级|UDC|编号)\s*:?\s*(.{0,30})$',
               re.IGNORECASE),
]

SECTION_CLASSIFICATION = {
    'abstract': ['摘要', 'Abstract'],
    'introduction': ['引言', 'Introduction', '绪论', '相关工作', 'Related Work'],
    'methods': ['方法', 'Method', 'Methodology'],
    'experiments': ['实验', 'Experiment', '结果', 'Results'],
    'discussion': ['讨论', 'Discussion'],
    'conclusion': ['结论', 'Conclusion', '总结', 'Summary', '展望', 'Future Work'],
    'appendix': ['致谢', 'Acknowledgement', '参考文献', 'References'],
    'front_matter': ['独创性声明', '学位论文原创性声明', '学位论文版权使用授权',
                     '授权书', 'Dissertation', 'Thesis', '分类号', '学校代码',
                     '学号', '密级', 'UDC', '编号'],
}


def _normalize(text: str) -> str:
    """去掉字符间多余空格（PDF 提取常见 artifact），方便正则匹配"""
    # "摘  要" → "摘要"，"1 . 1" → "1.1"
    import re as _re
    t = _re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)  # "1 . 1" → "1.1"
    t = _re.sub(r'(?<=[^\x00-\x7f])\s+(?=[^\x00-\x7f])', '', t)  # 中文字符间的空格去掉
    return t


def detect_section(text: str):
    """检测文本开头是否为章节标题，返回 (section_title, section_type) 或 (None, None)。

    PDF 抽取文本中章节标题可能在非首行（前有页码、页眉等），扫描前 5 行。
    """
    raw_lines = text.strip().split('\n')[:5]
    # 归一化后逐行匹配
    for raw_line in raw_lines:
        raw_line = raw_line.strip()
        # 过滤空行和明显太短的行（归一化后判断，因为 PDF 抽取可能插入空格）
        normalized = _normalize(raw_line)
        if len(normalized) < 2:
            continue
        first_line = normalized
        # 过滤明显不是章节标题的行
        if len(first_line) > 60:          # 太长（可能是句子）
            continue
        if first_line.endswith('。'):      # 以句号结尾（完整句子，非标题）
            continue
        if re.search(r'\[\s*[JCSTPM]\s*\]', first_line):  # 参考文献条目
            continue

        for pi, pattern in enumerate(SECTION_PATTERNS):
            m = pattern.match(first_line)
            if m:
                # m.group(1) 是编号/前缀，m.group(2)（如有）是标题内容
                if m.lastindex and m.lastindex >= 2 and m.group(2):
                    # 关键词模式（第3/4个）: 标题后跟随内容过长 → 是句子不是标题
                    if pi >= 2 and len(m.group(2).strip()) > 10:
                        continue
                    title = (m.group(1) + ' ' + m.group(2)).strip()
                else:
                    title = m.group(1).strip()

                # 去掉 TOC 中的点线 "...... 45"
                title = re.sub(r'\s*\.{3,}\s*\d*\s*$', '', title).strip()
                if len(title) < 2:
                    continue

                # 分类
                section_type = 'body'
                for stype, keywords in SECTION_CLASSIFICATION.items():
                    if any(kw.lower() in title.lower() for kw in keywords):
                        section_type = stype
                        break

                return title, section_type
    return None, None


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False)

    # split_documents 直接传 Document 列表以保留 metadata
    texts = text_splitter.split_documents(docs)

    # ---- L1 metadata 注入 ----
    # 1. 按 source 分组计算 chunk 位置
    by_source = defaultdict(list)
    for t in texts:
        by_source[t.metadata.get('source', t.metadata.get('title', ''))].append(t)

    for _source, chunks in by_source.items():
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i

    # 分组完成后清理 source（与 paper_title 冗余，仅 paper_title 进入下游）
    for t in texts:
        t.metadata.pop('source', None)

    # 2. 检测章节标题
    current_section_title = None
    current_section_type = 'body'
    for chunk in texts:
        # 每个 chunk 尝试检测是否为章节开头
        section_title, section_type = detect_section(chunk.page_content)
        if section_title:
            current_section_title = section_title
            current_section_type = section_type
        # 注入当前章节信息
        if current_section_title:
            chunk.metadata['section_title'] = current_section_title
        chunk.metadata['section_type'] = current_section_type

    n_chunks = len(texts)
    sections_found = sum(1 for t in texts if t.metadata.get('section_title'))
    print(f"Split into {n_chunks} chunks ({sections_found} with section labels)")
    return texts
