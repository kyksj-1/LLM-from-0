"""
HTML/文本清洗与提取

本模块实现了从原始 HTML 到清洁纯文本的提取和清洗功能，
是预训练数据管线中的关键前处理步骤。

主要功能:
- HTML 标签移除与正文提取
- Unicode 规范化（NFKC、零宽字符移除）
- 空白字符规范化
- 模板文本（boilerplate）检测与移除
- 文本分段与质量评估

Web 数据（如 Common Crawl）的原始格式通常是 HTML，
需要经过文本提取才能用于语言模型训练。

参考:
- trafilatura: 专业的 Web 正文提取工具
- jusText: 基于段落分类的提取算法
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """
    文本提取结果

    Attributes:
        text: 提取出的纯文本
        title: 页面标题（如果有）
        num_paragraphs: 段落数
        original_length: 原始 HTML 长度
        extracted_length: 提取文本长度
        extraction_ratio: 提取比例
    """
    text: str
    title: str = ""
    num_paragraphs: int = 0
    original_length: int = 0
    extracted_length: int = 0
    extraction_ratio: float = 0.0


class HTMLTextExtractor:
    """
    HTML 文本提取器

    从 HTML 文档中提取正文文本，去除标签、脚本、样式等非内容元素。

    处理流程:
    1. 移除 script/style/noscript 标签及其内容
    2. 移除 HTML 注释
    3. 处理常见的 HTML 实体
    4. 移除所有 HTML 标签
    5. 提取 title 标签内容
    6. 清理多余空白

    Args:
        min_paragraph_length: 最短段落长度（字符数），更短的段落被视为噪声
        keep_links: 是否保留链接文本
    """

    # 需要完全移除的标签（包括内容）
    REMOVE_TAGS_WITH_CONTENT = [
        'script', 'style', 'noscript', 'iframe', 'svg',
        'header', 'footer', 'nav', 'aside',
    ]

    # HTML 实体映射
    HTML_ENTITIES = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'",
        '&nbsp;': ' ',
        '&mdash;': '--',
        '&ndash;': '-',
        '&hellip;': '...',
        '&copy;': '(c)',
        '&reg;': '(R)',
    }

    def __init__(
        self,
        min_paragraph_length: int = 20,
        keep_links: bool = False,
    ):
        self.min_paragraph_length = min_paragraph_length
        self.keep_links = keep_links

        # 预编译正则表达式
        self._re_comments = re.compile(r'<!--.*?-->', re.DOTALL)
        self._re_tags_to_remove = re.compile(
            r'<(' + '|'.join(self.REMOVE_TAGS_WITH_CONTENT)
            + r')[^>]*>.*?</\1>',
            re.DOTALL | re.IGNORECASE,
        )
        self._re_all_tags = re.compile(r'<[^>]+>')
        self._re_title = re.compile(
            r'<title[^>]*>(.*?)</title>',
            re.DOTALL | re.IGNORECASE,
        )
        self._re_block_tags = re.compile(
            r'</(p|div|br|h[1-6]|li|tr|td|blockquote|pre)>',
            re.IGNORECASE,
        )
        self._re_br = re.compile(r'<br\s*/?\s*>', re.IGNORECASE)

    def extract(self, html: str) -> ExtractionResult:
        """
        从 HTML 中提取纯文本

        Args:
            html: 原始 HTML 字符串

        Returns:
            ExtractionResult 包含提取的文本和元数据
        """
        original_length = len(html)

        # 提取标题
        title_match = self._re_title.search(html)
        title = ""
        if title_match:
            title = self._re_all_tags.sub('', title_match.group(1)).strip()

        # 步骤 1: 移除注释
        text = self._re_comments.sub('', html)

        # 步骤 2: 移除 script/style 等标签及其内容
        text = self._re_tags_to_remove.sub('', text)

        # 步骤 3: 将块级标签替换为换行符
        text = self._re_block_tags.sub('\n', text)
        text = self._re_br.sub('\n', text)

        # 步骤 4: 处理 HTML 实体
        for entity, replacement in self.HTML_ENTITIES.items():
            text = text.replace(entity, replacement)
        # 处理数字实体 &#123;
        text = re.sub(
            r'&#(\d+);',
            lambda m: chr(int(m.group(1)))
            if int(m.group(1)) < 0x10000 else '',
            text,
        )
        # 处理十六进制实体 &#x1F;
        text = re.sub(
            r'&#x([0-9a-fA-F]+);',
            lambda m: chr(int(m.group(1), 16))
            if int(m.group(1), 16) < 0x10000 else '',
            text,
        )

        # 步骤 5: 移除所有剩余 HTML 标签
        text = self._re_all_tags.sub('', text)

        # 步骤 6: 清理空白
        text = self._normalize_whitespace(text)

        # 步骤 7: 过滤短段落
        paragraphs = [
            p.strip() for p in text.split('\n')
            if len(p.strip()) >= self.min_paragraph_length
        ]
        text = '\n\n'.join(paragraphs)

        extracted_length = len(text)

        return ExtractionResult(
            text=text,
            title=title,
            num_paragraphs=len(paragraphs),
            original_length=original_length,
            extracted_length=extracted_length,
            extraction_ratio=(
                extracted_length / original_length
                if original_length > 0 else 0.0
            ),
        )

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        # 行内空白: 连续空格/制表符合并为一个空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 连续空行合并为两个换行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)


class UnicodeNormalizer:
    """
    Unicode 规范化器

    统一文本的 Unicode 编码，处理以下问题:
    - 全角/半角字符不一致
    - 兼容字符（如连字、特殊符号）
    - 零宽字符
    - 控制字符
    - 异常的空白字符

    使用 NFKC 规范化，它同时执行兼容分解和规范组合。

    NFKC 的效果:
    - 全角字母 -> 半角: 'Ａ' -> 'A'
    - 兼容字符 -> 标准: 'fi' (连字) -> 'fi'
    - 上标/下标 -> 普通: 'x^2' -> 'x2' (需注意)
    """

    # 零宽字符
    ZERO_WIDTH_CHARS = set('\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad')

    # 异常空白字符映射到普通空格
    SPACE_CHARS = set(
        '\u00a0'  # 不换行空格
        '\u2000'  # En Quad
        '\u2001'  # Em Quad
        '\u2002'  # En Space
        '\u2003'  # Em Space
        '\u2004'  # Three-Per-Em Space
        '\u2005'  # Four-Per-Em Space
        '\u2006'  # Six-Per-Em Space
        '\u2007'  # Figure Space
        '\u2008'  # Punctuation Space
        '\u2009'  # Thin Space
        '\u200a'  # Hair Space
        '\u202f'  # Narrow No-Break Space
        '\u205f'  # Medium Mathematical Space
        '\u3000'  # 全角空格
    )

    def normalize(self, text: str) -> str:
        """
        对文本执行完整的 Unicode 规范化

        Args:
            text: 输入文本

        Returns:
            规范化后的文本
        """
        # 步骤 1: NFKC 规范化
        text = unicodedata.normalize('NFKC', text)

        # 步骤 2: 移除零宽字符
        text = ''.join(
            c for c in text if c not in self.ZERO_WIDTH_CHARS
        )

        # 步骤 3: 统一空白字符
        text = ''.join(
            ' ' if c in self.SPACE_CHARS else c for c in text
        )

        # 步骤 4: 移除控制字符（保留换行和制表）
        text = ''.join(
            c for c in text
            if c in ('\n', '\t', '\r') or not unicodedata.category(c).startswith('C')
        )

        # 步骤 5: 规范化换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        return text


class TextCleaner:
    """
    文本清洗器

    对提取出的纯文本进行深度清洗，去除残余噪声。

    清洗步骤:
    1. Unicode 规范化
    2. URL 移除或替换
    3. 重复空白规范化
    4. 模板文本检测（如导航栏、页脚）
    5. 短行过滤

    Args:
        remove_urls: 是否移除 URL
        url_replacement: URL 替换文本（当 remove_urls=True 时使用）
        min_line_words: 保留行的最少词数
        max_consecutive_newlines: 最大连续空行数
    """

    def __init__(
        self,
        remove_urls: bool = True,
        url_replacement: str = '',
        min_line_words: int = 3,
        max_consecutive_newlines: int = 2,
    ):
        self.remove_urls = remove_urls
        self.url_replacement = url_replacement
        self.min_line_words = min_line_words
        self.max_consecutive_newlines = max_consecutive_newlines

        self._normalizer = UnicodeNormalizer()
        self._re_url = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE
        )

        # 模板文本模式
        self._boilerplate_patterns = [
            re.compile(r'copyright\s*\d{4}', re.IGNORECASE),
            re.compile(r'all rights reserved', re.IGNORECASE),
            re.compile(r'terms (of|and) (use|service|conditions)', re.IGNORECASE),
            re.compile(r'privacy policy', re.IGNORECASE),
            re.compile(r'cookie (policy|settings|preferences)', re.IGNORECASE),
            re.compile(r'subscribe to (our )?newsletter', re.IGNORECASE),
            re.compile(r'follow us on', re.IGNORECASE),
            re.compile(r'share (this|on)', re.IGNORECASE),
            re.compile(r'click here to', re.IGNORECASE),
            re.compile(r'powered by', re.IGNORECASE),
        ]

    def clean(self, text: str) -> str:
        """
        对文本进行深度清洗

        Args:
            text: 输入文本

        Returns:
            清洗后的文本
        """
        # 步骤 1: Unicode 规范化
        text = self._normalizer.normalize(text)

        # 步骤 2: URL 处理
        if self.remove_urls:
            text = self._re_url.sub(self.url_replacement, text)

        # 步骤 3: 按行处理
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # 跳过空行（保留一个占位）
            if not line:
                cleaned_lines.append('')
                continue

            # 跳过太短的行
            if len(line.split()) < self.min_line_words:
                continue

            # 检测模板文本
            if self._is_boilerplate(line):
                continue

            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # 步骤 4: 规范化连续空行
        max_nl = '\n' * (self.max_consecutive_newlines + 1)
        replacement_nl = '\n' * self.max_consecutive_newlines
        while max_nl in text:
            text = text.replace(max_nl, replacement_nl)

        return text.strip()

    def _is_boilerplate(self, line: str) -> bool:
        """检测是否为模板文本"""
        for pattern in self._boilerplate_patterns:
            if pattern.search(line):
                return True
        return False


class DocumentProcessor:
    """
    文档处理器

    将 HTML 提取、Unicode 规范化、文本清洗整合为一个完整流程。

    使用方法:
        processor = DocumentProcessor()
        result = processor.process(html_content)
        clean_text = result.text

    Args:
        min_paragraph_length: 最短段落长度
        remove_urls: 是否移除 URL
        min_output_length: 处理后文本的最短长度（低于此值则丢弃）
    """

    def __init__(
        self,
        min_paragraph_length: int = 20,
        remove_urls: bool = True,
        min_output_length: int = 100,
    ):
        self.min_output_length = min_output_length
        self._extractor = HTMLTextExtractor(
            min_paragraph_length=min_paragraph_length
        )
        self._cleaner = TextCleaner(remove_urls=remove_urls)

    def process(self, html: str) -> Optional[ExtractionResult]:
        """
        处理单个 HTML 文档

        Args:
            html: 原始 HTML

        Returns:
            ExtractionResult 或 None（如果提取结果太短）
        """
        # 提取文本
        result = self._extractor.extract(html)

        # 深度清洗
        cleaned_text = self._cleaner.clean(result.text)

        # 长度检查
        if len(cleaned_text) < self.min_output_length:
            return None

        # 更新结果
        result.text = cleaned_text
        result.extracted_length = len(cleaned_text)
        result.extraction_ratio = (
            len(cleaned_text) / result.original_length
            if result.original_length > 0 else 0.0
        )
        result.num_paragraphs = len(
            [p for p in cleaned_text.split('\n\n') if p.strip()]
        )

        return result

    def process_batch(
        self, html_docs: List[str], verbose: bool = True
    ) -> List[ExtractionResult]:
        """
        批量处理 HTML 文档

        Args:
            html_docs: HTML 文档列表
            verbose: 是否打印进度

        Returns:
            成功提取的 ExtractionResult 列表
        """
        results = []
        failed = 0

        for i, html in enumerate(html_docs):
            result = self.process(html)
            if result is not None:
                results.append(result)
            else:
                failed += 1

        if verbose:
            print(f"[文本提取] 处理 {len(html_docs)} 个文档: "
                  f"成功 {len(results)}, 丢弃 {failed}")

        return results


if __name__ == '__main__':
    print("=" * 60)
    print("HTML/文本清洗与提取 演示")
    print("=" * 60)

    # 测试 HTML 文档
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Machine Learning Introduction</title>
        <style>body { font-family: Arial; }</style>
        <script>console.log('hello');</script>
    </head>
    <body>
        <header>
            <nav>Home | About | Contact</nav>
        </header>

        <div class="content">
            <h1>Introduction to Machine Learning</h1>

            <p>Machine learning is a branch of artificial intelligence (AI)
            that focuses on building applications that learn from data and
            improve their accuracy over time without being programmed to do so.</p>

            <p>In data science, an algorithm is a sequence of statistical
            processing steps. In machine learning, algorithms are trained to
            find patterns and features in massive amounts of data in order to
            make decisions and predictions based on new data.</p>

            <p>The better the algorithm, the more accurate the decisions and
            predictions will become as it processes more data.</p>

            <!-- This is a comment -->

            <p>Contact us at info@example.com or call 13800138000.</p>

            <p>Visit us at https://www.example.com/ml-intro for more details.</p>
        </div>

        <footer>
            <p>Copyright 2024 All Rights Reserved.</p>
            <p>Privacy Policy | Terms of Service</p>
            <p>Follow us on Twitter</p>
            <p>Powered by WordPress</p>
        </footer>
    </body>
    </html>
    """

    # 1. HTML 提取
    print("\n--- HTML 文本提取 ---")
    extractor = HTMLTextExtractor(min_paragraph_length=20)
    result = extractor.extract(sample_html)
    print(f"  标题: {result.title}")
    print(f"  段落数: {result.num_paragraphs}")
    print(f"  提取比例: {result.extraction_ratio:.2%}")
    print(f"  提取文本:\n{result.text}")

    # 2. Unicode 规范化
    print("\n--- Unicode 规范化 ---")
    normalizer = UnicodeNormalizer()
    test_cases = [
        ("全角字母", "Ｈｅｌｌｏ Ｗｏｒｌｄ"),
        ("零宽字符", "Hello\u200bWorld\u200c!"),
        ("特殊空格", "Hello\u3000World\u00a0!"),
    ]
    for name, text in test_cases:
        normalized = normalizer.normalize(text)
        print(f"  {name}: '{text}' -> '{normalized}'")

    # 3. 文本清洗
    print("\n--- 文本清洗 ---")
    cleaner = TextCleaner(remove_urls=True, min_line_words=3)
    raw_text = (
        "Introduction to Machine Learning\n\n"
        "Machine learning is a subset of artificial intelligence.\n"
        "Visit https://example.com for more.\n"
        "OK\n"  # 太短，会被移除
        "Copyright 2024 All Rights Reserved.\n"  # 模板文本
        "Deep learning uses neural networks with multiple layers.\n"
    )
    cleaned = cleaner.clean(raw_text)
    print(f"  原文行数: {len(raw_text.split(chr(10)))}")
    print(f"  清洗后:\n{cleaned}")

    # 4. 完整文档处理
    print("\n--- 完整文档处理管线 ---")
    processor = DocumentProcessor(
        min_paragraph_length=30,
        remove_urls=True,
        min_output_length=50,
    )
    final_result = processor.process(sample_html)
    if final_result:
        print(f"  标题: {final_result.title}")
        print(f"  段落数: {final_result.num_paragraphs}")
        print(f"  提取比例: {final_result.extraction_ratio:.2%}")
        print(f"  最终文本:\n{final_result.text}")
    else:
        print("  文档被丢弃（提取文本过短）")
