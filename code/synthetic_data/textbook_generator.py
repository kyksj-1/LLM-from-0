"""
教科书风格重写工具 (Textbook Generator)

本模块实现了 Phi-1 论文中的核心思路：将原始代码/文本重写为"教科书"风格的
高质量训练数据。通过系统化的 prompt 工程，将杂乱的网页/博客/代码片段转化为
结构化的教学材料。

核心功能:
1. 代码→教科书: 将代码文件重写为教科书章节
2. 文本→问答对: 将文章转化为结构化的问答数据
3. 风格迁移: 将专业文档改写为不同读者水平的版本
4. 批量处理: 支持批量重写并输出 JSONL 格式

依赖:
- openai: OpenAI 兼容 API 客户端

安装:
    pip install openai

参考:
- Gunasekar et al. (2023). Textbooks Are All You Need. (Phi-1)
- Li et al. (2023). Textbooks Are All You Need II: phi-1.5 technical report.
"""

import json
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 重写风格定义
# ============================================================

class RewriteStyle(str, Enum):
    """重写风格枚举"""
    TEXTBOOK = "textbook"           # 教科书风格
    QA_PAIRS = "qa_pairs"           # 问答对风格
    BEGINNER = "beginner"           # 初学者友好
    ADVANCED = "advanced"           # 高级/深入
    EXERCISE = "exercise"           # 练习题+解答


# ============================================================
# Prompt 模板库
# ============================================================

REWRITE_PROMPTS = {
    RewriteStyle.TEXTBOOK: """你是一位经验丰富的编程教科书作者。请将以下原始材料改写为教科书中的一个小节。

要求：
1. 以"为什么需要这个？"作为开头，用一个具体场景说明动机
2. 逐步引入概念，从简单到复杂，确保每一步都有充分的解释
3. 每个抽象概念都配有具体的代码示例
4. 在关键处添加"思考题"引导读者深入理解
5. 用类比帮助理解（如"这就像..."）
6. 以"小结"收尾，列出本节学到的要点
7. 代码示例必须可运行，包含输出注释

原始材料:
{source_text}

教科书风格输出:""",

    RewriteStyle.QA_PAIRS: """请将以下材料转化为一系列高质量的问答对。

要求：
1. 生成 5-8 个问答对
2. 问题应覆盖材料的核心知识点
3. 问题难度从易到难递进
4. 回答应准确、完整、有条理
5. 包含至少一个"为什么"类型的深层问题
6. 每个回答都应是自包含的（不依赖其他问答）

请严格按以下 JSON 格式输出：
[
  {{"question": "...", "answer": "...", "difficulty": "easy/medium/hard"}},
  ...
]

原始材料:
{source_text}

问答对 (JSON):""",

    RewriteStyle.BEGINNER: """你正在为零基础的编程初学者写一本入门读物。请将以下技术材料改写为初学者能理解的版本。

要求：
1. 避免使用专业术语，如果必须使用则立即解释
2. 用日常生活的类比来解释技术概念
3. 每个概念用一个简单的例子说明
4. 语言轻松友好，像是在和朋友聊天
5. 如果涉及代码，先用自然语言描述逻辑，再给出代码
6. 添加"小贴士"框来提醒常见误区

原始材料:
{source_text}

初学者版本:""",

    RewriteStyle.ADVANCED: """你是领域专家，请将以下材料改写为面向高级读者的深度分析文章。

要求：
1. 深入分析底层原理和设计决策
2. 讨论时间/空间复杂度和性能考量
3. 与同类方案进行对比分析
4. 指出常见的陷阱和最佳实践
5. 引用相关论文或技术文档
6. 讨论在生产环境中的实际考量

原始材料:
{source_text}

高级分析:""",

    RewriteStyle.EXERCISE: """请基于以下材料设计一组编程练习题。

要求：
1. 设计 3-5 道练习题，难度递进
2. 每题包含：题目描述、输入输出示例、提示
3. 最后提供参考答案和详细解析
4. 练习应覆盖材料中的核心技能点
5. 包含至少一道需要综合运用的"挑战题"

请按以下格式输出：
## 练习1: [题目名]（难度：⭐）
**题目**: ...
**示例**: ...
**提示**: ...

...

## 参考答案
...

原始材料:
{source_text}

练习题:""",
}


# ============================================================
# 源材料预处理
# ============================================================

@dataclass
class SourceMaterial:
    """源材料"""
    content: str
    source_type: str  # code / text / mixed
    language: str = "python"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SourcePreprocessor:
    """源材料预处理器"""

    @staticmethod
    def clean_code(code: str) -> Optional[str]:
        """
        清洗原始代码

        过滤条件:
        - 去除过长的文件（> 500 行）
        - 去除无函数/类定义的纯脚本
        - 去除注释比例过低的文件
        """
        lines = code.strip().split("\n")

        # 长度检查
        if len(lines) > 500:
            logger.debug("代码过长: %d 行", len(lines))
            return None

        if len(lines) < 5:
            logger.debug("代码过短: %d 行", len(lines))
            return None

        # 检查是否有函数/类定义
        has_definition = any(
            line.strip().startswith(("def ", "class ", "async def "))
            for line in lines
        )
        if not has_definition:
            logger.debug("代码无函数/类定义")
            return None

        return code.strip()

    @staticmethod
    def clean_text(text: str) -> Optional[str]:
        """清洗原始文本"""
        text = text.strip()

        # 长度检查
        if len(text) < 100:
            return None
        if len(text) > 20000:
            # 截断到前 20000 字符
            text = text[:20000] + "\n\n[内容截断]"

        return text

    def prepare(self, content: str, source_type: str = "auto") -> Optional[SourceMaterial]:
        """
        预处理源材料

        Args:
            content: 原始内容
            source_type: 源类型（auto/code/text）

        Returns:
            预处理后的 SourceMaterial，若不合格则返回 None
        """
        if source_type == "auto":
            # 自动检测：包含 def/class/import 且占比高则判为代码
            code_indicators = ["def ", "class ", "import ", "from ", "return "]
            code_line_count = sum(
                1 for line in content.split("\n")
                if any(line.strip().startswith(ind) for ind in code_indicators)
            )
            total_lines = len(content.split("\n"))
            source_type = "code" if code_line_count / max(total_lines, 1) > 0.2 else "text"

        if source_type == "code":
            cleaned = self.clean_code(content)
        else:
            cleaned = self.clean_text(content)

        if cleaned is None:
            return None

        return SourceMaterial(content=cleaned, source_type=source_type)


# ============================================================
# 教科书生成器
# ============================================================

@dataclass
class RewriteResult:
    """重写结果"""
    source: str
    rewritten: str
    style: str
    source_type: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextbookGenerator:
    """
    教科书风格数据生成器

    使用 LLM 将原始代码/文本重写为高质量的教学材料。
    支持多种重写风格，可批量处理。
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.preprocessor = SourcePreprocessor()

    def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """调用 LLM"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def rewrite(
        self,
        content: str,
        style: RewriteStyle = RewriteStyle.TEXTBOOK,
        source_type: str = "auto",
    ) -> Optional[RewriteResult]:
        """
        将源材料重写为指定风格

        Args:
            content: 原始内容
            style: 目标风格
            source_type: 源类型

        Returns:
            RewriteResult 或 None（若预处理失败）
        """
        # 预处理
        material = self.preprocessor.prepare(content, source_type)
        if material is None:
            logger.warning("源材料预处理失败，跳过")
            return None

        # 选择 prompt 模板
        prompt_template = REWRITE_PROMPTS[style]
        prompt = prompt_template.format(source_text=material.content)

        # 调用 LLM 重写
        rewritten = self._call_llm(prompt)

        return RewriteResult(
            source=material.content,
            rewritten=rewritten,
            style=style.value,
            source_type=material.source_type,
        )

    def rewrite_multi_style(
        self,
        content: str,
        styles: Optional[list[RewriteStyle]] = None,
        source_type: str = "auto",
    ) -> list[RewriteResult]:
        """
        将同一源材料重写为多种风格

        Args:
            content: 原始内容
            styles: 目标风格列表，默认为所有风格
            source_type: 源类型

        Returns:
            RewriteResult 列表
        """
        if styles is None:
            styles = list(RewriteStyle)

        results = []
        for style in styles:
            result = self.rewrite(content, style, source_type)
            if result:
                results.append(result)
                logger.info("完成 %s 风格重写", style.value)

        return results

    def batch_rewrite(
        self,
        sources: list[str],
        style: RewriteStyle = RewriteStyle.TEXTBOOK,
        source_type: str = "auto",
        output_file: Optional[str] = None,
    ) -> list[RewriteResult]:
        """
        批量重写

        Args:
            sources: 源材料列表
            style: 目标风格
            source_type: 源类型
            output_file: 输出文件路径（JSONL 格式），若指定则边生成边写入

        Returns:
            RewriteResult 列表
        """
        results = []
        output_path = Path(output_file) if output_file else None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        for i, source in enumerate(sources):
            logger.info("处理 %d / %d ...", i + 1, len(sources))
            result = self.rewrite(source, style, source_type)

            if result:
                results.append(result)

                # 实时写入 JSONL
                if output_path:
                    with open(output_path, "a", encoding="utf-8") as f:
                        record = {
                            "instruction": f"请将以下内容改写为{style.value}风格",
                            "input": result.source[:500],  # 截断源材料
                            "output": result.rewritten,
                            "style": result.style,
                            "source_type": result.source_type,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("批量重写完成: %d / %d 成功", len(results), len(sources))
        return results


# ============================================================
# 质量评估工具
# ============================================================

class TextbookQualityEvaluator:
    """评估重写后教科书数据的质量"""

    @staticmethod
    def evaluate(result: RewriteResult) -> dict:
        """
        评估重写质量

        指标:
        - 长度倍率: 重写后长度 / 原始长度
        - 结构化程度: 标题/列表/代码块的数量
        - 信息密度: 估算每 100 token 的概念数
        """
        source_len = len(result.source)
        rewritten_len = len(result.rewritten)

        # 结构化元素计数
        rewritten = result.rewritten
        num_headers = rewritten.count("#")
        num_lists = rewritten.count("\n- ") + rewritten.count("\n* ")
        num_code_blocks = rewritten.count("```")
        num_bold = rewritten.count("**") // 2

        # 估算概念密度（用粗体/标题数量近似）
        approx_tokens = rewritten_len / 2  # 粗略估算
        concept_density = (num_bold + num_headers) / max(approx_tokens / 100, 1)

        return {
            "source_length": source_len,
            "rewritten_length": rewritten_len,
            "length_ratio": round(rewritten_len / max(source_len, 1), 2),
            "num_headers": num_headers,
            "num_lists": num_lists,
            "num_code_blocks": num_code_blocks // 2,  # 开闭标记各一个
            "num_bold_terms": num_bold,
            "concept_density": round(concept_density, 3),
        }


# ============================================================
# 示例源材料
# ============================================================

SAMPLE_CODE = '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
'''

SAMPLE_TEXT = """
哈希表是一种通过哈希函数将键映射到存储位置的数据结构。它支持平均O(1)时间复杂度的
插入、删除和查找操作。哈希冲突可以通过链地址法或开放寻址法解决。负载因子是已存储
元素数与桶数的比值，当负载因子超过阈值（通常0.75）时需要扩容。Python的dict就是
基于哈希表实现的，它使用开放寻址法处理冲突。
"""


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="教科书风格数据生成器")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                        help="OpenAI 兼容 API 地址")
    parser.add_argument("--api-key", type=str, default="not-needed", help="API 密钥")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="模型名称")
    parser.add_argument("--style", type=str, default="textbook",
                        choices=[s.value for s in RewriteStyle],
                        help="重写风格")
    parser.add_argument("--input", type=str, default=None,
                        help="输入文件路径（代码或文本文件）")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="输入目录（批量处理所有 .py/.txt 文件）")
    parser.add_argument("--output", type=str, default="textbook_output.jsonl",
                        help="输出文件路径 (JSONL)")
    parser.add_argument("--demo", action="store_true",
                        help="使用内置示例运行演示")
    args = parser.parse_args()

    generator = TextbookGenerator(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
    )
    evaluator = TextbookQualityEvaluator()

    style = RewriteStyle(args.style)

    if args.demo:
        # 演示模式：使用内置示例
        print("=== 代码重写演示 ===\n")
        result = generator.rewrite(SAMPLE_CODE, style=style, source_type="code")
        if result:
            print(result.rewritten)
            print("\n=== 质量评估 ===")
            metrics = evaluator.evaluate(result)
            for k, v in metrics.items():
                print(f"  {k}: {v}")

        print("\n\n=== 文本重写演示 ===\n")
        result = generator.rewrite(SAMPLE_TEXT, style=style, source_type="text")
        if result:
            print(result.rewritten)
            print("\n=== 质量评估 ===")
            metrics = evaluator.evaluate(result)
            for k, v in metrics.items():
                print(f"  {k}: {v}")

    elif args.input:
        # 单文件处理
        content = Path(args.input).read_text(encoding="utf-8")
        result = generator.rewrite(content, style=style)
        if result:
            Path(args.output).write_text(
                json.dumps({"source": result.source, "rewritten": result.rewritten}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"结果已保存到 {args.output}")

    elif args.input_dir:
        # 批量处理
        input_path = Path(args.input_dir)
        sources = []
        for ext in ["*.py", "*.txt", "*.md"]:
            for f in input_path.glob(ext):
                content = f.read_text(encoding="utf-8", errors="ignore")
                sources.append(content)

        logger.info("找到 %d 个文件", len(sources))
        results = generator.batch_rewrite(sources, style=style, output_file=args.output)
        print(f"\n处理完成: {len(results)} / {len(sources)} 成功，结果保存到 {args.output}")

    else:
        print("请指定 --input、--input-dir 或 --demo 参数")
        parser.print_help()
