"""
BPE分词器完整实现
包含训练、编码、解码功能

作者: LLM学习教程
模块: 模块1 - Tokenization
"""

import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import unicodedata


class BPETokenizer:
    """
    从零实现的BPE（Byte Pair Encoding）分词器
    
    功能：
    - 训练：从语料库学习合并规则
    - 编码：将文本转换为词元ID序列
    - 解码：将词元ID序列还原为文本
    - 保存/加载：持久化分词器
    
    数学原理：
    BPE通过迭代合并最高频的相邻词元对来构建词汇表。
    目标是最小化编码长度：L(V) = -Σ c(w) log₂|V|
    其中c(w)是词元w的出现次数。
    """
    
    def __init__(self):
        # 核心属性
        self.vocab: Set[str] = set()  # 词汇表
        self.merges: List[Tuple[str, str]] = []  # 合并规则（按优先级排序）
        self.token_to_id: Dict[str, int] = {}  # 词元到ID的映射
        self.id_to_token: Dict[int, str] = {}  # ID到词元的映射
        
        # 特殊token
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,  # BOS
            '</s>': 3,  # EOS
        }
        
        # 正则表达式：用于预分词
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """
        预分词：将文本分割为词的列表
        
        Args:
            text: 输入文本
            
        Returns:
            词的列表
        """
        # 使用正则分割，保留标点和空格信息
        words = re.findall(self.pattern, text)
        return words
    
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """
        统计相邻词元对的频率
        
        数学形式：
        count((x, y)) = Σ freq(word) * occurrence((x,y) in word)
        
        Args:
            word_freqs: {(词元序列): 频率}
            
        Returns:
            {(词元1, 词元2): 频率}
        """
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, 
                    word_freqs: Dict[Tuple[str, ...], int], 
                    pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
        """
        在所有词中合并指定的词元对
        
        Args:
            word_freqs: {(词元序列): 频率}
            pair: 要合并的词元对 (x, y)
            
        Returns:
            更新后的词频字典，其中所有 (x, y) 都被替换为 xy
        """
        new_word_freqs = {}
        bigram = pair
        replacement = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                # 检查当前位置是否是要合并的对
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, 
              corpus: List[str], 
              vocab_size: int, 
              show_progress: bool = True,
              min_frequency: int = 2) -> None:
        """
        在语料库上训练BPE分词器
        
        算法流程：
        1. 初始化词汇表为所有唯一字符
        2. 统计所有相邻词元对的频率
        3. 选择频率最高的对进行合并
        4. 更新词汇表和合并规则
        5. 重复步骤2-4直到达到目标词汇量
        
        Args:
            corpus: 文本列表
            vocab_size: 目标词汇量
            show_progress: 是否显示训练进度
            min_frequency: 合并的最小频率阈值
        """
        if show_progress:
            print("="*50)
            print("开始BPE训练")
            print("="*50)
        
        # Step 1: 预分词并统计词频
        if show_progress:
            print("Step 1: 预分词...")
        
        word_freqs = defaultdict(int)
        for text in corpus:
            words = self._pre_tokenize(text)
            for word in words:
                # 将每个词转换为字符序列，添加结束标记
                word_chars = tuple(list(word) + ['</w>'])
                word_freqs[word_chars] += 1
        
        if show_progress:
            print(f"  唯一词数: {len(word_freqs)}")
        
        # Step 2: 初始化词汇表
        if show_progress:
            print("Step 2: 初始化词汇表...")
        
        self.vocab = set()
        for word in word_freqs:
            for char in word:
                self.vocab.add(char)
        
        # 添加特殊token
        for token in self.special_tokens:
            self.vocab.add(token)
        
        initial_vocab_size = len(self.vocab)
        
        if show_progress:
            print(f"  初始词汇表大小: {initial_vocab_size}")
            print(f"  目标词汇表大小: {vocab_size}")
        
        # Step 3: 迭代合并
        if show_progress:
            print("Step 3: 迭代合并...")
        
        num_merges = vocab_size - initial_vocab_size
        
        for i in range(num_merges):
            # 统计相邻对频率
            pairs = self._get_stats(word_freqs)
            
            if not pairs:
                if show_progress:
                    print(f"  第{i}轮：无更多可合并的对，提前停止")
                break
            
            # 过滤低频对
            pairs = {k: v for k, v in pairs.items() if v >= min_frequency}
            if not pairs:
                if show_progress:
                    print(f"  第{i}轮：所有对的频率都低于{min_frequency}，停止")
                break
            
            # 选择频率最高的对
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]
            
            # 合并
            word_freqs = self._merge_pair(word_freqs, best_pair)
            
            # 更新词汇表和合并规则
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add(new_token)
            self.merges.append(best_pair)
            
            if show_progress and (i + 1) % 100 == 0:
                print(f"  合并进度: {i + 1}/{num_merges}, "
                      f"词汇表大小: {len(self.vocab)}, "
                      f"最新合并: {best_pair} → '{new_token}' (频率: {best_freq})")
        
        # Step 4: 构建映射
        self._build_token_mappings()
        
        if show_progress:
            print("="*50)
            print(f"训练完成！")
            print(f"  最终词汇表大小: {len(self.vocab)}")
            print(f"  合并规则数量: {len(self.merges)}")
            print("="*50)
    
    def _build_token_mappings(self) -> None:
        """
        构建词元和ID的双向映射
        
        排序规则：
        1. 特殊token排在最前面
        2. 其他token按长度降序排列（确保贪婪匹配优先匹配长词元）
        """
        # 特殊token
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        # 其他token按长度排序
        other_tokens = sorted(
            [t for t in self.vocab if t not in self.special_tokens],
            key=lambda x: (-len(x), x)
        )
        
        start_id = len(self.special_tokens)
        for i, token in enumerate(other_tokens):
            self.token_to_id[token] = start_id + i
            self.id_to_token[start_id + i] = token
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            词元列表
        """
        # 预分词
        words = self._pre_tokenize(text)
        
        all_tokens = []
        for word in words:
            # 转换为字符序列
            word_tokens = list(word) + ['</w>']
            
            # 按顺序应用合并规则
            for merge_pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge_pair[0] and word_tokens[i + 1] == merge_pair[1]:
                        word_tokens = word_tokens[:i] + [merge_pair[0] + merge_pair[1]] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            all_tokens.extend(word_tokens)
        
        return all_tokens
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        将文本编码为ID序列
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加BOS/EOS
            
        Returns:
            ID列表
        """
        tokens = self.tokenize(text)
        
        ids = []
        if add_special_tokens:
            ids.append(self.special_tokens['<s>'])
        
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.special_tokens['<unk>'])
        
        if add_special_tokens:
            ids.append(self.special_tokens['</s>'])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将ID序列解码为文本
        
        Args:
            ids: ID列表
            skip_special_tokens: 是否跳过特殊token
            
        Returns:
            解码后的文本
        """
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append('<unk>')
        
        text = ''.join(tokens)
        # 移除词结束标记，还原空格
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def get_vocab_size(self) -> int:
        """返回词汇表大小"""
        return len(self.vocab)
    
    def save(self, path: str) -> None:
        """
        保存分词器到文件
        
        Args:
            path: 保存路径
        """
        data = {
            'vocab': list(self.vocab),
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'token_to_id': self.token_to_id,
        }
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"分词器已保存到: {path}")
    
    def load(self, path: str) -> None:
        """
        从文件加载分词器
        
        Args:
            path: 文件路径
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = set(data['vocab'])
        self.merges = [tuple(m) for m in data['merges']]
        self.special_tokens = data['special_tokens']
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data.get('id_to_token', {}).items()}
        
        if not self.id_to_token:
            self._build_token_mappings()
        
        print(f"分词器已从 {path} 加载")


class ByteBPETokenizer(BPETokenizer):
    """
    Byte-level BPE分词器（GPT-2风格）
    
    核心创新：
    将文本先编码为UTF-8字节序列，再进行BPE。
    这样可以处理任意Unicode字符，彻底解决OOV问题。
    
    优势：
    1. 词汇表大小可控（基础256个字节）
    2. 无OOV问题
    3. 跨语言通用
    """
    
    @staticmethod
    def bytes_to_unicode() -> Dict[int, str]:
        """
        返回字节到Unicode字符的映射
        
        目的：使分词结果更可读
        
        映射规则：
        - 可打印ASCII字符（!-~）和部分扩展拉丁字符保持不变
        - 其他字节映射到256+的Unicode码点
        
        Returns:
            {字节值: Unicode字符}
        """
        # 可打印字符保持原样
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))
        
        cs = bs[:]
        n = 0
        
        # 其他字节映射到更高码点
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def __init__(self):
        super().__init__()
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    def _text_to_byte_tokens(self, text: str) -> str:
        """
        将文本转换为字节级别的Unicode字符串
        
        Args:
            text: 原始文本
            
        Returns:
            字节映射后的Unicode字符串
        """
        return ''.join(self.byte_encoder[b] for b in text.encode('utf-8'))
    
    def _byte_tokens_to_text(self, byte_tokens: str) -> str:
        """
        将字节级别的Unicode字符串还原为原始文本
        
        Args:
            byte_tokens: 字节映射后的字符串
            
        Returns:
            原始文本
        """
        byte_values = bytes([self.byte_decoder[c] for c in byte_tokens])
        return byte_values.decode('utf-8', errors='replace')
    
    def tokenize(self, text: str) -> List[str]:
        """
        Byte-level BPE分词
        
        Args:
            text: 输入文本
            
        Returns:
            词元列表
        """
        bpe_tokens = []
        
        # 使用正则预分词
        for match in re.finditer(self.pattern, text):
            token = match.group()
            # 转换为字节级别
            byte_token = self._text_to_byte_tokens(token)
            # 应用BPE合并规则
            word_tokens = list(byte_token) + ['</w>']
            
            for merge_pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge_pair[0] and word_tokens[i + 1] == merge_pair[1]:
                        word_tokens = word_tokens[:i] + [merge_pair[0] + merge_pair[1]] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            bpe_tokens.extend(word_tokens)
        
        return bpe_tokens
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将ID序列解码为文本
        
        Args:
            ids: ID列表
            skip_special_tokens: 是否跳过特殊token
            
        Returns:
            解码后的文本
        """
        # 获取词元
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append('<unk>')
        
        # 合并词元
        byte_string = ''.join(tokens)
        # 移除结束标记
        byte_string = byte_string.replace('</w>', '')
        
        # 转换回原始文本
        try:
            return self._byte_tokens_to_text(byte_string)
        except:
            return byte_string  # 如果转换失败，返回原始字符串


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*60)
    print("BPE分词器演示")
    print("="*60)
    
    # 训练语料
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog sleeps all day long",
        "the quick brown fox is very fast and agile",
        "a lazy dog and a quick brown fox became friends",
        "the fox and the dog played together in the forest",
        "machine learning is transforming the world",
        "natural language processing enables computers to understand text",
        "tokenization is the first step in nlp pipelines",
    ]
    
    # 创建并训练分词器
    print("\n1. 训练BPE分词器")
    print("-"*60)
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=200, show_progress=True)
    
    # 测试编码解码
    print("\n2. 测试编码解码")
    print("-"*60)
    test_texts = [
        "the quick fox",
        "a lazy dog",
        "machine learning is great",
    ]
    
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        
        print(f"\n原文: {text}")
        print(f"词元: {tokens}")
        print(f"ID:   {ids}")
        print(f"解码: {decoded}")
        print(f"压缩率: {len(text)/len(tokens):.2f} chars/token")
    
    # 保存和加载
    print("\n3. 测试保存和加载")
    print("-"*60)
    tokenizer.save("my_tokenizer.json")
    
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("my_tokenizer.json")
    
    # 验证
    text = "the quick brown fox"
    assert tokenizer.encode(text) == new_tokenizer.encode(text)
    print("保存和加载验证通过！")
    
    # Byte-level BPE演示
    print("\n4. Byte-level BPE演示")
    print("-"*60)
    byte_tokenizer = ByteBPETokenizer()
    byte_tokenizer.train(corpus, vocab_size=200, show_progress=False)
    
    test_texts = [
        "Hello World!",
        "这是一个中文测试",
        "こんにちは",  # 日语
        "🎉🚀💯",  # Emoji
    ]
    
    for text in test_texts:
        tokens = byte_tokenizer.tokenize(text)
        ids = byte_tokenizer.encode(text)
        decoded = byte_tokenizer.decode(ids)
        
        print(f"\n原文: {text}")
        print(f"词元: {tokens[:10]}..." if len(tokens) > 10 else f"词元: {tokens}")
        print(f"解码: {decoded}")
        print(f"还原成功: {text == decoded}")
