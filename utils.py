import json
import os
from typing import List, Union
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from cdifflib import CSequenceMatcher


en_punc = set(list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
zh_punc = set(list("＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“‟”„…‧﹏﹑﹔·！？｡。"))


def read_txt(path: str, encode="utf-8"):
    data = []
    with open(path, 'r', encoding=encode) as f:
        for i in f:
            x = i.strip()
            if not x:
                continue
            data.append(x)
    return data


def read_multi_txt(path: str, length=None, splitter='\t', encode="utf-8"):
    data = []
    with open(path, 'r', encoding=encode) as f:
        for i in f:
            x = i.strip().split(splitter)
            if len(x) != length: continue
            data.append(x)
    return data


def read_jsonl(path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        for i in f.readlines():
            try:
                x = json.loads(i.strip())
            except:
                print("json load error")
                continue
            if not x: continue
            data.append(x)
    return data

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_csv(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.where(pd.notnull(df), None)
    data_list = df.values.tolist()
    return data_list

def write_jsonl(path, data):
    with open(path, 'w', encoding="utf-8") as f:
        for i in data:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

def write_json(path, data):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def write_txt(path, data):
    with open(path, 'w', encoding="utf-8") as f:
        for i in data:
            f.write(i + "\n")


def read_excel_and_to_list(path, sheet_name=0):
    data = pd.read_excel(path, keep_default_na=False, sheet_name=sheet_name)
    df = pd.DataFrame(data)
    df = np.array(df)
    df = df.tolist()
    return df


def find_num(text):
    # 找出所有数字
    result = re.findall(r'[0-9]+\.?[0-9]*', text)
    return result


def is_chinese_char(char):
    """判断是否是汉字"""
    if '\u4e00' <= char <= '\u9fa5':
        return True
    return False


def is_english_char(char):
    """判断是否是英文字符"""
    if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
        return True
    return False


def is_punctuation_char(char):
    """判断是否是标点符号"""
    if char in en_punc or char in zh_punc:
        return True
    return False


def is_number_char(char):
    """判断是否是数字字符"""
    if '\u0030' <= char <= '\u0039':
        return True
    return False


def count_chinese_char(text):
    """统计中文字符个数"""
    count = 0
    for char in text:
        if is_chinese_char(char):
            count += 1
    return count


def find_most_char(text, top=3):
    """找出文档中出现最多的字符"""
    char_dict = defaultdict(int)
    for char in text:
        char_dict[char] += 1
    sorted_char_dict = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_char_dict[0][0]  # [(char, num), (...), ...]


def get_ngrams_english(text, n):
    ngrams = defaultdict(int)
    text = text.lower().replace('\n', ' ')
    words = text.split(' ')
    for i in range(len(words)-n+1):
        gram = ' '.join(words[i:i+n])
        ngrams[gram] += 1
    return ngrams


def get_ngrams_chinese(text, n):
    ngrams = defaultdict(int)
    text = text.lower().replace('\n', '')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', '', text)
    words = list(text)
    for i in range(len(words)-n+1):
        gram = ''.join(words[i:i+n])
        ngrams[gram] += 1
    return ngrams, words


def longest_dup_substring(s: str) -> str:
    # 左指针
    left, right, res, n = 0, 1, "", len(s)
    while right < n:
        if s[left:right] in s[left + 1:]:
            if right - left > len(res):
                res = s[left:right]
            right += 1
            continue
        left += 1
        if left == right:
            right += 1
    return res


def remove_all_punctuation(text):
    """去除所有标点符号"""
    text = re.sub(r'[^\w\s]', '', text)
    return text


def analysis_nums(data):
    """统计数据中的数字"""
    data = pd.Series(data)
    print(data.describe(percentiles=[.25, .5, .75, .90, .95, .99]))


def to_excel(path: str, data: List[List], header: List) -> None:
    data = pd.DataFrame(data)
    data.to_excel(path, index=False, header=header)


def preprocess_sql(sql_text):
    """处理sql中的录题数据"""
    if not sql_text: return ""

    # if len(sql_text) > 160 or len(sql_text) < 8: return ""
    def process_sql_latex(sql_text):
        img_latex_pt = re.compile(r"<img.*?data-latex=\"\\\\(.*?)\"/>")
        m = img_latex_pt.search(sql_text)
        while m is not None:
            latex = m.group(1)
            sql_text = re.sub(img_latex_pt, ' ' + latex + ' ', sql_text)
            m = img_latex_pt.search(sql_text)
        return sql_text

    text = str(sql_text).strip()
    text = text.replace("\n", "")

    text = process_sql_latex(text)

    img_pt = re.compile(r"<img.*?>")
    div_pt = re.compile(r"<div.*?>")
    pstyle_pt = re.compile(r"<p style.*?>")
    span_pt = re.compile(r"<span.*?>")
    p1 = re.compile(r"<p.*?>")
    p2 = re.compile(r"<label.*?/label>")
    p3 = re.compile(r"<input.*?>")
    p4 = re.compile(r"<blk.*?/blk>")
    p5 = re.compile(r"<table.*?>")
    p6 = re.compile(r"<ul.*?>")
    p7 = re.compile(r"<li.*?/li>")
    p8 = re.compile(r"<td.*?>")
    p9 = re.compile(r"<em.*?>")
    p10 = re.compile(r"<dl.*?>")
    p11 = re.compile(r"<dl.*?>")
    p12 = re.compile(r"<.*?>")

    replace_pts = [img_pt, div_pt, pstyle_pt, span_pt, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    replace_symbs = ['<p>', '</p>', '</div>', '</span>', '&quot;', '&nbsp;', '<br/>',
                     '</td>', '</tr>', '</tbody>', '</table>', '<br>', '<tbody>',
                     '<tr>', '<td>', '<li>', '</li>', '</ul>', '<ul>', '<u>', '</u>',
                     '<sup>', '</sup>', '$', '\\t', '</em>', '<dd>', '</dd>', '</dl>',
                     '>>>>', '>>>', '<<<', '<<<<', '&ldquo;', '&rdquo;']

    for pt in replace_pts:
        text = re.sub(pt, ' ', text)

    for symb in replace_symbs:
        text = text.replace(symb, '')
    text = ' '.join(text.split())
    return text


def gen_batch_text(texts, batch_size=64):
    batch_texts = []
    for i, text in enumerate(texts):
        if i % batch_size == 0:
            batch_texts.append([])
        batch_texts[-1].append(text)
    return batch_texts


def split_sentences(text, min_len=10, max_len=50):
    def append_with_check(sub_sent):
        if len(result[-1]) < min_len:
            result[-1] += sub_sent
            result[-1] = result[-1][:max_len]
        else:
            result.append(sub_sent[:max_len])

    # 定义正则表达式，匹配句子结束符号
    pattern = r'(?<=[。！？；])'
    # 将文本按照句子结束符号进行分割
    sentences = re.split(pattern, text)
    # 对句子进行处理
    result = [""]
    temp_str = ""
    # print(sentences)
    for sentence in sentences:
        if len(sentence) < 1: continue
        # 如果句子长度大于50，则按照标点符号进行拆分
        if len(sentence) > max_len:
            sub_sentences = re.split(r'(?<=[，。！？；… ])', sentence)
            for sub_sentence in sub_sentences:
                if len(sub_sentence) < 1: continue
                append_with_check(sub_sentence)
        # 其他情况，直接添加到结果列表中
        else:
            append_with_check(sentence)
    return result

def split_chinese(rawstr):
    """
    以非中文字符作为切割，得到连续的中文子串
    :param rawstr: 原始字符串
    :return: 中文子串列表
    """
    if len(rawstr) < 5:
        return 0
    # 非中文字符范围
    no_chinese_range = u'[^\u4e00-\u9fa5]+'
    # 编译一个正则表达式对象并返回
    pattern = re.compile(no_chinese_range)
    # 以非中文字符为分隔符，将字符串分割为列表
    result = pattern.split(rawstr)
    # 去除列表中的空字符串
    result = [x for x in result if x != '']
    # 得到list中最长的元素的长度
    try:
        max_len = max([len(x) for x in result])
        return max_len
    except:
        # print("rawstr:{}\nresult:{}".format(rawstr, result))
        return 0


def convert_seconds_to_min_sec(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}分{remaining_seconds}秒" if minutes > 0 else f"{remaining_seconds}秒"


def convert_to_ranges(lst):
    if not lst: return []
    ranges = [[lst[0]]]
    for i in range(1, len(lst)):
        if lst[i] - lst[i-1] == 1:
            if len(ranges[-1]) == 1:
                ranges[-1].append(lst[i])
            else:
                ranges[-1][-1] = lst[i]
        else:
            ranges.append([lst[i]])
    return [f"{i[0]}-{i[1]}" if len(i) > 1 else f"{i[0]}" for i in ranges]


def find_common_str(query, doc):
    words = set()
    matches = CSequenceMatcher(None, query, doc).get_matching_blocks()
    for match in matches:
        if match.size > 0:
            words.add(doc[match.a:match.a + match.size])
    return list(words)

def get_dir_files(dir_path):
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    s = "冯滨（南充市中心医院心胸外科副主任医师、副教授），男，南充市中心医院心胸外科副主任医师、副教授。擅长动脉导管结扎术、慢性缩窄性心包炎包切除术、体外循环下房间隔缺损修补术、室间隔缺损修补术、部分心 内膜垫缺损修补术、双向格林手术、部分及完全肺静脉异位引流矫治术、二尖瓣置换术、主动脉瓣置换术、双瓣膜替换术、法乐四联症矫治术等心脏手术。擅长漏斗胸矫正 术、肺叶切除术、全肺切除术、支气管肺袖式切除术、食管癌切除食管胃吻合术、经食管裂孔食管内翻拔除术、纵隔肿瘤摘除术、重症肌无力胸腺切除、电视胸腔镜下肺大 疱切除术等普胸手术。熟悉冠状动脉搭桥术、复杂先心病矫治术等心血管手术。擅长动脉导管结扎术、慢性缩窄性心包炎包切除术、体外循环下房间隔缺损修补术、室间隔 缺损修补术、部分心内膜垫缺损修补术、双向格林手术、部分及完全肺静脉异位引流矫治术、二尖瓣置换术、主动脉瓣置换术、双瓣膜替换术、法乐四联症矫治术等心脏手 术。擅长漏斗胸矫正术、肺叶切除术、全肺切除术、支气管肺袖式切除术、食管癌切除食管胃吻合术、经食管裂孔食管内翻拔除术、纵隔肿瘤摘除术、重症肌无力胸腺切除 、电视胸腔镜下肺大疱切除术等普胸手术。熟悉冠状动脉搭桥术、复杂先心病矫治术等心血管手术。2007年度四川省科学技术进步三等奖。"
    # s = remove_all_punctuation(s)
    longest_dup = longest_dup_substring(s)
    new_s = s[::-1].replace(longest_dup[::-1], '', s.count(longest_dup)-1)[::-1]
    print(s)
    print(new_s)
