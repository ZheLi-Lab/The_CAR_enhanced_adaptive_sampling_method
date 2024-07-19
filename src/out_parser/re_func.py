import os
import re

def extract_folder_names(path):
    # 使用 os.path.normpath 规范化路径
    normalized_path = os.path.normpath(path)
    # print(normalized_path)

    # 使用 os.path.split 获取每个层级的文件夹名字
    folder_names = []
    while True:
        head, tail = os.path.split(normalized_path)
        if tail:
            folder_names.insert(0, tail)  # 在列表的开头插入
        else:
            if head:
                folder_names.insert(0, head)
            break
        normalized_path = head

    return folder_names

def extract_numbers_from_path(path):
    # print(path)
    # 使用正则表达式匹配所有数字或浮点数
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', path)
    
    # 将匹配到的字符串转换为浮点数并存储在列表中
    numbers = [float(match) for match in matches]
    if len(set(numbers)) > 1:
        raise ValueError("Inconsistent numbers found in paths.")
    return numbers[0]

def remove_g_and_get_numbers(input_string):
    # 使用正则表达式匹配 '_g' 后面的数字
    matches = re.findall(r'_g(\d+)', input_string)

    # 删除 '_g' 及其后面的数字
    result = re.sub(r'_g\d+', '', input_string)

    return result, matches