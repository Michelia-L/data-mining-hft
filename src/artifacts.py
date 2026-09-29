"""实验结果的 JSON 序列化工具。

字典与嵌套对象保留缩进，纯数字或字符串数组压成一行，避免数百个曲线点
占满 Git 差异页面。只改变文件排版，不降低数值精度，也不删减实验字段。"""
import json


def pretty_json(value, level=0):
    """递归生成可读且符合标准的 JSON 文本。

    参数 value 是待保存的字典、列表或基础值；level 是递归缩进层数。
    基础数组直接交给 json.dumps，含对象的数组逐项递归，空容器也由标准库处理。
    ensure_ascii=False 保留中文；allow_nan=False 遇到 NaN/Infinity 就报错，
    避免浏览器无法解析结果，或把计算异常伪装成有效实验输出。"""
    indent = '  ' * level
    child = '  ' * (level + 1)
    if isinstance(value, dict) and value:
        return '{\n' + ',\n'.join(child + json.dumps(k, ensure_ascii=False) + ': ' + pretty_json(v, level + 1)
                                  for k, v in value.items()) + '\n' + indent + '}'
    if isinstance(value, list) and any(isinstance(x, (dict, list)) for x in value):
        return '[\n' + ',\n'.join(child + pretty_json(v, level + 1) for v in value) + '\n' + indent + ']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
