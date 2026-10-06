"""从本地 Databento 索引选取一个 UTC 日期的 ESZ5 行情文件。

供唯一入口按协议所需日期逐文件准备，保留质量与来源信息。
UTC分区选择与后续session隔离分别处理，不用文件日期代替交易日。
文件日期是下载分区的 UTC 日期，不等于交易所 session 或全部 ts_event 的日期。
"""
from datetime import date
import json
from pathlib import Path

import pyarrow.parquet as pq

from src.config import BASE_DIR


def select_daily_source(index_path, data_date, include_degraded=False, data_root=BASE_DIR):
    """按明确日期选择文件，并在训练之前检查索引和 Parquet 是否一致。

    index_path 是索引 JSON；其中 parquet 相对路径以项目根目录为基准，与已有
    data/ESZ5/index.json 一致。data_root 允许测试或迁移数据时指定另一根目录。
    返回值均可写入 JSON，包含解析后的绝对路径；文件哈希由实验入口统一计算。

    本阶段只支持 GLBX.MDP3 / mbp-10 / ESZ5，以免其他品种误用 ES 的乘数、
    tick_size 和成本配置。degraded 必须显式允许；未知质量状态一律报错，
    防止索引拼写错误或新增状态被悄悄当作正常数据。
    """
    # ISO 标准解析也接受某些简写，因此额外比较格式，统一要求 YYYY-MM-DD。
    if date.fromisoformat(data_date).isoformat() != data_date:
        raise ValueError('data_date must use YYYY-MM-DD')
    index_path = Path(index_path).resolve()
    catalog = json.loads(index_path.read_text(encoding='utf-8'))
    expected = dict(dataset='GLBX.MDP3', schema='mbp-10', symbol='ESZ5', stype_in='raw_symbol')
    if any(catalog.get(key) != value for key, value in expected.items()):
        raise ValueError('Daily input currently requires GLBX.MDP3 / mbp-10 / ESZ5 / raw_symbol')
    matches = [entry for entry in catalog['files'] if entry['date'] == data_date]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one indexed file for {data_date}; found {len(matches)}')
    entry = matches[0]
    condition = entry['condition']
    if condition not in ('available', 'degraded'):
        raise ValueError(f'Unsupported data condition: {condition}')
    if condition == 'degraded' and not include_degraded:
        raise ValueError(f'{data_date} is degraded; use --include-degraded to include it explicitly')
    path = Path(entry['parquet'])
    if not path.is_absolute():
        path = Path(data_root) / path
    path = path.resolve(strict=True)
    if path.suffix != '.parquet':
        raise ValueError('Daily input must be a Parquet file')
    # 只读取 footer 即可核对行数；不为挑选一天的数据解码整个三个月数据集。
    rows = pq.ParquetFile(path).metadata.num_rows
    if type(entry['rows']) is not int or entry['rows'] <= 0 or rows != entry['rows']:
        raise ValueError(f'Parquet row count disagrees with index for {data_date}')
    size = path.stat().st_size
    if 'parquet_bytes' in entry and size != entry['parquet_bytes']:
        raise ValueError(f'Parquet file size disagrees with index for {data_date}')
    return dict(index_file=str(index_path), source_file=str(path), file_date_utc=data_date,
                condition=condition, source_rows=rows, source_bytes=size, **expected)
