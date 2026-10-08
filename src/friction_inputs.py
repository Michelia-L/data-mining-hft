"""全量核对缓存与原 prepare 的采样/特征函数；不改变奖励或交易策略。

按 UTC 文件并行独立原采样，每次只读一日；随后逐批比较全部列、顺序、
来源元数据和派生特征/标签。Parquet 行组不同不等于行情值不同。
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform


def compare_parquet(left, right):
    """保持行序逐批精确比较，跨不同 Parquet 行组边界，不抽样、不容差放宽。"""
    import pyarrow.parquet as pq
    from src.snapshot_dataset import sha256_file
    a, b = pq.ParquetFile(left), pq.ParquetFile(right)
    if not a.schema_arrow.equals(b.schema_arrow, check_metadata=True):
        raise ValueError(f'Parquet schema/source metadata differ: {left}')
    if a.metadata.num_rows != b.metadata.num_rows:
        raise ValueError(f'Parquet row counts differ: {left}')
    ia, ib = a.iter_batches(batch_size=65536), b.iter_batches(batch_size=65536)
    ca = cb = None
    count = 0
    while True:
        if ca is None or ca.num_rows == 0:
            ca = next(ia, None)
        if cb is None or cb.num_rows == 0:
            cb = next(ib, None)
        if ca is None or cb is None:
            if ca is not None or cb is not None or count != a.metadata.num_rows:
                raise ValueError('Parquet stream lengths differ')
            break
        n = min(ca.num_rows, cb.num_rows)
        if not ca.slice(0, n).equals(cb.slice(0, n), check_metadata=True):
            raise ValueError(f'Parquet values/order differ near row {count}: {left}')
        count += n
        ca, cb = ca.slice(n), cb.slice(n)
    return dict(cached_sha256=sha256_file(left), reference_sha256=sha256_file(right),
                rows=count, columns=a.schema_arrow.names, all_rows_compared=True,
                schema_and_metadata_equal=True, values_and_order_equal=True)


def _reference_snapshot(task):
    """直接使用原采样器读取全部原始消息，再与实际缓存比较；不复用宽年龄输出。"""
    from src.timed_snapshots import write_timed_snapshots
    source, cached, target, age, interval, expected_hash = task
    stats = write_timed_snapshots(source['source_file'], target, interval_ms=interval,
        max_age_ms=age, source_date=source['file_date_utc'], condition=source['condition'])
    if stats['source_records'] != source['source_rows']:
        raise ValueError('Reference sampler did not consume the complete source partition')
    comparison = compare_parquet(cached, target)
    if comparison['cached_sha256'] != expected_hash:
        raise ValueError('Cached snapshot no longer matches its frozen quality manifest')
    return dict(file_date_utc=source['file_date_utc'], source_records=stats['source_records'],
                max_age_ms=age, partial_prefix=False, **comparison)


def verify_friction_inputs(config_path, prepared_directory, output_directory, workers=4):
    """在新目录独立全量原采样并生成三个 prepared，再发布全量等价性证据。

    并行只加速相互独立的 UTC 文件；使用与公开 prepare 完全相同的采样和
    session 特征/标签函数。错误保留诊断文件，不发布 passed 报告。
    """
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    from src.config import BASE_DIR
    from src.course_experiment import load_config
    from src.data_catalog import select_daily_source
    from src.snapshot_dataset import prepare_snapshot_dataset, sha256_file
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('workers must be an integer from 1 to 8')
    config, calendar = load_config(config_path)
    cached, output = Path(prepared_directory).resolve(), Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite reference inputs: {output}')
    manifest = cached/'inputs.json'
    binding = json.loads(manifest.read_text())
    ages = config['library']['age_candidates_ms']
    days = [d for phase in ('train', 'calibration', 'validation', 'test')
            for d in config['protocol']['sessions'][phase]]
    dates = sorted({t.date().isoformat() for day in days for t in pd.date_range(
        calendar.sessions[day]['open'].normalize(), calendar.sessions[day]['close'].normalize(), freq='D')})
    index_path = BASE_DIR/config['data_index']
    config_sha, index_sha = sha256_file(config_path), sha256_file(index_path)
    manifest_sha = sha256_file(manifest)
    if (binding['config_sha256'] != config_sha or binding['data_index_sha256'] != index_sha
            or binding['source_dates_utc'] != dates or len(binding['datasets']) != len(ages)):
        raise ValueError('Cached input binding differs from the declared complete protocol')
    sources = [select_daily_source(index_path, d, config['protocol']['include_degraded']) for d in dates]
    code_files = ('src/timed_snapshots.py', 'src/snapshot_dataset.py', 'src/data_catalog.py',
                  'src/course_experiment.py', 'src/friction_inputs.py', 'run_project.py')
    code = {f:sha256_file(BASE_DIR/f) for f in code_files}
    qualities = {}
    for dataset in binding['datasets']:
        path = Path(dataset)
        quality = json.loads((path/'quality.json').read_text())
        age = quality['inputs'][0]['max_age_ms']
        if age in qualities or age not in ages or quality['partial_input']:
            raise ValueError('Cached input has duplicate ages or is only a prefix')
        qualities[age] = (path, quality, sha256_file(path/'quality.json'))
    output.mkdir(parents=True)
    tasks = []
    for age in ages:
        _, quality, _ = qualities[age]
        by_date = {i['metadata']['source_date_utc']:i for i in quality['inputs']}
        if len(by_date) != len(quality['inputs']) or sorted(by_date) != dates:
            raise ValueError('Cached UTC source partitions are incomplete or duplicated')
        for source in sources:
            d = source['file_date_utc']; item = by_date[d]
            target = output/'snapshots'/f'age-{age}'/f'{d}.parquet'
            target.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((source, item['path'], str(target), age, config['interval_ms'], item['sha256']))
    snapshots = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_reference_snapshot, t) for t in tasks]
        for future in as_completed(futures):
            row = future.result(); snapshots.append(row)
            print(f"全量原采样及字段一致：{row['file_date_utc']} / {row['max_age_ms']}ms"
                  f"（{len(snapshots)}/{len(tasks)}）", flush=True)
    cases, directories = [], []
    for age in ages:
        path, quality, quality_sha = qualities[age]
        target = output/f'age-{age}'
        reference = prepare_snapshot_dataset(
            [output/'snapshots'/f'age-{age}'/f'{d}.parquet' for d in dates], target,
            calendar_path=calendar.path, horizons_ms=config['protocol']['reward_horizons_ms'],
            allow_partial=config['protocol']['allow_partial'], include_degraded=config['protocol']['include_degraded'])
        compared = compare_parquet(path/'snapshots.parquet', target/'snapshots.parquet')
        # 路径/文件字节哈希和运行源码身份分别记录；实际来源元数据已逐文件比较。
        excluded = {'inputs', 'code_sha256'}
        if ({k:v for k,v in quality.items() if k not in excluded}
                != {k:v for k,v in reference.items() if k not in excluded}):
            raise ValueError('Derived quality/labels/feature coverage differ')
        if sha256_file(path/'quality.json') != quality_sha:
            raise ValueError('Cached quality changed during full comparison')
        directories.append(str(target))
        cases.append(dict(max_age_ms=age, snapshots=sorted(
            [r for r in snapshots if r['max_age_ms']==age], key=lambda r:r['file_date_utc']),
            dataset=dict(**compared, cached_quality_sha256=quality_sha,
                         reference_quality_sha256=sha256_file(target/'quality.json'),
                         quality_equal_except_input_paths_file_hashes_and_code=True)))
        print(f'全部派生特征/标签一致：{age}ms / {compared["rows"]}行', flush=True)
    if (sha256_file(config_path)!=config_sha or sha256_file(index_path)!=index_sha
            or sha256_file(manifest)!=manifest_sha or code!={f:sha256_file(BASE_DIR/f) for f in code_files}):
        raise ValueError('Reference implementation or input binding changed during verification')
    reference_binding = dict(config_sha256=config_sha, datasets=directories,
                             source_dates_utc=dates, data_index_sha256=index_sha)
    (output/'inputs.json').write_text(json.dumps(reference_binding,ensure_ascii=False,indent=2)+'\n')
    proof = dict(schema_version=1, verification_kind='full_original_sampler_and_prepared_dataset_equivalence',
        status='passed', config_sha256=config_sha, data_index_sha256=index_sha,
        calendar_sha256=calendar.sha256, source_dates_utc=dates, ages_ms=ages,
        cached_input_manifest_sha256=manifest_sha,
        reference_input_manifest_sha256=sha256_file(output/'inputs.json'),
        reference_code_sha256=code, workers=workers, cases=cases,
        reference_method='write_timed_snapshots_then_prepare_snapshot_dataset_same_as_public_prepare',
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyarrow=pa.__version__))
    (output/'input_equivalence.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    return proof
