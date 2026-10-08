"""费用实验的完整证据契约：标准库压缩、哈希、冻结输入绑定与检查。

只检查发布包，不改写冻结回放/模型或历史源码身份；无需原始行情即可审阅。
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def file_sha256(path):
    """流式文件身份，避免为了哈希把完整回放读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def archive_file(source, target):
    """gzip保存原字节；固定mtime且不记录文件名，同一输入重复导出可核对。"""
    with Path(source).open('rb') as incoming, Path(target).open('xb') as outgoing:
        with gzip.GzipFile(filename='', mode='wb', fileobj=outgoing, mtime=0) as zipped:
            shutil.copyfileobj(incoming, zipped)
    return file_sha256(target)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_input_equivalence(proof, plan):
    """全量比较记录必须绑定本次实际输入，不能拿其他日期/年龄的检查顶替。"""
    require(proof['schema_version']==1 and proof['status']=='passed'
            and proof['verification_kind']=='full_original_sampler_and_prepared_dataset_equivalence',
            'Need a successful complete input equivalence report')
    require(proof['config_sha256']==plan['config_sha256']
            and proof['data_index_sha256']==plan['data_index_sha256']
            and proof['cached_input_manifest_sha256']==plan['prepared_input_manifest']['sha256'],
            'Input verification is not bound to this replay')
    ages = plan['config']['library']['age_candidates_ms']
    dates = sorted({d for c in plan['cases'] for s in c['source_coverage']
                    for d in s['required_utc_partitions']})
    require(proof['ages_ms']==ages and proof['source_dates_utc']==dates
            and [c['max_age_ms'] for c in proof['cases']]==ages, 'Full input verification omits ages or UTC partitions')
    for case, binding in zip(proof['cases'],plan['cases']):
        require(proof['calendar_sha256']==binding['calendar']['sha256'], 'Verification calendar differs')
        require([r['file_date_utc'] for r in case['snapshots']]==dates, 'Snapshot verification has missing/duplicate dates')
        for row in case['snapshots']+[case['dataset']]:
            require(row['all_rows_compared'] and row['schema_and_metadata_equal']
                    and row['values_and_order_equal'] and row['rows']>0 and row['columns'],
                    'Input comparison is incomplete or has mismatched values')
        require(all(r['partial_prefix'] is False and r['source_records']>0
                    and r['max_age_ms']==case['max_age_ms'] for r in case['snapshots']), 'Prefix/age mismatch in verification')
        dataset = case['dataset']
        require(dataset['cached_sha256']==binding['dataset']['parquet_sha256']
                and dataset['cached_quality_sha256']==binding['dataset']['quality_sha256']
                and dataset['quality_equal_except_input_paths_file_hashes_and_code'], 'Verified dataset differs from frozen replay')


def _relative_file(directory, name):
    """包内引用须相对且不越界，使证据复制到仓库后仍可核对。"""
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Artifact reference must stay inside its bundle')
    target = directory/path
    require(target.is_file(), f'Missing evidence artifact: {name}')
    return target


def check_friction_bundle(directory):
    """任意正式导出包的基本端到端契约；课程数值/历史回归另由项目检查核对。"""
    directory = Path(directory).resolve()
    summary = json.loads((directory/'summary.json').read_text())
    source = summary['source']
    raw = {}
    for field, sha_field, content_sha, key in (
        ('archive_path','archive_sha256',source['sha256'],'result'),
        ('library_archive_path','library_archive_sha256',summary['plan']['library_sha256'],'library')):
        archive = _relative_file(directory,source[field])
        require(file_sha256(archive)==source[sha_field], 'Compressed artifact checksum differs')
        content = gzip.decompress(archive.read_bytes())
        require(hashlib.sha256(content).hexdigest()==content_sha, 'Compressed evidence changed original bytes')
        raw[key] = json.loads(content)
    replay = raw['result']
    require(replay['result_kind']==summary['result_kind']=='course_friction_reward_increment'
            and summary['new_replay_performed'] is True, 'Need an actual incremental replay')
    require(summary['plan']==replay['plan'] and summary['plan_sha256']==replay['plan_sha256'], 'Export changed frozen identity')
    plan = summary['plan']
    actual = hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k!='plan_sha256'},
        sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',', ':')).encode()).hexdigest()
    require(actual==summary['plan_sha256']==plan['plan_sha256'],'Frozen plan fingerprint differs')
    require(len(summary['age_cases'])==len(replay['age_cases']), 'Summary omits age cases')
    for case, original in zip(summary['age_cases'],replay['age_cases']):
        require(case['max_age_ms']==original['max_age_ms']
                and case['reward_fit']==original['reward_fit']
                and case['execution_gate']==original['execution_gate'], 'Summary changes fit/gate/age')
        require(set(case['phases'])==set(original['phases']), 'Summary omits phases')
        for phase, data in case['phases'].items():
            previous = original['phases'][phase]
            require(data['statistics']==previous['statistics'] and len(data['daily'])==len(previous['daily']), 'Summary changes results')
            for day, before in zip(data['daily'],previous['daily']):
                require(day['session_id']==before['session_id'] and len(day['results'])==len(before['results']), 'Summary omits days/arms')
                require(all(all(before['results'][i][k]==v for k,v in row.items())
                            for i,row in enumerate(day['results'])), 'Summary values differ from original replay')
    for name, sha in summary['artifact_sha256'].items():
        require(file_sha256(_relative_file(directory,name))==sha, 'Published figure checksum differs')
    if 'input_equivalence_path' in source:
        path = _relative_file(directory,source['input_equivalence_path'])
        require(file_sha256(path)==source['input_equivalence_sha256'], 'Input verification checksum differs')
        check_input_equivalence(json.loads(path.read_text()),summary['plan'])
    elif (summary.get('publication',{}).get('format_version',0)>=2
          and plan['prepared_input_manifest'].get('preparation_cache')):
        require(False,'Cached input publication requires complete bound equivalence evidence')
    return summary
