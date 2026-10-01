"""冻结报价年龄对照，解释采样缺格并审计长尺度标签；不运行收益选优。

论文物理第 3 页 §3.1 定义定时快照，第 4 页 Eq.(2)–(4) 需要多尺度
中间价差，最长超过一小时。原文没有公开报价年龄或缺失处理规则；这里比较
工程假设的影响，不能把更多有效标签或更旧的报价直接当作复现质量更高。
"""
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import tempfile

import pandas as pd

from src.artifacts import pretty_json
from src.config import BASE_DIR
from src.data_catalog import select_daily_source
from src.session_experiment import CODE_FILES, audit_horizons, fingerprint, read_day
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import DEFAULT_CALENDAR, SessionCalendar, prepare_snapshot_dataset, sha256_file
from src.timed_snapshots import write_timed_snapshots


REASONS = ('emitted', 'emitted_preopen_source', 'stale_completed_book', 'incomplete_event',
           'bad_book_flag', 'invalid_completed_book', 'unreliable_receive_state', 'no_completed_book')
AGE_BINS = ('le_500ms', '500_to_1000ms', '1000_to_2000ms', 'gt_2000ms', 'unknown')


def audit_code_hashes():
    """包含实际调用的数据准备/读取/序列化代码，冻结时记录源码内容。"""
    names = (*CODE_FILES, 'src/snapshot_audit.py', 'src/data_catalog.py', 'run_snapshot_audit.py')
    return {name: sha256_file(BASE_DIR / name) for name in names}


def result_fingerprint(result):
    """结果内容也绑定哈希，汇总时发现误改计数；同样不构成数字签名。"""
    return fingerprint({key: value for key, value in result.items() if key != 'result_sha256'})


def validate_config(config, calendar):
    """日期连续且所有候选预先声明；固定网格，不能运行中增加有利参数。"""
    keys = {'schema_version', 'purpose', 'session_dates', 'source_dates_utc', 'interval_ms',
            'max_age_candidates_ms', 'horizons_ms', 'include_degraded'}
    if set(config) != keys or config['schema_version'] != 1 or config['purpose'] != 'data_quality_diagnostic':
        raise ValueError('Unsupported diagnostic configuration')
    days = config['session_dates']
    if (not isinstance(days, list) or not days or any(d not in calendar.sessions for d in days)
            or days != sorted(set(days))
            or days != [d for d in calendar.sessions if days[0] <= d <= days[-1]]):
        raise ValueError('Need continuous ordered calendar sessions')
    interval = config['interval_ms']
    if type(interval) is not int or interval <= 0 or type(config['include_degraded']) is not bool:
        raise ValueError('Need positive grid milliseconds and explicit quality permission')
    for name in ('max_age_candidates_ms', 'horizons_ms'):
        values = config[name]
        if (not isinstance(values, list) or not values or values != sorted(set(values))
                or any(type(v) is not int or v <= 0 or v > 7 * 86400000 for v in values)):
            raise ValueError(f'Invalid ordered positive milliseconds: {name}')
    if any(h % interval for h in config['horizons_ms']):
        raise ValueError('Horizons must align to the fixed grid')
    if interval not in config['max_age_candidates_ms']:
        raise ValueError('Comparison must retain the one-grid-age baseline')
    needed = set()
    for day in days:
        session = calendar.sessions[day]
        needed.update(t.date().isoformat() for t in pd.date_range(
            session['open'].normalize(), session['close'].normalize(), freq='D'))
    if config['source_dates_utc'] != sorted(needed):
        raise ValueError('Source dates must exactly cover all adjacent UTC partitions')


def freeze_audit(config_path, index_path, calendar_path=DEFAULT_CALENDAR):
    """读取 footer 和文件哈希后冻结；此时不生成标签，不查看盈亏。"""
    config_path, index_path = Path(config_path).resolve(), Path(index_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    calendar = SessionCalendar(calendar_path)
    validate_config(config, calendar)
    sources = []
    for day in config['source_dates_utc']:
        source = select_daily_source(index_path, day, config['include_degraded'])
        sources.append(source | {'sha256': sha256_file(source['source_file'])})
    plan = dict(schema_version=1, plan_kind='frozen_snapshot_age_audit', config=config,
        config_source=dict(path=str(config_path), sha256=sha256_file(config_path)),
        index=dict(path=str(index_path), sha256=sha256_file(index_path)),
        calendar=dict(path=str(Path(calendar_path).resolve()), sha256=calendar.sha256),
        sources=sources, code_sha256=audit_code_hashes(),
        purpose='development_diagnostic_no_strategy_or_profit_selection')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def load_plan(path):
    """使用冻结配置副本；外部配置文件变化不能静默影响该实验。"""
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if (plan.get('plan_kind') != 'frozen_snapshot_age_audit' or plan.get('schema_version') != 1
            or plan.get('plan_sha256') != fingerprint(plan)):
        raise ValueError('Frozen audit plan integrity check failed')
    if audit_code_hashes() != plan['code_sha256']:
        raise ValueError('Source changed after freezing; use a new plan')
    calendar = SessionCalendar(plan['calendar']['path'])
    if calendar.sha256 != plan['calendar']['sha256']:
        raise ValueError('Calendar changed after freezing')
    validate_config(plan['config'], calendar)
    if sha256_file(plan['index']['path']) != plan['index']['sha256']:
        raise ValueError('Data index changed after freezing')
    return plan, calendar


def add_age_bins(counter, first, stop, step_ns, reference_ns):
    """对整数网格范围累计年龄分箱，不展开长空档；上界包含年龄等于阈值。"""
    if reference_ns is None:
        counter['unknown'] += stop - first
        return
    cursor = first
    for label, age in zip(AGE_BINS[:3], (500, 1000, 2000)):
        end = min(stop, max(cursor, (reference_ns + age * 1000000) // step_ns + 1))
        counter[label] += end - cursor
        cursor = end
    counter['gt_2000ms'] += stop - cursor


class GridAudit:
    """把采样器实际结算的网格与版本化交易时段相交，隔离休市和未知尾部。

    回调范围为 [start_ns,end_ns)，calendar 区间为 (open,close]。只解释流中
    已由下一条可信消息结算的边界；文件首尾未结算的计划网格记为 unknown，
    不能叫作数据丢失。异常原因描述采样状态，不证明数据供应商中断。
    """
    def __init__(self, calendar, days, interval_ms):
        self.step = interval_ms * 1000000
        self.intervals, self.ends, self.rows = [], [], {}
        for day in days:
            self.rows[day] = dict(expected_grid_points=0, reasons=Counter(),
                emitted_book_age_bins=Counter(), trusted_receive_silence_bins=Counter())
            for opening, closing in calendar.sessions[day]['intervals']:
                first, stop = opening.value // self.step + 1, closing.value // self.step + 1
                self.intervals.append((first, stop, day, opening.value))
                self.ends.append(stop)
                self.rows[day]['expected_grid_points'] += stop - first
        self.last_end = None

    def observe(self, start_ns, end_ns, reason, book_recv_ns, trusted_recv_ns):
        """只累计数量，绝不修改盘口、补格或把未来质量结果送给执行器。"""
        if (reason not in REASONS or start_ns % self.step or end_ns % self.step
                or start_ns >= end_ns or (self.last_end is not None and start_ns < self.last_end)):
            raise ValueError('Diagnostic grid ranges must be ordered, disjoint and aligned')
        self.last_end = end_ns
        first, stop = start_ns // self.step, end_ns // self.step
        index = bisect_right(self.ends, first)
        while index < len(self.intervals):
            opening, closing, day, opening_ns = self.intervals[index]
            if opening >= stop:
                break
            a, b = max(first, opening), min(stop, closing)
            if a < b:
                state = reason
                # 扩大年龄可能沿用休市前报价；已有 calendar.assign 会拒绝，
                # 诊断也单列这类生成行，不把它计入 session 有效覆盖。
                if state == 'emitted' and (book_recv_ns is None or book_recv_ns < opening_ns):
                    state = 'emitted_preopen_source'
                row = self.rows[day]
                row['reasons'][state] += b - a
                add_age_bins(row['trusted_receive_silence_bins'], a, b, self.step, trusted_recv_ns)
                if state == 'emitted':
                    add_age_bins(row['emitted_book_age_bins'], a, b, self.step, book_recv_ns)
            index += 1

    def result(self):
        """每个 session 的 emitted+各失效状态+未结算网格等于计划量。"""
        output = []
        for day, row in self.rows.items():
            resolved = sum(row['reasons'].values())
            unknown = row['expected_grid_points'] - resolved
            if unknown < 0:
                raise ValueError('Resolved grids exceed scheduled grids')
            output.append(dict(session_id=day, expected_grid_points=row['expected_grid_points'],
                resolved_grid_points=resolved, not_resolved_by_stream=unknown,
                grid_reasons={key: row['reasons'][key] for key in REASONS},
                emitted_book_age_bins={key: row['emitted_book_age_bins'][key] for key in AGE_BINS},
                trusted_receive_silence_bins={key: row['trusted_receive_silence_bins'][key] for key in AGE_BINS}))
        return output


def run_age_audit(plan_path, max_age_ms, output_dir):
    """完整读取相同原始文件并生成独立 prepared；一次只运行已冻结的一个候选。

    所有候选有相同日期、网格与标签规则；只改变完成盘口最大年龄。过期报价仍
    丢弃，未完成/损坏盘口和不可信接收时间始终拒绝，休市不可前填。默认基线
    不替换；保留每个候选产物，数据质量对照不做交易收益或自动选择默认值。
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite audit output')
    plan, calendar = load_plan(plan_path)
    config = plan['config']
    if type(max_age_ms) is not int or max_age_ms not in config['max_age_candidates_ms']:
        raise ValueError('Age candidate was not declared in the frozen plan')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.age-audit-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'output'
        staging.mkdir()
        snapshots = staging / 'snapshots'
        snapshots.mkdir()
        observer = GridAudit(calendar, config['session_dates'], config['interval_ms'])
        raw_stats, paths = [], []
        for source in plan['sources']:
            path = Path(source['source_file'])
            if path.stat().st_size != source['source_bytes'] or sha256_file(path) != source['sha256']:
                raise ValueError('Raw source changed after freezing')
            identity = (path.stat().st_size, path.stat().st_mtime_ns)
            destination = snapshots / (source['file_date_utc'] + '.parquet')
            stats = write_timed_snapshots(path, destination, interval_ms=config['interval_ms'],
                max_age_ms=max_age_ms, source_date=source['file_date_utc'], condition=source['condition'],
                grid_observer=observer.observe)
            if (path.stat().st_size, path.stat().st_mtime_ns) != identity:
                raise ValueError('Raw source changed during audit')
            paths.append(destination)
            raw_stats.append(dict(source_date_utc=source['file_date_utc'], **stats))
            print(f"完成 {max_age_ms}ms / {source['file_date_utc']}：{stats['snapshots']} 条快照", flush=True)
        quality = prepare_snapshot_dataset(paths, staging / 'prepared', calendar_path=calendar.path,
            horizons_ms=config['horizons_ms'], include_degraded=config['include_degraded'])
        reader = PreparedDatasetReader(staging / 'prepared', calendar=calendar,
                                       include_degraded=config['include_degraded'])
        grids, audits = observer.result(), []
        for row in grids:
            frame = read_day(reader, row['session_id'])
            if len(frame) != row['grid_reasons']['emitted']:
                raise ValueError('Sampling diagnostic disagrees with prepared session rows')
            audit = audit_horizons(frame, calendar, config['horizons_ms'])
            blocks = frame.groupby('continuous_block').ts_event.agg(['min', 'max'])
            audit['longest_continuous_ms'] = float(((blocks['max'] - blocks['min']).dt.total_seconds() * 1000).max())
            audits.append(audit)
        if audit_code_hashes() != plan['code_sha256']:
            raise ValueError('Source changed during audit')
        document = dict(schema_version=1, result_kind='snapshot_age_diagnostic',
            plan_sha256=plan['plan_sha256'], plan=plan, max_age_ms=max_age_ms,
            raw_statistics=raw_stats, session_grids=grids, horizon_audits=audits,
            prepared=dict(parquet_sha256=reader.provenance['parquet_sha256'],
                          quality_sha256=reader.provenance['quality_sha256']),
            environment=quality['environment'], default_age_changed=False,
            limits=['只比较数据质量；有效样本增多不证明报价仍代表真实可成交盘口。',
                    '超龄/接收静默是观测状态，不证明供应商丢包；未结算首尾另记。',
                    '不拟合 IRL、不训练模型、不计算收益、不选择默认年龄。'])
        # 质量来源路径使用最终位置；改 JSON 后重算质量哈希，避免报告指向已删除临时目录。
        quality_path = staging / 'prepared' / 'quality.json'
        saved = json.loads(quality_path.read_text(encoding='utf-8'))
        for source in saved['inputs']:
            source['path'] = str(output / 'snapshots' / Path(source['path']).name)
        quality_path.write_text(pretty_json(saved) + '\n', encoding='utf-8')
        document['prepared']['quality_sha256'] = sha256_file(quality_path)
        document['result_sha256'] = result_fingerprint(document)
        (staging / 'result.json').write_text(pretty_json(document) + '\n', encoding='utf-8')
        (staging / 'report.md').write_text(render_audit(document), encoding='utf-8')
        staging.rename(output)
    return document


def render_audit(result):
    """用同一 JSON 展示可用量与原因，报告没有策略收益或优胜候选。"""
    lines = ['# 报价年龄与长期奖励数据诊断', '', f"冻结计划：`{result['plan_sha256']}`；"
        f"最大年龄 {result['max_age_ms']}ms；默认年龄未改变。", '',
        '| session | 快照 | 特征有效 | 全尺度联合 | 最长连续 ms |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in result['horizon_audits']:
        lines.append(f"| {row['session_id']} | {row['observed_rows']} | {row['feature_valid_rows']} | "
                     f"{row['all_horizons_feature_valid_rows']} | {row['longest_continuous_ms']} |")
    lines += ['', '## 采样状态（只含已结算的交易网格）', '', '```json',
              pretty_json(result['session_grids']), '```', '',
              '可信接收静默超过 2 秒不是供应商中断的证明；首尾未结算网格单列。', '',
              '## 各尺度标签', '', '```json', pretty_json(result['horizon_audits']), '```', '']
    lines += ['- ' + limit for limit in result['limits']]
    return '\n'.join(lines) + '\n'


def compare_audits(plan_path, result_paths):
    """所有预声明候选必须齐备，不允许删去负对照或混用不同日期/代码。"""
    plan, _ = load_plan(plan_path)
    rows = [json.loads(Path(path).read_text(encoding='utf-8')) for path in result_paths]
    ages = [row.get('max_age_ms') for row in rows]
    if sorted(ages) != plan['config']['max_age_candidates_ms']:
        raise ValueError('Comparison must contain every frozen candidate exactly once')
    for row, path in zip(rows, result_paths):
        if (row.get('result_kind') != 'snapshot_age_diagnostic' or row.get('plan') != plan
                or row.get('plan_sha256') != plan['plan_sha256']
                or row.get('result_sha256') != result_fingerprint(row)):
            raise ValueError('Cannot mix diagnostic results from different plans')
        prepared = Path(path).parent / 'prepared'
        if any(sha256_file(prepared / name) != row['prepared'][key] for name, key in
               [('snapshots.parquet', 'parquet_sha256'), ('quality.json', 'quality_sha256')]):
            raise ValueError('Diagnostic prepared files changed')
    rows.sort(key=lambda r: r['max_age_ms'])
    result = dict(schema_version=1, result_kind='snapshot_age_comparison', plan=plan,
                  plan_sha256=plan['plan_sha256'], candidates=rows, selected_age_ms=None)
    lines = ['# 冻结的报价年龄对照', '', '未选择默认年龄；不计算策略收益。', '',
             '| 最大年龄 ms | session | 快照 | 特征有效 | 联合有效 | 最长连续 ms |',
             '| ---: | --- | ---: | ---: | ---: | ---: |']
    for candidate in rows:
        for day in candidate['horizon_audits']:
            lines.append(f"| {candidate['max_age_ms']} | {day['session_id']} | {day['observed_rows']} | "
                f"{day['feature_valid_rows']} | {day['all_horizons_feature_valid_rows']} | {day['longest_continuous_ms']} |")
    return result, '\n'.join(lines) + '\n'
