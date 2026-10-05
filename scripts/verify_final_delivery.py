"""只读核对最终交付的归档身份、收益表和引用，不重新训练或交易。

该入口仅依赖标准库，不需要行情或数值环境。它验证本项目已归档的三批
证据及同源图，而不是逐笔成交因果性、统计显著性或论文实盘收益。预期
文件哈希在 evidence_index.json 中人工审阅固定；哈希不是独立真实性证明。
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
AGES = [500, 1000, 2000]
GROUPS = ['equal', 'single_shortest', 'online_library']
# 这是已审阅交付的必检范围，独立于可编辑索引。若只遍历索引，删除条目
# 就会减少检查次数却继续报告成功；固定ID与路径，不以索引自身决定范围。
EXPERIMENT_PATHS = {
    'oe_development': 'results/final_report/development_snapshot.json',
    'four_combinations': 'results/final_report/four_combinations_snapshot.json',
    'frozen_ablation': 'results/final_report/frozen_ablation_snapshot.json',
}
FIGURE_PATHS = {
    'results/final_report/figures/daily_net_usd.png',
    'results/final_report/figures/cumulative_daily_net_usd.png',
}
REQUIRED_DELIVERY_DOCUMENTS = {
    'README.md', 'final_replication_report.md',
    'docs/final_delivery_plan_2026-10-04.md',
    'docs/final_delivery_validation_2026-10-05.md',
    'docs/paper_alignment.md', 'docs/frozen_ablation_validation_2026-10-05.md',
    'results/final_report/README.md',
}


def require(condition, message):
    """使用显式异常，使 python -O 也不会跳过交付检查。"""
    if not condition:
        raise ValueError(message)


def digest(raw):
    """文件字节SHA256；不先解码或重排JSON，否则不能发现字节改变。"""
    return hashlib.sha256(raw).hexdigest()


def fingerprint(document):
    """与研究入口相同的计划内容身份，仅排除最外层plan_sha256。"""
    content = {k: v for k, v in document.items() if k != 'plan_sha256'}
    return digest(json.dumps(content, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode())


def inside(root, relative):
    """索引与引用只能定位本仓库文件；不读取../逃逸或外部符号链接。"""
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f'路径超出交付目录：{relative}')
    return path


def money(value):
    """表内金额统一两位小数，阻断保持未观察而不改成现金0。"""
    return '未定义' if value is None else f'{value:.2f}'


def section(text, start, end=None):
    """使用固定中文章节定位本交付；不存在的章节必须失败。"""
    require(text.count(start) == 1, f'章节缺失或重复：{start}')
    part = text.split(start, 1)[1]
    if end:
        require(end in part, f'章节结束位置缺失：{end}')
        part = part.split(end, 1)[0]
    return part


def table_rows(text, expected):
    """检查预期行只出现一次，排版空格变化不视为研究数值改变。

    按单元格比较而不按排版空格比较。金额来自快照日结果重新加总，避免
    只比较两个已舍入的报告表格；差额先在美元原值相减，再展示两位小数。
    """
    rows = [tuple(cell.strip() for cell in line.strip().strip('|').split('|'))
            for line in text.splitlines() if line.strip().startswith('|')]
    for row in expected:
        require(rows.count(tuple(row)) == 1, f'表格数值缺失、重复或不符：{row}')


def strategy_rows(text, names, totals, blocked_label):
    """总表策略顺序必须完整，拒绝增加、删除、重复或改名的策略行。"""
    body = section(text, '| 策略 |').split('\n\n', 1)[0]
    names_in_table = [line.strip().strip('|').split('|')[0].strip()
                      for line in body.splitlines()[2:] if line.strip().startswith('|')]
    require(names_in_table == names, '主策略表条目或顺序不符')
    table_rows(body, [[name] + [blocked_label if case[name]['run_status'] == 'blocked'
               else money(case[name]['net_pnl_usd']) for case in totals] for name in names])


def audit_index(index):
    """先锁定必检清单，再读取证据，拒绝删除、重复或偷换检查项。

    三批实验与两图采用精确集合，ID和文件路径必须一一对应。文档允许新增，
    但原七份必要文档必须全部存在且无重复。索引是身份记录，不能同时充当
    “哪些文件可以不检查”的授权；主报告也不能被换成另一个可编辑文档。
    """
    require(index['report'] == 'final_replication_report.md', '主报告路径改变')
    experiments = index['experiments']
    require(Counter(item['id'] for item in experiments) == Counter(EXPERIMENT_PATHS.keys())
            and {item['id']: item['path'] for item in experiments} == EXPERIMENT_PATHS,
            '实验检查清单缺项、重复或ID/路径改变')
    require(Counter(item['path'] for item in index['figures']) == Counter(FIGURE_PATHS),
            '图像检查清单缺项、重复或路径改变')
    documents = index['delivery_documents']
    require(len(documents) == len(set(documents))
            and REQUIRED_DELIVERY_DOCUMENTS <= set(documents),
            '交付文档检查清单缺少必要文档或存在重复')


def audit_inherited_days(old, four):
    """逐日核对旧24项OE/静态参照，拒绝总额不变的跨日改写。

    按年龄、阶段、session和原策略顺序对齐，金额/成交数/奖励及完整动态审计
    都比较原JSON值，不先舍入或只取字段交集。两份紧凑快照有已知投影差异：
    原OE独有terminal_position、holding_ms_mean、pnl_aggregation_defined；
    四组合独有blocking_reasons、scored_history_windows。只排除这些已核实字段，
    共有字段缺失、增加未知字段或更改周版本/反馈计数均失败。

    日级校准身份从calibration_sha256显式对应oe_calibration_sha256；预测审计
    保持一致。此核对约束证据继承，不重新证明Eq.(5)或逐笔成交因果性。
    """
    names = old['strategies']
    require(four['plan']['strategies'][:len(names)] == names, '四组合原24项策略身份改变')
    old_only = {'terminal_position', 'holding_ms_mean', 'pnl_aggregation_defined'}
    four_only = {'blocking_reasons', 'scored_history_windows'}
    for prior, case in zip(old['age_cases'], four['age_cases']):
        require(prior['max_age_ms'] == case['max_age_ms'], '四组合继承报价年龄不符')
        require(prior['phases'].keys() == case['phases'].keys(), '四组合继承阶段改变')
        for phase, previous in prior['phases'].items():
            current = case['phases'][phase]
            require(previous['statistics'] == current['statistics'][:len(names)],
                    '四组合改写了原24项统计')
            require([d['session_id'] for d in previous['daily']]
                    == [d['session_id'] for d in current['daily']], '四组合继承阶段日期改变')
            for left, right in zip(previous['daily'], current['daily']):
                context = f'{prior["max_age_ms"]}ms/{phase}/{left["session_id"]}'
                require(left['prediction_audit'] == right['prediction_audit']
                        and left['calibration_sha256'] == right['oe_calibration_sha256'],
                        f'四组合原24项日级预测/校准审计改变：{context}')
                for before, after in zip(left['results'], right['results'][:len(names)]):
                    original = {k: v for k, v in before.items() if k not in old_only}
                    inherited = {k: v for k, v in after.items() if k not in four_only}
                    require(original == inherited,
                            f'四组合改写了原24项逐日结果：{context}/{before["strategy"]}')
                for name in names:
                    require(name in left['dynamic_audits'] and name in right['dynamic_audits']
                            and left['dynamic_audits'][name] == right['dynamic_audits'][name],
                            f'四组合改写了原24项逐日审计：{context}/{name}')


def audit_case(case, strategies, expected_days):
    """从七个独立日账户复算统计；不同阶段不重复计入同一评价。

    形状为7日×预声明策略数。旧证据分validation/test两段；冻结后段一段。
    不将None变零，不将未平仓日累入正式总净利；毛利−摩擦=净利逐日验证。
    返回每策略总金额、反馈计数以及执行/阻断计数，不重算奖励或交易。
    """
    phases = list(case['phases'].values()) if 'phases' in case else [case]
    days = [day for phase in phases for day in phase['daily']]
    require([day['session_id'] for day in days] == expected_days, '评价日期或顺序不符')
    counts = Counter()
    totals = {}
    for day in days:
        require([row['strategy'] for row in day['results']] == strategies, '逐日策略缺项或次序改变')
        for row in day['results']:
            status = row.get('run_status', 'executed')
            require(status in ('executed', 'blocked'), '出现未披露的执行状态')
            counts[status] += 1
            if status == 'blocked':
                require(all(row[k] is None for k in ('gross_pnl_usd', 'friction_usd', 'net_pnl_usd')),
                        '阻断被写成金额')
            else:
                require(row['terminal_position_liquidated'], '归档存在未平仓日，须重新审阅累计口径')
                require(math.isclose(row['gross_pnl_usd'] - row['friction_usd'],
                                     row['net_pnl_usd'], abs_tol=1e-8), '日账本金额不守恒')
                if row['strategy'] == 'cash':
                    require(row['net_pnl_usd'] == row['total_fills'] == 0, '现金参照不是无交易0')
    for name in strategies:
        rows = [next(x for x in day['results'] if x['strategy'] == name) for day in days]
        summaries = [next(x for x in phase['statistics'] if x['strategy'] == name) for phase in phases]
        blocked = all(x.get('run_status') == 'blocked' for x in rows)
        require(blocked or all(x.get('run_status', 'executed') == 'executed' for x in rows),
                '混合阻断状态不能直接用本交付七日总表')
        total = dict(run_status='blocked' if blocked else 'executed')
        for key in ('gross_pnl_usd', 'friction_usd', 'net_pnl_usd'):
            value = None if blocked else sum(row[key] for row in rows)
            recorded = None if blocked else sum(x[key] for x in summaries)
            require(value == recorded, f'逐日与汇总不一致：{name}/{key}')
            if blocked:
                require(all(x[key] is None for x in summaries), '阻断汇总不是None')
            total[key] = value
        for key in ('mature_periods', 'scored_windows'):
            value = None if blocked else sum(day['dynamic_audits'].get(name, {}).get(key, 0) for day in days)
            if not blocked:
                require(value == sum(x.get(key, 0) for x in summaries), f'反馈汇总不一致：{name}/{key}')
            total[key] = value
        totals[name] = total
    return totals, counts


def verify(root, source_commits=False):
    """固定三份证据的交付核对，不扩展成实验框架；所有操作均为只读。"""
    root = Path(root).resolve()
    index = json.loads(inside(root, 'results/final_report/evidence_index.json').read_text())
    require(index['schema_version'] == 1 and index['delivery_status'] == 'partial_replication_final',
            '需要本项目最终交付索引')
    audit_index(index)
    report = inside(root, index['report']).read_text()
    artifacts = {}
    audited = {}
    for item in index['experiments']:
        raw = inside(root, item['path']).read_bytes()
        require(digest(raw) == item['sha256'], f'快照字节哈希不符：{item["path"]}')
        evidence = json.loads(raw)
        for key in ('evidence_kind', 'source_result_sha256', 'plan_sha256'):
            require(evidence[key] == item[key], f'索引与快照身份不符：{item["id"]}/{key}')
        if 'plan' in evidence:
            require(fingerprint(evidence['plan']) == evidence['plan_sha256'], '冻结计划内容身份不符')
        strategies = evidence.get('strategies', evidence.get('plan', {}).get('strategies'))
        require(len(strategies) == item['strategy_count'], '策略总数不符')
        require([c['max_age_ms'] for c in evidence['age_cases']] == AGES, '报价年龄缺项或选优')
        count = Counter()
        totals = []
        for case in evidence['age_cases']:
            values, observed = audit_case(case, strategies, item['evaluation_sessions'])
            totals.append(values)
            count.update(observed)
        require(dict(count) == item['strategy_day_statuses'], '策略日执行/阻断总数不符')
        artifacts[item['id']] = evidence
        audited[item['id']] = totals
        if source_commits:
            # 可选：完整克隆保留历史数值提交时检查Git对象，不依赖当前文档提交。
            # shallow clone可能没有这些对象，明确失败，不能伪称完成源码核对。
            codes = evidence.get('source_code_sha256', evidence.get('plan', {}).get('code_sha256'))
            for path, expected in codes.items():
                original = subprocess.run(['git', 'show', f'{item["source_commit"]}:{path}'],
                    cwd=root, capture_output=True, check=False)
                require(original.returncode == 0 and digest(original.stdout) == expected,
                        f'历史源码对象缺失或身份不符：{item["source_commit"]}/{path}')
        print(f'{item["id"]}：{len(strategies)}策略，{sum(count.values())}策略日，'
              f'{count["executed"]}执行/{count["blocked"]}阻断，金额与反馈一致')

    old = artifacts['oe_development']
    four = artifacts['four_combinations']
    frozen = artifacts['frozen_ablation']
    require(frozen['plan']['prior_result']['sha256'] == four['source_result_sha256']
            and frozen['plan']['prior_plan_sha256'] == four['plan_sha256'], '后段没有继承旧结果身份')
    audit_inherited_days(old, four)
    for prior, case in zip(four['age_cases'], frozen['age_cases']):
        for kind in ('ME', 'OE'):
            groups = prior['me_calibration']['scopes'] if kind == 'ME' else prior['oe_calibration']
            for scope in ('all_candidates', 'online_library'):
                group = groups[scope]
                inherited = case['rewards'][kind][scope]
                require(inherited['weights'] == group['active_reward_weights']
                        and inherited['calibration_scope_sha256'] == fingerprint(dict(group=group)),
                        '后段校准权重或门控身份改变')
    for path, expected in frozen['plan']['code_sha256'].items():
        require(digest(inside(root, path).read_bytes()) == expected, f'当前数值源码改变：{path}')

    old_table = section(report, '## 附表：')
    old_names = four['plan']['strategies']
    strategy_rows(old_table, old_names, audited['four_combinations'], '阻断（未观察）')
    recent = section(report, '### 5.5 ', '## 6.')
    new_names = frozen['plan']['strategies']
    strategy_rows(recent, new_names, audited['frozen_ablation'], '阻断')
    for age, totals in zip(AGES, audited['frozen_ablation']):
        for kind in ('ME', 'OE'):
            a, b, learned = [totals[f'{kind}-{g}-UCB'] for g in GROUPS]
            an, bn, ln = [x['net_pnl_usd'] for x in (a, b, learned)]
            table_rows(recent, [[str(age), kind] + list(map(money,
                [an, bn, ln, bn-an, None if ln is None else ln-an]))])
            if ln is not None:
                table_rows(recent, [[str(age), kind] + [money(learned[k]-a[k])
                             for k in ('gross_pnl_usd', 'friction_usd', 'net_pnl_usd')]])
            feedback = []
            for selector, key in (('UCB', 'mature_periods'), ('ARS-matured', 'scored_windows')):
                values = [totals[f'{kind}-{g}-{selector}'] for g in GROUPS]
                feedback.append('/'.join('阻断' if v['run_status'] == 'blocked' else str(v[key]) for v in values))
            table_rows(recent, [[str(age), kind] + feedback])
    for item in index['figures']:
        raw = inside(root, item['path']).read_bytes()
        require(digest(raw) == item['sha256'] and raw.startswith(b'\x89PNG\r\n\x1a\n'),
                f'图像身份不符：{item["path"]}')
        require(item['source_result_sha256'] == frozen['source_result_sha256'], '图像源结果身份不符')
    for name in index['delivery_documents']:
        document = inside(root, name)
        plain = re.sub(r'```.*?```', '', document.read_text(), flags=re.S)
        for target in re.findall(r'\]\(([^\s)]+)\)', plain):
            if '://' in target or target.startswith('#'):
                continue
            relative = document.parent.relative_to(root) / target.split('#')[0]
            require(inside(root, relative).is_file(), f'本地引用不存在：{name} → {target}')
    print('三批身份/继承关系、当前36项数值文件、两张图、34×3及27×3收益、'
          '赋权/成本差与反馈表、本地引用核对通过。')
    if source_commits:
        print('三批历史数值提交的Git对象逐文件核对通过。')
    print('该核对不替代原行情和完整日账本审计，不证明盈利或完整论文复现。')


def main():
    """默认定位脚本所在仓库；--root供另存交付副本核对，失败退出码为1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-commits', action='store_true', help='额外核对三批历史Git数值对象')
    args = parser.parse_args()
    try:
        verify(args.root, args.source_commits)
    except (ValueError, KeyError, OSError, StopIteration) as error:
        parser.exit(1, f'交付核对失败：{error}\n')


if __name__ == '__main__':
    main()
