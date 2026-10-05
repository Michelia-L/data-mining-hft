"""交付校验器的标准库回归测试：检查清单与逐日继承均不能被删改绕过。

使用已归档证据的临时副本，不训练、不回放、不修改正式结果。负例主动
更新被改写快照在临时索引中的 SHA，以验证语义检查，而非仅让字节哈希
提前拒绝。尤其保持阶段金额汇总不变，检验旧24项逐日继承的独立约束。
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/verify_final_delivery.py'
INDEX = Path('results/final_report/evidence_index.json')
FOUR = Path('results/final_report/four_combinations_snapshot.json')


class FinalDeliveryTests(unittest.TestCase):
    """从真实归档构造可核对交付；每个案例只修改自己的临时副本。"""

    def setUp(self):
        """只复制小型交付文件及数值源码，不复制行情、Git对象或外部账本。"""
        directory = tempfile.TemporaryDirectory(prefix='fmato-delivery-test-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        files = {ROOT / '1679894.pdf'}
        for pattern in ('*.md', '*.py', '*.txt', '*.html', 'docs/*.md', 'results/*.md', 'results/*.json',
                        'src/*.py', 'tests/*.py', 'scripts/*.py', 'config/*.json',
                        '.github/*.md', '.github/workflows/*.yml'):
            files.update(ROOT.glob(pattern))
        files.update(p for p in (ROOT / 'results/final_report').rglob('*') if p.is_file())
        for source in files:
            target = self.root / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        self.index = json.loads((self.root / INDEX).read_text())
        self.four = json.loads((self.root / FOUR).read_text())

    def run_verifier(self):
        """与CI同样禁用site-packages；CLI失败应返回1并给出可定位原因。"""
        return subprocess.run([sys.executable, '-S', str(SCRIPT), '--root', str(self.root)],
                              capture_output=True, text=True, check=False)

    def write_index(self, index):
        """索引本身没有外部字节哈希，删除检查项的负例直接修改该文件。"""
        (self.root / INDEX).write_text(json.dumps(index, ensure_ascii=False))

    def write_four(self, evidence):
        """只刷新临时四组合快照的文件SHA；原结果、计划和金额汇总均不动。"""
        raw = json.dumps(evidence, ensure_ascii=False).encode()
        (self.root / FOUR).write_bytes(raw)
        index = deepcopy(self.index)
        item = next(x for x in index['experiments'] if x['id'] == 'four_combinations')
        item['sha256'] = hashlib.sha256(raw).hexdigest()
        self.write_index(index)

    def assert_rejected(self, reason):
        """必须命中指定语义检查，避免负例被无关缺文件或旧SHA误伤而假通过。"""
        result = self.run_verifier()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(reason, result.stderr)
        self.assertNotIn('本地引用核对通过', result.stdout)

    def test_standard_library_cli_preserves_original_bundle(self):
        """完整归档无需数值依赖即可通过，文件集合和全部内容保持只读。"""
        def contents():
            return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.root.rglob('*') if p.is_file()}
        before = contents()
        result = self.run_verifier()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('本地引用核对通过', result.stdout)
        self.assertEqual(before, contents())

    def test_removing_any_required_manifest_entry_is_rejected(self):
        """逐项删除三批实验、两张图或任一必要文档，不能缩减检查范围后报成功。"""
        reasons = {'experiments': '实验检查清单', 'figures': '图像检查清单',
                   'delivery_documents': '交付文档检查清单'}
        for collection, reason in reasons.items():
            for position in range(len(self.index[collection])):
                with self.subTest(collection=collection, position=position):
                    index = deepcopy(self.index)
                    del index[collection][position]
                    self.write_index(index)
                    self.assert_rejected(reason)
            with self.subTest(collection=collection, empty=True):
                index = deepcopy(self.index)
                index[collection] = []
                self.write_index(index)
                self.assert_rejected(reason)

    def test_duplicate_or_substituted_manifest_entries_are_rejected(self):
        """条目数量相同也不能用重复、未知ID或换路径替代必须检查的文件。"""
        changes = [('experiments', '实验检查清单'), ('figures', '图像检查清单'),
                   ('delivery_documents', '交付文档检查清单')]
        for collection, reason in changes:
            with self.subTest(collection=collection, duplicate=True):
                index = deepcopy(self.index)
                index[collection][1] = deepcopy(index[collection][0])
                self.write_index(index)
                self.assert_rejected(reason)
        for field, replacement in (('id', 'unknown_experiment'),
                                   ('path', 'results/final_report/four_combinations_snapshot.json')):
            with self.subTest(field=field):
                index = deepcopy(self.index)
                index['experiments'][0][field] = replacement
                self.write_index(index)
                self.assert_rejected('实验检查清单')

    def test_balanced_daily_pnl_changes_cannot_preserve_inheritance(self):
        """两日毛利/净利各加减1美元，账本恒等式与阶段汇总不变仍必须拒绝。"""
        evidence = deepcopy(self.four)
        days = evidence['age_cases'][0]['phases']['validation']['daily']
        for day, delta in zip(days[:2], (1., -1.)):
            for field in ('gross_pnl_usd', 'net_pnl_usd'):
                day['results'][0][field] += delta
        self.write_four(evidence)
        self.assert_rejected('原24项逐日结果')

    def test_balanced_daily_fill_changes_cannot_preserve_inheritance(self):
        """成交数在两日之间迁移，总成交数不变也不能冒称旧24项原样继承。"""
        evidence = deepcopy(self.four)
        days = evidence['age_cases'][0]['phases']['validation']['daily']
        days[0]['results'][0]['total_fills'] += 1
        days[1]['results'][0]['total_fills'] -= 1
        self.write_four(evidence)
        self.assert_rejected('原24项逐日结果')

    def test_daily_audit_changes_are_rejected(self):
        """金额不变时，选择次数或所用周版本仍必须与原日审计一致。"""
        for field in ('selections', 'selected_version_ids'):
            with self.subTest(field=field):
                evidence = deepcopy(self.four)
                days = evidence['age_cases'][0]['phases']['validation']['daily']
                audit = days[0]['dynamic_audits']['Period-OE-UCB']
                if field == 'selections':
                    audit[field] += 1
                    days[1]['dynamic_audits']['Period-OE-UCB'][field] -= 1
                else:
                    audit[field] = ['changed_model_version']
                self.write_four(evidence)
                self.assert_rejected('原24项逐日审计')

    def test_missing_shared_result_fields_or_audits_are_rejected(self):
        """共有奖励字段及单策略审计被删除时，不能通过取字段交集跳过继承检查。"""
        for field in ('reward_weights', 'matured_order_count', 'period_status_counts', 'dynamic_audits'):
            with self.subTest(field=field):
                evidence = deepcopy(self.four)
                day = evidence['age_cases'][0]['phases']['validation']['daily'][0]
                if field == 'dynamic_audits':
                    del day[field]['Period-OE-UCB']['selected_version_ids']
                    reason = '原24项逐日审计'
                else:
                    del day['results'][0][field]
                    reason = '原24项逐日结果'
                self.write_four(evidence)
                self.assert_rejected(reason)


if __name__ == '__main__':
    unittest.main()
