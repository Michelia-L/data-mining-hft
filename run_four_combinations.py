"""冻结并运行收尾2/4的最小ME/OE四组合，保护原始OE结果和旧报告证据。"""
import argparse
import json
from pathlib import Path

from src.artifacts import pretty_json
from src.four_combinations import compact_evidence, freeze_four, render_four, run_four
from src.session_experiment import publish_bundle
from src.snapshot_dataset import sha256_file


def main(argv=None):
    """freeze绑定配置和原OE；run仅消费计划，另存完整结果和报告证据快照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze'); freeze.add_argument('--config', required=True)
    freeze.add_argument('--oe-result', required=True)
    run = commands.add_parser('run'); run.add_argument('--plan', required=True)
    run.add_argument('--detail', action='store_true'); run.add_argument('--checkpoint-dir')
    for command in (freeze, run):
        command.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    if args.command == 'freeze':
        result = freeze_four(args.config, args.oe_result)
        publish_bundle(result, '# 冻结四组合计划\n\n```json\n'+pretty_json(result)+'\n```\n',
                       args.output_dir, 'plan.json')
    else:
        result = run_four(args.plan, detail=args.detail, checkpoint_dir=args.checkpoint_dir)
        publish_bundle(result, render_four(result), args.output_dir, 'result.json')
        path = Path(args.output_dir)/'result.json'
        snapshot = compact_evidence(result, sha256_file(path))
        with (path.parent/'evidence.json').open('x', encoding='utf-8') as handle:
            json.dump(snapshot, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            handle.write('\n')
    print(f'已保存 {args.command} 产物：{args.output_dir}', flush=True)
    return result


if __name__ == '__main__':
    main()
