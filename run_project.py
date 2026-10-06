"""课程项目唯一入口：默认查看既有结果；重跑时显式指定新输出。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    """结果查看/核对只依赖标准库；数值与绘图依赖在对应命令中加载。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('show', help='查看七日课程结果，不读取行情或重新训练')
    check = sub.add_parser('check', help='核对结果、文档与证据身份')
    check.add_argument('--core', action='store_true', help='另做小合成时序/账本检查')
    export = sub.add_parser('export', help='从冻结快照导出摘要与两张结果图')
    export.add_argument('--output-dir', required=True, help='不存在的新输出目录')
    for command in ('prepare', 'run'):
        p = sub.add_parser(command, help='准备协议所需新数据' if command=='prepare' else '重跑核心OE-UCB；另存新结果')
        p.add_argument('--config', default=str(ROOT / 'config/course_experiment.json'))
        p.add_argument('--output-dir', required=True, help='不存在的新输出目录')
        if command=='run':
            p.add_argument('--prepared-dir', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in (None, 'show'):
            from src.course_delivery import build_summary, show_summary
            show_summary(build_summary(ROOT))
        elif args.command == 'check':
            command = [sys.executable, str(ROOT / 'scripts/check_project.py')]
            if args.core:
                command.append('--core')
            return subprocess.run(command, cwd=ROOT).returncode
        elif args.command == 'export':
            from src.course_delivery import export_course
            export_course(ROOT, args.output_dir)
            print(f'已导出：{Path(args.output_dir).resolve()}')
        else:
            from src.course_experiment import prepare_course, run_course
            if args.command == 'prepare':
                prepare_course(args.config, args.output_dir)
            else:
                result = run_course(args.config, args.prepared_dir, args.output_dir)
                (Path(args.output_dir) / 'result.json').write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
                print(f'新结果已保存：{Path(args.output_dir).resolve()}')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
