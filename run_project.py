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
    check.add_argument('--friction-output-dir', help='另外核对一个独立费用实验导出包')
    export = sub.add_parser('export', help='从冻结快照导出摘要与两张结果图')
    export.add_argument('--output-dir', required=True, help='不存在的新输出目录')
    friction_export = sub.add_parser('export-friction', help='从真实费用实验重新导出完整证据，不重跑或改写冻结身份')
    friction_export.add_argument('--result-dir', required=True)
    friction_export.add_argument('--output-dir', required=True)
    friction_export.add_argument('--input-equivalence', help='全量输入等价性报告，必须绑定这次冻结输入')
    verification = sub.add_parser('verify-friction-inputs', help='用原采样器全量核对缓存与公开 prepare 的产物')
    verification.add_argument('--config', default=str(ROOT / 'config/course_experiment.json'))
    verification.add_argument('--prepared-dir', required=True)
    verification.add_argument('--output-dir', required=True, help='不存在的参考输入目录')
    verification.add_argument('--workers', type=int, default=4, help='独立源分区进程数，1至8')
    for command in ('prepare', 'run', 'run-friction'):
        help_text = ('准备协议所需新数据' if command=='prepare' else
                     '增量实验：原OE与内化实际摩擦的OE共用冻结权重' if command=='run-friction' else
                     '重跑核心OE-UCB；另存新结果')
        p = sub.add_parser(command, help=help_text)
        p.add_argument('--config', default=str(ROOT / 'config/course_experiment.json'))
        p.add_argument('--output-dir', required=True, help='不存在的新输出目录')
        if command in ('run', 'run-friction'):
            p.add_argument('--prepared-dir', required=True)
        if command == 'run-friction':
            p.add_argument('--input-equivalence', help='可选的全量输入等价性报告')
    args = parser.parse_args(argv)
    try:
        if args.command in (None, 'show'):
            from src.course_delivery import build_summary, show_summary
            show_summary(build_summary(ROOT))
        elif args.command == 'check':
            if args.friction_output_dir:
                from src.friction_artifacts import check_friction_bundle
                check_friction_bundle(args.friction_output_dir)
                print('通过：独立费用实验压缩包、摘要及输入核对身份。')
            command = [sys.executable, str(ROOT / 'scripts/check_project.py')]
            if args.core:
                command.append('--core')
            return subprocess.run(command, cwd=ROOT).returncode
        elif args.command == 'export':
            from src.course_delivery import export_course
            export_course(ROOT, args.output_dir)
            print(f'已导出：{Path(args.output_dir).resolve()}')
        elif args.command == 'export-friction':
            from src.friction_delivery import reexport_friction_result
            reexport_friction_result(args.result_dir, args.output_dir, input_equivalence=args.input_equivalence)
        elif args.command == 'verify-friction-inputs':
            from src.friction_inputs import verify_friction_inputs
            verify_friction_inputs(args.config, args.prepared_dir, args.output_dir, args.workers)
        else:
            from src.course_experiment import prepare_course, run_course
            if args.command == 'prepare':
                prepare_course(args.config, args.output_dir)
            else:
                result = run_course(args.config, args.prepared_dir, args.output_dir,
                                    friction_experiment=args.command == 'run-friction')
                (Path(args.output_dir) / 'result.json').write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
                if args.command == 'run-friction':
                    from src.friction_delivery import export_friction_result
                    export_friction_result(result,args.output_dir,input_equivalence=args.input_equivalence)
                print(f'新结果已保存：{Path(args.output_dir).resolve()}')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
