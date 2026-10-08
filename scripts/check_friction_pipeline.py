"""用小合成消息实际执行公开 CLI 的准备、建库、回放和完整证据检查。

短时间参数只为核对发布流程，不是正式交易实验或盈利证据。
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check_friction_pipeline(root, require):
    """真实调用 prepare/run-friction/check；核对归档完备、篡改与覆盖拒绝。"""
    import gzip
    import os
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.friction_artifacts import check_friction_bundle, file_sha256
    from src.friction_inputs import compare_parquet
    from src.snapshot_dataset import SessionCalendar
    from src.timed_snapshots import F_LAST, PRICE_FIELDS, SIZE_FIELDS
    root = Path(root)
    calendar = SessionCalendar()
    with tempfile.TemporaryDirectory(prefix='friction-cli-pipeline-') as temporary:
        work = Path(temporary)
        config = json.loads((root/'config/course_experiment.json').read_text())
        phases = dict(train=['2025-10-02','2025-10-03'],calibration=['2025-10-06'],
                      validation=['2025-10-07'],test=['2025-10-08'])
        horizons = [500,1000,1500,2000,2500,3000,3500]
        config['protocol'].update(sessions=phases,prediction_horizon_ms=500,
            reward_horizons_ms=horizons,holding_review_ms=1000)
        config['library'].update(bootstrap_session='2025-10-06',through_session='2025-10-08',
            evaluation_sessions=['2025-10-06','2025-10-07','2025-10-08'],
            prediction_horizon_ms=500,purge_ms=3500)
        config['selection_period_ms'] = 5000
        days = [d for values in phases.values() for d in values]
        dates = sorted({t.date().isoformat() for day in days for t in pd.date_range(
            calendar.sessions[day]['open'].normalize(),calendar.sessions[day]['close'].normalize(),freq='D')})
        rows = {d:[] for d in dates}
        for j, day in enumerate(days):
            session = calendar.sessions[day]
            for start in (session['open'],session['close']-pd.Timedelta(seconds=120)):
                for i in range(241):
                    stamp = start+pd.Timedelta(milliseconds=500*i+400)
                    price = 100+j+.25*round(10*np.sin(i/12)+i/15)
                    row = dict(ts_recv=stamp,ts_event=stamp,flags=F_LAST,sequence=i,
                               instrument_id=294973,symbol='ESZ5')
                    for level in range(5):
                        row.update({f'bid_px_{level:02d}':price-.25*level,
                            f'ask_px_{level:02d}':price+.25*(level+1),
                            f'bid_sz_{level:02d}':10+i%5,f'ask_sz_{level:02d}':12+i%7})
                    rows[stamp.date().isoformat()].append(row)
        fields = [pa.field('ts_recv',pa.timestamp('ns',tz='UTC')),
            pa.field('ts_event',pa.timestamp('ns',tz='UTC')),pa.field('flags',pa.uint8()),
            pa.field('sequence',pa.uint32()),pa.field('instrument_id',pa.uint32()),
            pa.field('symbol',pa.string()),*[pa.field(f,pa.float64()) for f in PRICE_FIELDS],
            *[pa.field(f,pa.uint32()) for f in SIZE_FIELDS]]
        files = []
        for day, values in rows.items():
            target = work/f'{day}.parquet'
            pq.write_table(pa.Table.from_pylist(sorted(values,key=lambda r:r['ts_recv']),schema=pa.schema(fields)),target)
            files.append(dict(date=day,condition='available',parquet=str(target),rows=len(values)))
        index = work/'index.json'
        index.write_text(json.dumps(dict(dataset='GLBX.MDP3',schema='mbp-10',symbol='ESZ5',stype_in='raw_symbol',files=files)))
        config['data_index'] = str(index)
        path = work/'config.json'; path.write_text(json.dumps(config))
        prepared, result = work/'prepared', work/'result'
        env = os.environ.copy(); env['MPLCONFIGDIR'] = str(work/'matplotlib')
        def run(args, success=True):
            completed = subprocess.run([sys.executable,str(root/'run_project.py'),*args],
                cwd=root,env=env,capture_output=True,text=True)
            require((completed.returncode==0)==success,
                    '费用实验CLI流程状态错误：'+completed.stdout[-1000:]+completed.stderr[-2000:])
            return completed
        run(['prepare','--config',str(path),'--output-dir',str(prepared)])
        run(['run-friction','--config',str(path),'--prepared-dir',str(prepared),'--output-dir',str(result)])
        summary = check_friction_bundle(result)
        require(all(c['execution_gate']['usable_for_execution'] for c in summary['age_cases']),
                '端到端样本须实际执行两臂，不能仅检查全阻断包')
        require(all(r['total_fills']>0 for c in summary['age_cases'] for phase in c['phases'].values()
                    for r in phase['statistics']), '端到端样本没有真实成交')
        require(gzip.decompress((result/'experiment_snapshot.json.gz').read_bytes())==(result/'result.json').read_bytes()
                and gzip.decompress((result/'model_library.json.gz').read_bytes())==(result/'library.json').read_bytes(),
                '端到端归档改变原字节')
        run(['check','--friction-output-dir',str(result)])
        reference = work/'reference'
        run(['verify-friction-inputs','--config',str(path),'--prepared-dir',str(prepared),
             '--output-dir',str(reference),'--workers','2'])
        proof = reference/'input_equivalence.json'
        copied = work/'republished'
        run(['export-friction','--result-dir',str(result),'--output-dir',str(copied),
             '--input-equivalence',str(proof)])
        require('input_equivalence_path' in check_friction_bundle(copied)['source'],
                '正式导出未纳入绑定的全量核对证据')
        wrong = json.loads(proof.read_text()); wrong['cached_input_manifest_sha256']='0'*64
        wrong_path = work/'wrong-proof.json'; wrong_path.write_text(json.dumps(wrong))
        run(['export-friction','--result-dir',str(result),'--output-dir',str(work/'wrong-publication'),
             '--input-equivalence',str(wrong_path)],success=False)
        require(file_sha256(copied/'experiment_snapshot.json.gz')==file_sha256(result/'experiment_snapshot.json.gz')
                and file_sha256(copied/'model_library.json.gz')==file_sha256(result/'model_library.json.gz'),
                '确定性归档不一致')
        before = file_sha256(result/'summary.json')
        run(['export-friction','--result-dir',str(result),'--output-dir',str(result)],success=False)
        require(file_sha256(result/'summary.json')==before,'拒绝覆盖时仍修改了产物')
        archive = copied/'model_library.json.gz'; archive.write_bytes(archive.read_bytes()+b'tampered')
        run(['check','--friction-output-dir',str(copied)],success=False)
        # 核对全量比较跨不同行组，且能发现末行（前缀检查覆盖不到）差异。
        a, b = work/'a.parquet', work/'b.parquet'
        table = pa.table({'value':range(12),'label':[None,1.]*6})
        pq.write_table(table,a,row_group_size=2); pq.write_table(table,b,row_group_size=5)
        require(compare_parquet(a,b)['rows']==12,'行组变化不应造成错误不等价')
        changed = pa.table({'value':[*range(11),999],'label':[None,1.]*6})
        pq.write_table(changed,b)
        try:
            compare_parquet(a,b)
        except ValueError:
            pass
        else:
            require(False,'全量检查漏过了末行差异')
    print('通过：公开 prepare→run-friction→check 实际成交、完整归档、全量输入核对绑定、篡改/覆盖拒绝及末行差异检出。')
