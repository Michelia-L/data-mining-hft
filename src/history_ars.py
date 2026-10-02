"""Algorithm 3 的独立历史窗口回测与因果 ARS 工程实现。

论文第 5 页例子为最近 30 分钟回测其他模型、交易 5 分钟；超过一小时
奖励如何成熟没有公开。本模块区分严格最近窗口与向前平移 H_max 的成熟窗口，
不把平移窗口说成论文唯一规定。每次窗口从空仓/空意图开始，重新预热特征。
"""
from collections import Counter

import numpy as np
import pandas as pd

from src.config import INSTRUMENT_CONFIG, QUANTITY
from src.period_ucb import PeriodOESelector
from src.snapshot_dataset import build_session_dataset, utc_ns
from src.time_execution import TimeAccount
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import predict_library


class HistoryWindowBacktester:
    """在当前可见前缀内重回放窗口，所有模型共享行情准备但账户完全独立。

    回测区间 [s,e)，订单评价可使用 e 之后但不晚于 now 的价格；观察扩展
    不再下单。窗口开始为空仓；e-step 若有可信可交易报价则主动退出，否则
    保留未平仓并拒绝完整评分。该边界退出为历史子回测工程假设。
    """
    def __init__(self, frame, versions, calendar, reward, *, interval_ms=500,
                 latency_ms=500, holding_review_ms=15000, threshold=.000015):
        self.frame=frame.reset_index(drop=True)
        self.times=utc_ns(self.frame.ts_event)
        self.versions, self.calendar, self.reward=versions,calendar,reward
        self.step=interval_ms*1000000;self.interval_ms=interval_ms
        self.latency_ns=latency_ms*1000000;self.holding_ns=holding_review_ms*1000000
        self.threshold=threshold
        settings=(interval_ms,latency_ms,holding_review_ms)
        if (any(type(n) is not int or n<=0 for n in settings) or self.frame.ts_event.dt.tz is None
                or (np.diff(self.times)<=0).any() or (self.times%self.step).any()
                or reward.definition!='paper_price_difference' or reward.deduct_cost
                or not np.isfinite(threshold) or threshold<0
                or any(type(h) is not int or h<=0 or h%interval_ms for h in reward.horizons)):
            raise ValueError('Valid causal millisecond window protocol required')
        if not versions:
            raise ValueError('Scheduled model versions required')
        self.model_ids=[m['model_id'] for m in versions[0]['models']]

    def evaluate(self, now, session, version_id, window_ms, alignment, owners, *, detail=False):
        """仅重回放请求的其他模型，返回按模型排列的完整窗口 OE 或显式失败。

        recent: [now-W,now)；matured: [now-H_max-W,now-H_max)。价格观察
        严格止于 now。不裁掉未成熟订单，不用窗口外旧持仓、特征或反馈状态。
        """
        if (type(window_ms) is not int or window_ms<=0 or window_ms%self.interval_ms
                or alignment not in ('recent','matured') or not 0<len(owners)<=len(self.model_ids)
                or len(set(owners))!=len(owners) or any(type(i) is not int or not 0<=i<len(self.model_ids) for i in owners)):
            raise ValueError('Grid-aligned window and valid unique model owners required')
        longest=max(self.reward.horizons)*1000000
        end=int(now)-(longest if alignment=='matured' else 0);start=end-window_ms*1000000
        metadata=dict(window_start_ns=start,window_end_ns=end,known_through_ns=int(now),
            reward_observation_latest_required_ns=end-self.step+longest,alignment=alignment,
            model_version_id=version_id,session_id=session,initial_account='flat_no_intents',
            feature_history='rebuilt_inside_window_only',source='independent_history_replay')
        version=next((v for v in self.versions if v['version_sha256']==version_id),None)
        if version is None or pd.Timestamp(version['available_at_utc']).value>now:
            raise ValueError('Current visible model version required')
        visible=[v for v in self.versions if pd.Timestamp(v['available_at_utc']).value<=now]
        if visible[-1]['version_sha256']!=version_id:
            raise ValueError('Do not score an obsolete model version as the current version')
        info=self.calendar.sessions[session]
        reason=None
        if start<info['open'].value or end>info['close'].value:reason='insufficient_session_history'
        elif start<pd.Timestamp(version['available_at_utc']).value:reason='window_before_current_version'
        if reason:return {i:metadata|dict(owner=i,status=reason,score=None,total_fills=None) for i in owners}
        # 输入容器可能保存整日，但实际交易窗与观察容器都只取当前可见前缀。
        stop=int(np.searchsorted(self.times,int(now),side='right'))
        lo=int(np.searchsorted(self.times[:stop],start));hi=int(np.searchsorted(self.times[:stop],end))
        if hi-lo<2:return {i:metadata|dict(owner=i,status='insufficient_quotes',score=None,total_fills=None) for i in owners}
        raw=self.frame.iloc[lo:hi][SNAPSHOT_SCHEMA.names+['session_id','segment_id']].copy()
        received=utc_ns(raw.source_ts_recv)
        if raw.source_ts_recv.dt.tz is None or raw.source_ts_recv.isna().any() or (received>=self.times[lo:hi]).any():
            raise ValueError('Historical replay quotes must already be received')
        assigned,outside,preopen=self.calendar.assign(raw)
        if (outside or preopen or not assigned.session_id.equals(raw.session_id)
                or not assigned.segment_id.equals(raw.segment_id)):
            raise ValueError('Historical quotes disagree with session calendar')
        quotes=raw[['bid_px_00','ask_px_00']].to_numpy(float)
        if not np.isfinite(quotes).all() or (quotes[:,0]<=0).any() or (quotes[:,1]<quotes[:,0]).any():
            raise ValueError('Finite valid historical quotes required')
        # 只重建因果特征；生成的一格标签不参与预测、决策或 OE 可用性。
        historical,_,_=build_session_dataset(raw,self.calendar,self.interval_ms,(self.interval_ms,))
        predictions,_=predict_library([version],historical,self.interval_ms)
        clocks=self.times[lo:hi];n=len(clocks)
        segments=historical.segment_id.to_numpy()
        ends=dict(zip(self.calendar.segments.segment_id,self.calendar.ends))
        tradable=np.array([t<ends[s] for t,s in zip(clocks,segments)])
        changed=np.r_[False,(np.diff(clocks)!=self.step)|(segments[1:]!=segments[:-1])]
        scheduled=tradable & np.array([t+self.step>=ends[s] for t,s in zip(clocks,segments)])
        full_start=clocks[0]==start or start==info['open'].value and clocks[0]==start+self.step
        full_end=clocks[-1]==end-self.step and tradable[-1]
        risks=changed|scheduled
        if full_end:risks[-1]=True
        # 观察连续性用完整可见报价前缀计算；未来标签列从未读取。
        observed_times=self.times[:stop]
        observed=self.frame.iloc[:stop]
        blocks=np.cumsum(np.r_[False,(np.diff(observed_times)!=self.step)
            |(observed.segment_id.to_numpy()[1:]!=observed.segment_id.to_numpy()[:-1])
            |(observed.session_id.to_numpy()[1:]!=observed.session_id.to_numpy()[:-1])])
        mids=observed[['bid_px_00','ask_px_00']].mean(axis=1).to_numpy(float)
        results={}
        for owner in owners:
            if version['models'][owner]['status']!='fitted':
                results[owner]=metadata|dict(owner=owner,status='model_not_fitted',score=None,total_fills=None);continue
            available=historical.feature_valid.to_numpy() & np.isfinite(predictions[:,owner])
            available &= tradable & ~scheduled
            if full_end:available[-1]=False
            origins=np.flatnonzero(available)
            destinations=np.searchsorted(clocks,clocks[origins]+self.latency_ns)
            safe=np.minimum(destinations,n-1)
            local_blocks=np.cumsum(changed)
            keep=(destinations<n)&(local_blocks[safe]==local_blocks[origins])&~risks[safe]&tradable[safe]
            origins,destinations=origins[keep],destinations[keep]
            # 重复到期落在同一报价时只保留最新意图，与真实时间执行器一致。
            execute=np.full(n,-1,dtype=int);execute[destinations]=origins
            actions=np.zeros(n,dtype=int);rows=np.flatnonzero(execute>=0)
            values=predictions[execute[rows],owner]
            actions[rows]=(values>self.threshold).astype(int)-(values<-self.threshold).astype(int)
            account=TimeAccount(INSTRUMENT_CONFIG['CME_ES'],self.holding_ns,QUANTITY)
            events=np.flatnonzero((execute>=0)|risks);event_times=clocks[events]
            meaningful=np.flatnonzero((actions[events]!=0)|risks[events])
            risk_times=clocks[np.flatnonzero(risks)]
            cursor=0
            while cursor<len(events):
                row=int(events[cursor]);clock=int(clocks[row])
                if risks[row] or not tradable[row]:
                    if tradable[row] and account.position:
                        fills=account.advance(clock,0,owner,historical.mid_price.iloc[row],
                            historical.bid_px_00.iloc[row],historical.ask_px_00.iloc[row],terminal=True)
                    else:fills=[]
                elif not account.position and actions[row]==0:fills=[]
                else:
                    fills=account.advance(clock,int(actions[row]),owner,historical.mid_price.iloc[row],
                        historical.bid_px_00.iloc[row],historical.ask_px_00.iloc[row])
                for fill in fills:
                    fill['window_row']=row;fill['model_version_id']=version_id
                cursor+=1
                # 跳过确实不可能改变账户的 tick，不跳过缺格/暂停/窗口退出。
                # 空仓零信号无效果；持仓复核前不能改方向。原始毫秒时钟不压缩。
                if not account.position:
                    j=np.searchsorted(meaningful,cursor)
                    cursor=int(meaningful[j]) if j<len(meaningful) else len(events)
                else:
                    target=account.last_review_step+self.holding_ns
                    j=np.searchsorted(risk_times,clock,side='right')
                    if j<len(risk_times):target=min(target,int(risk_times[j]))
                    cursor=max(cursor,int(np.searchsorted(event_times,target)))
            statuses=Counter();feature_sum=np.zeros(len(self.reward.horizons))
            hs=np.asarray(self.reward.horizons,dtype=np.int64)*1000000
            for fill in account.fills:
                origin=lo+fill['window_row'];targets=observed_times[origin]+hs
                if targets[-1]>now:status='pending_orders'
                else:
                    indices=np.searchsorted(observed_times,targets);safe=np.minimum(indices,stop-1)
                    if (indices>=stop).any() or not np.array_equal(observed_times[safe],targets):status='missing_target'
                    elif (blocks[safe]!=blocks[origin]).any():status='crossed_gap_or_session'
                    else:
                        status='matured'
                        feature_sum+=self.reward.features_from_prices(mids[origin],mids[safe],fill['side'])
                statuses[status]+=1
            count=len(account.fills)
            status=('incomplete_window_boundary' if not full_start or not full_end else
                    'unsettled_account' if account.position else 'no_orders' if not count else
                    'pending_orders' if statuses['pending_orders'] else
                    'incomplete_orders' if statuses['matured']!=count else 'matured')
            mean=feature_sum/count if status=='matured' else None
            results[owner]=metadata|dict(owner=owner,status=status,score=self.reward.score(mean) if mean is not None else None,
                total_fills=count,matured_order_count=statuses['matured'],order_statuses=dict(statuses),
                order_feature_expectation=mean.tolist() if mean is not None else None,
                terminal_position=account.position,window_end_liquidated=account.position==0,
                gross_pnl_usd=account.gross_equity_change(historical.mid_price.iloc[-1]),
                friction_usd=account.costs,net_pnl_usd=account.gross_equity_change(historical.mid_price.iloc[-1])-account.costs,
                decision_rows=int(available.sum()))
            if detail:
                # 子回测审计保留实际成交时刻与美元成本，供原执行器逐单核对。
                results[owner]['fills']=account.fills
        return results


class HistoryARSSelector(PeriodOESelector):
    """当前模型用真实账户成熟期间 OE，其他模型每次重回放固定历史窗。

    评分仅在新的选择期间计算，期间内锁定；W 不可观察时为 None，不拿初始
    零压过负奖励。全部不可评分时按模型声明顺序轮换，这是明确冷启动约定。
    """
    def __init__(self,name,models,backtester,*,period_ms=300000,window_ms=1800000,
                 alignment='recent',tie_tolerance=1e-9):
        super().__init__(name,models,period_ms=period_ms,c=0.)
        if (alignment not in ('recent','matured') or type(window_ms) is not int or window_ms<=0
                or window_ms%backtester.interval_ms or type(tie_tolerance) not in (float,int)
                or not np.isfinite(tie_tolerance) or tie_tolerance<0):
            raise ValueError('Valid ARS history window, alignment and tie tolerance required')
        self.backtester,self.window_ms,self.alignment=backtester,window_ms,alignment
        self.tie_tolerance=tie_tolerance;self.mode='history_ars_'+alignment
        self.latest_live,self.previous_owner={},{}

    def observe_period(self,version,owner,reward,*,context=None):
        """旧期间反馈保留实际时刻；晚到较老期间不覆盖该臂较新的成熟期间。"""
        if context is None:raise ValueError('ARS live feedback needs observed time and period identity')
        super().observe_period(version,owner,reward,context=context)
        key=(version,owner)
        if key not in self.latest_live or context['period_index']>=self.latest_live[key]['period_index']:
            self.latest_live[key]=dict(context,reward=float(reward))

    def select_period(self,session,version,period,clock_ns):
        """Algorithm 3 的其他臂额外历史反馈，不是从回放起点连续运行的影子账户。"""
        key=(session,version,period)
        if key not in self.choices:
            state=self.state(version);previous=self.previous_owner.get(version,0)
            other=[i for i in range(self.k) if i!=previous]
            evaluations=self.backtester.evaluate(int(clock_ns),session,version,self.window_ms,self.alignment,other) if other else {}
            live=self.latest_live.get((version,previous))
            if live and live['observed_at_ns']>clock_ns:
                raise ValueError('Live period reward is not visible yet')
            evaluations[previous]=dict(owner=previous,model_version_id=version,source='live_selected_model',
                status='matured' if live else 'no_matured_live_period',score=live['reward'] if live else None,
                live_period=live)
            scores=[evaluations[i]['score'] for i in range(self.k)]
            eligible=[i for i,s in enumerate(scores) if s is not None]
            if eligible:
                best=max(scores[i] for i in eligible)
                tied=[i for i in eligible if abs(scores[i]-best)<=self.tie_tolerance];owner=tied[0]
            else:tied=[];owner=int(state['visits'].sum()%self.k)
            self.choices[key]=owner;self.previous_owner[version]=owner
            self.period_selections.append(dict(session_id=session,model_version_id=version,period_index=int(period),
                selected_at_ns=int(clock_ns),owner=owner,previous_owner=previous,
                visits_before=state['visits'].tolist(),feedback_counts_before=state['feedback_counts'].tolist(),
                scores_before=scores,tied_best_owners=tied,cold_start=not eligible,mode=self.mode,
                evaluations=[evaluations[i] for i in range(self.k)]))
            state['visits'][owner]+=1
        return self.record(self.choices[key])
