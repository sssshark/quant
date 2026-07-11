export const meta = {
  name: 'quant-critical-audit',
  description: '5-lens adversarial audit of quant project conclusions/code/data/diag-scripts, verified before fixing',
  phases: [
    { title: 'Audit', detail: '5 critical lenses over the project, each returns structured findings' },
    { title: 'Verify', detail: 'adversarially refute each finding; only confirmed survive' },
  ],
}

// ---- schemas ----
const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string', description: 'which lens (conclusion-staleness/overfitting/code/data/scripts)' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'short slug like code-qfq-offbyone' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          file: { type: 'string' },
          line: { type: 'number' },
          summary: { type: 'string', description: 'one-sentence statement of the defect/concern' },
          failure_scenario: { type: 'string', description: 'concrete inputs/state -> wrong output/crash; for a stale conclusion, the specific claim that no longer holds + evidence' },
          evidence: { type: 'string', description: 'exact code lines / IMPROVEMENTS quote / function names that support this' },
          proposed_fix: { type: 'string', description: 'concrete fix or the re-run needed to confirm' },
        },
        required: ['id', 'severity', 'file', 'summary', 'failure_scenario', 'evidence', 'proposed_fix'],
      },
    },
  },
  required: ['lens', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'PLAUSIBLE'] },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    reasoning: { type: 'string', description: 'why this survives/dies scrutiny; cite exact code lines read' },
    corrected_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'], description: 'your independent severity assessment' },
    notes_for_fix: { type: 'string', description: 'specific guidance for the fixing agent, or empty if refuted' },
  },
  required: ['verdict', 'confidence', 'reasoning', 'corrected_severity'],
}

// ---- 5 lenses ----
const LENSES = [
  {
    key: 'conclusion-staleness',
    label: 'audit:conclusion-staleness',
    prompt: `你是一个极度批判的审计员，审查 /home/poisson/program/quant_program/quant 这个量化项目。

【你的专属视角：结论过时（conclusion-staleness）】
用户的核心引子："项目中记录的结论都是中间的版本中得出来的，在最终的版本中也能维持这个结论吗？"

IMPROVEMENTS.md 记录了 79+ 条结论，很多标注"基于去污前数据/污染期/中间口径"，后来经历了 3 次重定基线（E3/E4 污染期→pre_close 法重定）。你的任务是逐条核对这些结论的**时效性**：
- IMPROVEMENTS.md 里的"解决记录"表中，哪些行的"结果"数字是基于已经被废弃的中间版本（如"基于去污前数据，绝对值见重定基线说明"这类）但仍然写在表里、未更新到最终口径？
- 正文各条目（A1-A9/B1-B6/C1-C10/D1-D7/E1-E7/F1-F8/G1-G6/H1-H11/I1-I4/J1-J7/M1-M9）里引用的夏普/回撤/年化数字，哪些是中间版本、与最终版代码/数据下重跑的结果可能不一致？
- 是否有结论本身已被后续条目修正（如 M6 被 M7 修正"M6 过强"）但 IMPROVEMENTS 顶部"进度块"或"图例"没同步？
- 顶部"重定基线"说明里声明"绝对值作废、需重跑"——那表里保留的绝对值是否会误导读者？

方法：
1. 读 IMPROVEMENTS.md 全文（384 行），列出每条结论引用的具体数字 + 它标注的数据版本。
2. 读 cta.py 的常量块（L19-116）和 decide_targets（L236+），确认"最终版"参数/逻辑。
3. 重点：找出"声称已验证但数字来自被废弃口径"的条目，以及"结论可能随最终版漂移甚至翻转"的条目。
4. 不要轻信文档自述的"稳健"——要用最终版代码逻辑反推。

只报真实缺陷。每条 finding 必须给 file/line + failure_scenario（哪个结论在哪个数据版本下得出、最终版下为何可能不成立）+ evidence（IMPROVEMENTS 原文引用 + 代码行）+ proposed_fix。没有缺陷就返回空 findings。宁缺毋滥，但别放过。`,
  },
  {
    key: 'overfitting',
    label: 'audit:overfitting',
    prompt: `你是一个极度批判的审计员，专审"过拟合"——用户的核心质疑之一："真的不会有过拟合吗？"

【你的专属视角：过拟合的真假】
项目自述 PBO=0.67（>0.5 过拟合警告），但又声称 pcv 衰减 0.89、DSR@50=0.961、月频 DSR=0.999。你要批判性核查这些"可信"声明：
1. PBO=0.67：读 engine.py run_pbo(L904)/pbo(L873)。Bailey-LdP 排名法的实现正确吗？K=6 折、label_horizon=21、embargo=21——参数是否被调到让结果好看？ISC 最优参数在 test 排名 [4,2,6,6,5,3] 4/6 落下半=0.67，这个计算有无 bug（比如排名方向反了、ISC 口径泄漏）？
2. purged_cv 衰减 0.89：读 purged_cv(L807)。purge/embargo 是否真的防了动量标签泄漏？还是 purge 窗口太短（21 日 < 126 日动量窗）导致仍有泄漏、把 OOS 衰减做高？
3. DSR：读 deflated_sharpe(L1085)/_norm_ppf/_norm_cdf。纯 Python 实现的 _norm_ppf(Acklam)/_norm_cdf(erfc) 数学对吗？n_trials 取值是否合理（50/100 是否偏低估了真实多重比较次数——整个 backlog 就是 A/B 选参记录，真实 trials 远超 100）？
4. walk_forward：读 walk_forward(L723)。grid 只有 8 个参数，是否"挑了一个不太烂的 grid"？
5. 月频 DSR=0.999：读 _monthly_returns(L1113)/run_dsr(L1120)。"月频 n=152 仍按月独立"被项目自己列为 caveat——那 0.999 是否高估？
6. hold_all/J2 选股无 alpha P=0.148：读 bootstrap_selection(L981)。配对 block bootstrap 的 seed=7 固定——是否恰好挑了个不显著的 seed？

方法：读代码数学实现，逐函数核验。对每个"可信"声明，给"它可能在哪一步被做高"的具体机制。
没有缺陷就返回空 findings。每条给 evidence（代码行 + 数学论证）+ proposed_fix（如"把 purge 窗口从 21 提到 max(LOOKBACKS)=126 重跑"）。`,
  },
  {
    key: 'code-correctness',
    label: 'audit:code-correctness',
    prompt: `你是一个极度批判的代码审计员。用户的核心质疑："实现的代码都没有系统审查过，代码确定正确吗？"

【你的专属视角：代码正确性】
项目从未被系统审查过。逐个核验 engine.py(1702行)/cta.py/multi_strategy.py 的核心函数数学与逻辑正确性：

1. _qfq_from_pre_close (engine.py L61-75): f[t]=pre_close[t]/close[t-1], g[i]=∏f[t≥i] cumprod 反向, adj[i]=close[i]×g.shift(-1)。核验：① cumprod 反向索引方向对吗？② g.shift(-1) 的 fillna(1.0) 会不会把除权日当天本身漏算？③ "最新价=close末值不变"是否真的成立（边界）？④ 除权日 f<1 会把历史价下压——但 pre_close 对除权日当天是否=除权后基准？给具体反例。
2. backtest (L270-359): T+1 pending 缓冲。核验 pending 写入逻辑：信号 T 日算 weights→pending→T+1 成交。closes 用于收益是从 T+2 起？还是 T+1？有无 off-by-one（信号日用持仓算收益=未来函数）？
3. limit_masks (L180-194) + _apply_target_with_limits (L247): 涨停价 round(pre_close*(1+lim),2)。核验：① A 股涨停价是"前收盘×1.1 四舍五入到分"对吗？还是 ST 板不同？② cant_buy 用 close>=涨停价——但 ETF 尾盘封板 close==涨停价会成立，盘中封板尾盘开板则不触发，这符合 D2 设计。但有无"复权价 vs 真实价"混淆（raw_close 必须未复权）？
4. factor_attribution_multi (L452) / factor_attribution (L400): OLS via np.linalg.lstsq。核验：① t-stat = pinv(X'X)·σ²ε 的公式对吗（自由度、是否含截距列）？② 年化口径（自然日 vs 交易日）是否一致？③ SMB/VMG/BND/GLD/NSDQ 因子从 px 构造，有无构造 bug（如 NSDQ=纳指-300 但纳指本身在池里=自回归）？
5. _circ_block_idx (L917) / _ann_sharpe (L926): circular block bootstrap。核验：① 循环索引（block 跨越序列边界）实现正确吗？② _ann_sharpe 的年化（×√252?）口径与 perf 一致吗？
6. _seg_perf (L601) / run_regime: 按 regime 切片重建净值——有无"切片内 ffill 引入泄漏"或"起点选择偏倚"？
7. perf (L376): Sortino 下行偏差、Calmar、Omega 公式核验。
8. decide_targets (cta.py L236): hold_all 分支(L285)与正常分支候选集是否"同时点一致"（J2 声称的对照公平性）？绝对动量≤0 切防守 + vol_target + trend 叠加顺序有无逻辑漏洞？

方法：逐函数读源码，用具体输入构造反例。每条 finding 给 file:line + failure_scenario（输入→错误输出）+ 数学/逻辑论证 evidence + proposed_fix。
没有缺陷返回空。`,
  },
  {
    key: 'data',
    label: 'audit:data',
    prompt: `你是一个极度批判的数据审计员。用户引子隐含：数据是否真的干净？

【你的专属视角：数据正确性】
项目经历了 3 次重定基线（污染期→E3/E4→pre_close 法）。核验数据管线是否真"干净"：

1. _load_via_tushare (engine.py L78-138): ① fund_daily 返回的 close 是未复权，pre_close 也是未复权——但 tushare 的 pre_close 列在除权日当天是"除权后调整的前收"还是"昨收原值"？这直接决定 _qfq_from_pre_close 的 f 因子对不对。② time.sleep(0.6) 限速——但 8 只 ETF 串行拉，有无某只失败静默返回 NaN 被后段 ffill 掩盖？③ .ffill().dropna(how="all") 会不会把停牌/未上市段 ffill 成虚假平稳价？
2. _qfq_from_pre_close 数学（同 code 审计员交叉）：除权日 f<1 压历史价，但若是**份额拆分**（1拆2），pre_close[t] 应≈close[t-1]/2，f≈0.5——核验项目对 159928/513500/512010 拆分的处理是否正确。
3. _data_quality_guard (L200): thr_mult=1.5，即 |日收益|>1.5×涨跌停才告警。但 ETF 涨跌停是 ±10%/±20%，1.5×=±15%/±30%——这意味着 ±12% 的伪跳变**不会被告警**。是否阈值太松、放过中等伪迹？
4. _anchor_start (L219): ready[0]=BENCH+DEFENSE 首个共同非NaN日=2013-03-25。但此时 512100/518880/513100 未上市——动量排序 dropna 跳过，但**这是否意味着早期只有 4-5 只可选、top_n=3 几乎=等权**，早期样本代表性被静默稀释？
5. 缓存 vs 实盘：有无 px_cache.pkl 之类缓存，与 live 数据不一致？diag 脚本的数据加载路径与 engine.load_real 是否一致（diag 可能用不同 loader 导致口径漂移）？
6. 国债 511010：项目自述"国债 +2.9%/年 债牛"。但 511010 是国债 ETF，其收益是否被 _qfq 正确复权（国债 ETF 也有分红/折价）？J4 bondstress 把它替换成 0%/-2% 合成序列——合成序列的构造（_synth_defense_series L1239）有无日复利 bug？
7. crossmarket.py 美股数据：yfinance 取数，有无 survivorship/split 未处理？

方法：读 loader 代码 + 用 grep 找缓存文件 + 核验数学。每条给 file:line + failure_scenario + evidence + proposed_fix（含"需重跑 X 核验"）。没有缺陷返回空。`,
  },
  {
    key: 'scripts',
    label: 'audit:diag-scripts',
    prompt: `你是一个极度批判的脚本审计员。用户的核心质疑："调研的脚本能经得起深度解析吗？"

【你的专属视角：诊断脚本深度】
/home/poisson/program/quant_program/quant/diag/ 下有 14 个诊断脚本。逐个核验它们的结论是否**经得起深度解析**——尤其那些"负结果"（证伪）脚本，负结果可能是**脚本 bug 造成的伪证伪**，让本应有效的方法被错误丢弃。

重点脚本（按风险排序）：
1. diag_portfolio_opt.py (EPO 协方差优化，负结果): hand-written ledoit_wolf_cov。核验：① δ=clip((π−ρ)/(γ·T),0,1) 的 π/ρ/γ 定义对吗？项目用"ρ≈trace(pi_mat) 对角近似"——这是 Ledoit-Wolf 2004 的常数相关模型 ρ 还是误用了？② min_var_weights 的"负权重→等权 fallback"是否把 EPO 退化成等权、人为压低 EPO 表现？③ δ=0.000（N=2 不收缩）的结论是否是"样本太小导致 shrinkage 退化"的 bug 而非真无收益？
2. diag_rsrs.py (RSRS 择时，负结果): rsrs_beta_r2 OLS slope。核验：① OLS slope 公式对吗？② z-score×r² 修正分——r² 在哪算（全样本还是滚动）？有无前视？③ 多头占比 21.5% 太保守——是否阈值 0.7 设太高、或信号方向（多/空）反了导致本应高暴露变低暴露？④ timing_backtest 的 T+1/成本口径与 engine.backtest 一致吗？
3. diag_riskparity.py (risk-parity 三层): 静态/滚动/backtest 三口径。核验：① 日频 1.91 vs 月频 1.45 的差异是否因为日频有前视（用了全样本 σ）？② 1/σ 权重滚动窗口对齐（无前视）真的无前视吗？
4. diag_sector*.py (M5/M6/M7/M8 全负): 行业选股全挂。核验：① "三道关"里的 |corr|<0.3 用的是全样本 corr 还是滚动？全样本 corr 有无前视？② 行业 ETF 复权是否走 _qfq_from_pre_close？③ 回撤 -54% 这类极端值是否单只行业拆分未复权造成？
5. diag_mr.py / diag_pairs.py / diag_bonds.py: 均值回归/配对/国债候选诊断。核验三道关计算。

方法：逐个读 diag 脚本源码，找数学/前视/口径漂移 bug。每条给 file:line + failure_scenario（哪个负结果可能是 bug 造成的）+ evidence + proposed_fix。
**特别警惕**：负结果脚本最危险——bug 会让真有效的方法被丢弃。对每个"证伪"结论，问"如果是 bug，正确实现会翻转吗？"
没有缺陷返回空。宁缺毋滥。`,
  },
]

// ---- pipeline: each lens audits, then its findings get verified adversarially ----
phase('Audit')
phase('Verify')

const results = await pipeline(
  LENSES,
  // stage 1: audit
  (lens) => agent(lens.prompt, {
    label: lens.label,
    phase: 'Audit',
    schema: FINDINGS_SCHEMA,
    effort: 'xhigh',
  }),
  // stage 2: adversarially verify EACH finding from this lens
  (audit, lens) => {
    if (!audit || !audit.findings || audit.findings.length === 0) {
      return { lens: lens.key, confirmed: [], all: [] }
    }
    return parallel(audit.findings.map(f => () =>
      agent(
        `你是一个对抗式验证员（adversarial verifier），任务是**反驳**下面这条审计发现。默认怀疑态度：如果无法用代码/数据确凿证明它成立，就判 REFUTED 或 PLAUSIBLE，只有确凿无疑才 CONFIRMED。

【审计发现】
lens: ${lens.key}
id: ${f.id}
severity: ${f.severity}
file: ${f.file}${f.line ? ':' + f.line : ''}
summary: ${f.summary}
failure_scenario: ${f.failure_scenario}
evidence: ${f.evidence}
proposed_fix: ${f.proposed_fix}

【你的任务】
1. 打开 /home/poisson/program/quant_program/quant 实际读 ${f.file} 的相关行（以及 IMPROVEMENTS.md 如涉及结论）。
2. 独立判断这条发现是否成立。重点：
   - 代码行号是否准确？引用的代码是否真的如 evidence 所述？
   - failure_scenario 是否真的能触发？数学/逻辑论证是否正确？
   - 是否是"误报"——比如把向后兼容设计当成 bug、把已修复的旧 bug 当现存问题、把有意设计当缺陷？
   - 对"负结果可能是 bug"类发现：正确实现真的会翻转结论吗？还是 bug 不影响结论方向？
3. 给出 verdict（CONFIRMED/REFUTED/PLAUSIBLE）+ corrected_severity + reasoning（引用你读到的真实代码行）+ notes_for_fix。
4. CONFIRMED 只用于确凿无疑的真实缺陷；PLAUSIBLE 用于"可能但不确凿"；REFUTED 用于误报或已不成立。`,
        {
          label: `verify:${f.id}`,
          phase: 'Verify',
          schema: VERDICT_SCHEMA,
          effort: 'xhigh',
        }
      ).then(v => ({ ...f, verdict: v, lens: lens.key }))
    )).then(verified => ({
      lens: lens.key,
      all: audit.findings,
      confirmed: verified.filter(Boolean).filter(v =>
        v.verdict && (v.verdict.verdict === 'CONFIRMED' || v.verdict.verdict === 'PLAUSIBLE')),
    }))
  }
)

// ---- collect ----
const allConfirmed = results.filter(Boolean).flatMap(r => r.confirmed || [])
const allFindings = results.filter(Boolean).flatMap(r => r.all || [])
const byLens = {}
for (const r of results.filter(Boolean)) {
  byLens[r.lens] = { total: (r.all || []).length, confirmed: (r.confirmed || []).length }
}

log(`Round-1 audit complete: ${allFindings.length} raw findings, ${allConfirmed.length} survived verification`)
for (const [k, v] of Object.entries(byLens)) {
  log(`  ${k}: ${v.confirmed}/${v.total} confirmed`)
}

return {
  round: 1,
  summary: `${allConfirmed.length} confirmed/plausible findings (of ${allFindings.length} raw)`,
  byLens,
  confirmed: allConfirmed.map(f => ({
    id: f.id, lens: f.lens, severity: f.verdict.corrected_severity || f.severity,
    file: f.file, line: f.line, summary: f.summary, failure_scenario: f.failure_scenario,
    evidence: f.evidence, proposed_fix: f.proposed_fix,
    verdict: f.verdict.verdict, confidence: f.verdict.confidence,
    verify_reasoning: f.verdict.reasoning, notes_for_fix: f.verdict.notes_for_fix || '',
  })),
  refuted_count: allFindings.length - allConfirmed.length,
}
