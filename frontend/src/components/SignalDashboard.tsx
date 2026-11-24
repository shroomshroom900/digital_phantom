import { useMemo, useState } from 'react'
import type { SimResult } from '../api'
import { fmt, mm, ms, uV } from '../units'
import Plot from './Plot'

export interface Change { key: string; a: any; b: any }

function stacked(result: SimResult, mode: 'diff' | 'mono', selected: number | null) {
  const sig = mode === 'diff' ? result.diff_V : result.mono_V
  const labels = mode === 'diff' ? result.electrodes.channel_labels.map(l => l.split(':')[0]) : sig.map((_, k) => `e${k + 1}`)
  let amax = 0
  for (const s of sig) for (const v of s) amax = Math.max(amax, Math.abs(v))
  const gap = uV(amax) * 1.1 || 1
  const t = result.time_s.map(ms)
  const n = sig.length
  const data: any[] = sig.map((s, k) => ({
    x: t, y: s.map(v => uV(v) + gap * (n - 1 - k)), type: 'scatter', mode: 'lines', name: labels[k],
    line: { width: k === selected ? 2.2 : 1, color: k === selected ? '#ffd166' : '#9fb3c8' },
    hovertemplate: `${labels[k]}: %{customdata:.3f} µV at %{x:.2f} ms<extra></extra>`, customdata: s.map(uV),
  }))
  const layout: any = {
    title: { text: mode === 'diff' ? 'Differential channels (µV, offset)' : 'Monopolar electrodes (µV, offset; zero-mean reference)', font: { size: 12 } },
    showlegend: false, xaxis: { title: { text: 'time (ms)' } },
    yaxis: { tickvals: labels.map((_, k) => gap * (n - 1 - k)), ticktext: labels, zeroline: false },
    margin: { l: 48, r: 12, t: 28, b: 40 },
    annotations: [{ xref: 'paper', yref: 'paper', x: 1, y: 1.04, showarrow: false, text: `spacing ${fmt(gap)} µV`, font: { size: 10 } }],
  }
  return { data, layout }
}

export default function SignalDashboard({ result, baseline, timeIndex, setTimeIndex, selectedChannel, setSelectedChannel,
  changes, onSetBaseline, onClearBaseline, isBaselineCurrent }: {
  result: SimResult; baseline: SimResult | null; timeIndex: number; setTimeIndex: (i: number) => void
  selectedChannel: number | null; setSelectedChannel: (j: number | null) => void
  changes: Change[] | null; onSetBaseline: () => void; onClearBaseline: () => void; isBaselineCurrent: boolean
}) {
  const [mode, setMode] = useState<'diff' | 'mono'>('diff')
  const ch = selectedChannel ?? Math.min(8, result.diff_V.length - 1)
  const cursor = ms(result.time_s[timeIndex] ?? 0)
  const st = useMemo(() => stacked(result, mode, mode === 'diff' ? ch : null), [result, mode, ch])

  const single = useMemo(() => {
    const data: any[] = [{
      x: result.time_s.map(ms), y: result.diff_V[ch].map(uV), type: 'scatter', mode: 'lines',
      name: `current: ${result.electrodes.channel_labels[ch]}`, line: { color: '#ffd166', width: 2 },
    }]
    if (baseline && baseline.diff_V[ch]) {
      data.push({ x: baseline.time_s.map(ms), y: baseline.diff_V[ch].map(uV), type: 'scatter', mode: 'lines',
        name: `baseline (job ${baseline.seq})`, line: { color: '#5ce1e6', width: 1.5, dash: 'dot' } })
    }
    const [a, b] = result.electrodes.pairs[ch]
    for (const [k, col] of [[a, '#a78bfa'], [b, '#f472b6']] as const)
      data.push({ x: result.time_s.map(ms), y: result.mono_V[k].map(uV), type: 'scatter', mode: 'lines',
        name: `φ e${k + 1} (monopolar)`, line: { color: col, width: 1 }, visible: 'legendonly' })
    return { data, layout: { title: { text: `Channel ${ch + 1}: V_EMG = φ(e${a + 1}) − φ(e${b + 1})`, font: { size: 12 } },
      xaxis: { title: { text: 'time (ms)' } }, yaxis: { title: { text: 'µV' } },
      legend: { x: 1, xanchor: 'right', y: 1, yanchor: 'top', bgcolor: 'rgba(20,23,28,0.8)', font: { size: 10 } } } as any }
  }, [result, baseline, ch])

  const bars = useMemo(() => {
    const x = result.electrodes.channel_labels.map((_, j) => `ch${j + 1}`)
    const data: any[] = [
      { x, y: result.metrics.diff_p2p_V.map(uV), type: 'bar', name: 'peak-to-peak', marker: { color: '#ffd166' } },
      { x, y: result.metrics.diff_rms_V.map(uV), type: 'bar', name: 'RMS', marker: { color: '#f59e0b' } },
    ]
    if (baseline && baseline.metrics.diff_p2p_V.length === x.length) {
      data.push({ x, y: baseline.metrics.diff_p2p_V.map(uV), type: 'scatter', mode: 'markers', name: 'baseline p2p', marker: { color: '#5ce1e6', symbol: 'line-ew-open', size: 14, line: { width: 2 } } })
      data.push({ x, y: baseline.metrics.diff_rms_V.map(uV), type: 'scatter', mode: 'markers', name: 'baseline RMS', marker: { color: '#38bdf8', symbol: 'line-ew-open', size: 14, line: { width: 2 } } })
    }
    return { data, layout: { barmode: 'group', title: { text: 'Amplitude per differential channel (µV)', font: { size: 12 } } } as any }
  }, [result, baseline])

  const cv = result.conduction_velocity
  const s = result.solver
  const m = result.metrics
  const bp = baseline?.metrics
  const pct = (a: number, b: number) => (b ? `${a >= b ? '+' : ''}${fmt((100 * (a - b)) / b, 3)}%` : '–')

  return (
    <div className="dashboard">
      <div className="cards">
        <div className="card">
          <h4>Recording</h4>
          <table className="kv"><tbody>
            <tr><td>duration</td><td>{fmt(ms(m.duration_s))} ms ({m.n_samples} samples @ {fmt(m.fs_Hz / 1e3)} kHz)</td></tr>
            <tr><td>channel {ch + 1} p2p / RMS</td><td>{fmt(uV(m.diff_p2p_V[ch]))} / {fmt(uV(m.diff_rms_V[ch]))} µV</td></tr>
            <tr><td>max p2p (all channels)</td><td>{fmt(uV(Math.max(...m.diff_p2p_V)))} µV</td></tr>
          </tbody></table>
        </div>
        <div className="card">
          <h4>Conduction velocity</h4>
          <table className="kv"><tbody>
            <tr><td>configured (input)</td><td><b>{fmt(cv.configured_m_s)} m/s</b></td></tr>
            <tr><td>estimated from signals</td><td>{cv.estimated_m_s != null ? <b>{fmt(cv.estimated_m_s)} m/s</b> : <span className="muted">{cv.note}</span>}</td></tr>
            {cv.channels && <tr><td>channels / distance</td><td>ch{cv.channels[0]} → ch{cv.channels[1]}, {fmt(mm(cv.distance_m ?? 0))} mm</td></tr>}
          </tbody></table>
          <div className="muted small">{cv.method}</div>
        </div>
        <div className="card">
          <h4>Solver</h4>
          <table className="kv"><tbody>
            <tr><td>mesh</td><td>{s.n_vertices.toLocaleString()} vertices, {s.n_tets.toLocaleString()} tets</td></tr>
            <tr><td>CG rtol / max residual</td><td>{s.cg_rtol.toExponential(0)} / {s.max_rel_residual.toExponential(2)}</td></tr>
            <tr><td>CG iterations / row</td><td>{fmt(s.rows.reduce((a, r) => a + r.iterations, 0) / s.rows.length, 4)}</td></tr>
            <tr><td>fibre current balance</td><td>{s.current_conservation.toExponential(1)}</td></tr>
            <tr><td>reuse</td><td>mesh {result.cache.mesh}, A {result.cache.stiffness}, adjoint rows {result.cache.adjoint_rows_reused} reused / {result.cache.adjoint_rows_solved} solved</td></tr>
            <tr><td>time</td><td>{fmt(result.timings.total_s)} s</td></tr>
          </tbody></table>
        </div>
        <div className="card">
          <h4>Provenance</h4>
          <table className="kv"><tbody>
            <tr><td>job</td><td>#{result.seq} {result.label ?? ''}</td></tr>
            <tr><td>model version</td><td><code>{result.model_version}</code></td></tr>
            <tr><td>config hash</td><td><code>{result.config_hash}</code></td></tr>
          </tbody></table>
          <div className="row">
            <a href={`/api/simulations/${result.job_id}/export.json`} download>JSON</a>
            <a href={`/api/simulations/${result.job_id}/export.csv`} download>CSV</a>
            <a href={`/api/simulations/${result.job_id}/config.json`} download>config</a>
          </div>
        </div>
      </div>
      {result.warnings.length > 0 && <div className="warn">{result.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}</div>}

      <div className="row">
        <span>Show:</span>
        <button className={mode === 'diff' ? 'active' : ''} onClick={() => setMode('diff')}>differential</button>
        <button className={mode === 'mono' ? 'active' : ''} onClick={() => setMode('mono')}>monopolar</button>
        <span className="muted small">click a trace to select a channel and set the time cursor</span>
      </div>
      <div className="plots">
        <div className="plot-tall">
          <Plot data={st.data} layout={st.layout} cursor={cursor} height={480}
            onClickPoint={(curve, x) => {
              if (mode === 'diff') setSelectedChannel(curve)
              const i = result.time_s.findIndex(t => ms(t) >= x)
              if (i >= 0) setTimeIndex(i)
            }} />
        </div>
        <div className="plot-col">
          <Plot data={single.data} layout={single.layout} cursor={cursor} height={235} />
          <Plot data={bars.data} layout={bars.layout} height={235} />
        </div>
      </div>

      <div className="card compare">
        <div className="row">
          <h4>Baseline comparison</h4>
          <button onClick={onSetBaseline} disabled={isBaselineCurrent}>Use this result as baseline</button>
          {baseline && <button onClick={onClearBaseline}>Clear baseline</button>}
        </div>
        {!baseline && <div className="muted">Pin a result as baseline, change parameters, run again, and compare here.</div>}
        {baseline && (
          <>
            <div className="muted small">baseline = job #{baseline.seq} (config {baseline.config_hash}); current = job #{result.seq} (config {result.config_hash})</div>
            {changes && changes.length === 0 && <div>No parameter differences.</div>}
            {changes && changes.length > 0 && (
              <table className="grid"><thead><tr><th>parameter</th><th>baseline</th><th>current</th></tr></thead>
                <tbody>{changes.map(c => <tr key={c.key}><td><code>{c.key}</code></td><td>{JSON.stringify(c.a)}</td><td>{JSON.stringify(c.b)}</td></tr>)}</tbody>
              </table>
            )}
            {bp && bp.diff_p2p_V.length === m.diff_p2p_V.length && (
              <table className="grid"><thead><tr><th>metric (channel {ch + 1})</th><th>baseline</th><th>current</th><th>change</th></tr></thead>
                <tbody>
                  <tr><td>p2p (µV)</td><td>{fmt(uV(bp.diff_p2p_V[ch]))}</td><td>{fmt(uV(m.diff_p2p_V[ch]))}</td><td>{pct(m.diff_p2p_V[ch], bp.diff_p2p_V[ch])}</td></tr>
                  <tr><td>RMS (µV)</td><td>{fmt(uV(bp.diff_rms_V[ch]))}</td><td>{fmt(uV(m.diff_rms_V[ch]))}</td><td>{pct(m.diff_rms_V[ch], bp.diff_rms_V[ch])}</td></tr>
                  <tr><td>estimated CV (m/s)</td><td>{fmt(baseline.conduction_velocity.estimated_m_s)}</td><td>{fmt(cv.estimated_m_s)}</td><td>{cv.estimated_m_s && baseline.conduction_velocity.estimated_m_s ? pct(cv.estimated_m_s, baseline.conduction_velocity.estimated_m_s) : '–'}</td></tr>
                </tbody>
              </table>
            )}
          </>
        )}
      </div>
    </div>
  )
}
