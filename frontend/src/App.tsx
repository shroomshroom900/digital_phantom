import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  api, setPath, stableStringify,
  type Geometry, type JobView, type Json, type ParamMeta, type Preview, type SimConfig, type SimResult, type UnsupportedMeta,
} from './api'
import Viewer3D, { type LayerDisplay } from './components/Viewer3D'
import ParamPanel from './components/ParamPanel'
import ElectrodePanel from './components/ElectrodePanel'
import Legend from './components/Legend'
import SignalDashboard, { type Change } from './components/SignalDashboard'
import Verification from './components/Verification'
import { fmt, ms } from './units'

const DEFAULT_DISPLAY: Record<string, LayerDisplay> = {
  skin: { visible: true, opacity: 0.18 }, fat: { visible: true, opacity: 0.22 },
  muscle: { visible: true, opacity: 0.3 }, bone: { visible: true, opacity: 1 },
  biceps: { visible: false, opacity: 1 }, brachialis: { visible: false, opacity: 1 },
  triceps: { visible: false, opacity: 1 }, stage4_belly: { visible: false, opacity: 1 },
}

function leafDiff(a: any, b: any, prefix = '', out: string[] = []): string[] {
  const keys = new Set([...Object.keys(a ?? {}), ...Object.keys(b ?? {})])
  for (const k of keys) {
    const va = a?.[k], vb = b?.[k], key = prefix + k
    if (va && vb && typeof va === 'object' && typeof vb === 'object') leafDiff(va, vb, key + '.', out)
    else if (stableStringify(va) !== stableStringify(vb)) out.push(key)
  }
  return out
}

export default function App() {
  const [tab, setTab] = useState<'phantom' | 'verification' | 'model'>('phantom')
  const [params, setParams] = useState<ParamMeta[]>([])
  const [unsupported, setUnsupported] = useState<UnsupportedMeta[]>([])
  const [model, setModel] = useState<Json | null>(null)
  const [defaults, setDefaults] = useState<SimConfig | null>(null)
  const [config, setConfig] = useState<SimConfig | null>(null)
  const [bootErr, setBootErr] = useState<string | null>(null)

  const [preview, setPreview] = useState<Preview | null>(null)
  const [previewErr, setPreviewErr] = useState<string | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [geometry, setGeometry] = useState<Geometry | null>(null)
  const geomKey = useRef<string | null>(null)

  const [display, setDisplayState] = useState(DEFAULT_DISPLAY)
  const [showLabels, setShowLabels] = useState(false)
  const [resetCam, setResetCam] = useState(0)
  const [selectedElectrode, setSelectedElectrode] = useState<number | null>(null)
  const [placing, setPlacing] = useState(false)
  const [selectedChannel, setSelectedChannel] = useState<number | null>(8)

  const [pending, setPending] = useState<JobView[]>([])        // submitted, not finished
  const [lastJob, setLastJob] = useState<JobView | null>(null) // newest submitted job (status display)
  const [runErr, setRunErr] = useState<string | null>(null)
  const [result, setResult] = useState<SimResult | null>(null)
  const [baseline, setBaseline] = useState<SimResult | null>(null)
  const [changes, setChanges] = useState<Change[] | null>(null)
  const latestSubmitted = useRef(0)

  const [timeIndex, setTimeIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(5)                       // simulated ms per real second

  // ------------------------------------------------------------------ boot
  useEffect(() => {
    Promise.all([api.parameters(), api.defaultConfig(), api.model()])
      .then(([p, d, m]) => { setParams(p.parameters); setUnsupported(p.unsupported); setDefaults(d); setConfig(d); setModel(m) })
      .catch(e => setBootErr(`Cannot reach the backend (${e.message}). Start it with:  python -m app`))
  }, [])

  // ------------------------------------------------------------------ preview (debounced, stale-safe)
  const previewSeq = useRef(0)
  const resultSeq = result?.seq ?? 0          // re-preview after each result so the reuse plan is current
  useEffect(() => {
    if (!config) return
    const my = ++previewSeq.current
    setPreviewing(true)
    const h = setTimeout(() => {
      api.preview(config)
        .then(p => { if (my === previewSeq.current) { setPreview(p); setPreviewErr(null) } })
        .catch(e => { if (my === previewSeq.current) { setPreviewErr(e.message); setPreview(null) } })
        .finally(() => { if (my === previewSeq.current) setPreviewing(false) })
    }, 300)
    return () => clearTimeout(h)
  }, [config, resultSeq])

  useEffect(() => {
    const key = preview?.mesh_key
    if (!key || key === geomKey.current || !preview?.mesh) return
    geomKey.current = key
    api.geometry(key).then(g => { if (geomKey.current === key) setGeometry(g) }).catch(e => setPreviewErr(e.message))
  }, [preview])

  // ------------------------------------------------------------------ jobs
  const run = async () => {
    if (!config) return
    setRunErr(null)
    try {
      const j = await api.submit(config)
      latestSubmitted.current = j.seq
      setLastJob(j)
      setPending(p => [...p, j])          // older queued jobs come back 'superseded' and drop out
    } catch (e: any) { setRunErr(e.message) }
  }

  useEffect(() => {
    if (pending.length === 0) return
    const h = setInterval(async () => {
      const views = await Promise.all(pending.map(j => api.job(j.id).catch(() => null)))
      const still: JobView[] = []
      for (const v of views) {
        if (!v) continue
        if (v.seq === latestSubmitted.current) setLastJob(v)
        if (v.status === 'succeeded' && v.result) {
          const r = v.result
          // never let an older job overwrite a newer displayed result
          setResult(cur => (!cur || r.seq > cur.seq ? r : cur))
          if (v.seq === latestSubmitted.current) setRunErr(null)
        } else if (v.status === 'failed') {
          if (v.seq === latestSubmitted.current) setRunErr(`${v.error?.stage}: ${v.error?.message}`)
        } else if (v.status === 'queued' || v.status === 'running') still.push(v)
      }
      setPending(still)
    }, 400)
    return () => clearInterval(h)
  }, [pending])

  useEffect(() => { if (result) setTimeIndex(i => Math.min(i, result.time_s.length - 1)) }, [result])

  useEffect(() => {
    if (!baseline || !result) { setChanges(null); return }
    api.diff(baseline.config, result.config).then(d => setChanges(d.changes)).catch(() => setChanges(null))
  }, [baseline, result])

  // ------------------------------------------------------------------ animation (simulated time = backend time axis)
  useEffect(() => {
    if (!playing || !result) return
    let raf = 0, last = performance.now()
    let tSim = result.time_s[timeIndex] ?? 0
    const fs = result.metrics.fs_Hz, n = result.time_s.length
    const step = (now: number) => {
      tSim += ((now - last) / 1000) * speed * 1e-3
      last = now
      let i = Math.floor(tSim * fs + 1e-9)
      if (i >= n) { tSim = 0; i = 0 }
      setTimeIndex(i)
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, result, speed])

  // ------------------------------------------------------------------ config edits
  const change = useCallback((key: string, v: any) => setConfig(c => (c ? setPath(c, key, v) : c)), [])
  const setOverride = (k: number, theta: number, z: number) => {
    const th = Math.atan2(Math.sin(theta), Math.cos(theta))
    change(`electrodes.overrides.${k}`, { theta: th, z })
  }
  const clearOverride = (k: number) => setConfig(c => {
    if (!c) return c
    const o = { ...c.electrodes.overrides }; delete o[String(k)]
    return setPath(c, 'electrodes.overrides', o)
  })
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') setPlacing(false) }
    window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h)
  }, [])
  useEffect(() => {   // array size change: drop overrides that no longer exist
    if (!config) return
    const bad = Object.keys(config.electrodes.overrides).filter(k => Number(k) >= config.electrodes.array.n)
    if (bad.length) setConfig(c => {
      const o = { ...c!.electrodes.overrides }; bad.forEach(k => delete o[k]); return setPath(c!, 'electrodes.overrides', o)
    })
    if (selectedElectrode != null && selectedElectrode >= config.electrodes.array.n) setSelectedElectrode(null)
    if (selectedChannel != null && selectedChannel >= config.electrodes.array.n - 1) setSelectedChannel(null)
  }, [config, selectedElectrode, selectedChannel])

  const dirtyKeys = useMemo(() => new Set(result && config ? leafDiff(result.config, config) : []), [result, config])
  const dirty = !!result && dirtyKeys.size > 0
  const busy = pending.length > 0
  const canRun = !!config && !!preview?.valid && !previewing

  if (bootErr) return <div className="boot err">{bootErr}</div>
  if (!config || !defaults) return <div className="boot">Loading model…</div>

  const plan = preview?.plan
  return (
    <div className="app">
      <header>
        <div className="title">
          <b>emg_twin</b> upper-arm digital phantom
          <span className="muted small"> · single-fibre sEMG (Stages 1–3) · model {preview?.model_version ?? model?.model_version}</span>
        </div>
        <nav>
          {(['phantom', 'verification', 'model'] as const).map(t => (
            <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>
              {t === 'phantom' ? 'Phantom' : t === 'verification' ? 'Verification' : 'Model & stages'}
            </button>
          ))}
        </nav>
      </header>

      {tab === 'verification' && <div className="page"><Verification /></div>}
      {tab === 'model' && model && <ModelPage model={model} unsupported={unsupported} />}

      <div className="main" style={{ display: tab === 'phantom' ? 'grid' : 'none' }}>
        <aside className="left">
          <ParamPanel params={params} unsupported={unsupported} config={config} onChange={change} changedKeys={dirtyKeys} />
          <button className="wide" onClick={() => setConfig(defaults)}>Reset all parameters to defaults</button>
        </aside>

        <section className="center">
          <div className="runbar">
            <button className="primary" disabled={!canRun} onClick={run}
              title={!preview?.valid ? 'fix the validation errors first' : ''}>▶ Run simulation</button>
            <div className="status">
              {busy && <span className="running">● running: {lastJob?.progress}</span>}
              {!busy && runErr && <span className="err">✖ {runErr}</span>}
              {!busy && !runErr && result && !dirty && <span className="ok">✔ result #{result.seq} matches the controls</span>}
              {!busy && dirty && <span className="dirtytxt">◆ controls changed since result #{result!.seq} ({dirtyKeys.size} value{dirtyKeys.size > 1 ? 's' : ''}) – run again</span>}
              {!busy && !result && !runErr && <span className="muted">no result yet</span>}
            </div>
            {plan && (
              <div className="plan small muted" title="what the next run will reuse">
                next run: mesh {plan.mesh}, A {plan.stiffness}, adjoint {plan.adjoint_rows_cached} cached / {plan.adjoint_rows_to_solve} to solve
                {plan.adjoint_rows_to_solve > 0 && ` (~${Math.round(plan.adjoint_rows_to_solve * (preview!.mesh!.n_vertices / 116000) * 1.1)} s)`}
              </div>
            )}
          </div>
          {(previewErr || (preview && (preview.errors.length > 0 || preview.warnings.length > 0))) && (
            <div className="messages">
              {previewErr && <div className="err">✖ {previewErr}</div>}
              {preview?.errors.map((e, i) => <div key={i} className="err">✖ {e}</div>)}
              {preview?.warnings.map((w, i) => <div key={i} className="warn">⚠ {w}</div>)}
            </div>
          )}
          <div className="viewer">
            <Viewer3D geometry={geometry} display={display} electrodes={preview?.electrodes} fibre={preview?.fibre}
              result={result} timeIndex={timeIndex} selectedElectrode={selectedElectrode}
              onSelectElectrode={setSelectedElectrode} placing={placing && selectedElectrode != null}
              onPlace={(th, z) => { if (selectedElectrode != null) setOverride(selectedElectrode, th, z); setPlacing(false) }}
              selectedChannel={selectedChannel} showLabels={showLabels} resetSignal={resetCam} />
            <div className="overlay">
              <Legend geometry={geometry} display={display}
                setDisplay={(k, d) => setDisplayState(s => ({ ...s, [k]: { ...s[k], ...d } }))}
                showLabels={showLabels} setShowLabels={setShowLabels} onResetCamera={() => setResetCam(x => x + 1)} />
            </div>
            {placing && selectedElectrode != null && <div className="placehint">Click on the skin to move e{selectedElectrode + 1} · Esc cancels</div>}
            {previewing && <div className="busyhint">updating…</div>}
          </div>
          {result && (
            <div className="timebar">
              <button onClick={() => setPlaying(p => !p)}>{playing ? '❚❚' : '▶'}</button>
              <input type="range" min={0} max={result.time_s.length - 1} value={timeIndex}
                onChange={e => { setPlaying(false); setTimeIndex(Number(e.target.value)) }} />
              <span className="mono">t = {fmt(ms(result.time_s[timeIndex]), 4)} ms</span>
              <label className="small">speed <select value={speed} onChange={e => setSpeed(Number(e.target.value))}>
                {[1, 2, 5, 10, 30].map(s => <option key={s} value={s}>{s} ms/s</option>)}
              </select></label>
              <span className="muted small">animation = result #{result.seq} (v = {result.config.fibre.cv} m/s, same time axis as the plots)</span>
            </div>
          )}
          {result
            ? <SignalDashboard result={result} baseline={baseline} timeIndex={timeIndex} setTimeIndex={setTimeIndex}
                selectedChannel={selectedChannel} setSelectedChannel={setSelectedChannel} changes={changes}
                onSetBaseline={() => setBaseline(result)} onClearBaseline={() => setBaseline(null)}
                isBaselineCurrent={baseline?.job_id === result.job_id} />
            : <div className="card muted">Press <b>Run simulation</b> to compute the electrode signals with the FEM model.
                The first run on a new mesh solves one adjoint problem per electrode (~1 s each on the default mesh); later runs reuse them.</div>}
        </section>

        <aside className="right">
          <h3>Electrodes</h3>
          <ElectrodePanel config={config} preview={preview} selected={selectedElectrode} onSelect={setSelectedElectrode}
            placing={placing} setPlacing={setPlacing} onOverride={setOverride} onClearOverride={clearOverride}
            onResetAll={() => change('electrodes', defaults.electrodes)} selectedChannel={selectedChannel}
            onSelectChannel={setSelectedChannel} />
          <h3>Stage 4 · motor units</h3>
          <div className="card muted small">
            Motor-unit pools, recruitment and combined muscle activity exist as scripts (<code>muscle.py</code>,
            <code> stage4_muscle.py</code>) but are <b>not yet integrated</b> in this app. Motor-unit membership will be
            shown on the fibres once the backend supplies it (<code>POST /api/simulations/future</code> is reserved).
          </div>
          {preview?.mesh && (
            <div className="small muted">mesh: {preview.mesh.n_vertices.toLocaleString()} vertices, {preview.mesh.n_tets.toLocaleString()} tets · config {preview.config_hash}</div>
          )}
        </aside>
      </div>
    </div>
  )
}

function ModelPage({ model, unsupported }: { model: Json; unsupported: UnsupportedMeta[] }) {
  return (
    <div className="page">
      <section className="card">
        <h3>Numerical model</h3>
        <p>{model.reference}</p>
        <table className="kv"><tbody>
          {Object.entries(model.formulation).map(([k, v]) => <tr key={k}><td>{k}</td><td>{String(v)}</td></tr>)}
          <tr><td>coordinates</td><td>{model.coordinate_system}</td></tr>
          <tr><td>units</td><td>{Object.entries(model.units).map(([k, v]) => `${k}: ${v}`).join(', ')}</td></tr>
          <tr><td>model version</td><td><code>{model.model_version}</code> (hash of config.py, mesh.py, fem.py, fibre.py + app {model.app_version}); git {model.git_commit ?? '–'}</td></tr>
        </tbody></table>
      </section>
      <section className="card">
        <h3>Stages</h3>
        <table className="grid"><thead><tr><th>stage</th><th>content</th><th>status in app</th><th>code</th></tr></thead>
          <tbody>{model.stages.map((s: Json) => (
            <tr key={s.stage}><td>{s.stage}</td><td>{s.name}</td>
              <td className={s.status === 'integrated' ? 'ok' : 'warn'}>{s.status}{s.integration_point ? <div className="small muted">{s.integration_point}</div> : null}</td>
              <td><code>{s.code.join(', ')}</code></td></tr>
          ))}</tbody>
        </table>
      </section>
      <section className="card">
        <h3>Tissues</h3>
        <table className="grid"><tbody>{model.tissues.map((t: Json) => <tr key={t.key}><td>{t.name}</td><td>{t.in_fem ? 'in FEM' : 'not in FEM'}</td><td>{t.note}</td></tr>)}</tbody></table>
      </section>
      <section className="card">
        <h3>Not available in the current solver</h3>
        <table className="grid"><tbody>{unsupported.map(u => <tr key={u.key}><td>{u.label}</td><td>{u.reason}</td></tr>)}</tbody></table>
      </section>
    </div>
  )
}
