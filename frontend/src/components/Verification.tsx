import { useEffect, useState } from 'react'
import { api, type Json } from '../api'
import { fmt } from '../units'
import Plot from './Plot'

function ParamInput({ p, value, onChange }: { p: Json; value: any; onChange: (v: any) => void }) {
  const scale = p.display_scale ?? 1
  if (Array.isArray(p.default)) {
    const rows: Json[] = value
    const cols: Json[] = p.columns
    return (
      <div>
        <table className="grid">
          <thead><tr>{cols.map(c => <th key={c.key}>{c.label}{c.display_unit ? ` (${c.display_unit})` : ''}</th>)}<th /></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {cols.map(c => (
                  <td key={c.key}>
                    <input type={c.unit ? 'number' : 'text'} value={c.unit ? r[c.key] * (c.display_scale ?? 1) : r[c.key]}
                      onChange={e => {
                        const nr = rows.map(x => ({ ...x }))
                        nr[i][c.key] = c.unit ? Number(e.target.value) / (c.display_scale ?? 1) : e.target.value
                        onChange(nr)
                      }} />
                  </td>
                ))}
                <td><button className="link" onClick={() => onChange(rows.filter((_, j) => j !== i))}>remove</button></td>
              </tr>
            ))}
          </tbody>
        </table>
        <button onClick={() => onChange([...rows, { name: `layer ${rows.length + 1}`, length: 2e-3, sigma: 0.1 }])}>Add layer</button>
      </div>
    )
  }
  return (
    <span>
      <input type="number" value={Number((value * scale).toPrecision(8))}
        onChange={e => onChange(p.integer ? Math.round(Number(e.target.value)) : Number(e.target.value) / scale)} />
      {' '}{p.display_unit}
    </span>
  )
}

function Experiment({ exp }: { exp: Json }) {
  const init = Object.fromEntries(exp.params.map((p: Json) => [p.key, structuredClone(p.default)]))
  const [params, setParams] = useState<Json>(init)
  const [res, setRes] = useState<Json | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true); setErr(null)
    try { setRes(await api.runVerification(exp.name, params)) } catch (e: any) { setErr(e.message); setRes(null) }
    setBusy(false)
  }
  return (
    <section className="card vexp">
      <h3>{exp.title}</h3>
      <div className="formula"><code>{exp.formula}</code></div>
      {exp.assumptions?.length > 0 && <ul className="small muted">{exp.assumptions.map((a: string) => <li key={a}>{a}</li>)}</ul>}
      <table className="kv"><tbody>
        {exp.params.map((p: Json) => (
          <tr key={p.key}><td>{p.label}</td><td><ParamInput p={p} value={params[p.key]} onChange={v => setParams(s => ({ ...s, [p.key]: v }))} /></td></tr>
        ))}
      </tbody></table>
      <div className="row">
        <button className="primary" onClick={run} disabled={busy}>{busy ? 'Running…' : 'Run check'}</button>
        <button onClick={() => { setParams(init); setRes(null) }}>Defaults</button>
      </div>
      {err && <div className="err">✖ {err}</div>}
      {res && (
        <div>
          <div className={res.passed ? 'ok big' : 'err big'}>
            {res.passed ? '✔ PASSED' : '✖ FAILED'} <span className="muted small">(tolerance {res.tolerance}, {fmt(res.runtime_s)} s)</span>
          </div>
          <table className="grid"><tbody>
            {res.summary.map((r: Json) => <tr key={r.quantity}><td>{r.quantity}</td><td>{typeof r.value === 'number' ? fmt(r.value, 8) : String(r.value)}</td><td>{r.unit}</td></tr>)}
          </tbody></table>
          {res.layers && (
            <table className="grid">
              <thead><tr><th>layer</th><th>L nominal (mm)</th><th>L used (mm)</th><th>σ (S/m)</th><th>R (Ω)</th><th>ΔV analytical (V)</th><th>ΔV FEM (V)</th><th>rel. diff</th></tr></thead>
              <tbody>{res.layers.map((l: Json) => (
                <tr key={l.layer}><td>{l.layer}</td><td>{fmt(l.length_nominal_m * 1e3, 4)}</td><td>{fmt(l.length_effective_m * 1e3, 4)}</td>
                  <td>{fmt(l.sigma_S_m)}</td><td>{fmt(l.R_ohm, 6)}</td><td>{fmt(l.dV_analytic_V, 8)}</td><td>{fmt(l.dV_fem_V, 8)}</td><td>{fmt(l.rel_diff, 2)}</td></tr>
              ))}</tbody>
            </table>
          )}
          {res.profile && (
            <Plot height={260} data={res.profile.series.map((s: Json, i: number) => ({
              x: res.profile.x.map((v: number) => v * 1e3), y: s.y, name: s.name, type: 'scatter',
              mode: i === 0 ? 'markers' : 'lines', marker: { size: 4 },
            }))} layout={{ title: { text: 'Potential along the stack', font: { size: 12 } }, xaxis: { title: { text: 'z (mm), bottom face = 0' } }, yaxis: { title: { text: 'V' } } } as any} />
          )}
          {res.notes && <ul className="small muted">{res.notes.map((n: string) => <li key={n}>{n}</li>)}</ul>}
        </div>
      )}
    </section>
  )
}

export default function Verification() {
  const [list, setList] = useState<Json[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { api.verificationList().then(r => setList(r.experiments)).catch(e => setErr(e.message)) }, [])
  return (
    <div className="verification">
      <p className="muted">
        Each check compares the project's FEM (fem.assemble_stiffness / fem.cg_solve / fem.adjoint_K) against an
        analytical or independent reference. New formulas are added as registered experiments in
        <code> app/verification/</code>.
      </p>
      {err && <div className="err">{err}</div>}
      {list?.map(e => <Experiment key={e.name} exp={e} />)}
    </div>
  )
}
