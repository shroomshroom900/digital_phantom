import { useEffect, useState } from 'react'
import type { ParamMeta, SimConfig, UnsupportedMeta } from '../api'
import { getPath } from '../api'
import { fmt } from '../units'

const RECOMPUTE_LABEL: Record<string, string> = {
  mesh: 'remesh + A + adjoint',
  stiffness: 'reassemble A + adjoint',
  adjoint: 'adjoint rows (changed electrodes)',
  basis: 'fibre basis points (ms)',
  signal: 'signal only (ms)',
  display: 'display only',
}

function NumberField({ meta, value, onChange }: { meta: ParamMeta; value: number; onChange: (v: number) => void }) {
  const scale = meta.display_scale
  const [text, setText] = useState(fmt(value * scale, 6))
  useEffect(() => { setText(fmt(value * scale, 6)) }, [value, scale])
  const commit = (s: string) => {
    const v = Number(s)
    if (s.trim() === '' || !Number.isFinite(v)) { setText(fmt(value * scale, 6)); return }
    const si = meta.integer ? Math.round(v) : v / scale
    onChange(si)
  }
  if (meta.key === 'solver.cg_rtol') {
    return (
      <select value={value} onChange={e => onChange(Number(e.target.value))}>
        {[1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12].map(v => <option key={v} value={v}>{v.toExponential(0)}</option>)}
      </select>
    )
  }
  const out = value < meta.min || value > meta.max
  return (
    <div className="numfield">
      <input type="range" min={meta.min * scale} max={meta.max * scale}
        step={meta.step ? meta.step * scale : 'any'} value={value * scale}
        onChange={e => onChange(meta.integer ? Math.round(Number(e.target.value)) : Number(e.target.value) / scale)} />
      <input type="text" className={out ? 'bad' : ''} value={text}
        onChange={e => setText(e.target.value)} onBlur={e => commit(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter') commit((e.target as HTMLInputElement).value) }} />
      <span className="unit">{meta.display_unit}</span>
    </div>
  )
}

export default function ParamPanel({ params, unsupported, config, onChange, changedKeys }: {
  params: ParamMeta[]; unsupported: UnsupportedMeta[]; config: SimConfig
  onChange: (key: string, v: number) => void; changedKeys: Set<string>
}) {
  const groups = [...new Set([...params.map(p => p.group), ...unsupported.map(u => u.group)])]
  const [open, setOpen] = useState<Record<string, boolean>>({ Geometry: true, Conductivity: true, Fibre: true })
  return (
    <div className="params">
      {groups.map(g => (
        <section key={g} className="pgroup">
          <h3 onClick={() => setOpen(o => ({ ...o, [g]: !o[g] }))}>{open[g] ? '▾' : '▸'} {g}</h3>
          {open[g] && (
            <>
              {params.filter(p => p.group === g).map(p => {
                const v = getPath(config, p.key) as number
                const changed = v !== p.default
                return (
                  <div key={p.key} className={`param${changedKeys.has(p.key) ? ' dirty' : ''}`}>
                    <div className="plabel">
                      <span title={p.note}>{p.label}</span>
                      <span className={`tag ${p.provenance.slice(1, -1).toLowerCase()}`} title={p.provenance_note}>
                        {p.provenance}
                      </span>
                    </div>
                    <NumberField meta={p} value={v} onChange={nv => onChange(p.key, nv)} />
                    <div className="pmeta">
                      <span className="rc" title="what a change forces the pipeline to recompute">
                        ↻ {RECOMPUTE_LABEL[p.recompute]}
                      </span>
                      {changed && (
                        <button className="link" onClick={() => onChange(p.key, p.default)}
                          title={`default ${fmt(p.default * p.display_scale, 6)} ${p.display_unit}`}>reset</button>
                      )}
                    </div>
                  </div>
                )
              })}
              {unsupported.filter(u => u.group === g).map(u => (
                <div key={u.key} className="param unsupported" title={u.reason}>
                  <div className="plabel"><span>{u.label}</span><span className="tag na">not available</span></div>
                  <div className="reason">{u.reason}</div>
                </div>
              ))}
            </>
          )}
        </section>
      ))}
    </div>
  )
}
