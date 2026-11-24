import type { Preview, SimConfig } from '../api'
import { deg, fmt, mm, rad } from '../units'

export default function ElectrodePanel({ config, preview, selected, onSelect, placing, setPlacing,
  onOverride, onClearOverride, onResetAll, selectedChannel, onSelectChannel }: {
  config: SimConfig; preview: Preview | null; selected: number | null; onSelect: (k: number | null) => void
  placing: boolean; setPlacing: (b: boolean) => void
  onOverride: (k: number, theta: number, z: number) => void; onClearOverride: (k: number) => void
  onResetAll: () => void; selectedChannel: number | null; onSelectChannel: (j: number | null) => void
}) {
  const el = preview?.electrodes
  const n = config.electrodes.array.n
  const overrides = config.electrodes.overrides
  const sel = selected != null && el ? selected : null
  return (
    <div className="electrodes">
      <div className="row">
        <button onClick={onResetAll} title="Restore the default 16-electrode array and remove all moved electrodes">
          Reset electrodes
        </button>
        <span className="muted">{Object.keys(overrides).length} moved individually</span>
      </div>
      <p className="muted small">
        Point electrodes ({el?.sampling ?? 'barycentric P1 sampling'}). Discs in 3-D are drawn for visibility only;
        contact area is not a solver input.
      </p>
      <div className="egrid">
        {Array.from({ length: n }, (_, k) => {
          const tissue = el?.tissue[k]
          return (
            <button key={k} className={`echip${k === selected ? ' sel' : ''}${overrides[String(k)] ? ' moved' : ''}${tissue && tissue !== 'skin' ? ' bad' : ''}`}
              onClick={() => onSelect(k === selected ? null : k)}>e{k + 1}</button>
          )
        })}
      </div>
      {sel != null && el && (
        <div className="ebox">
          <div className="row">
            <b>Electrode e{sel + 1}</b>
            <span className={el.tissue[sel] === 'skin' ? 'ok' : 'err'}>on {el.tissue[sel]}</span>
          </div>
          <table className="kv">
            <tbody>
              <tr><td>θ (around arm)</td><td>
                <input type="number" step={1} value={Number(deg(el.surface[sel].theta).toFixed(2))}
                  onChange={e => onOverride(sel, rad(Number(e.target.value)), el.surface[sel].z)} /> deg</td></tr>
              <tr><td>z (along arm)</td><td>
                <input type="number" step={0.5} value={Number(mm(el.surface[sel].z).toFixed(2))}
                  onChange={e => onOverride(sel, el.surface[sel].theta, Number(e.target.value) / 1e3)} /> mm</td></tr>
              <tr><td>x, y, z (solver)</td><td>{el.positions_m[sel].map(v => fmt(mm(v), 4)).join(', ')} mm</td></tr>
              <tr><td>inset below nominal skin</td><td>{fmt(mm(el.inset_below_nominal_skin_m[sel]), 3)} mm
                <span className="muted"> (staircase surface)</span></td></tr>
            </tbody>
          </table>
          <div className="row">
            <button className={placing ? 'active' : ''} onClick={() => setPlacing(!placing)}>
              {placing ? 'Click the skin… (Esc to cancel)' : 'Place on skin'}
            </button>
            {overrides[String(sel)] && <button onClick={() => onClearOverride(sel)}>Back to array position</button>}
          </div>
        </div>
      )}
      <h4>Differential channels <span className="muted small">V = φ₁ − φ₂</span></h4>
      <div className="chlist">
        {el?.channel_labels.map((lab, j) => (
          <button key={j} className={`chchip${j === selectedChannel ? ' sel' : ''}`}
            onClick={() => onSelectChannel(j === selectedChannel ? null : j)}>{lab}</button>
        ))}
      </div>
    </div>
  )
}
