import { useState } from 'react'
import type { Geometry } from '../api'
import { fmt } from '../units'
import type { LayerDisplay } from './Viewer3D'

export default function Legend({ geometry, display, setDisplay, showLabels, setShowLabels, onResetCamera }: {
  geometry: Geometry | null; display: Record<string, LayerDisplay>
  setDisplay: (key: string, d: Partial<LayerDisplay>) => void
  showLabels: boolean; setShowLabels: (b: boolean) => void; onResetCamera: () => void
}) {
  const [open, setOpen] = useState(true)
  if (!geometry) return <div className="legend muted">loading geometry…</div>
  if (!open) return (
    <div className="legend"><div className="row"><button onClick={() => setOpen(true)}>▸ Tissues &amp; legend</button>
      <button onClick={onResetCamera}>Reset camera</button></div></div>
  )
  const items = [
    ...geometry.tissues.map(t => ({ ...t, lines: false })),
    ...geometry.illustrations.map(i => ({ ...i, lines: true })),
  ]
  return (
    <div className="legend">
      <div className="row">
        <button className="link" onClick={() => setOpen(false)} title="collapse">▾</button><b>Tissues</b>
        <button onClick={onResetCamera}>Reset camera</button>
        <label className="small"><input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} /> labels</label>
      </div>
      {items.map(it => {
        const d = display[it.key] ?? { visible: false, opacity: 1 }
        return (
          <div key={it.key} className={`lrow${it.in_fem ? '' : ' illus'}`} title={`${it.status}\n${it.note}`}>
            <input type="checkbox" checked={d.visible} onChange={e => setDisplay(it.key, { visible: e.target.checked })} />
            <span className={`swatch${it.lines ? ' line' : ''}`} style={{ background: it.lines ? 'transparent' : it.color, borderColor: it.color }} />
            <span className="lname">{it.name}</span>
            <span className={`tag ${it.in_fem ? 'fem' : 'na'}`}>{it.in_fem ? 'FEM' : 'not in FEM'}</span>
            {!it.lines && (
              <input type="range" min={0.05} max={1} step={0.05} value={d.opacity}
                onChange={e => setDisplay(it.key, { opacity: Number(e.target.value) })} title="opacity" />
            )}
          </div>
        )
      })}
      <div className="muted small">
        Surfaces are the mesh's own staircase tissue boundaries ({geometry.mesh.n_vertices.toLocaleString()} vertices,
        h = {fmt(geometry.mesh.h_xy_m * 1e3)} × {fmt(geometry.mesh.h_z_m * 1e3)} mm). Concentric-cylinder phantom: no
        anatomical validation is implied. Axes: {geometry.coordinate_system.axes}.
      </div>
      <div className="muted small">
        Fibre (drawn 0.6 mm thick and on top of the tissues for visibility; model radius 25 µm): red = current leaving the fibre (source), blue = sink,
        white dots = wavefronts at z₀ ± v·t; yellow = NMJ; rings = tendon ends.
      </div>
    </div>
  )
}
