// Typed client for the FastAPI backend (app/server.py).  All numbers are SI.

export type Json = Record<string, any>

export interface SurfacePoint { theta: number; z: number }
export interface SimConfig {
  kind: 'single_fibre'
  geometry: { bone_radius: number; muscle_radius: number; fat_thickness: number; skin_thickness: number; length: number }
  mesh: { h_xy: number; h_z: number }
  conductivity: { bone: number; muscle_transverse: number; muscle_longitudinal: number; fat: number; skin: number }
  fibre: { depth: number; theta: number; nmj_z: number; l1: number; l2: number; cv: number; taper: number; radius: number; sigma_in: number; dz: number }
  electrodes: {
    array: { n: number; spacing: number; theta_center: number; z_center: number; orientation: number }
    overrides: Record<string, SurfacePoint>
    pairing: 'consecutive'
  }
  timing: { duration: number; fs: number }
  solver: { cg_rtol: number }
}

export interface ParamMeta {
  key: string; label: string; unit: string; display_unit: string; display_scale: number
  min: number; max: number; step: number | null; default: number
  provenance: '[PAPER]' | '[PROMPT]' | '[ASSUMED]'; provenance_note: string; note: string
  supported: boolean; recompute: string; group: string; integer: boolean
}
export interface UnsupportedMeta { key: string; label: string; group: string; reason: string }

export interface ElectrodeBlock {
  positions_m: number[][]
  surface: SurfacePoint[]
  inset_below_nominal_skin_m: number[]
  tissue: string[]
  pairs: [number, number][]
  channel_labels: string[]
  sampling: string
}
export interface FibreBlock {
  x_m: number; y_m: number; stored_z_m: [number, number]; active_z_m: [number, number]; nmj_z_m: number
  cell_centres_z_m: number[]; dz_m: number; cv_m_s: number; radius_m: number
  cell_current_A?: number[][] | null
}
export interface Preview {
  valid: boolean; errors: string[]; warnings: string[]; mesh_key: string; estimated_vertices: number
  model_version: string; config_hash: string
  mesh?: { n_vertices: number; n_tets: number }
  electrodes?: ElectrodeBlock
  fibre?: FibreBlock
  plan?: { mesh: string; stiffness: string; adjoint_rows_cached: number; adjoint_rows_to_solve: number; per_electrode_cached: boolean[] }
}
export interface TissueSurface {
  key: string; name: string; color: string; in_fem: boolean; note: string; status: string
  n_triangles: number; volume_m3: number; positions_m: number[]
}
export interface Illustration { key: string; name: string; color: string; in_fem: boolean; status: string; note: string; lines_m: number[] }
export interface Geometry {
  coordinate_system: { axes: string; unit: string }
  bounds_m: { x: number[]; y: number[]; z: number[] }
  radii_m: Record<string, number>
  mesh: { n_vertices: number; n_tets: number; h_xy_m: number; h_z_m: number; element: string }
  tissues: TissueSurface[]
  illustrations: Illustration[]
}
export interface SimResult {
  job_id: string; seq: number; label: string | null
  kind: string; model_version: string; config: SimConfig; config_hash: string
  time_s: number[]; mono_V: number[][]; diff_V: number[][]
  metrics: { mono_p2p_V: number[]; mono_rms_V: number[]; diff_p2p_V: number[]; diff_rms_V: number[]; duration_s: number; fs_Hz: number; n_samples: number }
  conduction_velocity: { configured_m_s: number; estimated_m_s: number | null; method: string; note: string; channels?: number[]; distance_m?: number; lag_s?: number }
  electrodes: ElectrodeBlock
  fibre: FibreBlock & { cell_current_note: string }
  solver: { method: string; cg_rtol: number; n_vertices: number; n_tets: number; n_basis_nodes: number; max_rel_residual: number; current_conservation: number; rows: { electrode: number; cached: boolean; iterations: number; rel_residual: number }[] }
  cache: { mesh: string; stiffness: string; adjoint_rows_reused: number; adjoint_rows_solved: number; basis_points: string; signal: string }
  timings: Record<string, number>
  warnings: string[]
}
export interface JobView {
  id: string; seq: number; label: string | null; status: 'queued' | 'running' | 'succeeded' | 'failed' | 'superseded'
  progress: string; config_hash: string; error: { message: string; stage: string } | null
  latest_seq: number; result?: SimResult
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...init })
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const body = await r.json()
      if (typeof body.detail === 'string') msg = body.detail
      else if (Array.isArray(body.detail))
        msg = body.detail.map((d: any) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
    } catch { /* not JSON */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

export const api = {
  model: () => req<Json>('/api/model'),
  parameters: () => req<{ parameters: ParamMeta[]; unsupported: UnsupportedMeta[]; recompute_levels: string[] }>('/api/parameters'),
  defaultConfig: () => req<SimConfig>('/api/config/default'),
  preview: (c: SimConfig) => req<Preview>('/api/preview', { method: 'POST', body: JSON.stringify(c) }),
  geometry: (key: string) => req<Geometry>(`/api/geometry/${key}`),
  submit: (c: SimConfig, label?: string) =>
    req<JobView>('/api/simulations', { method: 'POST', body: JSON.stringify({ config: c, label }) }),
  job: (id: string) => req<JobView>(`/api/simulations/${id}`),
  diff: (a: SimConfig, b: SimConfig) =>
    req<{ changes: { key: string; a: any; b: any }[]; recompute: string | null }>('/api/config/diff', { method: 'POST', body: JSON.stringify({ a, b }) }),
  verificationList: () => req<{ experiments: Json[] }>('/api/verification'),
  runVerification: (name: string, params: Json) =>
    req<Json>(`/api/verification/${name}`, { method: 'POST', body: JSON.stringify(params) }),
}

// ---------------------------------------------------------------- config helpers
export function getPath(obj: any, key: string): any {
  return key.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj)
}
export function setPath<T>(obj: T, key: string, value: any): T {
  const parts = key.split('.')
  const out: any = structuredClone(obj)
  let o = out
  for (const p of parts.slice(0, -1)) o = o[p]
  o[parts[parts.length - 1]] = value
  return out
}
/** Key-order-independent JSON, used to decide whether controls differ from a result's config. */
export function stableStringify(x: any): string {
  if (Array.isArray(x)) return `[${x.map(stableStringify).join(',')}]`
  if (x && typeof x === 'object')
    return `{${Object.keys(x).sort().map(k => `${JSON.stringify(k)}:${stableStringify(x[k])}`).join(',')}}`
  return JSON.stringify(x)
}
