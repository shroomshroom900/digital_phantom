// The ONE place where SI values from the API are converted for display.
// API: metres, seconds, volts, amperes, radians.  Scene: millimetres (model axes, no swap).

export const M_TO_MM = 1e3
export const S_TO_MS = 1e3
export const V_TO_UV = 1e6
export const RAD_TO_DEG = 180 / Math.PI

export const mm = (m: number) => m * M_TO_MM
export const ms = (s: number) => s * S_TO_MS
export const uV = (v: number) => v * V_TO_UV
export const deg = (rad: number) => rad * RAD_TO_DEG
export const rad = (d: number) => d / RAD_TO_DEG

/** Format a number with a sensible number of significant digits. */
export function fmt(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '–'
  if (x === 0) return '0'
  const a = Math.abs(x)
  if (a >= 1e5 || a < 1e-3) return x.toExponential(Math.max(1, digits - 1))
  return Number(x.toPrecision(digits)).toString()
}
