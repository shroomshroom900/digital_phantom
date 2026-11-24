import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'

interface Props {
  data: Partial<Plotly.Data>[]
  layout: Partial<Plotly.Layout>
  /** time cursor (x value) drawn as a vertical line; updated cheaply via relayout */
  cursor?: number | null
  height?: number
  onClickPoint?: (curveNumber: number, x: number) => void
}

const BASE: Partial<Plotly.Layout> = {
  margin: { l: 56, r: 12, t: 28, b: 40 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { size: 11, color: '#d5d9e0' },
  xaxis: { gridcolor: '#2c323c', zerolinecolor: '#3a414c' },
  yaxis: { gridcolor: '#2c323c', zerolinecolor: '#3a414c' },
  legend: { orientation: 'h', y: -0.22 },
}

function cursorShape(x: number): Partial<Plotly.Shape> {
  return { type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: '#ffd166', width: 1 } }
}

export default function Plot({ data, layout, cursor, height = 260, onClickPoint }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const cb = useRef(onClickPoint)
  cb.current = onClickPoint

  useEffect(() => {
    const el = ref.current!
    const l: Partial<Plotly.Layout> = {
      ...BASE, ...layout, height,
      xaxis: { ...BASE.xaxis, ...layout.xaxis }, yaxis: { ...BASE.yaxis, ...layout.yaxis },
      shapes: [...(layout.shapes ?? []), ...(cursor != null ? [cursorShape(cursor)] : [])],
    }
    Plotly.react(el, data as Plotly.Data[], l, { displaylogo: false, responsive: true })
    const anyEl = el as any
    anyEl.removeAllListeners?.('plotly_click')
    anyEl.on?.('plotly_click', (ev: any) => {
      const p = ev.points?.[0]
      if (p && cb.current) cb.current(p.curveNumber, p.x)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, layout, height])

  useEffect(() => {
    const el = ref.current as any
    if (!el?._fullLayout) return
    Plotly.relayout(el, { shapes: [...(layout.shapes ?? []), ...(cursor != null ? [cursorShape(cursor)] : [])] } as any)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursor])

  useEffect(() => () => { if (ref.current) Plotly.purge(ref.current) }, [])
  return <div ref={ref} style={{ width: '100%', height }} />
}
