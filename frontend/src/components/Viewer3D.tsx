import { useEffect, useMemo, useRef } from 'react'
import { Canvas, type ThreeEvent } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, Html, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { ElectrodeBlock, FibreBlock, Geometry, SimResult } from '../api'
import { M_TO_MM } from '../units'

export interface LayerDisplay { visible: boolean; opacity: number }

interface Props {
  geometry: Geometry | null
  display: Record<string, LayerDisplay>
  electrodes: ElectrodeBlock | undefined      // current controls (preview)
  fibre: FibreBlock | undefined               // current controls (preview)
  result: SimResult | null                    // drives the animation
  timeIndex: number
  selectedElectrode: number | null
  onSelectElectrode: (k: number | null) => void
  placing: boolean
  onPlace: (theta: number, z: number) => void
  selectedChannel: number | null
  showLabels: boolean
  resetSignal: number
}

const RENDER_ORDER: Record<string, number> = { bone: 1, muscle: 2, fat: 3, skin: 4 }

function TissueMesh({ positions, color, opacity, order, onClick, onMove }: {
  positions: number[]; color: string; opacity: number; order: number
  onClick?: (e: ThreeEvent<MouseEvent>) => void; onMove?: (e: ThreeEvent<PointerEvent>) => void
}) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(positions, v => v * M_TO_MM), 3))
    g.computeVertexNormals()
    return g
  }, [positions])
  useEffect(() => () => geom.dispose(), [geom])
  const opaque = opacity >= 0.99
  return (
    <mesh geometry={geom} renderOrder={order} onClick={onClick} onPointerMove={onMove}>
      <meshStandardMaterial color={color} transparent={!opaque} opacity={opacity} depthWrite={opaque}
        side={THREE.DoubleSide} roughness={0.75} metalness={0} />
    </mesh>
  )
}

function Lines({ segments, color }: { segments: number[]; color: string }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(segments, v => v * M_TO_MM), 3))
    return g
  }, [segments])
  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial color={color} />
    </lineSegments>
  )
}

const UP = new THREE.Vector3(0, 1, 0)

function Electrodes({ block, selected, onSelect, placing, pairHighlight, showLabels }: {
  block: ElectrodeBlock; selected: number | null; onSelect: (k: number | null) => void
  placing: boolean; pairHighlight: [number, number] | null; showLabels: boolean
}) {
  return (
    <group>
      {block.positions_m.map((p, k) => {
        const pos = new THREE.Vector3(p[0], p[1], p[2]).multiplyScalar(M_TO_MM)
        const normal = new THREE.Vector3(p[0], p[1], 0).normalize()
        const q = new THREE.Quaternion().setFromUnitVectors(UP, normal)
        const inPair = pairHighlight && (pairHighlight[0] === k || pairHighlight[1] === k)
        const onSkin = block.tissue[k] === 'skin'
        const color = !onSkin ? '#ff3b3b' : k === selected ? '#ffd166' : inPair ? '#5ce1e6' : '#20242b'
        return (
          <group key={k} position={pos} quaternion={q}>
            <mesh position={[0, 0.35, 0]}
              onClick={(e) => { if (placing) return; e.stopPropagation(); onSelect(k === selected ? null : k) }}>
              <cylinderGeometry args={[1.2, 1.2, 0.7, 20]} />
              <meshStandardMaterial color={color} emissive={k === selected ? '#664400' : '#000'} />
            </mesh>
            {(showLabels || k === selected) && (
              <Html position={[0, 3.2, 0]} center style={{ pointerEvents: 'none' }}>
                <div className={`elabel${k === selected ? ' sel' : ''}`}>e{k + 1}</div>
              </Html>
            )}
          </group>
        )
      })}
      {block.pairs.map(([a, b], j) => {
        if (!pairHighlight || pairHighlight[0] !== a || pairHighlight[1] !== b) return null
        const pa = block.positions_m[a], pb = block.positions_m[b]
        const seg = [pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]].map((v, i) => v * (i % 3 === 2 ? 1 : 1.04))
        return <Lines key={j} segments={seg} color="#5ce1e6" />
      })}
    </group>
  )
}

/** Diverging colour for membrane current: red = source (current into tissue), blue = sink. */
function currentColor(v: number, vmax: number, out: THREE.Color) {
  const s = Math.max(-1, Math.min(1, v / (vmax || 1)))
  const grey = new THREE.Color('#6b6f78')
  const tgt = new THREE.Color(s >= 0 ? '#ff4d4d' : '#4d8bff')
  return out.copy(grey).lerp(tgt, Math.sqrt(Math.abs(s)))
}

function FibreAnim({ fibre, current, timeIndex, cv, t }: {
  fibre: FibreBlock; current: number[][] | null | undefined; timeIndex: number; cv: number; t: number
}) {
  const ref = useRef<THREE.InstancedMesh>(null)
  const n = fibre.cell_centres_z_m.length
  const x = fibre.x_m * M_TO_MM, y = fibre.y_m * M_TO_MM
  const vmax = useMemo(() => {
    if (!current) return 1
    let m = 0
    for (const row of current) for (const v of row) m = Math.max(m, Math.abs(v))
    return m
  }, [current])

  useEffect(() => {
    const mesh = ref.current
    if (!mesh) return
    const mtx = new THREE.Matrix4()
    const rot = new THREE.Quaternion().setFromUnitVectors(UP, new THREE.Vector3(0, 0, 1))
    const len = fibre.dz_m * M_TO_MM
    fibre.cell_centres_z_m.forEach((z, i) => {
      mtx.compose(new THREE.Vector3(x, y, z * M_TO_MM), rot, new THREE.Vector3(1, len, 1))
      mesh.setMatrixAt(i, mtx)
    })
    mesh.instanceMatrix.needsUpdate = true
  }, [fibre, x, y])

  useEffect(() => {
    const mesh = ref.current
    if (!mesh) return
    const c = new THREE.Color()
    const [a, b] = fibre.active_z_m
    fibre.cell_centres_z_m.forEach((z, i) => {
      if (current) currentColor(current[i][timeIndex] ?? 0, vmax, c)
      else c.set(z >= a && z <= b ? '#b03a48' : '#4a4d55')
      mesh.setColorAt(i, c)
    })
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [fibre, current, timeIndex, vmax])

  // wavefront positions from the SAME parameters the backend used: z0 +/- v t, clipped to the tendons
  const z0 = fibre.nmj_z_m, [zl, zr] = fibre.active_z_m
  const fr = z0 + cv * t, fl = z0 - cv * t
  return (
    <group>
      {/* drawn on top of the tissues (depthTest off) so the buried fibre stays visible */}
      <instancedMesh ref={ref} args={[undefined, undefined, n]} renderOrder={20}>
        <cylinderGeometry args={[0.6, 0.6, 1, 8]} />
        <meshBasicMaterial depthTest={false} transparent />
      </instancedMesh>
      <mesh position={[x, y, z0 * M_TO_MM]} renderOrder={21}>
        <sphereGeometry args={[1.1, 16, 16]} />
        <meshBasicMaterial color="#ffd166" depthTest={false} transparent />
      </mesh>
      <Html position={[x, y + 2.5, z0 * M_TO_MM]} center style={{ pointerEvents: 'none' }}>
        <div className="elabel nmj">NMJ</div>
      </Html>
      {[zl, zr].map((z, i) => (
        <mesh key={i} position={[x, y, z * M_TO_MM]} renderOrder={21}>
          <torusGeometry args={[1.1, 0.25, 8, 20]} />
          <meshBasicMaterial color="#e0e0e0" depthTest={false} transparent />
        </mesh>
      ))}
      {current && t > 0 && [fr <= zr ? fr : null, fl >= zl ? fl : null].map((z, i) => z === null ? null : (
        <mesh key={`f${i}`} position={[x, y, z * M_TO_MM]} renderOrder={22}>
          <sphereGeometry args={[0.9, 12, 12]} />
          <meshBasicMaterial color="#ffffff" depthTest={false} transparent />
        </mesh>
      ))}
    </group>
  )
}

function FibreOutline({ fibre }: { fibre: FibreBlock }) {
  const x = fibre.x_m, y = fibre.y_m
  const [a, b] = fibre.active_z_m
  return (
    <group>
      <Lines segments={[x, y, fibre.stored_z_m[0], x, y, fibre.stored_z_m[1]]} color="#777" />
      <Lines segments={[x, y, a, x, y, b]} color="#ffd166" />
    </group>
  )
}

export default function Viewer3D(p: Props) {
  const controls = useRef<any>(null)
  useEffect(() => { controls.current?.reset() }, [p.resetSignal])
  const pair = p.selectedChannel != null && p.electrodes ? p.electrodes.pairs[p.selectedChannel] ?? null : null
  const resultFibre = p.result?.fibre
  const t = p.result ? p.result.time_s[p.timeIndex] ?? 0 : 0
  const fibreChanged = !!(p.result && p.fibre && (
    Math.abs(p.fibre.x_m - p.result.fibre.x_m) > 1e-12 || Math.abs(p.fibre.y_m - p.result.fibre.y_m) > 1e-12 ||
    p.fibre.active_z_m.some((v, i) => Math.abs(v - p.result!.fibre.active_z_m[i]) > 1e-12)))

  const handleSkin = (e: ThreeEvent<MouseEvent>) => {
    if (!p.placing) return
    e.stopPropagation()
    const { x, y, z } = e.point
    p.onPlace(Math.atan2(y, x), z / M_TO_MM)
  }

  return (
    <Canvas camera={{ position: [150, 80, 40], fov: 36, near: 1, far: 5000 }}
      onPointerMissed={() => !p.placing && p.onSelectElectrode(null)}
      style={{ cursor: p.placing ? 'crosshair' : 'grab' }}>
      <color attach="background" args={['#14171c']} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[120, 160, 100]} intensity={1.1} />
      <directionalLight position={[-120, -60, -100]} intensity={0.35} />
      <OrbitControls ref={controls} makeDefault enableDamping={false} />
      {p.geometry?.tissues.map(ts => {
        const d = p.display[ts.key]
        if (!d?.visible) return null
        return <TissueMesh key={ts.key} positions={ts.positions_m} color={ts.color} opacity={d.opacity}
          order={RENDER_ORDER[ts.key] ?? 0} onClick={ts.key === 'skin' ? handleSkin : undefined} />
      })}
      {p.geometry?.illustrations.map(il => p.display[il.key]?.visible
        ? <Lines key={il.key} segments={il.lines_m} color={il.color} /> : null)}
      {resultFibre && (
        <FibreAnim fibre={resultFibre} current={resultFibre.cell_current_A} timeIndex={p.timeIndex}
          cv={p.result!.config.fibre.cv} t={t} />
      )}
      {p.fibre && (!resultFibre || fibreChanged) && <FibreOutline fibre={p.fibre} />}
      {p.electrodes && (
        <Electrodes block={p.electrodes} selected={p.selectedElectrode} onSelect={p.onSelectElectrode}
          placing={p.placing} pairHighlight={pair} showLabels={p.showLabels} />
      )}
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={['#ff6b6b', '#6bff8e', '#6bb5ff']} labelColor="#111" />
      </GizmoHelper>
    </Canvas>
  )
}
