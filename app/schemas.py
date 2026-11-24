"""
Request / response schemas.  All values SI (metres, seconds, S/m, A, V).

A SimConfig is an immutable snapshot of everything a single-fibre run depends
on.  Field defaults come from config.py, so with no overrides a run reproduces
stage2_forward.py + stage3_single_fibre.py exactly.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config as C

_FROZEN = ConfigDict(frozen=True, extra="forbid")


def _r(x: float) -> float:
    """Round derived lengths to 1e-12 m so e.g. 0.020 + 0.003 is bit-identical to 0.023."""
    return round(x, 12)


class _Finite(BaseModel):
    model_config = _FROZEN

    @model_validator(mode="after")
    def _all_finite(self):
        for name, v in self:
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        return self


# ----------------------------------------------------------------------------- geometry
class GeometryConfig(_Finite):
    """Concentric-cylinder volume conductor.  Radii are derived from thicknesses."""
    bone_radius: float = Field(C.R_BONE, gt=0, le=0.05)
    muscle_radius: float = Field(C.R_MUSCLE, gt=0, le=0.08)
    fat_thickness: float = Field(_r(C.R_FAT - C.R_MUSCLE), gt=0, le=0.03)
    skin_thickness: float = Field(_r(C.R_SKIN - C.R_FAT), gt=0, le=0.01)
    length: float = Field(C.LENGTH, ge=0.06, le=0.30)

    @model_validator(mode="after")
    def _order(self):
        if not self.bone_radius < self.muscle_radius:
            raise ValueError("bone radius must be smaller than the outer muscle radius")
        return self

    def radii(self):
        """(R_bone, R_muscle, R_fat, R_skin) in metres, as mesh.build_cylinder_mesh expects."""
        r_fat = _r(self.muscle_radius + self.fat_thickness)
        return (self.bone_radius, self.muscle_radius, r_fat, _r(r_fat + self.skin_thickness))


class MeshConfig(_Finite):
    h_xy: float = Field(C.H_XY, ge=0.5e-3, le=2.0e-3)
    h_z: float = Field(C.H_Z, ge=1.0e-3, le=4.0e-3)


class ConductivityConfig(_Finite):
    """Diagonal conductivity per tissue (S/m).  Muscle is transversely isotropic:
    sigma_x = sigma_y = transverse, sigma_z = longitudinal (fibres run along z)."""
    bone: float = Field(C.SIGMA[C.BONE][0], gt=0, le=10)
    muscle_transverse: float = Field(C.SIGMA[C.MUSCLE][0], gt=0, le=10)
    muscle_longitudinal: float = Field(C.SIGMA[C.MUSCLE][2], gt=0, le=10)
    fat: float = Field(C.SIGMA[C.FAT][0], gt=0, le=10)
    skin: float = Field(C.SIGMA[C.SKIN][0], gt=0, le=10)

    def table(self):
        """{label: (sx, sy, sz)} -- the format of config.SIGMA."""
        return {
            C.BONE: (self.bone,) * 3,
            C.MUSCLE: (self.muscle_transverse, self.muscle_transverse, self.muscle_longitudinal),
            C.FAT: (self.fat,) * 3,
            C.SKIN: (self.skin,) * 3,
        }


# ----------------------------------------------------------------------------- fibre
class FibreConfig(_Finite):
    """One straight fibre parallel to z.  Lateral position: depth below the outer
    muscle surface and angle theta around the axis."""
    depth: float = Field(C.FIBRE_DEPTH, gt=0, le=0.05)
    theta: float = Field(0.0, ge=-math.pi, le=math.pi)
    nmj_z: float = Field(C.NMJ_Z, ge=-0.15, le=0.15)
    l1: float = Field(C.L1, gt=0, le=0.15)
    l2: float = Field(C.L2, gt=0, le=0.15)
    cv: float = Field(C.CV, ge=1.0, le=10.0)
    taper: float = Field(C.TAPER, ge=0.5e-3, le=20e-3)
    radius: float = Field(C.FIBRE_RADIUS, ge=5e-6, le=100e-6)
    sigma_in: float = Field(C.SIGMA_IN, gt=0, le=10)
    dz: float = Field(C.FIBRE_DZ, ge=0.1e-3, le=2e-3)


# ----------------------------------------------------------------------------- electrodes
class SurfacePoint(_Finite):
    """A point on the skin, in cylinder surface coordinates."""
    theta: float = Field(ge=-math.pi, le=math.pi)   # rad
    z: float = Field(ge=-0.15, le=0.15)             # m


class ElectrodeArrayConfig(_Finite):
    """Linear array laid on the (unrolled) skin surface.

    Electrode k (0-based) sits at arc-length offset s_k = (k - (n-1)/2) * spacing
    from the centre, along a direction making angle `orientation` with the z axis:
        z_k     = z_center + s_k cos(orientation)
        theta_k = theta_center + s_k sin(orientation) / R_skin
    orientation = 0 reproduces the stage 2/3 array (a line along z)."""
    n: int = Field(C.N_ELEC_LINEAR, ge=2, le=32)
    spacing: float = Field(C.ELEC_SPACING, ge=1e-3, le=30e-3)
    theta_center: float = Field(0.0, ge=-math.pi, le=math.pi)
    z_center: float = Field(0.0, ge=-0.15, le=0.15)
    orientation: float = Field(0.0, ge=-math.pi / 2, le=math.pi / 2)


class ElectrodeConfig(_Finite):
    array: ElectrodeArrayConfig = ElectrodeArrayConfig()
    # individually moved electrodes: {electrode index (0-based): surface point}
    overrides: dict[int, SurfacePoint] = Field(default_factory=dict)
    # differential channel j = phi[pairs[j][0]] - phi[pairs[j][1]]  (V_EMG = phi_1 - phi_2)
    pairing: Literal["consecutive"] = "consecutive"

    @field_validator("overrides")
    @classmethod
    def _idx(cls, v):
        for k in v:
            if k < 0:
                raise ValueError("electrode override index must be >= 0")
        return v

    @model_validator(mode="after")
    def _idx_range(self):
        bad = [k for k in self.overrides if k >= self.array.n]
        if bad:
            raise ValueError(f"electrode override index {bad} >= number of electrodes {self.array.n}")
        return self

    def surface_points(self, r_skin: float):
        """[(theta, z)] for every electrode, applying overrides."""
        a = self.array
        out = []
        for k in range(a.n):
            if k in self.overrides:
                p = self.overrides[k]
                out.append((p.theta, p.z))
                continue
            s = (k - (a.n - 1) / 2) * a.spacing
            th = a.theta_center + s * math.sin(a.orientation) / r_skin
            th = math.atan2(math.sin(th), math.cos(th)) if abs(th) > math.pi else th
            out.append((th, a.z_center + s * math.cos(a.orientation)))
        return out

    def pairs(self):
        return [(k, k + 1) for k in range(self.array.n - 1)]


# ----------------------------------------------------------------------------- timing / solver
class TimingConfig(_Finite):
    duration: float = Field(C.DURATION_SINGLE, ge=5e-3, le=0.1)
    fs: float = Field(C.FS_SINGLE, ge=1000.0, le=20000.0)

    def n_samples(self):
        # the same expression as stage3 (int(0.030 * FS_SINGLE)), with the round-off
        # guard of muscle.spike_trains so e.g. 0.035 s * 10 kHz gives 350, not 349
        return int(self.duration * self.fs + 1e-6)

    @model_validator(mode="after")
    def _n(self):
        if not 10 <= self.n_samples() <= 2000:
            raise ValueError("duration x sampling rate must give between 10 and 2000 samples")
        return self


class SolverConfig(_Finite):
    cg_rtol: float = Field(C.CG_RTOL, ge=1e-12, le=1e-6)


# ----------------------------------------------------------------------------- top level
class SimConfig(_Finite):
    """Everything one simulation depends on (immutable snapshot)."""
    kind: Literal["single_fibre"] = "single_fibre"
    geometry: GeometryConfig = GeometryConfig()
    mesh: MeshConfig = MeshConfig()
    conductivity: ConductivityConfig = ConductivityConfig()
    fibre: FibreConfig = FibreConfig()
    electrodes: ElectrodeConfig = ElectrodeConfig()
    timing: TimingConfig = TimingConfig()
    solver: SolverConfig = SolverConfig()

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: SimConfig
    label: Optional[str] = Field(None, max_length=80)


class FutureSimRequest(BaseModel):
    """Placeholder request shape for Stage 4 (motor units) / Stage 5 integration."""
    model_config = ConfigDict(extra="allow")
    kind: Literal["motor_unit_pool", "muscle_ramp", "timing"]
