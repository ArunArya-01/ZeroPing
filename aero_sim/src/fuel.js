// Fuel-burn prediction for each flight segment.
//
// Two engines:
//  1. ONNX Runtime Web — loads the exported AeroTwin student model
//     (`public/models/large_mlp.onnx` + `large_mlp.preproc.json`) and runs the
//     real trained model in-browser. Feature vector is built to match the
//     preprocessing contract emitted by `experiments/11_onnx_deploy/export_onnx.py`.
//  2. Physics fallback — a simple OpenAP-like approximation used when the ONNX
//     model is not present, so the demo always runs.

import * as ort from 'onnxruntime-web'

const GRAVITY = 9.80665
const R_EARTH_M = 6371000.0

function haversineM(lat1, lon1, lat2, lon2) {
  const p1 = (lat1 * Math.PI) / 180
  const p2 = (lat2 * Math.PI) / 180
  const dp = ((lat2 - lat1) * Math.PI) / 180
  const dl = ((lon2 - lon1) * Math.PI) / 180
  const a =
    Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2
  return 2 * R_EARTH_M * Math.asin(Math.sqrt(Math.min(1, a)))
}

function physicsFuelKg(segment) {
  // Crude cruise burn ~0.35 kg/s plus climb/descent overhead.
  const cruise = 0.35 * segment.durationS
  const vertical = Math.max(0, segment.altitudeEndM - segment.altitudeStartM) * 0.02
  const dist = segment.distanceM || 1
  const speedPenalty = Math.min(1.5, Math.max(0.85, segment.speedMps / 240)) ** 3
  return cruise * speedPenalty + vertical
}

export class FuelPredictor {
  constructor() {
    this.session = null
    this.preproc = null
    this.inDim = 0
    this.engine = 'physics-demo'
  }

  get engineName() {
    return this.engine === 'onnx' ? 'ONNX Runtime Web' : 'Physics fallback (demo)'
  }

  async init() {
    try {
      // Tell onnxruntime-web where to find its wasm binaries.
      // Single-threaded mode avoids SharedArrayBuffer / MountedFiles requirements.
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/'
      ort.env.wasm.numThreads = 1
      ort.env.wasm.simd = true
      const [preprocRes, modelRes] = await Promise.all([
        fetch('/models/large_mlp.preproc.json'),
        fetch('/models/large_mlp.onnx'),
      ])
      const isJson = preprocRes.headers.get('content-type')?.includes('application/json')
      const isHtml = modelRes.headers.get('content-type')?.includes('text/html')
      if (!preprocRes.ok || !modelRes.ok || !isJson || isHtml) {
        throw new Error('model files not found')
      }
      const preproc = await preprocRes.json()
      // Pass the URL so onnxruntime-web can resolve the external data file
      // (large_mlp.onnx.data) which holds the model weights.
      this.session = await ort.InferenceSession.create('/models/large_mlp.onnx', {
        executionProviders: ['wasm'],
      })
      this.preproc = preproc
      this.inDim =
        preproc.numeric_columns.length +
        preproc.onehot_categories.reduce((acc, c) => acc + c.length, 0)
      this.engine = 'onnx'
      return true
    } catch (e) {
      console.warn('[FuelPredictor] ONNX load failed, using physics fallback:', e)
      this.session = null
      this.engine = 'physics-demo'
      return false
    }
  }

  // Build the full model input row from a segment + route context.
  _buildRow(segment, route) {
    const p = this.preproc
    const num = []
    const numericCols = p.numeric_columns
    for (const col of numericCols) {
      num.push(this._valueFor(col, segment, route))
    }
    const cats = []
    for (const c of p.categorical_columns) {
      if (c === 'aircraft_type') cats.push(route.aircraftType)
      else if (c === 'method') cats.push('acars')
      else if (c === 'origin_icao') cats.push(route.origin)
      else if (c === 'destination_icao') cats.push(route.destination)
      else cats.push('missing')
    }
    const catEncoded = []
    p.onehot_categories.forEach((categories, i) => {
      const label = cats[i] || 'missing'
      const idx = categories.indexOf(label)
      for (let k = 0; k < categories.length; k++) {
        catEncoded.push(idx === k ? 1 : 0)
      }
    })
    return num.concat(catEncoded)
  }

  _valueFor(col, segment, route) {
    const median =
      this.preproc.median_impute[this.preproc.numeric_columns.indexOf(col)] ?? 0
    switch (col) {
      case 'duration_s':
        return segment.durationS
      case 'n_traj_pts':
        return 20
      case 'has_acars_in_window':
        return 1
      case 'mean_altitude':
      case 'median_altitude':
        return (segment.altitudeStartM + segment.altitudeEndM) / 2
      case 'max_altitude':
        return Math.max(segment.altitudeStartM, segment.altitudeEndM)
      case 'std_altitude':
        return Math.abs(segment.altitudeEndM - segment.altitudeStartM) / 4
      case 'mean_groundspeed':
      case 'max_groundspeed':
        return segment.speedMps * 1.94384 // knots
      case 'std_groundspeed':
        return segment.speedMps * 1.94384 * 0.05
      case 'mean_vertical_rate':
        return (segment.altitudeEndM - segment.altitudeStartM) / Math.max(segment.durationS, 1)
      case 'std_vertical_rate':
        return 1.5
      case 'climb_fraction':
        return segment.altitudeEndM > segment.altitudeStartM ? 0.25 : 0.05
      case 'cruise_fraction':
        return 0.6
      case 'descent_fraction':
        return segment.altitudeEndM < segment.altitudeStartM ? 0.25 : 0.05
      case 'ref_mass_kg':
        return 64000
      case 'physics_fuel_kg':
        return physicsFuelKg(segment)
      case 'energy_efficiency':
        return 0.5
      case 'headwind_mps':
        return 5
      case 'crosswind_mps':
        return 2
      case 'temperature_k':
        return 220
      case 'pressure_pa':
        return 24000
      case 'isa_deviation_k':
        return 3
      case 'density_altitude_m':
        return segment.altitudeStartM * 1.1
      default:
        return median
    }
  }

  async predictSegments(segments, route) {
    if (this.session && this.preproc) {
      try {
        const n = segments.length
        const flat = new Float32Array(n * this.inDim)
        segments.forEach((seg, i) => {
          const row = this._buildRow(seg, route)
          row.forEach((v, j) => {
            flat[i * this.inDim + j] = Number.isFinite(v) ? v : 0
          })
        })
        const tensor = new ort.Tensor('float32', flat, [n, this.inDim])
        const results = await this.session.run({ input: tensor })
        const preds = Array.from(results.fuel_kg.data)
        return segments.map((seg, i) => ({
          fuelKg: Math.max(0, preds[i]),
          engine: this.engine,
        }))
      } catch (err) {
        console.warn('ONNX inference failed, falling back to physics', err)
      }
    }
    return segments.map((seg) => ({
      fuelKg: physicsFuelKg(seg),
      engine: this.engine,
    }))
  }
}
