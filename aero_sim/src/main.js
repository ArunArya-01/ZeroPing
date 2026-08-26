// AeroSim — AeroTwin model-validation simulator.
//
// Primary mode: loads a real PRC test flight with ground-truth fuel,
// OpenAP physics baseline, and R3 model predictions. Shows the achievement:
// R3 dramatically outperforms the physics baseline.

import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'

import { loadDemoFlight } from './data/routes.js'

const SPEED_STEPS = [10, 50, 100, 250, 500, 1000, 2000]

const state = {
  viewer: null,
  speedIdx: 2,
  paused: false,
  cameraMode: 'overview',
  running: false,
  // Demo flight data.
  intervals: [],
  totalFuelKg: 0,
  totalDurationS: 0,
  // Chart data.
  chartActual: [],
  chartPhysics: [],
  chartR3: [],
}

const els = {}
function grab() {
  ;[
    'hudOrigin', 'hudOriginName', 'hudDest', 'hudDestName',
    'hudAircraft', 'hudDistance', 'hudElapsed', 'hudFuelUsed',
    'hudFuelRemaining', 'hudEngine', 'hudProgress',
    'progressBarFill', 'btnPause', 'speedDown', 'speedUp', 'speedLabel',
    'thrustVal', 'thrustFill', 'dragVal', 'dragFill', 'liftVal', 'liftFill',
    'massTakeoff', 'massCurrent', 'massLanding', 'hudFuelRemaining', 'massFuelBurn',
    'massBurnRate', 'massFuelFrac', 'massRate', 'massWingLoad', 'massPhase',
    'predGt', 'predOpenap', 'predR3', 'predOpenapErr', 'predR3Err', 'predR3Rel', 'predChart',
  ].forEach((id) => (els[id] = document.getElementById(id)))
}

function fmtTime(sec) {
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  const pad = (x) => String(x).padStart(2, '0')
  return `${pad(h)}:${pad(m)}:${pad(s)}`
}

async function main() {
  grab()
  const viewer = new Cesium.Viewer('cesiumContainer', {
    timeline: false,
    animation: false,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    infoBox: false,
    selectionIndicator: false,
    fullscreenButton: false,
  })
  state.viewer = viewer
  viewer.scene.globe.enableLighting = true
  viewer.scene.globe.atmosphereLightFactor = 1.2
  viewer.scene.skyAtmosphere.show = true

  // Load demo flight.
  const flight = await loadDemoFlight()
  state.intervals = flight.intervals
  state.totalFuelKg = flight.intervals.reduce((a, iv) => a + iv.groundTruth, 0)
  state.totalDurationS = flight.intervals.reduce((a, iv) => a + iv.durationS, 0)

  els.hudOrigin.textContent = flight.origin
  els.hudOriginName.textContent = flight.originName
  els.hudDest.textContent = flight.destination
  els.hudDestName.textContent = flight.destinationName
  els.hudAircraft.textContent = flight.aircraftType
  els.hudDistance.textContent = `${(state.totalDurationS * 0.24 / 1000).toFixed(0)} km`
  els.hudEngine.textContent = 'R3 Model (Live)'

  bindControls(viewer)
  buildScene(viewer, flight)

  // Reset clock.
  const now = Cesium.JulianDate.fromDate(new Date())
  viewer.clock.startTime = now
  viewer.clock.stopTime = Cesium.JulianDate.addSeconds(now, state.totalDurationS, new Cesium.JulianDate())
  viewer.clock.currentTime = Cesium.JulianDate.clone(now)
  viewer.clock.shouldAnimate = !state.paused
  updateClock(viewer)
  state.running = true

  viewer.clock.onTick.addEventListener((clock) => tick(viewer, clock))
  viewer.clock.onTick.addEventListener((clock) => updateCamera(viewer, clock))
}

function buildScene(viewer, flight) {
  // Build a smooth great-circle path with realistic altitude profile.
  const startLat = 51.47, startLon = -0.45   // London Heathrow
  const endLat = 40.64, endLon = -73.78      // New York JFK
  const nSamples = 300

  // Great-circle interpolation (slerp on sphere).
  const latLonAlt = []
  const phi1 = Cesium.Math.toRadians(startLat)
  const lambda1 = Cesium.Math.toRadians(startLon)
  const phi2 = Cesium.Math.toRadians(endLat)
  const lambda2 = Cesium.Math.toRadians(endLon)
  const d = 2 * Math.asin(Math.sqrt(
    Math.sin((phi2 - phi1) / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin((lambda2 - lambda1) / 2) ** 2
  ))

  // Smooth easing functions for altitude transitions.
  const easeInOut = (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
  const easeOut = (t) => 1 - Math.pow(1 - t, 3)
  const easeIn = (t) => t * t * t

  for (let i = 0; i <= nSamples; i++) {
    const f = i / nSamples
    // Realistic altitude profile with smooth transitions.
    const cruiseAlt = 11000
    let alt
    if (f < 0.06) {
      // Takeoff: ground to 2000m with ease-out.
      alt = easeOut(f / 0.06) * 2000
    } else if (f < 0.15) {
      // Climb: 2000m to cruise with ease-in-out.
      alt = 2000 + easeInOut((f - 0.06) / 0.09) * (cruiseAlt - 2000)
    } else if (f > 0.90) {
      // Descent: cruise to 2000m with ease-in-out.
      const descFrac = (f - 0.90) / 0.08
      alt = 2000 + (1 - easeInOut(descFrac)) * (cruiseAlt - 2000)
    } else if (f > 0.97) {
      // Approach: 2000m to ground with ease-in.
      alt = (1 - easeIn((f - 0.97) / 0.03)) * 2000
    } else {
      // Cruise: gentle altitude variation.
      alt = cruiseAlt + Math.sin(f * Math.PI * 4) * 150 + Math.sin(f * Math.PI * 7) * 50
    }

    // Great-circle interpolation.
    const A = Math.sin((1 - f) * d) / Math.sin(d)
    const B = Math.sin(f * d) / Math.sin(d)
    const x = A * Math.cos(phi1) * Math.cos(lambda1) + B * Math.cos(phi2) * Math.cos(lambda2)
    const y = A * Math.cos(phi1) * Math.sin(lambda1) + B * Math.cos(phi2) * Math.sin(lambda2)
    const z = A * Math.sin(phi1) + B * Math.sin(phi2)
    const lat = Math.atan2(z, Math.sqrt(x * x + y * y))
    const lon = Math.atan2(y, x)

    latLonAlt.push({
      lat: Cesium.Math.toDegrees(lat),
      lon: Cesium.Math.toDegrees(lon),
      alt: alt,
    })
  }

  // Route polyline (glow) — only show the airborne portion.
  const routePositions = latLonAlt.map((p) => Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt))
  viewer.entities.add({
    polyline: {
      positions: routePositions,
      width: 3,
      material: new Cesium.PolylineGlowMaterialProperty({
        glowPower: 0.25,
        color: Cesium.Color.fromCssColorString('#dc1414').withAlpha(0.7),
      }),
    },
  })

  // Sampled position property for smooth animation with auto-orientation.
  const positionProperty = new Cesium.SampledPositionProperty()
  const startTime = viewer.clock.startTime
  const totalSeconds = state.totalDurationS

  for (let i = 0; i <= nSamples; i++) {
    const f = i / nSamples
    const time = Cesium.JulianDate.addSeconds(startTime, f * totalSeconds, new Cesium.JulianDate())
    const p = latLonAlt[i]
    positionProperty.addSample(time, Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt))
  }

  // Aircraft entity with velocity-based orientation.
  state.plane = viewer.entities.add({
    position: positionProperty,
    orientation: new Cesium.VelocityOrientationProperty(positionProperty),
    point: {
      pixelSize: 16,
      color: Cesium.Color.fromCssColorString('#ff3333'),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 2,
    },
    label: {
      text: '✈',
      font: 'bold 22px sans-serif',
      fillColor: Cesium.Color.WHITE,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      pixelOffset: new Cesium.Cartesian2(0, -12),
      scaleByDistance: new Cesium.NearFarScalar(1e6, 1.2, 5e6, 0.6),
    },
    path: {
      leadTime: 0,
      trailTime: Math.min(totalSeconds * 0.25, 1800),
      width: 3,
      resolution: 1,
      material: new Cesium.PolylineGlowMaterialProperty({
        glowPower: 0.25,
        color: Cesium.Color.fromCssColorString('#dc1414').withAlpha(0.5),
      }),
    },
  })

  // Origin / destination markers.
  viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(startLon, startLat, 0),
    point: { pixelSize: 8, color: Cesium.Color.WHITE, disableDepthTestDistance: Number.POSITIVE_INFINITY },
    label: { text: 'LHR', font: 'bold 12px sans-serif', fillColor: Cesium.Color.WHITE, style: Cesium.LabelStyle.FILL_AND_OUTLINE, outlineWidth: 2, outlineColor: Cesium.Color.BLACK, disableDepthTestDistance: Number.POSITIVE_INFINITY, pixelOffset: new Cesium.Cartesian2(0, 14) },
  })
  viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(endLon, endLat, 0),
    point: { pixelSize: 8, color: Cesium.Color.WHITE, disableDepthTestDistance: Number.POSITIVE_INFINITY },
    label: { text: 'JFK', font: 'bold 12px sans-serif', fillColor: Cesium.Color.WHITE, style: Cesium.LabelStyle.FILL_AND_OUTLINE, outlineWidth: 2, outlineColor: Cesium.Color.BLACK, disableDepthTestDistance: Number.POSITIVE_INFINITY, pixelOffset: new Cesium.Cartesian2(0, 14) },
  })

  viewer.camera.flyTo({ destination: Cesium.Cartesian3.fromDegrees(-40, 48, 5000000) })
}

// Smooth camera tracking with lerping to avoid jumps.
let lastCamTime = 0
function updateCamera(viewer, clock) {
  const pos = state.plane.position.getValue(clock.currentTime)
  if (!pos || state.cameraMode !== 'follow') return

  const now = performance.now()
  const dt = Math.min((now - lastCamTime) / 1000, 0.1)
  lastCamTime = now

  // Smooth look-at with offset that follows behind and above the aircraft.
  const offset = new Cesium.Cartesian3(0, -8000, 12000)
  const targetPos = Cesium.Cartesian3.add(pos, offset, new Cesium.Cartesian3())

  // Get current camera position and lerp toward target.
  const cam = viewer.camera
  const currentPos = cam.position
  const lerpFactor = 1 - Math.exp(-3 * dt) // Smooth exponential interpolation

  const newPos = new Cesium.Cartesian3(
    currentPos.x + (targetPos.x - currentPos.x) * lerpFactor,
    currentPos.y + (targetPos.y - currentPos.y) * lerpFactor,
    currentPos.z + (targetPos.z - currentPos.z) * lerpFactor
  )

  cam.setView({
    destination: newPos,
    orientation: {
      heading: Cesium.Math.toRadians(0),
      pitch: Cesium.Math.toRadians(-25),
      roll: 0,
    },
  })
}

function updateClock(viewer) {
  viewer.clock.multiplier = SPEED_STEPS[state.speedIdx]
  els.speedLabel.textContent = String(SPEED_STEPS[state.speedIdx])
}

function bindControls(viewer) {
  els.btnPause.addEventListener('click', () => {
    state.paused = !state.paused
    viewer.clock.shouldAnimate = !state.paused
    els.btnPause.textContent = state.paused ? '▶' : '⏸'
    els.btnPause.classList.toggle('active', !state.paused)
  })
  els.speedUp.addEventListener('click', () => { state.speedIdx = Math.min(SPEED_STEPS.length - 1, state.speedIdx + 1); updateClock(viewer) })
  els.speedDown.addEventListener('click', () => { state.speedIdx = Math.max(0, state.speedIdx - 1); updateClock(viewer) })
  document.querySelectorAll('#cameraMode .cn-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.cameraMode = btn.dataset.cam
      document.querySelectorAll('#cameraMode .cn-btn').forEach((b) => b.classList.toggle('active', b === btn))
    })
  })
}

function tick(viewer, clock) {
  if (!state.running) return
  const t = Math.max(0, Math.min(1, Cesium.JulianDate.secondsDifference(clock.currentTime, clock.startTime) / state.totalDurationS))
  const ints = state.intervals
  const fp = t * (ints.length - 1)
  const idx = Math.min(ints.length - 1, Math.floor(fp))
  const frac = fp - Math.floor(fp)
  const iv = ints[idx]

  // Cumulative fuel up to current interval.
  let cumGt = 0, cumPhys = 0, cumR3 = 0
  for (let i = 0; i < idx; i++) {
    cumGt += ints[i].groundTruth
    cumPhys += ints[i].physicsFuelKg
    cumR3 += ints[i].r3Prediction
  }
  cumGt += iv.groundTruth * frac
  cumPhys += iv.physicsFuelKg * frac
  cumR3 += iv.r3Prediction * frac

  const elapsed = t * state.totalDurationS

  els.hudProgress.textContent = `${(t * 100).toFixed(1)}%`
  els.progressBarFill.style.width = `${t * 100}%`
  els.hudElapsed.textContent = fmtTime(elapsed)
  els.hudFuelUsed.textContent = `${cumGt.toFixed(0)} kg`
  els.hudFuelRemaining.textContent = `${Math.max(0, state.totalFuelKg - cumGt).toFixed(0)} kg`

  // Force meters from flight state.
  updateForceMeters(iv)
  updateMassPanel(iv, cumGt)

  // Prediction comparison.
  updatePredictionComparison(iv, idx, cumGt, cumPhys, cumR3)
}

function updateForceMeters(iv) {
  if (!iv) return
  const speedMs = iv.groundSpeedMps || 240
  const altM = iv.altitudeM || 10000
  const vRate = iv.verticalRateMps || 0
  const densityRatio = Math.exp(-altM / 8500)
  const climbFactor = 0.45 + 0.55 * Math.max(0, Math.min(1, 0.5 + vRate / 12))
  const thrust = Math.round(Math.min(100, climbFactor * 100))
  const dragRaw = densityRatio * Math.pow(speedMs / 240, 2)
  const drag = Math.round(Math.min(100, Math.max(8, dragRaw * 90 + 10)))
  const liftFactor = 0.5 + 0.5 * Math.max(0, Math.min(1, 0.5 + vRate / 15))
  const lift = Math.round(Math.min(100, liftFactor * 100))
  els.thrustVal.textContent = `${thrust}%`
  els.thrustFill.style.width = `${thrust}%`
  els.dragVal.textContent = `${drag}%`
  els.dragFill.style.width = `${drag}%`
  els.liftVal.textContent = `${lift}%`
  els.liftFill.style.width = `${lift}%`
  return thrust
}

function updateMassPanel(iv, cumGt) {
  if (!iv) return
  const oew = 42600
  const takeoff = oew + 15000 + state.totalFuelKg
  const current = takeoff - cumGt
  const landing = oew + 15000 + state.totalFuelKg * 0.08
  const remaining = state.totalFuelKg - cumGt
  const burnRate = iv.groundTruth / Math.max(iv.durationS, 1)
  const vRate = iv.verticalRateMps || 0
  let phase = iv.phase && iv.phase !== 'unknown' ? iv.phase : 'Cruise'
  if (vRate > 1.5) phase = 'Climb'
  else if (vRate < -1.5) phase = 'Descent'
  els.massTakeoff.textContent = `${(takeoff / 1000).toFixed(1)} t`
  els.massCurrent.textContent = `${(current / 1000).toFixed(1)} t`
  els.massLanding.textContent = `${(landing / 1000).toFixed(1)} t`
  els.hudFuelRemaining.textContent = `${remaining.toFixed(0)} kg`
  els.massFuelBurn.textContent = `${cumGt.toFixed(0)} kg`
  els.massBurnRate.textContent = `${burnRate.toFixed(2)} kg/s`
  els.massFuelFrac.textContent = `${((remaining / takeoff) * 100).toFixed(1)}%`
  els.massRate.textContent = `${(-burnRate).toFixed(2)} kg/s`
  els.massWingLoad.textContent = `${(current / 122.6).toFixed(0)} kg/m²`
  els.massPhase.textContent = phase.charAt(0).toUpperCase() + phase.slice(1)
}

function updatePredictionComparison(iv, idx, cumGt, cumPhys, cumR3) {
  if (!iv) return
  const gt = iv.groundTruth
  const phys = iv.physicsFuelKg
  const r3 = iv.r3Prediction
  const physErr = phys - gt
  const r3Err = r3 - gt
  const r3RelErr = gt > 0 ? (Math.abs(r3Err) / gt) * 100 : 0

  els.predGt.textContent = `${gt.toFixed(0)} kg`
  els.predOpenap.textContent = `${phys.toFixed(0)} kg`
  els.predR3.textContent = `${r3.toFixed(0)} kg`
  els.predOpenapErr.textContent = `${physErr >= 0 ? '+' : ''}${physErr.toFixed(0)} kg`
  els.predR3Err.textContent = `${r3Err >= 0 ? '+' : ''}${r3Err.toFixed(0)} kg`
  els.predR3Rel.textContent = `${r3RelErr.toFixed(1)}%`

  // Chart data (sample every few intervals).
  if (state.chartActual.length <= idx) {
    state.chartActual.push(cumGt)
    state.chartPhysics.push(cumPhys)
    state.chartR3.push(cumR3)
    drawChart()
  }
}

function drawChart() {
  const canvas = els.predChart
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const W = canvas.width
  const H = canvas.height
  ctx.clearRect(0, 0, W, H)

  const actual = state.chartActual
  const phys = state.chartPhysics
  const r3 = state.chartR3
  if (actual.length === 0) return

  const maxVal = Math.max(actual[actual.length - 1], phys[phys.length - 1], r3[r3.length - 1], 1)

  // OpenAP (white, dashed).
  ctx.strokeStyle = 'rgba(255,255,255,0.5)'
  ctx.lineWidth = 1
  ctx.setLineDash([3, 3])
  ctx.beginPath()
  phys.forEach((v, i) => {
    const x = (i / Math.max(1, phys.length - 1)) * (W - 8) + 4
    const y = H - 6 - (v / maxVal) * (H - 12)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()

  // Ground truth (white, solid).
  ctx.strokeStyle = '#ffffff'
  ctx.lineWidth = 1.5
  ctx.setLineDash([])
  ctx.beginPath()
  actual.forEach((v, i) => {
    const x = (i / Math.max(1, actual.length - 1)) * (W - 8) + 4
    const y = H - 6 - (v / maxVal) * (H - 12)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()

  // R3 (red).
  ctx.strokeStyle = '#dc1414'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  r3.forEach((v, i) => {
    const x = (i / Math.max(1, r3.length - 1)) * (W - 8) + 4
    const y = H - 6 - (v / maxVal) * (H - 12)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()
}

async function launch() {
  const home = document.getElementById('homeOverlay')
  const simUI = document.getElementById('simUI')
  home.classList.add('hidden')
  simUI.classList.add('ready')
  await new Promise((r) => requestAnimationFrame(() => r()))
  await main()
}

document.getElementById('launchBtn').addEventListener('click', () => {
  launch().catch((err) => {
    console.error(err)
    const e = document.getElementById('hudEngine')
    if (e) e.textContent = 'Error: ' + err.message
  })
})

// Back button: stop the viewer and return to home.
document.getElementById('backBtn').addEventListener('click', () => {
  // Stop the clock and destroy the viewer to free WebGL resources.
  if (state.viewer) {
    state.viewer.clock.shouldAnimate = false
    state.viewer.destroy()
    state.viewer = null
  }
  state.running = false
  // Show home overlay, hide sim UI.
  document.getElementById('homeOverlay').classList.remove('hidden')
  document.getElementById('simUI').classList.remove('ready')
})

if (new URLSearchParams(location.search).get('autolaunch') === '1') {
  document.getElementById('launchBtn')?.click()
}
