// AeroSim — Cesium / Three.js / ONNX fuel-burn simulator entry point.
//
// The simulation only boots after the user clicks "Launch Simulation" on the
// home page, so WebGL / ONNX assets stay idle until needed.

import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'

import { SAMPLE_ROUTES } from './data/routes.js'
import { FuelPredictor } from './fuel.js'
import { FuelGauge3D } from './threeScene.js'

const SPEED_STEPS = [10, 50, 100, 250, 500, 1000, 2000]

const state = {
  route: SAMPLE_ROUTES[0],
  speedIdx: 2,
  paused: false,
  cameraMode: 'overview', // 'follow' | 'overview'
  segPredictions: [],
  segments: [],
  totalFuelKg: 0,
  totalDurationS: 0,
  running: false,
}

const els = {}
function grab() {
  ;[
    'hudOrigin', 'hudOriginName', 'hudDest', 'hudDestName',
    'hudAircraft', 'hudDistance', 'hudElapsed', 'hudFuelUsed',
    'hudFuelRemaining', 'hudFuelTotal', 'hudEngine', 'hudProgress',
    'progressBarFill', 'fuelPct', 'routeSelect', 'btnPause',
    'speedDown', 'speedUp', 'speedLabel',
  ].forEach((id) => (els[id] = document.getElementById(id)))
}

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000
  const p1 = (lat1 * Math.PI) / 180
  const p2 = (lat2 * Math.PI) / 180
  const dp = ((lat2 - lat1) * Math.PI) / 180
  const dl = ((lon2 - lon1) * Math.PI) / 180
  const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(Math.min(1, a)))
}

function buildSegments(route) {
  const wp = route.waypoints
  const totalDur = route.durationMin * 60
  const n = wp.length - 1
  const durationS = totalDur / n
  const segs = []
  for (let i = 0; i < n; i++) {
    const a = wp[i]
    const b = wp[i + 1]
    const distanceM = haversineM(a.lat, a.lon, b.lat, b.lon)
    segs.push({
      durationS,
      distanceM,
      speedMps: distanceM / Math.max(durationS, 1),
      altitudeStartM: a.alt * 0.3048,
      altitudeEndM: b.alt * 0.3048,
      lat1: a.lat,
      lon1: a.lon,
      lat2: b.lat,
      lon2: b.lon,
    })
  }
  return segs
}

function fmtTime(sec) {
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  const pad = (x) => String(x).padStart(2, '0')
  return `${pad(h)}:${pad(m)}:${pad(s)}`
}

function initRouteSelect(viewer) {
  els.routeSelect.innerHTML = ''
  SAMPLE_ROUTES.forEach((r, i) => {
    const opt = document.createElement('option')
    opt.value = String(i)
    opt.textContent = `${r.origin} → ${r.destination} · ${r.aircraftType}`
    els.routeSelect.appendChild(opt)
  })
  els.routeSelect.value = '0'
  els.routeSelect.addEventListener('change', () => loadRoute(viewer, SAMPLE_ROUTES[Number(els.routeSelect.value)]))
}

async function loadRoute(viewer, route) {
  state.route = route
  state.segments = buildSegments(route)
  state.totalDurationS = state.segments.reduce((a, s) => a + s.durationS, 0)

  const fuel = new FuelPredictor()
  await fuel.init()
  els.hudEngine.textContent = fuel.engineName

  state.segPredictions = await fuel.predictSegments(state.segments, route)
  state.totalFuelKg = state.segPredictions.reduce((a, p) => a + p.fuelKg, 0)

  // Clear prior entities.
  viewer.entities.removeAll()

  // Flight card values.
  els.hudOrigin.textContent = route.origin
  els.hudOriginName.textContent = route.originName
  els.hudDest.textContent = route.destination
  els.hudDestName.textContent = route.destinationName
  els.hudAircraft.textContent = route.aircraftType
  els.hudDistance.textContent = `${(
    state.segments.reduce((a, s) => a + s.distanceM, 0) / 1000
  ).toFixed(0)} km`
  els.hudFuelUsed.textContent = '0 kg'
  els.hudFuelRemaining.textContent = `${state.totalFuelKg.toFixed(0)} kg`
  els.hudFuelTotal.textContent = `${state.totalFuelKg.toFixed(0)} kg`
  els.hudProgress.textContent = '0%'
  els.progressBarFill.style.width = '0%'
  els.fuelPct.innerHTML = '100<span>%</span>'

  // Route polyline with glow.
  const positions = route.waypoints.map((w) => Cesium.Cartesian3.fromDegrees(w.lon, w.lat, w.alt))
  viewer.entities.add({
    polyline: {
      positions,
      width: 3,
      material: new Cesium.PolylineGlowMaterialProperty({
        glowPower: 0.22,
        color: Cesium.Color.fromCssColorString('#dc1414').withAlpha(0.85),
      }),
    },
  })

  // Segment markers colored by burn.
  const maxFuel = Math.max(...state.segPredictions.map((p) => p.fuelKg), 1e-6)
  state.segments.forEach((seg, i) => {
    const midLat = (seg.lat1 + seg.lat2) / 2
    const midLon = (seg.lon1 + seg.lon2) / 2
    const midAlt = (seg.altitudeStartM + seg.altitudeEndM) / 2
    const burnFrac = state.segPredictions[i].fuelKg / maxFuel
    const color =
      burnFrac < 0.35
        ? Cesium.Color.fromCssColorString('#ffffff').withAlpha(0.8)
        : burnFrac < 0.7
          ? Cesium.Color.fromCssColorString('#dc1414').withAlpha(0.6)
          : Cesium.Color.fromCssColorString('#dc1414')
    viewer.entities.add({
      point: {
        position: Cesium.Cartesian3.fromDegrees(midLon, midLat, midAlt),
        pixelSize: 7,
        color,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        outlineColor: Cesium.Color.WHITE.withAlpha(0.25),
        outlineWidth: 1,
      },
    })
  })

  // Origin / destination labels.
  const start = route.waypoints[0]
  const end = route.waypoints[route.waypoints.length - 1]
  viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(start.lon, start.lat, start.alt + 2000),
    point: { pixelSize: 10, color: Cesium.Color.WHITE, disableDepthTestDistance: Number.POSITIVE_INFINITY },
    label: {
      text: route.origin,
      font: 'bold 13px sans-serif',
      fillColor: Cesium.Color.WHITE,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      outlineWidth: 3,
      outlineColor: Cesium.Color.BLACK,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      pixelOffset: new Cesium.Cartesian2(0, 18),
    },
  })
  viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(end.lon, end.lat, end.alt + 2000),
    point: { pixelSize: 10, color: Cesium.Color.WHITE, disableDepthTestDistance: Number.POSITIVE_INFINITY },
    label: {
      text: route.destination,
      font: 'bold 13px sans-serif',
      fillColor: Cesium.Color.WHITE,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      outlineWidth: 3,
      outlineColor: Cesium.Color.BLACK,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      pixelOffset: new Cesium.Cartesian2(0, 18),
    },
  })

  // Aircraft with heading arrow + trail, using a plane billboard.
  state.plane = viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(start.lon, start.lat, start.alt + 1000),
    point: {
      pixelSize: 14,
      color: Cesium.Color.fromCssColorString('#dc1414'),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 2,
    },
    label: {
      text: '✈',
      font: '20px sans-serif',
      fillColor: Cesium.Color.WHITE,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      pixelOffset: new Cesium.Cartesian2(0, -10),
    },
    path: {
      leadTime: 0,
      trailTime: Math.min(state.totalDurationS, 3600),
      width: 2,
      material: new Cesium.PolylineGlowMaterialProperty({
        glowPower: 0.15,
        color: Cesium.Color.fromCssColorString('#dc1414').withAlpha(0.7),
      }),
    },
  })

  // Reset clock.
  const now = Cesium.JulianDate.fromDate(new Date())
  viewer.clock.startTime = now
  viewer.clock.stopTime = Cesium.JulianDate.addSeconds(now, state.totalDurationS, new Cesium.JulianDate())
  viewer.clock.currentTime = Cesium.JulianDate.clone(now)
  viewer.clock.shouldAnimate = !state.paused
  updateClock(viewer)
  state.running = true

  // Initial camera framing.
  const pos = Cesium.Cartesian3.fromDegrees(
    (start.lon + end.lon) / 2,
    (start.lat + end.lat) / 2,
    0,
  )
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(
      (start.lon + end.lon) / 2,
      (start.lat + end.lat) / 2,
      4000000,
    ),
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
  els.speedUp.addEventListener('click', () => {
    state.speedIdx = Math.min(SPEED_STEPS.length - 1, state.speedIdx + 1)
    updateClock(viewer)
  })
  els.speedDown.addEventListener('click', () => {
    state.speedIdx = Math.max(0, state.speedIdx - 1)
    updateClock(viewer)
  })
  document.querySelectorAll('#cameraMode .cn-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.cameraMode = btn.dataset.cam
      document.querySelectorAll('#cameraMode .cn-btn').forEach((b) => b.classList.toggle('active', b === btn))
    })
  })
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
  viewer.scene.globe.enableLighting = true
  viewer.scene.globe.atmosphereLightFactor = 1.2
  viewer.scene.skyAtmosphere.show = true

  initRouteSelect(viewer)
  bindControls(viewer)
  await loadRoute(viewer, state.route)

  // Progress + HUD tick.
  viewer.clock.onTick.addEventListener((clock) => {
    if (!state.running) return
    const t = Math.max(
      0,
      Math.min(1, Cesium.JulianDate.secondsDifference(clock.currentTime, clock.startTime) / state.totalDurationS),
    )
    const segs = state.segments
    const fp = t * (segs.length - 1)
    const segIdx = Math.min(segs.length - 1, Math.floor(fp))
    const segFrac = fp - Math.floor(fp)
    const seg = segs[segIdx]

    const lat = seg.lat1 + (seg.lat2 - seg.lat1) * segFrac
    const lon = seg.lon1 + (seg.lon2 - seg.lon1) * segFrac
    const alt = seg.altitudeStartM + (seg.altitudeEndM - seg.altitudeStartM) * segFrac + 500
    state.plane.position = new Cesium.ConstantPositionProperty(Cesium.Cartesian3.fromDegrees(lon, lat, alt))

    const elapsed = t * state.totalDurationS
    const fuelUsed =
      state.segPredictions.slice(0, segIdx).reduce((a, p) => a + p.fuelKg, 0) +
      state.segPredictions[segIdx].fuelKg * segFrac
    const remaining = Math.max(0, state.totalFuelKg - fuelUsed)
    const pct = (remaining / state.totalFuelKg) * 100

    els.hudProgress.textContent = `${(t * 100).toFixed(1)}%`
    els.progressBarFill.style.width = `${t * 100}%`
    els.hudElapsed.textContent = fmtTime(elapsed)
    els.hudFuelUsed.textContent = `${fuelUsed.toFixed(0)} kg`
    els.hudFuelRemaining.textContent = `${remaining.toFixed(0)} kg`
    els.fuelPct.innerHTML = `${pct.toFixed(0)}<span>%</span>`
    els.fuelPct.style.color = pct > 40 ? '#ffffff' : pct > 20 ? '#dc1414' : '#dc1414'
  })

  // Camera follow / overview.
  viewer.clock.onTick.addEventListener(() => {
    const pos = state.plane.position.getValue(viewer.clock.currentTime)
    if (!pos || state.cameraMode !== 'follow') return
    viewer.camera.lookAt(pos, new Cesium.Cartesian3(0, 0, 12000))
  })

  // Three.js gauge.
  const gauge = new FuelGauge3D(document.getElementById('fuelGaugeWrap'))
  window.addEventListener('resize', () => gauge.onResize())
  setInterval(() => {
    const remaining = parseFloat(els.hudFuelRemaining.textContent) || 0
    gauge.updateFuel(remaining / state.totalFuelKg)
  }, 200)
}

// Boot the simulation: reveal the UI, then run the existing init flow.
async function launch() {
  const home = document.getElementById('homeOverlay')
  const simUI = document.getElementById('simUI')
  home.classList.add('hidden')
  simUI.classList.add('ready')
  // Wait one frame so the container is laid out before Cesium measures it.
  await new Promise((r) => requestAnimationFrame(() => r()))
  await main()
}

// Wire the launch button once the DOM is ready.
document.getElementById('launchBtn').addEventListener('click', () => {
  launch().catch((err) => {
    console.error(err)
    const e = document.getElementById('hudEngine')
    if (e) e.textContent = 'Error: ' + err.message
  })
})
