// Main entry: Cesium globe + Three.js fuel gauge + ONNX/Physics fuel model.

import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'

import { SAMPLE_ROUTES } from './data/routes.js'
import { FuelPredictor } from './fuel.js'
import { FuelGauge3D } from './threeScene.js'

const ROUTE = SAMPLE_ROUTES[0]

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
  const segs = []
  for (let i = 0; i < wp.length - 1; i++) {
    const a = wp[i]
    const b = wp[i + 1]
    const distanceM = haversineM(a.lat, a.lon, b.lat, b.lon)
    const durationS = totalDur / (wp.length - 1)
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

async function main() {
  const viewer = new Cesium.Viewer('cesiumContainer', {
    timeline: true,
    animation: true,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: true,
    sceneModePicker: false,
    navigationHelpButton: false,
    infoBox: false,
    selectionIndicator: false,
  })
  viewer.scene.globe.enableLighting = true
  viewer.scene.globe.depthTestAgainstTerrain = true
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(-20, 50, 1800000),
  })

  const segments = buildSegments(ROUTE)
  const totalDurationS = segments.reduce((acc, s) => acc + s.durationS, 0)

  // Fuel model (ONNX in-browser or physics fallback).
  const fuel = new FuelPredictor()
  await fuel.init()

  const segPredictions = await fuel.predictSegments(segments, ROUTE)
  const totalFuelKg = segPredictions.reduce((acc, p) => acc + p.fuelKg, 0)

  document.getElementById('hudRoute').textContent = `${ROUTE.origin} → ${ROUTE.destination}`
  document.getElementById('hudAircraft').textContent = ROUTE.aircraftType
  document.getElementById('hudEngine').textContent = fuel.engineName
  document.getElementById('hudFuelUsed').textContent = '0 kg'
  document.getElementById('hudFuelRemaining').textContent = `${totalFuelKg.toFixed(0)} kg`
  document.getElementById('hudDistance').textContent = `${(
    segments.reduce((a, s) => a + s.distanceM, 0) / 1000
  ).toFixed(0)} km`

  // Build Cesium positions (Cartesian3) from waypoints.
  const positions = ROUTE.waypoints.map((w) =>
    Cesium.Cartesian3.fromDegrees(w.lon, w.lat, w.alt),
  )

  // Polyline route.
  viewer.entities.add({
    polyline: {
      positions,
      width: 4,
      material: Cesium.Color.WHITE.withAlpha(0.7),
    },
  })

  // Segment markers, colored by predicted fuel burn.
  const maxFuel = Math.max(...segPredictions.map((p) => p.fuelKg), 1e-6)
  segments.forEach((seg, i) => {
    const midLat = (seg.lat1 + seg.lat2) / 2
    const midLon = (seg.lon1 + seg.lon2) / 2
    const midAlt = (seg.altitudeStartM + seg.altitudeEndM) / 2
    const burnFrac = segPredictions[i].fuelKg / maxFuel
    const color =
      burnFrac < 0.35
        ? Cesium.Color.LIMEGREEN
        : burnFrac < 0.7
          ? Cesium.Color.ORANGE
          : Cesium.Color.RED
    viewer.entities.add({
      point: {
        position: Cesium.Cartesian3.fromDegrees(midLon, midLat, midAlt),
        pixelSize: 6,
        color,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `${segPredictions[i].fuelKg.toFixed(0)} kg`,
        font: '10px sans-serif',
        fillColor: Cesium.Color.WHITE,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        pixelOffset: new Cesium.Cartesian2(0, -14),
      },
    })
  })

  // Animated aircraft along the route.
  const startPos = positions[0]
  const plane = viewer.entities.add({
    position: startPos,
    point: {
      pixelSize: 12,
      color: Cesium.Color.DODGERBLUE,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 2,
    },
  })

  let lastTick = null

  // Set up the clock to run the route.
  const now = Cesium.JulianDate.fromDate(new Date())
  viewer.clock.startTime = now
  viewer.clock.stopTime = Cesium.JulianDate.addSeconds(now, totalDurationS, new Cesium.JulianDate())
  viewer.clock.currentTime = Cesium.JulianDate.clone(now)
  viewer.clock.shouldAnimate = true
  viewer.clock.multiplier = 100 // 100x real time

  viewer.clock.onTick.addEventListener((clock) => {
    if (lastTick === null) lastTick = clock.currentTime
    const t = Math.max(
      0,
      Math.min(1, Cesium.JulianDate.secondsDifference(clock.currentTime, clock.startTime) / totalDurationS),
    )
    lastTick = clock.currentTime

    // Interpolate along the route.
    const fp = t * (segments.length - 1)
    const segIdx = Math.min(segments.length - 1, Math.floor(fp))
    const segFrac = fp - Math.floor(fp)
    const seg = segments[segIdx]
    const lat = seg.lat1 + (seg.lat2 - seg.lat1) * segFrac
    const lon = seg.lon1 + (seg.lon2 - seg.lon1) * segFrac
    const alt = seg.altitudeStartM + (seg.altitudeEndM - seg.altitudeStartM) * segFrac + 500

    plane.position = new Cesium.ConstantPositionProperty(
      Cesium.Cartesian3.fromDegrees(lon, lat, alt),
    )

    // HUD updates.
    const fuelUsed =
      segPredictions.slice(0, segIdx).reduce((a, p) => a + p.fuelKg, 0) +
      segPredictions[segIdx].fuelKg * segFrac
    document.getElementById('hudProgress').textContent = `${(t * 100).toFixed(1)}%`
    document.getElementById('hudFuelUsed').textContent = `${fuelUsed.toFixed(0)} kg`
    document.getElementById('hudFuelRemaining').textContent = `${(totalFuelKg - fuelUsed).toFixed(0)} kg`
  })

  // Follow the aircraft.
  viewer.clock.onTick.addEventListener(() => {
    const pos = plane.position.getValue(viewer.clock.currentTime)
    if (pos) {
      viewer.camera.lookAt(pos, new Cesium.Cartesian3(0, 0, 30000))
    }
  })

  // Three.js fuel gauge overlay.
  const gaugeContainer = document.getElementById('fuelGaugeWrap')
  const gauge = new FuelGauge3D(gaugeContainer)
  window.addEventListener('resize', () => gauge.onResize())

  setInterval(() => {
    const remaining = parseFloat(document.getElementById('hudFuelRemaining').textContent) || 0
    gauge.updateFuel(remaining / totalFuelKg)
  }, 250)
}

main().catch((err) => {
  console.error(err)
  document.getElementById('hudEngine').textContent = 'Error: ' + err.message
})
