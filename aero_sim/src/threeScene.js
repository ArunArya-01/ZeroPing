// Three.js fuel-tank gauge: a transparent cylindrical vessel whose liquid level
// rises and falls linearly. The outer shell and measurement rings rotate gently for
// a polished look, but the liquid itself stays axis-aligned so it reads as a real fill.

import * as THREE from 'three'

// Tank geometry (units).
const SHELL_RADIUS = 0.58
const SHELL_HALF_HEIGHT = 0.78 // shell spans y: -0.78 .. +0.78
const LIQUID_RADIUS = 0.50 // slightly inside the shell wall
const LIQUID_HALF_HEIGHT = SHELL_HALF_HEIGHT - 0.02

export class FuelGauge3D {
  constructor(container) {
    this.container = container
    this.width = container.clientWidth || 190
    this.height = container.clientHeight || 190
    this.fraction = 1

    this.scene = new THREE.Scene()
    this.scene.background = null

    // Slightly wider view for the tank to breathe inside the panel.
    this.camera = new THREE.PerspectiveCamera(40, this.width / this.height, 0.1, 100)
    this.camera.position.set(2.0, 1.3, 2.6)
    this.camera.lookAt(0, 0.1, 0)

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setSize(this.width, this.height)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.append(this.renderer.domElement)

    this._buildLights()
    this._buildTank()
    this._setLevel(1, true)
    this._animate()
  }

  _buildLights() {
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
    const key = new THREE.DirectionalLight(0xffffff, 0.95)
    key.position.set(2, 3, 3)
    const fill = new THREE.DirectionalLight(0x46d7a0, 0.6)
    fill.position.set(-1.5, 1, -1)
    this.scene.add(key, fill)
  }

  _buildTank() {
    // --- Rotating group: shell + caps + measurement rings (no liquid). ---
    const outer = new THREE.Group()

    const glass = new THREE.Mesh(
      new THREE.CylinderGeometry(SHELL_RADIUS, SHELL_RADIUS, SHELL_HALF_HEIGHT * 2, 40, 1, true),
      new THREE.MeshPhysicalMaterial({
        color: 0x9fc1e8,
        transparent: true,
        opacity: 0.14,
        roughness: 0.12,
        metalness: 0.2,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    )
    outer.add(glass)

    for (const y of [-0.55, 0.55]) {
      const band = new THREE.Mesh(
        new THREE.TorusGeometry(SHELL_RADIUS, 0.012, 10, 48),
        new THREE.MeshPhongMaterial({
          color: 0x3a4a66,
          emissive: 0x141c2a,
          transparent: true,
          opacity: 0.5,
        }),
      )
      band.rotation.x = Math.PI / 2
      band.position.y = y
      outer.add(band)
    }

    const cap = new THREE.Mesh(
      new THREE.CylinderGeometry(SHELL_RADIUS + 0.04, SHELL_RADIUS + 0.04, 0.1, 40),
      new THREE.MeshPhongMaterial({ color: 0x232f45, emissive: 0x0c1320, shininess: 40 }),
    )
    cap.rotation.x = Math.PI / 2
    cap.position.y = SHELL_HALF_HEIGHT + 0.05
    outer.add(cap)

    this.scene.add(outer)
    this.outer = outer

    // Liquid: fixed (non-rotating) cylinder clipped inside the shell.
    this.liquid = new THREE.Mesh(
      new THREE.CylinderGeometry(LIQUID_RADIUS, LIQUID_RADIUS, LIQUID_HALF_HEIGHT * 2, 40, 1),
      new THREE.MeshPhongMaterial({
        color: 0xdc1414,
        transparent: true,
        opacity: 0.9,
        shininess: 110,
        side: THREE.DoubleSide,
        emissive: 0x3a0606,
      }),
    )
    // CylinderGeometry's axis is already along Y (up), so the cylinder is
    // vertical by default. We reposition it in _setLevel to match the fill line.
    this.scene.add(this.liquid)

    // Glowing fill light that follows the liquid surface.
    this.glow = new THREE.PointLight(0xdc1414, 1.4, 3)
    this.glow.position.y = -0.3
    this.scene.add(this.glow)
  }

  // Set the liquid level. f in [0, 1]. Liquid occupies the shell bottom upward;
  // the cylinder is centered at the fill midpoint so only the volume between the
  // bottom of the shell and the fill line is opaque.
  _setLevel(f, instant = false) {
    f = Math.min(1, Math.max(0, f))
    const bottom = -SHELL_HALF_HEIGHT
    const top = bottom + f * (SHELL_HALF_HEIGHT * 2 - 0.04)
    const mid = (bottom + top) / 2
    const cylHalf = Math.max(0.002, (top - bottom) / 2)

    if (instant) {
      this.liquid.position.y = mid
      this.liquid.scale.y = cylHalf / LIQUID_HALF_HEIGHT
    } else {
      this._targetMid = mid
      this._targetCylHalf = cylHalf
    }

    // Color: bright red when full, dimming toward dark as fuel depletes.
    const lightness = 0.15 + f * 0.35
    this.liquid.material.color.setHSL(0, 0.85, lightness)
    this.liquid.material.emissive.setHSL(0, 0.9, lightness * 0.4)
    this.glow.color.setHSL(0, 0.85, 0.5)
    this.glow.intensity = 0.5 + f * 1.8
  }

  updateFuel(fraction) {
    this.fraction = Math.min(1, Math.max(0, fraction))
    this._setLevel(this.fraction)
  }

  _animate() {
    requestAnimationFrame(() => this._animate())

    // Rotate only the vessel for a living, cinematic look.
    this.outer.rotation.y += 0.004

    if (this._targetMid !== undefined) {
      const dMid = this._targetMid - this.liquid.position.y
      const dScale = this._targetCylHalf / LIQUID_HALF_HEIGHT - this.liquid.scale.y
      this.liquid.position.y += dMid * 0.1
      this.liquid.scale.y += dScale * 0.1
    }
    // Keep liquid red, dimming as it empties.
    const lightness = 0.15 + this.fraction * 0.35
    this.liquid.material.color.setHSL(0, 0.85, lightness)
    this.liquid.material.emissive.setHSL(0, 0.9, lightness * 0.4)
    this.glow.intensity = 0.5 + this.fraction * 1.8
    this.renderer.render(this.scene, this.camera)
  }

  onResize() {
    this.width = this.container.clientWidth || 190
    this.height = this.container.clientHeight || 190
    this.camera.aspect = this.width / this.height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(this.width, this.height)
  }
}
