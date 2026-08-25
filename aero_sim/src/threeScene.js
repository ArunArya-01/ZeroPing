// Three.js fuel-tank overlay with live fill, gradient liquid, and measurement ring.

import * as THREE from 'three'

export class FuelGauge3D {
  constructor(container) {
    this.container = container
    const width = container.clientWidth || 190
    const height = container.clientHeight || 190
    this.fraction = 1

    this.scene = new THREE.Scene()
    this.scene.background = null // transparent, panel provides bg

    this.camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100)
    this.camera.position.set(2.4, 1.7, 3.0)
    this.camera.lookAt(0, 0, 0)

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setSize(width, height)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.append(this.renderer.domElement)

    this._buildLights()
    this._buildTank()
    this._animate()
  }

  _buildLights() {
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.5))
    const key = new THREE.DirectionalLight(0xffffff, 1.0)
    key.position.set(2, 3, 2)
    const rim = new THREE.DirectionalLight(0x42a5f5, 0.5)
    rim.position.set(-2, 1.5, -2)
    const under = new THREE.DirectionalLight(0xffa726, 0.25)
    under.position.set(0, -2, 1)
    this.scene.add(key, rim, under)
  }

  _buildTank() {
    const group = new THREE.Group()

    // Glass cylinder shell.
    const shellGeo = new THREE.CylinderGeometry(0.62, 0.62, 1.6, 40, 1, true)
    const shellMat = new THREE.MeshPhysicalMaterial({
      color: 0x9fc1e8,
      transparent: true,
      opacity: 0.16,
      roughness: 0.1,
      metalness: 0.3,
      side: THREE.DoubleSide,
      depthWrite: false,
    })
    const shell = new THREE.Mesh(shellGeo, shellMat)
    shell.rotation.x = Math.PI / 2
    group.add(shell)

    // Measurement ring base.
    const ringGeo = new THREE.TorusGeometry(0.62, 0.02, 12, 48)
    const ringMat = new THREE.MeshPhongMaterial({ color: 0x46d7a0, emissive: 0x0a4433 })
    const ring = new THREE.Mesh(ringGeo, ringMat)
    ring.rotation.x = Math.PI / 2
    group.add(ring)

    // Cap.
    const capGeo = new THREE.CylinderGeometry(0.65, 0.65, 0.1, 40)
    const capMat = new THREE.MeshPhongMaterial({ color: 0x2a3648, emissive: 0x101820 })
    const cap = new THREE.Mesh(capGeo, capMat)
    cap.rotation.x = Math.PI / 2
    cap.position.y = 0.8
    group.add(cap)

    // Rounded liquid (with top gloss).
    const liquidGeo = new THREE.SphereGeometry(0.54, 40, 32)
    const liquidMat = new THREE.MeshPhongMaterial({
      color: 0x46d7a0,
      transparent: true,
      opacity: 0.85,
      shininess: 90,
      emissive: 0x0a2f1f,
    })
    this.liquid = new THREE.Mesh(liquidGeo, liquidMat)
    this.liquid.scale.set(1, 1.4, 1)
    this.liquid.position.y = -0.7
    group.add(this.liquid)

    // Inner glow light for empty fuel.
    this.glow = new THREE.PointLight(0x46d7a0, 1.5, 3)
    this.glow.position.y = -0.4
    group.add(this.glow)

    this.scene.add(group)
    this.group = group
    this._updateMesh(1)
  }

  _updateMesh(fraction) {
    const f = Math.max(0, Math.min(1, fraction))
    // Liquid rises inside the shell.
    const liquidScaleY = Math.max(0.001, f * 1.35 + 0.05)
    this.liquid.scale.y = liquidScaleY
    this.liquid.position.y = -0.7 + (liquidScaleY / 2 - 0.05) + (1 - f) * 0.08
    this.liquid.visible = f > 0.002

    // Color shifts red (empty) -> green (full).
    const hue = f * 0.36
    this.liquid.material.color.setHSL(hue, 0.8, 0.5)
    this.liquid.material.emissive.setHSL(hue, 0.9, 0.12)
    this.glow.color.setHSL(hue, 0.9, 0.5)
    this.glow.intensity = 0.6 + f * 1.8
  }

  updateFuel(fraction) {
    this.fraction = Math.min(1, Math.max(0, fraction))
  }

  _animate() {
    requestAnimationFrame(() => this._animate())
    this.group.rotation.y += 0.004
    // Smoothly approach target fill for a nicer motion.
    const current = this.liquid.scale.y
    const target = Math.max(0.001, this.fraction * 1.35 + 0.05)
    const ease = current + (target - current) * 0.08
    this.liquid.scale.y = ease
    this.liquid.position.y = -0.7 + (ease / 2 - 0.05) + (1 - this.fraction) * 0.08
    this.liquid.material.color.setHSL(this.fraction * 0.36, 0.8, 0.5)
    this.liquid.material.emissive.setHSL(this.fraction * 0.36, 0.9, 0.12)
    this.glow.color.setHSL(this.fraction * 0.36, 0.9, 0.5)
    this.glow.intensity = 0.6 + this.fraction * 1.8
    this.renderer.render(this.scene, this.camera)
  }

  onResize() {
    const width = this.container.clientWidth || 190
    const height = this.container.clientHeight || 190
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height)
  }
}
