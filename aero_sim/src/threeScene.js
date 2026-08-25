// Three.js fuel-tank overlay rendered in the corner HUD.
// A 3D tank whose fill level animates with predicted remaining fuel.

import * as THREE from 'three'

export class FuelGauge3D {
  constructor(container) {
    this.container = container
    const width = container.clientWidth
    const height = container.clientHeight

    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x0a0e16)

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100)
    this.camera.position.set(2.6, 1.8, 3.2)
    this.camera.lookAt(0, 0, 0)

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setSize(width, height)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.prepend(this.renderer.domElement)

    this._buildTank()
    this._buildLights()
    this._animate()
  }

  _buildLights() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.55)
    const key = new THREE.DirectionalLight(0xffffff, 0.9)
    key.position.set(2, 3, 2)
    const rim = new THREE.DirectionalLight(0x42a5f5, 0.4)
    rim.position.set(-2, 1, -2)
    this.scene.add(ambient, key, rim)
  }

  _buildTank() {
    const tankGeo = new THREE.CylinderGeometry(0.62, 0.62, 1.7, 32, 1, true)
    const tankMat = new THREE.MeshPhongMaterial({
      color: 0x3a4a5e,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
    })
    this.tank = new THREE.Mesh(tankGeo, tankMat)
    this.tank.rotation.x = Math.PI / 2
    this.scene.add(this.tank)

    const liquidGeo = new THREE.CylinderGeometry(0.54, 0.54, 1.0, 32)
    const liquidMat = new THREE.MeshPhongMaterial({ color: 0x3ddc84 })
    this.liquid = new THREE.Mesh(liquidGeo, liquidMat)
    this.liquid.rotation.x = Math.PI / 2
    this.liquid.position.y = -0.35
    this.scene.add(this.liquid)

    const capGeo = new THREE.CylinderGeometry(0.66, 0.66, 0.08, 32)
    const capMat = new THREE.MeshPhongMaterial({ color: 0x93a4bc, metalness: 0.6 })
    this.cap = new THREE.Mesh(capGeo, capMat)
    this.cap.rotation.x = Math.PI / 2
    this.cap.position.y = 0.85
    this.scene.add(this.cap)
  }

  updateFuel(remainingFraction) {
    const f = Math.min(1, Math.max(0, remainingFraction))
    this.liquid.scale.y = Math.max(0.001, f)
    this.liquid.position.y = -0.35 + f * 0.7
    const hue = f * 0.35 // 0 red-ish -> 0.35 green
    this.liquid.material.color.setHSL(hue, 0.75, 0.5)
  }

  _animate() {
    requestAnimationFrame(() => this._animate())
    this.tank.rotation.y += 0.003
    this.renderer.render(this.scene, this.camera)
  }

  onResize() {
    const width = this.container.clientWidth
    const height = this.container.clientHeight
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height)
  }
}
