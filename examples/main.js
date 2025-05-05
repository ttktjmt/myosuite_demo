import * as THREE from 'three';
import { GUI } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import { RLController, isRLSupported } from './rlUtils.js';
import load_mujoco from '../dist/mujoco_wasm.js';

const mujoco = await load_mujoco();

// Set up Emscripten's Virtual File System
var initialScene = "myo_sim/hand/myo_hand_combined.xml";
// var initialScene = "myo_sim/arm/myoarm_bionic_bimanual.mjb";
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
await downloadExampleScenesFolder(mujoco);

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;
    this.rlController = new RLController();

    this.params = {
      scene: initialScene,
      paused: false,
      help: false,
      ctrlnoiserate: 0.0,
      ctrlnoisestd: 0.0,
      keyframeNumber: 0,
      rlControl: false,
      policy: "baseline",
      rlUpdateInterval: 100
    };

    this.mujoco_time = 0.0;
    this.lastRLUpdateTime = 0;
    this.bodies = {};
    this.lights = {};
    this.tmpVec = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];
    this.model = null;
    this.state = null;
    this.simulation = null;

    this.container = document.createElement('div');
    document.body.appendChild(this.container);

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100);
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(0.3, 1.5, 1.2);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5);

    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
    this.ambientLight.name = 'AmbientLight';
    this.scene.add(this.ambientLight);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    this.renderer.setAnimationLoop(this.render.bind(this));

    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(-0.2, 1.4, 0.4);
    this.controls.panSpeed = 1;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.state, this.simulation, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, initialScene, this);

    this.gui = new GUI();
    setupGUI(this);

    // Load the RL model if the initial scene supports it
    if (isRLSupported(this.params.scene)) {
      try {
        // Point directly to the baseline.onnx file in the models directory
        const modelPath = `./examples/models/${this.params.policy}.onnx`;
        console.log(`Attempting to load RL model from: ${modelPath}`);

        // Use fetch to check if the model file exists before trying to load it
        try {
          const response = await fetch(modelPath, { method: 'HEAD' });
          if (!response.ok) {
            console.error(`RL model file not found at ${modelPath}`, response.status);
            return;
          }
        } catch (error) {
          console.error(`Error checking RL model file: ${modelPath}`, error);
          return;
        }

        // Once we've confirmed the file exists, try to load it
        const modelLoaded = await this.rlController.loadModel(modelPath);
        console.log(`RL model ${modelLoaded ? 'successfully loaded' : 'failed to load'}`);

        // Get and display model diagnostics to help troubleshoot issues
        if (modelLoaded) {
          console.log('Running model diagnostics:');
          const diagnostics = this.rlController.getModelDiagnostics();
          console.log('Model input shape:', diagnostics.inputShapes);
          console.log('Model output shape:', diagnostics.outputShapes);
        }
      } catch (error) {
        console.error('Failed to load RL model:', error);
      }
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  async updateRLControl(timeMS) {
    // Check if RL control is enabled and enough time has passed since the last update
    if (!this.params.rlControl ||
      timeMS - this.lastRLUpdateTime < this.params.rlUpdateInterval ||
      !this.rlController.isModelLoaded) {
      return;
    }

    try {
      const observation = this.rlController.getObservation(this.simulation, this.model);
      const action = await this.rlController.runInference(observation);

      if (action) {
        this.rlController.applyAction(this.simulation, action);

        // Update the sliders in the GUI to reflect the new control values
        for (let i = 0; i < this.simulation.ctrl.length; i++) {
          // Get the actual actuator name from the model
          const textDecoder = new TextDecoder("utf-8");
          const nullChar = textDecoder.decode(new ArrayBuffer(1));
          const actuatorName = textDecoder.decode(
            this.model.names.subarray(this.model.name_actuatoradr[i])
          ).split(nullChar)[0];

          if (this.params[actuatorName] !== undefined) {
            // Ensure the value is within the valid range
            const ctrlValue = this.simulation.ctrl[i];
            const actRange = this.model.actuator_ctrlrange;
            if (actRange && i * 2 + 1 < actRange.length) {
              const minValue = actRange[i * 2];
              const maxValue = actRange[i * 2 + 1];
              // Clamp the value to the actuator's range
              const clampedValue = Math.max(minValue, Math.min(maxValue, ctrlValue));
              this.params[actuatorName] = clampedValue;
            } else {
              this.params[actuatorName] = ctrlValue;
            }
          }
        }
      }
      this.lastRLUpdateTime = timeMS;
    } catch (error) {
      console.error('[RL Control] Error:', error);
    }
  }

  render(timeMS) {
    this.controls.update();

    if (!this.params["paused"]) {
      let timestep = this.model.getOptions().timestep;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }

      // Update RL control if enabled
      this.updateRLControl(timeMS);

      while (this.mujoco_time < timeMS) {

        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.simulation.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.simulation.qfrc_applied.length; i++) { this.simulation.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition(this.simulation.xpos, b, this.bodies[b].position);
              getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(250));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);

          // TODO: Apply pose perturbations (mocap bodies only).
        }

        this.simulation.step();

        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update(); // Update the world-space force origin
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition(this.simulation.xpos, b, this.tmpVec, false); // Get raw coordinate from MuJoCo
        getQuaternion(this.simulation.xquat, b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          // Set the root body's mocap position...
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos = this.simulation.mocap_pos;
          pos[addr + 0] += offset.x;
          pos[addr + 1] += offset.y;
          pos[addr + 2] += offset.z;
        } else {
          // Set the root body's position directly...
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos = this.simulation.qpos;
          pos[addr + 0] += offset.x;
          pos[addr + 1] += offset.y;
          pos[addr + 2] += offset.z;
        }
      }

      this.simulation.forward();
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.simulation.xpos, b, this.bodies[b].position);
        getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.simulation.light_xpos, l, this.lights[l].position);
        getPosition(this.simulation.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Update tendon transforms.
    let numWraps = 0;
    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      let mat = new THREE.Matrix4();
      for (let t = 0; t < this.model.ntendon; t++) {
        let startW = this.simulation.ten_wrapadr[t];
        let r = this.model.tendon_width[t];
        for (let w = startW; w < startW + this.simulation.ten_wrapnum[t] - 1; w++) {
          let tendonStart = getPosition(this.simulation.wrap_xpos, w, new THREE.Vector3());
          let tendonEnd = getPosition(this.simulation.wrap_xpos, w + 1, new THREE.Vector3());
          let tendonAvg = new THREE.Vector3().addVectors(tendonStart, tendonEnd).multiplyScalar(0.5);

          let validStart = tendonStart.length() > 0.01;
          let validEnd = tendonEnd.length() > 0.01;

          if (validStart) { this.mujocoRoot.spheres.setMatrixAt(numWraps, mat.compose(tendonStart, new THREE.Quaternion(), new THREE.Vector3(r, r, r))); }
          if (validEnd) { this.mujocoRoot.spheres.setMatrixAt(numWraps + 1, mat.compose(tendonEnd, new THREE.Quaternion(), new THREE.Vector3(r, r, r))); }
          if (validStart && validEnd) {
            mat.compose(tendonAvg, new THREE.Quaternion().setFromUnitVectors(
              new THREE.Vector3(0, 1, 0), tendonEnd.clone().sub(tendonStart).normalize()),
              new THREE.Vector3(r, tendonStart.distanceTo(tendonEnd), r));
            this.mujocoRoot.cylinders.setMatrixAt(numWraps, mat);
            numWraps++;
          }
        }
      }
      this.mujocoRoot.cylinders.count = numWraps;
      this.mujocoRoot.spheres.count = numWraps > 0 ? numWraps + 1 : 0;
      this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
      this.mujocoRoot.spheres.instanceMatrix.needsUpdate = true;
    }

    // Render!
    this.renderer.render(this.scene, this.camera);
  }
}

let demo = new MuJoCoDemo();
await demo.init();
