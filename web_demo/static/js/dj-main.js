// Live Mixer App Logic - adapted from main.js to use DJMixer instead of AudioCapture

// Global mixer instance
let djMixer = null;

function updateMeter(channelIndex, value) {
    const meter = document.getElementById(`meter${channelIndex}`);
    if (!meter) return;
    const segments = meter.querySelectorAll('.meter-segment');
    const litCount = Math.round((parseInt(value) / 100) * segments.length);
    segments.forEach((seg, i) => {
        seg.classList.toggle('lit', i < litCount);
    });
}

function onFaderInput(channelIndex, value) {
    const intValue = parseInt(value);
    const valueEl = document.getElementById(`faderValue${channelIndex}`);
    if (valueEl) valueEl.textContent = intValue;
    updateMeter(channelIndex, intValue);

    // Initialize mixer on first interaction (user gesture required for AudioContext)
    if (djMixer && !djMixer.isInitialized) {
        djMixer.init().then(() => {
            djMixer.onFaderChange(channelIndex, intValue);
            updateChannelHighlight();
        });
    } else if (djMixer) {
        djMixer.onFaderChange(channelIndex, intValue);
        updateChannelHighlight();
    }
}

function updateChannelHighlight() {
    if (!djMixer) return;
    for (let i = 0; i < djMixer.channels.length; i++) {
        const strip = document.getElementById(`channelStrip${i}`);
        if (strip) {
            if (djMixer.channels[i].currentValue > 10) {
                strip.classList.add('active');
            } else {
                strip.classList.remove('active');
            }
        }
    }
}

async function toggleSpeakerCapture() {
    if (!djMixer) return;

    if (!djMixer.isInitialized) {
        await djMixer.init();
    }

    const btn = document.getElementById('speakerCaptureBtn');

    if (djMixer.isCapturing) {
        djMixer.stopSpeakerCapture();
        btn.textContent = 'Capture';
        btn.classList.remove('active');
    } else {
        const ok = await djMixer.startSpeakerCapture();
        if (ok) {
            btn.textContent = 'Stop';
            btn.classList.add('active');
        }
    }
}


class DJMotionApp {
    constructor() {
        this.isRunning = false;
        this.targetFps = 30;
        this.frameInterval = 1000 / this.targetFps;
        this.lastFetchTime = 0;
        this.nextFetchTime = 0;
        this.frameCount = 0;
        this.fpsCounter = 0;
        this.fpsUpdateTime = 0;

        this.motionFrameCount = 0;
        this.motionFpsCounter = 0;
        this.motionFpsUpdateTime = 0;

        this.isFetchingFrame = false;
        this.consecutiveWaiting = 0;

        this.batchSize = 4;
        this.frameQueue = [];

        this.sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

        this.lastUserInteraction = 0;
        this.autoFollowDelay = 2000;
        this.currentRootPos = new THREE.Vector3(0, 1, 0);

        this.generationMode = 'audio';

        this.initThreeJS();
        this.initUI();
        this.updateStatus();
        this.setupBeforeUnload();

        // Create mixer
        djMixer = new DJMixer();

        console.log('Session ID:', this.sessionId);
    }

    setupBeforeUnload() {
        window.addEventListener('beforeunload', () => {
            if (!this.isIdle) {
                const blob = new Blob(
                    [JSON.stringify({ session_id: this.sessionId })],
                    { type: 'application/json' }
                );
                navigator.sendBeacon('/api/reset', blob);
            }
        });

        document.addEventListener('visibilitychange', () => {
            if (document.hidden && !this.isIdle && this.isRunning) {
                console.log('Tab hidden while generating');
            }
        });
    }

    initThreeJS() {
        const canvas = document.getElementById('renderCanvas');
        const container = document.getElementById('canvas-container');

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a12);

        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(3, 1.5, 3);
        this.camera.lookAt(0, 1, 0);

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
        keyLight.position.set(5, 8, 3);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 50;
        keyLight.shadow.camera.left = -5;
        keyLight.shadow.camera.right = 5;
        keyLight.shadow.camera.top = 5;
        keyLight.shadow.camera.bottom = -5;
        keyLight.shadow.bias = -0.0001;
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-3, 5, -3);
        this.scene.add(fillLight);

        const groundGeometry = new THREE.PlaneGeometry(1000, 1000);
        const groundMaterial = new THREE.ShadowMaterial({ opacity: 0.15 });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        const gridHelper = new THREE.GridHelper(1000, 1000, 0x222233, 0x1a1a28);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);

        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.update();

        const updateInteractionTime = () => { this.lastUserInteraction = Date.now(); };
        canvas.addEventListener('mousedown', updateInteractionTime);
        canvas.addEventListener('wheel', updateInteractionTime);
        canvas.addEventListener('touchstart', updateInteractionTime);

        this.skeleton = new Skeleton3D(this.scene);

        window.addEventListener('resize', () => this.onWindowResize());
        this.animate();
    }

    initUI() {
        this.motionText = document.getElementById('motionText');
        this.historyLength = document.getElementById('historyLength');
        this.denoiseSteps = document.getElementById('denoiseSteps');
        this.smoothingAlpha = document.getElementById('smoothingAlpha');
        this.smoothingValue = document.getElementById('smoothingValue');
        this.currentSmoothing = document.getElementById('currentSmoothing');
        this.currentSteps = document.getElementById('currentSteps');
        this.startResetBtn = document.getElementById('startResetBtn');
        this.updateBtn = document.getElementById('updateBtn');
        this.pauseResumeBtn = document.getElementById('pauseResumeBtn');
        this.statusEl = document.getElementById('status');
        this.bufferSizeEl = document.getElementById('bufferSize');
        this.fpsEl = document.getElementById('fps');
        this.frameCountEl = document.getElementById('frameCount');
        this.conflictWarning = document.getElementById('conflictWarning');
        this.forceTakeoverBtn = document.getElementById('forceTakeoverBtn');
        this.cancelTakeoverBtn = document.getElementById('cancelTakeoverBtn');

        this.statPrepareEl = document.getElementById('statPrepare');
        this.statModelEl = document.getElementById('statModel');
        this.statVAEEl = document.getElementById('statVAE');
        this.statFrameEl = document.getElementById('statFrame');
        this.statPostEl = document.getElementById('statPost');
        this.statResampleEl = document.getElementById('statResample');
        this.statExtractEl = document.getElementById('statExtract');
        this.statTotalEl = document.getElementById('statTotal');

        this.isPaused = false;
        this.isIdle = true;
        this.isProcessing = false;
        this.pendingStartRequest = null;

        this.startResetBtn.addEventListener('click', () => this.toggleStartReset());
        this.pauseResumeBtn.addEventListener('click', () => this.togglePauseResume());
        if (this.forceTakeoverBtn) {
            this.forceTakeoverBtn.addEventListener('click', () => this.handleForceTakeover());
        }
        if (this.cancelTakeoverBtn) {
            this.cancelTakeoverBtn.addEventListener('click', () => this.handleCancelTakeover());
        }

        if (this.smoothingAlpha) {
            this.smoothingAlpha.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value).toFixed(2);
                if (this.smoothingValue) this.smoothingValue.textContent = value;
            });
        }
    }

    async toggleStartReset() {
        if (this.isProcessing) return;
        if (this.isIdle) {
            await this.startAudioGeneration();
        } else {
            await this.reset();
        }
    }

    async startAudioGeneration(force = false) {
        if (this.isProcessing) return;

        // Ensure mixer is initialized
        if (djMixer && !djMixer.isInitialized) {
            this.statusEl.textContent = 'Initializing mixer...';
            await djMixer.init();
        }

        if (!djMixer || !djMixer.isInitialized) {
            alert('Mixer not initialized. Please interact with a fader first.');
            return;
        }

        const historyLength = parseInt(this.historyLength.value) || 30;
        if (historyLength < 10 || historyLength > 200) {
            alert('History length must be between 10 and 200');
            return;
        }

        const denoiseSteps = parseInt(this.denoiseSteps.value) || 10;
        if (denoiseSteps < 5 || denoiseSteps > 50) {
            alert('Denoising steps must be between 5 and 50');
            return;
        }
        if (denoiseSteps % 5 !== 0) {
            alert('Denoising steps must be a multiple of 5');
            return;
        }

        const smoothingAlpha = parseFloat(this.smoothingAlpha.value);

        this.isProcessing = true;
        this.statusEl.textContent = 'Initializing Audio Mode...';

        try {
            const response = await fetch('/api/start_audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                    denoise_steps: denoiseSteps,
                    audio_window_size: 32,
                    sample_rate: djMixer.sampleRate || 44100,
                    force: force
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Enable audio motion mode on the mixer
                djMixer.setAudioMotionMode(true, this.sessionId);

                this.isRunning = true;
                this.isPaused = false;
                this.isIdle = false;
                this.frameCount = 0;
                this.motionFrameCount = 0;
                this.motionFpsCounter = 0;
                this.motionFpsUpdateTime = performance.now();
                this.isFetchingFrame = false;
                this.consecutiveWaiting = 0;
                this.startResetBtn.textContent = 'Reset';
                this.startResetBtn.classList.remove('btn-primary');
                this.startResetBtn.classList.add('btn-danger');
                this.pauseResumeBtn.disabled = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.statusEl.textContent = 'Running (Live Audio)';

                this.startFrameLoop();
                console.log('Audio generation started');
            } else if (response.status === 409 && data.conflict) {
                this.statusEl.textContent = 'Conflict - Another user is generating';
                this.conflictWarning.style.display = 'block';
                this.pendingStartRequest = { mode: 'audio', history_length: historyLength };
                return;
            } else {
                alert('Error: ' + data.message);
                this.statusEl.textContent = 'Idle';
                this.isIdle = true;
                this.isRunning = false;
            }
        } catch (error) {
            console.error('Error starting generation:', error);
            alert('Failed to start generation: ' + error.message);
            this.statusEl.textContent = 'Idle';
            this.isIdle = true;
            this.isRunning = false;
        } finally {
            this.isProcessing = false;
        }
    }

    async togglePauseResume() {
        if (this.isProcessing) return;
        if (this.isPaused) {
            await this.resumeGeneration();
        } else {
            await this.pauseGeneration();
        }
    }

    async pauseGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/pause', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId })
            });
            const data = await response.json();
            if (data.status === 'success') {
                this.isRunning = false;
                this.isPaused = true;
                this.pauseResumeBtn.textContent = 'Resume';
                this.statusEl.textContent = 'Paused';
            }
        } catch (error) {
            console.error('Error pausing:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async resumeGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId })
            });
            const data = await response.json();
            if (data.status === 'success') {
                this.isRunning = true;
                this.isPaused = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.statusEl.textContent = 'Running (Live Audio)';
                this.startFrameLoop();
            }
        } catch (error) {
            console.error('Error resuming:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async reset() {
        if (this.isProcessing) return;

        const historyLength = parseInt(this.historyLength.value) || 30;
        const smoothingAlpha = parseFloat(this.smoothingAlpha.value);
        const denoiseSteps = parseInt(this.denoiseSteps.value) || 10;

        this.isProcessing = true;
        try {
            if (djMixer) {
                djMixer.setAudioMotionMode(false, null);
            }

            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                    denoise_steps: denoiseSteps
                })
            });

            const data = await response.json();
            if (data.status === 'success') {
                this.isRunning = false;
                this.isPaused = false;
                this.isIdle = true;
                this.frameCount = 0;
                this.motionFrameCount = 0;
                this.motionFpsCounter = 0;
                this.isFetchingFrame = false;
                this.consecutiveWaiting = 0;
                this.startResetBtn.textContent = 'Start';
                this.startResetBtn.classList.remove('btn-danger');
                this.startResetBtn.classList.add('btn-primary');
                this.pauseResumeBtn.disabled = true;
                this.pauseResumeBtn.textContent = 'Pause';
                this.statusEl.textContent = 'Idle';
                this.bufferSizeEl.textContent = '0 / 4';
                this.frameCountEl.textContent = '0';
                this.fpsEl.textContent = '0';

                if (this.skeleton) {
                    this.skeleton.clearTrail();
                }

                console.log('Reset complete');
            }
        } catch (error) {
            console.error('Error resetting:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async handleForceTakeover() {
        this.conflictWarning.style.display = 'none';
        if (!this.pendingStartRequest) return;
        this.isProcessing = false;
        await this.startAudioGeneration(true);
        this.pendingStartRequest = null;
    }

    handleCancelTakeover() {
        this.conflictWarning.style.display = 'none';
        this.statusEl.textContent = 'Idle';
        this.isProcessing = false;
        this.pendingStartRequest = null;
    }

    startFrameLoop() {
        const now = performance.now();
        this.lastFetchTime = now;
        this.nextFetchTime = now + this.frameInterval;
        this.frameQueue = [];
        this.consumeFrame();
        this.fetchFramesBatch();
    }

    consumeFrame() {
        if (!this.isRunning) return;

        const now = performance.now();

        if (now >= this.nextFetchTime) {
            this.nextFetchTime += this.frameInterval;
            if (this.nextFetchTime < now) {
                this.nextFetchTime = now + this.frameInterval;
            }

            if (this.frameQueue.length > 0) {
                const joints = this.frameQueue.shift();
                this.skeleton.updatePose(joints);
                this.frameCount++;
                this.frameCountEl.textContent = this.frameCount;

                this.motionFrameCount++;
                this.motionFpsCounter++;

                this.currentRootPos.set(joints[0][0], joints[0][1], joints[0][2]);
                this.updateAutoFollow();
                this.consecutiveWaiting = 0;
            }
        }

        requestAnimationFrame(() => this.consumeFrame());
    }

    fetchFramesBatch() {
        if (!this.isRunning) return;

        if (this.frameQueue.length < this.batchSize && !this.isFetchingFrame) {
            this.isFetchingFrame = true;

            fetch(`/api/get_frame?session_id=${this.sessionId}&count=${this.batchSize}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        if (data.frames) {
                            this.frameQueue.push(...data.frames);
                        } else if (data.joints) {
                            this.frameQueue.push(data.joints);
                        }
                        this.consecutiveWaiting = 0;
                    } else if (data.status === 'waiting') {
                        this.consecutiveWaiting++;
                    }
                })
                .catch(error => {
                    console.error('Error fetching frames:', error);
                })
                .finally(() => {
                    this.isFetchingFrame = false;
                });
        }

        setTimeout(() => this.fetchFramesBatch(), 10);
    }

    updateAutoFollow() {
        const timeSinceInteraction = Date.now() - this.lastUserInteraction;

        if (timeSinceInteraction > this.autoFollowDelay) {
            const currentOffset = new THREE.Vector3().subVectors(
                this.camera.position,
                this.controls.target
            );

            const newTarget = this.currentRootPos.clone();
            newTarget.y = 1.0;

            const newCameraPos = newTarget.clone().add(currentOffset);

            this.controls.target.lerp(newTarget, 0.2);
            this.camera.position.lerp(newCameraPos, 0.2);
        }
    }

    async updateStatus() {
        try {
            const response = await fetch(`/api/status?session_id=${this.sessionId}`);
            const data = await response.json();

            if (data.initialized) {
                this.bufferSizeEl.textContent = `${data.buffer_size} / ${data.target_size}`;

                if (data.smoothing_alpha !== undefined && this.currentSmoothing) {
                    this.currentSmoothing.textContent = data.smoothing_alpha.toFixed(2);
                }
                if (data.denoise_steps !== undefined && this.currentSteps) {
                    this.currentSteps.textContent = data.denoise_steps;
                }

                if (data.perf_stats) {
                    const stats = data.perf_stats;
                    const prepareSec = (stats.prepare_time || 0).toFixed(3);
                    const modelSec = stats.model_inference_time.toFixed(3);
                    const vaeSec = stats.vae_decode_time.toFixed(3);
                    const frameSec = stats.frame_process_time.toFixed(3);
                    const postSec = (stats.post_time || 0).toFixed(3);
                    const resampleSec = (stats.audio_resample_time || 0).toFixed(3);
                    const extractSec = (stats.audio_feature_time || 0).toFixed(3);

                    const totalSec = (stats.prepare_time || 0) + stats.model_inference_time +
                        stats.vae_decode_time + stats.frame_process_time +
                        (stats.post_time || 0) +
                        (stats.audio_resample_time || 0) + (stats.audio_feature_time || 0);

                    if (this.statPrepareEl) this.statPrepareEl.textContent = `${prepareSec}s`;
                    if (this.statModelEl) this.statModelEl.textContent = `${modelSec}s`;
                    if (this.statVAEEl) this.statVAEEl.textContent = `${vaeSec}s`;
                    if (this.statFrameEl) this.statFrameEl.textContent = `${frameSec}s`;
                    if (this.statPostEl) this.statPostEl.textContent = `${postSec}s`;
                    if (this.statResampleEl) this.statResampleEl.textContent = `${resampleSec}s`;
                    if (this.statExtractEl) this.statExtractEl.textContent = `${extractSec}s`;
                    if (this.statTotalEl) this.statTotalEl.textContent = `${totalSec.toFixed(3)}s`;
                }
            }

            const now = performance.now();
            if (now - this.motionFpsUpdateTime > 1000) {
                this.fpsEl.textContent = this.motionFpsCounter;
                this.motionFpsCounter = 0;
                this.motionFpsUpdateTime = now;
            }
        } catch (error) {
            // Silently fail
        }

        setTimeout(() => this.updateStatus(), 500);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const container = document.getElementById('canvas-container');
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    window.app = new DJMotionApp();
});
