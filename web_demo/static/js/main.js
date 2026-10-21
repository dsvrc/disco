// Main App Logic & 3D Rendering

class MotionApp {
    constructor() {
        this.isRunning = false;
        this.targetFps = 30; // Model generates data at 30fps
        this.frameInterval = 1000 / this.targetFps; // 33.3ms
        this.lastFetchTime = 0;
        this.nextFetchTime = 0;  // Scheduled time for next fetch
        this.frameCount = 0;
        this.fpsCounter = 0;
        this.fpsUpdateTime = 0;
        this.lastRenderTime = 0;
        
        // Motion FPS tracking (frame consumption rate)
        this.motionFrameCount = 0;
        this.motionFpsCounter = 0;
        this.motionFpsUpdateTime = 0;
        
        // Request throttling
        this.isFetchingFrame = false;  // Prevent concurrent requests
        this.consecutiveWaiting = 0;   // Count consecutive 'waiting' responses
        
        // Batch frame fetching
        this.batchSize = 4;  // Fetch 4 frames per request
        this.frameQueue = [];  // Local queue of fetched frames
        
        // Session management
        this.sessionId = this.generateSessionId();
        
        // Camera follow settings
        this.lastUserInteraction = 0;
        this.autoFollowDelay = 2000; // Auto-follow after 2 seconds of inactivity (reduced from 3s)
        this.currentRootPos = new THREE.Vector3(0, 1, 0);
        
        // ============ Generation Mode ============
        // Default to audio mode since this is a music-to-dance demo
        this.generationMode = 'audio';
        
        this.initThreeJS();
        this.initUI();
        this.updateStatus();
        this.setupBeforeUnload();
        
        console.log('Session ID:', this.sessionId);
    }
    
    generateSessionId() {
        // Generate a simple unique session ID
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    setupBeforeUnload() {
        // Handle page close/refresh - send reset request
        window.addEventListener('beforeunload', () => {
            // Send synchronous reset if we're generating
            if (!this.isIdle) {
                // Use Blob to set correct Content-Type for JSON
                const blob = new Blob(
                    [JSON.stringify({session_id: this.sessionId})],
                    {type: 'application/json'}
                );
                navigator.sendBeacon('/api/reset', blob);
                console.log('Sent reset beacon on page unload');
            }
        });
        
        // Also handle visibility change (tab hidden, mobile app switch)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && !this.isIdle && this.isRunning) {
                // User switched away while generating - they might not come back
                // Note: Don't reset immediately, let the frame consumption monitor handle it
                console.log('Tab hidden while generating - consumption monitor will auto-reset if needed');
            }
        });
    }
    
    initThreeJS() {
        // Get canvas
        const canvas = document.getElementById('renderCanvas');
        const container = document.getElementById('canvas-container');
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);  // White background
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(3, 1.5, 3);
        this.camera.lookAt(0, 1, 0);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        
        // Add lights - bright and soft
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
        
        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-3, 5, -3);
        this.scene.add(fillLight);
        
        // Add ground plane - light gray, very large
        const groundGeometry = new THREE.PlaneGeometry(1000, 1000);
        const groundMaterial = new THREE.ShadowMaterial({
            opacity: 0.15
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);
        
        // Add infinite-looking grid - very large grid
        const gridHelper = new THREE.GridHelper(1000, 1000, 0xdddddd, 0xeeeeee);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);
        
        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.update();
        
        // Listen for user interaction - record time
        const updateInteractionTime = () => {
            this.lastUserInteraction = Date.now();
        };
        canvas.addEventListener('mousedown', updateInteractionTime);
        canvas.addEventListener('wheel', updateInteractionTime);
        canvas.addEventListener('touchstart', updateInteractionTime);
        
        // Create skeleton
        this.skeleton = new Skeleton3D(this.scene);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start render loop
        this.animate();
    }
    
    initUI() {
        // Get UI elements
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
        
        // Performance stats elements
        this.statPrepareEl = document.getElementById('statPrepare');
        this.statModelEl = document.getElementById('statModel');
        this.statVAEEl = document.getElementById('statVAE');
        this.statFrameEl = document.getElementById('statFrame');
        this.statPostEl = document.getElementById('statPost');
        this.statResampleEl = document.getElementById('statResample');
        this.statExtractEl = document.getElementById('statExtract');
        this.statTotalEl = document.getElementById('statTotal');
        
        // ============ Mode selection elements ============
        this.textModeBtn = document.getElementById('textModeBtn');
        this.audioModeBtn = document.getElementById('audioModeBtn');
        this.modeHint = document.getElementById('modeHint');
        this.currentModeEl = document.getElementById('currentMode');
        this.audioModeStatusEls = document.querySelectorAll('.audio-mode-status');
        
        // Track state
        this.isPaused = false;
        this.isIdle = true;
        this.isProcessing = false;  // Prevent concurrent API calls
        this.pendingStartRequest = null;  // Store pending start request data
        
        // Attach event listeners
        this.startResetBtn.addEventListener('click', () => this.toggleStartReset());
        if (this.updateBtn) {
            this.updateBtn.addEventListener('click', () => this.updateText());
        }
        this.pauseResumeBtn.addEventListener('click', () => this.togglePauseResume());
        if (this.forceTakeoverBtn) {
            this.forceTakeoverBtn.addEventListener('click', () => this.handleForceTakeover());
        }
        if (this.cancelTakeoverBtn) {
            this.cancelTakeoverBtn.addEventListener('click', () => this.handleCancelTakeover());
        }
        
        // ============ Mode selection listeners ============
        if (this.textModeBtn) {
            this.textModeBtn.addEventListener('click', () => this.setGenerationMode('text'));
        }
        if (this.audioModeBtn) {
            this.audioModeBtn.addEventListener('click', () => this.setGenerationMode('audio'));
        }
        
        // Update smoothing value display when slider changes
        if (this.smoothingAlpha) {
            this.smoothingAlpha.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value).toFixed(2);
                if (this.smoothingValue) {
                    this.smoothingValue.textContent = value;
                }
            });
        }
        
        // Since default mode is audio, update UI accordingly
        if (this.updateBtn) {
            this.updateBtn.style.display = 'none';  // Hide update text button in audio mode
        }
        // Show audio mode status elements
        this.audioModeStatusEls.forEach(el => el.style.display = '');
        if (this.currentModeEl) {
            this.currentModeEl.textContent = 'Audio';
        }
        
        console.log('UI initialized, default mode: audio');
    }
    
    // ============ Mode Selection Methods ============
    setGenerationMode(mode) {
        if (!this.isIdle) {
            alert('Please reset before changing mode.');
            return;
        }
        
        this.generationMode = mode;
        
        // Update UI
        if (mode === 'text') {
            this.textModeBtn.classList.add('active');
            this.audioModeBtn.classList.remove('active');
            this.modeHint.textContent = 'Text mode: Enter motion description to generate';
            this.updateBtn.style.display = '';  // Show update text button
            // Hide audio mode status
            this.audioModeStatusEls.forEach(el => el.style.display = 'none');
        } else {
            this.textModeBtn.classList.remove('active');
            this.audioModeBtn.classList.add('active');
            this.modeHint.textContent = 'Audio mode: Start audio capture, then click Start to generate motion from music';
            this.updateBtn.style.display = 'none';  // Hide update text button in audio mode
            // Show audio mode status
            this.audioModeStatusEls.forEach(el => el.style.display = '');
        }
        
        if (this.currentModeEl) {
            this.currentModeEl.textContent = mode === 'text' ? 'Text' : 'Audio';
        }
        
        console.log('Generation mode set to:', mode);
    }
    // ============ End Mode Selection Methods ============
    
    async toggleStartReset() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        if (this.isIdle) {
            // Currently idle, so start based on mode
            if (this.generationMode === 'audio') {
                await this.startAudioGeneration();
            } else {
                await this.startGeneration();
            }
        } else {
            // Currently running/paused, so reset
            await this.reset();
        }
    }
    
    // ============ Audio Mode Generation ============
    async startAudioGeneration(force = false) {
        if (this.isProcessing) return;
        
        console.log('[Audio Mode] Starting audio generation...');
        console.log('[Audio Mode] window.audioCapture:', window.audioCapture);
        console.log('[Audio Mode] isCapturing:', window.audioCapture ? window.audioCapture.isCapturing : 'N/A');
        
        // Auto-start audio capture if not running
        if (window.audioCapture && !window.audioCapture.isCapturing) {
            console.log('[Audio Mode] Audio capture not running, starting automatically...');
            this.statusEl.textContent = 'Starting audio capture...';
            this.statusEl.style.color = '#ff9800';  // Orange color for intermediate state
            
            try {
                await window.audioCapture.startCapture();
                console.log('[Audio Mode] Audio capture started successfully');
                await new Promise(resolve => setTimeout(resolve, 100));  // Brief delay for audio system to stabilize
            } catch (error) {
                console.error('[Audio Mode] Failed to start audio capture:', error);
                this.statusEl.textContent = 'Idle';
                this.statusEl.style.color = '';
                alert(`Failed to start audio capture: ${error.message}\n\nPlease grant microphone permission and try again.`);
                return;
            }
        } else if (!window.audioCapture) {
            alert('Audio capture system not initialized. Please refresh the page.');
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
        
        console.log('[Audio Mode] Sending /api/start_audio request...');
        
        try {
            const response = await fetch('/api/start_audio', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                    denoise_steps: denoiseSteps,
                    audio_window_size: 32,
                    sample_rate: window.audioCapture.sampleRate || 44100,
                    force: force
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Enable audio motion mode in AudioCapture with current session
                if (window.audioCapture) {
                    window.audioCapture.setAudioMotionMode(true, this.sessionId);
                    console.log('Audio motion mode enabled for session:', this.sessionId);
                }
                
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
                this.updateBtn.disabled = true;  // No text update in audio mode
                this.pauseResumeBtn.disabled = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.statusEl.textContent = 'Running (Audio)';
                
                if (this.currentModeEl) {
                    this.currentModeEl.textContent = 'Audio';
                }
                
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
            console.error('Error starting audio generation:', error);
            alert('Failed to start audio generation: ' + error.message);
            this.statusEl.textContent = 'Idle';
            this.isIdle = true;
            this.isRunning = false;
        } finally {
            this.isProcessing = false;
        }
    }
    // ============ End Audio Mode Generation ============
    
    async startGeneration(force = false) {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        const text = this.motionText.value.trim();
        if (!text) {
            alert('Please enter a motion description');
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
            alert('Denoising steps must be a multiple of 5 (e.g., 5, 10, 15, 20...)');
            return;
        }
        
        const smoothingAlpha = parseFloat(this.smoothingAlpha.value);
        
        this.isProcessing = true;
        this.statusEl.textContent = 'Initializing...';
        
        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                    denoise_steps: denoiseSteps,
                    force: force
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
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
                this.updateBtn.disabled = false;
                this.pauseResumeBtn.disabled = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.statusEl.textContent = 'Running';
                this.startFrameLoop();
            } else if (response.status === 409 && data.conflict) {
                // Another session is running, show warning UI
                this.statusEl.textContent = 'Conflict - Another user is generating';
                this.conflictWarning.style.display = 'block';
                
                // Store request data for later
                this.pendingStartRequest = {
                    text: text,
                    history_length: historyLength
                };
                
                return;
            } else {
                // Other errors
                alert('Error: ' + data.message);
                this.statusEl.textContent = 'Idle';
                this.isIdle = true;
                this.isRunning = false;
                this.isPaused = false;
            }
        } catch (error) {
            console.error('Error starting generation:', error);
            alert('Failed to start generation: ' + error.message);
            this.statusEl.textContent = 'Idle';
            // Keep idle state on error
            this.isIdle = true;
            this.isRunning = false;
            this.isPaused = false;
        } finally {
            this.isProcessing = false;
        }
    }
    
    async updateText() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        const text = this.motionText.value.trim();
        if (!text) {
            alert('Please enter a motion description');
            return;
        }
        
        this.isProcessing = true;
        try {
            const response = await fetch('/api/update_text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('Text updated:', text);
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error updating text:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async togglePauseResume() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        if (this.isPaused) {
            // Currently paused, so resume
            await this.resumeGeneration();
        } else {
            // Currently running, so pause
            await this.pauseGeneration();
        }
    }
    
    async pauseGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/pause', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: this.sessionId})
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = false;
                this.isPaused = true;
                this.pauseResumeBtn.textContent = 'Resume';
                this.pauseResumeBtn.classList.remove('btn-warning');
                this.pauseResumeBtn.classList.add('btn-success');
                this.updateBtn.disabled = true;
                this.statusEl.textContent = 'Paused';
                console.log('Generation paused (state preserved)');
            }
        } catch (error) {
            console.error('Error pausing generation:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async resumeGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/resume', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: this.sessionId})
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = true;
                this.isPaused = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.pauseResumeBtn.classList.remove('btn-success');
                this.pauseResumeBtn.classList.add('btn-warning');
                this.updateBtn.disabled = false;
                this.statusEl.textContent = 'Running';
                this.startFrameLoop();
                console.log('Generation resumed');
            }
        } catch (error) {
            console.error('Error resuming generation:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async reset() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        const historyLength = parseInt(this.historyLength.value) || 30;
        const smoothingAlpha = parseFloat(this.smoothingAlpha.value);
        const denoiseSteps = parseInt(this.denoiseSteps.value) || 10;
        
        this.isProcessing = true;
        try {
            // Disable audio motion mode if it was enabled
            if (window.audioCapture) {
                window.audioCapture.setAudioMotionMode(false, null);
            }
            
            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
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
                this.updateBtn.disabled = true;
                this.pauseResumeBtn.disabled = true;
                this.pauseResumeBtn.textContent = 'Pause';
                this.pauseResumeBtn.classList.remove('btn-success');
                this.pauseResumeBtn.classList.add('btn-warning');
                this.statusEl.textContent = 'Idle';
                this.bufferSizeEl.textContent = '0 / 4';
                this.frameCountEl.textContent = '0';
                this.fpsEl.textContent = '0';
                
                // Clear trail
                if (this.skeleton) {
                    this.skeleton.clearTrail();
                }
                
                // Keep audioCapture running but update session when restarting
                // Don't disable audioMotionMode here, it will be re-enabled on next start
                
                console.log('Reset complete - all state cleared');
            }
        } catch (error) {
            console.error('Error resetting:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async handleForceTakeover() {
        // Hide warning
        this.conflictWarning.style.display = 'none';
        
        if (!this.pendingStartRequest) return;
        
        // Retry with force=true
        this.isProcessing = false;
        await this.startGeneration(true);
        
        this.pendingStartRequest = null;
    }
    
    handleCancelTakeover() {
        // Hide warning
        this.conflictWarning.style.display = 'none';
        this.statusEl.textContent = 'Idle';
        this.isProcessing = false;
        this.pendingStartRequest = null;
    }
    
    startFrameLoop() {
        const now = performance.now();
        this.lastFetchTime = now;
        this.nextFetchTime = now + this.frameInterval;
        this.frameQueue = [];  // Clear local queue
        this.consumeFrame();
        this.fetchFramesBatch();
    }
    
    consumeFrame() {
        if (!this.isRunning) return;
        
        const now = performance.now();
        
        // Consume frame from local queue at fixed rate
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
        
        // Fetch more frames when local queue is running low
        if (this.frameQueue.length < this.batchSize && !this.isFetchingFrame) {
            this.isFetchingFrame = true;
            
            fetch(`/api/get_frame?session_id=${this.sessionId}&count=${this.batchSize}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        if (data.frames) {
                            // Batch response
                            this.frameQueue.push(...data.frames);
                        } else if (data.joints) {
                            // Single frame response
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
        
        // Check again soon
        setTimeout(() => this.fetchFramesBatch(), 10);
    }
    
    // Keep old fetchFrame for backward compatibility (not used)
    fetchFrame() {
        if (!this.isRunning) return;
        
        const now = performance.now();
        
        // Check if it's time to fetch next frame AND we're not already fetching
        if (now >= this.nextFetchTime && !this.isFetchingFrame) {
            // Schedule next fetch (maintain fixed rate regardless of delays)
            this.nextFetchTime += this.frameInterval;
            
            // If we've fallen behind, catch up
            if (this.nextFetchTime < now) {
                this.nextFetchTime = now + this.frameInterval;
            }
            
            // Mark as fetching to prevent concurrent requests
            this.isFetchingFrame = true;
            
            fetch(`/api/get_frame?session_id=${this.sessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        this.skeleton.updatePose(data.joints);
                        this.frameCount++;
                        this.frameCountEl.textContent = this.frameCount;
                        
                        // Update motion FPS counter (only when frame consumed)
                        this.motionFrameCount++;
                        this.motionFpsCounter++;
                        
                        // Update current root position
                        this.currentRootPos.set(
                            data.joints[0][0],
                            data.joints[0][1],
                            data.joints[0][2]
                        );
                        
                        // Auto-follow (if user hasn't interacted for a while)
                        this.updateAutoFollow();
                        
                        // Reset waiting counter on success
                        this.consecutiveWaiting = 0;
                    } else if (data.status === 'waiting') {
                        // No frame available, slow down requests if this happens repeatedly
                        this.consecutiveWaiting++;
                        
                        // If buffer is consistently empty, back off a bit
                        if (this.consecutiveWaiting > 5) {
                            // Add a small delay to reduce server load
                            this.nextFetchTime = now + this.frameInterval * 1.5;
                            this.consecutiveWaiting = 0;
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching frame:', error);
                })
                .finally(() => {
                    // Always mark as done fetching
                    this.isFetchingFrame = false;
                });
        }
        
        // Use requestAnimationFrame for continuous checking
        requestAnimationFrame(() => this.fetchFrame());
    }
    
    updateAutoFollow() {
        const timeSinceInteraction = Date.now() - this.lastUserInteraction;
        
        // Auto-follow if user hasn't interacted for more than 3 seconds
        if (timeSinceInteraction > this.autoFollowDelay) {
            // Calculate camera offset relative to current target
            const currentOffset = new THREE.Vector3().subVectors(
                this.camera.position, 
                this.controls.target
            );
            
            // New target position (character position, waist height)
            const newTarget = this.currentRootPos.clone();
            newTarget.y = 1.0;
            
            // Calculate new camera position (maintain relative offset)
            const newCameraPos = newTarget.clone().add(currentOffset);
            
            // Smooth interpolation follow (increased lerp factor for more obvious following)
            // 0.2 = more aggressive following, 0.05 = gentle following
            this.controls.target.lerp(newTarget, 0.2);
            this.camera.position.lerp(newCameraPos, 0.2);
            
            // Debug log (comment out in production)
            // console.log('Auto-follow active, tracking:', newTarget);
        }
    }
    
    async updateStatus() {
        try {
            const response = await fetch(`/api/status?session_id=${this.sessionId}`);
            const data = await response.json();
            
            if (data.initialized) {
                this.bufferSizeEl.textContent = `${data.buffer_size} / ${data.target_size}`;
                
                // Update current smoothing display
                if (data.smoothing_alpha !== undefined) {
                    this.currentSmoothing.textContent = data.smoothing_alpha.toFixed(2);
                }
                
                // Update current denoising steps display
                if (data.denoise_steps !== undefined) {
                    this.currentSteps.textContent = data.denoise_steps;
                }
                
                // Update performance stats
                if (data.perf_stats) {
                    const stats = data.perf_stats;
                    const prepareSec = (stats.prepare_time || 0).toFixed(3);
                    const modelSec = stats.model_inference_time.toFixed(3);
                    const vaeSec = stats.vae_decode_time.toFixed(3);
                    const frameSec = stats.frame_process_time.toFixed(3);
                    const postSec = (stats.post_time || 0).toFixed(3);
                    const resampleSec = (stats.audio_resample_time || 0).toFixed(3);
                    const extractSec = (stats.audio_feature_time || 0).toFixed(3);
                    
                    // Total = prepare + model + vae + frame + post + resample + extract
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
            
            // Update motion FPS (frame consumption rate)
            const now = performance.now();
            if (now - this.motionFpsUpdateTime > 1000) {
                this.fpsEl.textContent = this.motionFpsCounter;
                this.motionFpsCounter = 0;
                this.motionFpsUpdateTime = now;
            }
        } catch (error) {
            // Silently fail for status updates
        }
        
        // Update status every 500ms
        setTimeout(() => this.updateStatus(), 500);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls
        this.controls.update();
        
        // Render scene
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
    window.app = new MotionApp();
});

