// Audio Capture & Streaming

class AudioCapture {
    constructor() {
        this.ws = null;
        this.isCapturing = false;
        this.audioBuffers = [];
        this.sampleRate = 44100;
        this.audioSource = 'system';  // Default to system audio (for music playback)
        
        // Browser audio capture
        this.mediaStream = null;
        this.audioContext = null;
        this.analyser = null;
        this.scriptProcessor = null;
        this.currentAudioData = null;
        
        // ============ Audio-to-Motion Mode ============
        this.audioMotionMode = false;  // Whether to send audio to motion API
        this.sessionId = null;  // Session ID for motion generation
        this.sendInterval = null;  // Interval for sending audio data
        this.pendingAudioSamples = [];  // Buffer for pending audio samples
        
        // Canvas elements
        this.waveformCanvas = document.getElementById('waveformCanvas');
        this.spectrumCanvas = document.getElementById('spectrumCanvas');
        
        this.waveformCtx = this.waveformCanvas.getContext('2d');
        this.spectrumCtx = this.spectrumCanvas.getContext('2d');
        
        this.setupEventListeners();
        // WebSocket is optional (only for legacy visualization server)
        // Don't block if it fails to connect
        try { this.connectWebSocket(); } catch(e) { console.warn('WebSocket not available:', e); }
    }
    
    // ============ Audio-to-Motion API Methods ============
    setAudioMotionMode(enabled, sessionId = null) {
        this.audioMotionMode = enabled;
        this.sessionId = sessionId;
        console.log(`Audio motion mode: ${enabled}, session: ${sessionId}`);
    }
    
    async sendAudioToMotionAPI(audioSamples) {
        if (!this.audioMotionMode || !this.sessionId) return;
        
        try {
            const response = await fetch('/api/audio_data', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    samples: audioSamples,
                    sample_rate: this.sampleRate
                })
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                // Update audio status display if available
                const featureFramesEl = document.getElementById('audioFeatureFrames');
                if (featureFramesEl) {
                    featureFramesEl.textContent = data.feature_frames || 0;
                }
            }
        } catch (error) {
            console.error('Error sending audio to motion API:', error);
        }
    }
    // ============ End Audio-to-Motion API Methods ============
    
    setupEventListeners() {
        document.getElementById('audioToggleBtn').addEventListener('click', () => {
            this.toggleCapture();
        });
        
        document.getElementById('clearAudioBtn').addEventListener('click', () => {
            this.clearBuffer();
        });
        
        document.getElementById('audioSourceSelect').addEventListener('change', (e) => {
            this.handleAudioSourceChange(e.target.value);
        });
    }
    
    handleAudioSourceChange(source) {
    this.audioSource = source;
    
    const hintElement = document.getElementById('audioSourceHint');
    if (source === 'microphone') {
        hintElement.innerHTML = 'ℹ️ <strong>Microphone:</strong> Capture audio from microphone input.';
    } else {
        hintElement.innerHTML = 'ℹ️ <strong>System Audio:</strong> Share a browser tab/window to capture its audio output.';
    }
    
    if (this.isCapturing) {
        alert('Please stop current capture before changing audio source.');
        document.getElementById('audioSourceSelect').value = 
            this.audioSource === 'microphone' ? 'system' : 'microphone';
        this.audioSource = this.audioSource === 'microphone' ? 'system' : 'microphone';
    }
    
    console.log('Audio source changed to:', source);
}
    connectWebSocket() {
        const wsHost = window.location.hostname;
        const wsPort = wsHost === 'localhost' || wsHost === '127.0.0.1' ? 8766 : 8765;
        this.ws = new WebSocket(`ws://${wsHost}:${wsPort}`);
        
        this.ws.onopen = () => {
            console.log('✅ Connected to audio server');
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            console.log('Server message:', message);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('❌ Disconnected from audio server');
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }
    
    toggleCapture() {
        if (this.isCapturing) {
            this.stopCapture();
        } else {
            this.startCapture();
        }
    }
    
async startCapture() {
    console.log('🔍 startCapture called');

    try {
        let stream;
        
        if (this.audioSource === 'system') {
            stream = await navigator.mediaDevices.getDisplayMedia({
                video: true,
                audio: true
            });
            
            const audioTracks = stream.getAudioTracks();
            const videoTracks = stream.getVideoTracks();
            
            videoTracks.forEach(track => track.stop());
            
            stream = new MediaStream(audioTracks);
            
            console.log('✅ Got system audio access');
        }
         else {
            const constraints = {
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: this.sampleRate
                }
            };
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log('✅ Got microphone access');
        }
        
        this.mediaStream = stream;
        
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: this.sampleRate
        });
        
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 2048;
        source.connect(this.analyser);
        
        const bufferSize = 4096;
        this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        this.scriptProcessor.onaudioprocess = (e) => {
            if (!this.isCapturing) return;
            
            const inputData = e.inputBuffer.getChannelData(0);
            this.currentAudioData = new Float32Array(inputData);
            
            const audioData = {
                timestamp: new Date().toISOString(),
                data: Array.from(inputData),
                sample_rate: this.sampleRate,
                duration: inputData.length / this.sampleRate
            };
            
            this.audioBuffers.push(audioData);
            
            if (this.audioBuffers.length > 30) {
                this.audioBuffers.shift();
            }
            
            // ============ Send to Motion API if in audio motion mode ============
            if (this.audioMotionMode && this.sessionId) {
                // Accumulate samples and send periodically
                this.pendingAudioSamples.push(...audioData.data);
                
                // Send when we have enough samples (every ~100ms worth)
                const samplesPerSend = Math.floor(this.sampleRate * 0.1);  // 100ms
                if (this.pendingAudioSamples.length >= samplesPerSend) {
                    const samplesToSend = this.pendingAudioSamples.splice(0, samplesPerSend);
                    this.sendAudioToMotionAPI(samplesToSend);
                }
            }
            // ============ End Motion API ============
            
            // Original WebSocket send (for visualization server)
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'audio_data',
                    data: audioData
                }));
            }
            
            this.updateDuration();
        };
        
        source.connect(this.scriptProcessor);
        this.scriptProcessor.connect(this.audioContext.destination);
        
        this.isCapturing = true;
        this.updateUI(true);
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                command: 'start',
                source: this.audioSource
            }));
        }
        
        document.getElementById('sampleRate').textContent = `${this.audioContext.sampleRate} Hz`;
        
        this.animate();
        console.log('✅ Audio capture started');
        
    } catch (error) {
        console.error('❌ Failed to start capture:', error);
        alert(`Failed to access audio: ${error.message}`);
    }
}
    
    stopCapture() {
        console.log('⏹️ Stopping capture');
        
        this.isCapturing = false;
        
        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }
        
        if (this.analyser) {
            this.analyser.disconnect();
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command: 'stop' }));
        }
        
        this.updateUI(false);
    }
    
    animate() {
        if (!this.isCapturing) return;
        
        requestAnimationFrame(() => this.animate());
        
        if (this.currentAudioData && this.analyser) {
            this.drawWaveform();
            this.drawSpectrum();
        }
    }
    
    drawWaveform() {
        const canvas = this.waveformCanvas;
        const ctx = this.waveformCtx;
        const width = canvas.width;
        const height = canvas.height;
        const data = this.currentAudioData;
        
        ctx.fillStyle = 'rgb(10, 10, 10)';
        ctx.fillRect(0, 0, width, height);
        
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#00ff88';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00ff88';
        
        ctx.beginPath();
        
        const step = Math.ceil(data.length / width);
        
        for (let i = 0; i < width; i++) {
            const index = i * step;
            const value = data[index] || 0;
            const y = (0.5 + value * 0.5) * height;
            
            if (i === 0) {
                ctx.moveTo(i, y);
            } else {
                ctx.lineTo(i, y);
            }
        }
        
        ctx.stroke();
        ctx.shadowBlur = 0;
    }
    
    drawSpectrum() {
        const canvas = this.spectrumCanvas;
        const ctx = this.spectrumCtx;
        const width = canvas.width;
        const height = canvas.height;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);
        
        ctx.fillStyle = 'rgb(10, 10, 10)';
        ctx.fillRect(0, 0, width, height);
        
        const barWidth = width / bufferLength * 2.5;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * height;
            const hue = (i / bufferLength) * 360;
            
            ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
            ctx.shadowBlur = 5;
            ctx.shadowColor = `hsl(${hue}, 100%, 50%)`;
            
            ctx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
            x += barWidth;
        }
        ctx.shadowBlur = 0;
    }
    
    clearBuffer() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command: 'clear' }));
        }
        
        this.audioBuffers = [];
        this.updateDuration();
    }
    
    updateUI(isActive) {
        const btn = document.getElementById('audioToggleBtn');
        const icon = document.getElementById('audioIcon');
        const text = document.getElementById('audioText');
        const status = document.getElementById('audioStatus');
        const clearBtn = document.getElementById('clearAudioBtn');
        
        if (isActive) {
            btn.classList.add('active');
            icon.textContent = '⏹️';
            text.textContent = 'Stop Capture';
            status.textContent = 'Recording...';
            status.style.color = '#ff4444';
            clearBtn.disabled = false;
        } else {
            btn.classList.remove('active');
            icon.textContent = '🎤';
            text.textContent = 'Start Capture';
            status.textContent = 'Idle';
            status.style.color = '#888';
            clearBtn.disabled = false;
        }
    }
    
    updateDuration() {
        const duration = this.audioBuffers.length;
        document.getElementById('audioDuration').textContent = `${duration}.0s`;
    }
    
    getAudioData() {
        return this.audioBuffers;
    }
}

// Initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.audioCapture = new AudioCapture();
    });
} else {
    window.audioCapture = new AudioCapture();
}