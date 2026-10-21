// Multi-Source Audio Mixer - 4-channel audio mixer with mutual exclusion faders

class DJMixer {
    constructor() {
        this.audioCtx = null;
        this.channels = []; // {source, gain, element, type, label}
        this.merger = null;
        this.analyser = null;
        this.scriptProcessor = null;
        this.isInitialized = false;

        // Audio-to-Motion mode
        this.audioMotionMode = false;
        this.sessionId = null;
        this.pendingAudioSamples = [];
        this.sampleRate = 44100;

        // Visualization
        this.waveformCanvas = document.getElementById('waveformCanvas');
        this.spectrumCanvas = document.getElementById('spectrumCanvas');
        this.waveformCtx = this.waveformCanvas ? this.waveformCanvas.getContext('2d') : null;
        this.spectrumCtx = this.spectrumCanvas ? this.spectrumCanvas.getContext('2d') : null;
        this.currentAudioData = null;
        this.isAnimating = false;

        // Speaker capture state
        this.speakerStream = null;
        this.isCapturing = false;
    }

    async init() {
        if (this.isInitialized) return;

        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: this.sampleRate
        });

        // === Split audio graph: playback path vs capture path ===
        // Playback path: only Track channels play through speakers
        this.playbackGain = this.audioCtx.createGain();
        this.playbackGain.gain.value = 1.0;
        this.playbackGain.connect(this.audioCtx.destination);

        // Capture path: all channels (tracks + speaker) feed into analyser + scriptProcessor
        this.captureGain = this.audioCtx.createGain();
        this.captureGain.gain.value = 1.0;

        // Analyser for visualization
        this.analyser = this.audioCtx.createAnalyser();
        this.analyser.fftSize = 2048;
        this.captureGain.connect(this.analyser);

        // ScriptProcessor for capturing PCM data to send to backend
        const bufferSize = 4096;
        this.scriptProcessor = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);

        this.scriptProcessor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            this.currentAudioData = new Float32Array(inputData);

            // Send to motion API if enabled
            if (this.audioMotionMode && this.sessionId) {
                this.pendingAudioSamples.push(...inputData);
                const samplesPerSend = Math.floor(this.sampleRate * 0.1); // 100ms
                if (this.pendingAudioSamples.length >= samplesPerSend) {
                    const samplesToSend = this.pendingAudioSamples.splice(0, samplesPerSend);
                    this.sendAudioToMotionAPI(samplesToSend);
                }
            }
        };

        this.captureGain.connect(this.scriptProcessor);
        // ScriptProcessor must connect to destination to stay alive, use silent output
        this.silentGain = this.audioCtx.createGain();
        this.silentGain.gain.value = 0;
        this.scriptProcessor.connect(this.silentGain);
        this.silentGain.connect(this.audioCtx.destination);

        // Initialize 3 track channels
        for (let i = 0; i < 3; i++) {
            const audioEl = document.getElementById(`track${i + 1}Audio`);
            const gainNode = this.audioCtx.createGain();
            gainNode.gain.value = 0;

            const source = this.audioCtx.createMediaElementSource(audioEl);
            source.connect(gainNode);
            // Tracks connect to BOTH playback (hear it) and capture (send to backend)
            gainNode.connect(this.playbackGain);
            gainNode.connect(this.captureGain);

            this.channels.push({
                type: 'track',
                label: `Track ${i + 1}`,
                source: source,
                gain: gainNode,
                element: audioEl,
                currentValue: 0
            });

            // Start playback (looped, muted by gain=0)
            audioEl.loop = true;
            audioEl.play().catch(e => console.warn(`Track ${i + 1} autoplay blocked:`, e));
        }

        // Speaker channel: capture ONLY (no playback = no echo)
        const speakerGain = this.audioCtx.createGain();
        speakerGain.gain.value = 0;
        speakerGain.connect(this.captureGain);

        this.channels.push({
            type: 'speaker',
            label: 'Speaker',
            source: null,
            gain: speakerGain,
            element: null,
            currentValue: 0
        });

        this.isInitialized = true;
        this.sampleRate = this.audioCtx.sampleRate;
        this.startVisualization();
        console.log('Mixer initialized, sample rate:', this.sampleRate);
    }

    async startSpeakerCapture() {
        if (this.isCapturing) return;

        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: true,
                audio: true
            });

            // Stop video tracks, keep audio
            stream.getVideoTracks().forEach(track => track.stop());
            const audioStream = new MediaStream(stream.getAudioTracks());

            this.speakerStream = audioStream;

            const source = this.audioCtx.createMediaStreamSource(audioStream);
            const speakerChannel = this.channels[3];

            // Disconnect old source if any
            if (speakerChannel.source) {
                try { speakerChannel.source.disconnect(); } catch(e) {}
            }

            source.connect(speakerChannel.gain);
            speakerChannel.source = source;
            this.isCapturing = true;

            // Handle track ending (user stops sharing)
            audioStream.getAudioTracks()[0].addEventListener('ended', () => {
                this.stopSpeakerCapture();
            });

            console.log('Speaker capture started');
            return true;
        } catch (error) {
            console.error('Failed to capture speaker:', error);
            return false;
        }
    }

    stopSpeakerCapture() {
        if (!this.isCapturing) return;

        const speakerChannel = this.channels[3];
        if (speakerChannel.source) {
            try { speakerChannel.source.disconnect(); } catch(e) {}
            speakerChannel.source = null;
        }

        if (this.speakerStream) {
            this.speakerStream.getTracks().forEach(track => track.stop());
            this.speakerStream = null;
        }

        this.isCapturing = false;
        speakerChannel.currentValue = 0;

        // Update fader UI
        const faderEl = document.getElementById('fader3');
        if (faderEl) faderEl.value = 0;
        const valueEl = document.getElementById('faderValue3');
        if (valueEl) valueEl.textContent = '0';
        if (typeof updateMeter === 'function') updateMeter(3, 0);

        console.log('Speaker capture stopped');
    }

    onFaderChange(channelIndex, newValue) {
        if (!this.isInitialized || channelIndex < 0 || channelIndex >= this.channels.length) return;

        // Resume audio context if suspended (browser autoplay policy)
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }

        const now = this.audioCtx.currentTime;
        const fadeTime = 0.5; // 500ms transition

        const normalizedValue = newValue / 100; // fader is 0-100, gain is 0-1

        // Set current channel volume
        const currentChannel = this.channels[channelIndex];
        currentChannel.gain.gain.cancelScheduledValues(now);
        currentChannel.gain.gain.setValueAtTime(currentChannel.gain.gain.value, now);
        currentChannel.gain.gain.linearRampToValueAtTime(normalizedValue, now + fadeTime);
        currentChannel.currentValue = newValue;

        // Mutual exclusion: reduce other channels
        if (newValue > 0) {
            for (let i = 0; i < this.channels.length; i++) {
                if (i === channelIndex) continue;

                const otherChannel = this.channels[i];
                // Calculate target: other channels share the remaining headroom
                const targetGain = Math.max(0, (1 - normalizedValue)) * (otherChannel.currentValue / 100);

                otherChannel.gain.gain.cancelScheduledValues(now);
                otherChannel.gain.gain.setValueAtTime(otherChannel.gain.gain.value, now);
                otherChannel.gain.gain.linearRampToValueAtTime(targetGain, now + fadeTime);

                // Update the stored value and fader UI
                const newOtherValue = Math.round(targetGain * 100);
                otherChannel.currentValue = newOtherValue;

                const faderEl = document.getElementById(`fader${i}`);
                if (faderEl) faderEl.value = newOtherValue;
                const valueEl = document.getElementById(`faderValue${i}`);
                if (valueEl) valueEl.textContent = newOtherValue;

                // Update LED meter
                if (typeof updateMeter === 'function') updateMeter(i, newOtherValue);
            }
        }
    }

    // Audio-to-Motion API
    setAudioMotionMode(enabled, sessionId = null) {
        this.audioMotionMode = enabled;
        this.sessionId = sessionId;
        this.pendingAudioSamples = [];
        console.log(`Mixer audio motion mode: ${enabled}, session: ${sessionId}`);
    }

    async sendAudioToMotionAPI(audioSamples) {
        if (!this.audioMotionMode || !this.sessionId) return;

        try {
            const response = await fetch('/api/audio_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    samples: Array.from(audioSamples),
                    sample_rate: this.sampleRate
                })
            });

            const data = await response.json();
            if (data.status === 'success') {
                const featureFramesEl = document.getElementById('audioFeatureFrames');
                if (featureFramesEl) {
                    featureFramesEl.textContent = data.feature_frames || 0;
                }
            }
        } catch (error) {
            console.error('Error sending audio to motion API:', error);
        }
    }

    // Visualization
    startVisualization() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.animateVisuals();
    }

    animateVisuals() {
        if (!this.isAnimating) return;
        requestAnimationFrame(() => this.animateVisuals());

        if (this.currentAudioData && this.analyser) {
            this.drawWaveform();
            this.drawSpectrum();
        }
    }

    drawWaveform() {
        if (!this.waveformCtx) return;
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
            if (i === 0) ctx.moveTo(i, y);
            else ctx.lineTo(i, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    drawSpectrum() {
        if (!this.spectrumCtx || !this.analyser) return;
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

    dispose() {
        this.isAnimating = false;
        this.stopSpeakerCapture();

        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }
        if (this.playbackGain) {
            this.playbackGain.disconnect();
        }
        if (this.captureGain) {
            this.captureGain.disconnect();
        }
        if (this.silentGain) {
            this.silentGain.disconnect();
        }
        if (this.audioCtx) {
            this.audioCtx.close();
            this.audioCtx = null;
        }
        this.isInitialized = false;
    }
}
