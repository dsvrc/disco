# Real-time Music2Dance Web Demo

## Overview
Real-time 3D dance motion generation from audio input. The system captures audio from your microphone or system audio, extracts music features (rhythm, melody, beats), and generates synchronized dance movements in real-time.

## Quick Start (Optimized Workflow)

**New Feature**: The system now auto-starts audio capture when you click "Start", eliminating manual setup steps!

1. Start the server (see Setup below)
2. Open the web interface
3. **Simply click "Start"** - audio capture will start automatically
4. Grant microphone permission when prompted
5. Watch the 3D character dance to your audio!

**Note**: The first ~1 second may show minimal movement while the audio buffer fills, then motion quality rapidly improves as real audio features arrive.

## Architecture

**Browser (Client):**
- Uses Web Audio API to capture microphone
- Processes audio with `ScriptProcessorNode`
- Sends audio data via HTTP POST to Flask server
- Visualizes audio in real-time

**Server (Flask :5000):**
- Receives raw audio samples from browser
- Extracts music features (VQ-PAE or librosa, auto-selected by `music_dim` in config)
- Diffusion model generates dance motion in real-time
- Returns joint positions to frontend

## Dependencies

### Python Packages
```bash
pip install -r requirements.txt
```

### System Requirements
- Python 3.10+
- Modern browser (Chrome/Firefox/Edge) with Web Audio API support
- SSH client (for remote access with tunnel, built-in on Windows 10+)

## Setup & Usage

### Scenario 1: Local Development (Same Machine)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start server:
```bash
cd web_demo
bash server.sh start
```

3. Open browser:
   - `http://localhost:5000` — Standard audio capture mode
   - `http://localhost:5000/mixer` — Live mixer mode (see below)
4. Click "Start" (audio capture will start automatically)

### Scenario 2: Remote Server via SSH (Current Setup)

**Step 1 — Start the server on the remote machine:**

```bash
# SSH into the server
ssh root@<ip_address>

# Start the service
cd web_demo
bash server.sh start
```

Server management commands:
```bash
bash server.sh stop      # Stop the server
bash server.sh restart   # Restart the server
bash server.sh status    # Check running status
bash server.sh log       # View logs
```

**Step 2 — Set up SSH port forwarding from your local terminal:**

```bash
# Run this on your LOCAL machine (PowerShell / Terminal), NOT on the server
ssh -L 5000:localhost:5000 root@<ip_address>
```

> Keep this terminal window open — closing it will disconnect the tunnel.
> If password login is inconvenient, set up SSH key authentication for passwordless access.


## Live Mixer Mode (`/mixer`)

An alternative interface for mixing multiple audio sources with a multi-channel console.

**Features:**
- **4 audio channels**: Track 1/2/3 (preset music) + Speaker (system audio capture)
- **Vertical faders**: Drag to control channel volume (0–100)
- **Mutual exclusion**: Raising one fader automatically fades out the others with a smooth 500ms transition
- **Speaker capture**: Click "Capture" on the Speaker channel to share system audio via `getDisplayMedia`
- **Same backend**: Uses the same `/api/audio_data` endpoint — all generated dance motions are identical to standard mode

**Usage:**
1. Open `http://localhost:5000/mixer`
2. Drag a track fader up to start hearing preset music
3. Switch between tracks by dragging another fader — crossfade happens automatically
4. Click "Start" to begin motion generation from the mixed audio
5. Optionally click "Capture" on the Speaker channel to mix in system audio

## Technical Details

### Audio Processing

Server auto-selects processing mode based on `music_dim` in config:

| | Librosa (music_dim=35) | VQ-PAE (music_dim=16) |
|---|---|---|
| Sample Rate | 15360 Hz | 16000 Hz |
| Feature Dim | 35 | 16 |
| Window Size | 32 frames | 180 frames (6s) |
| Compute | CPU librosa | GPU VQ-PAE forward |

Current config (`configs/stream_demo.yaml`) uses VQ-PAE mode.

### Network
- Flask web server port: 5000

### Performance Optimizations

**Fast Startup**: The system uses two optimizations to minimize startup latency:

1. **Auto-start Audio Capture**: When you click "Start", audio capture begins automatically - no manual setup needed
2. **Silence Pre-filling**: The audio buffer is pre-filled with 1 second of silence, allowing motion generation to start immediately

**Startup Timeline**:
- **First run (cold start)**: ~6.5 seconds (PyTorch JIT compilation)
- **Subsequent runs**: ~200ms to first frame (15x faster!)
- **Initial motion quality**: Minimal for first 1-2 seconds (silence-based), then rapidly improves as real audio arrives

**Real-time Performance**:
- Model inference: ~85ms per step
- Generation rate: 47 FPS (4 frames per step)
- Display rate: 20 FPS (smooth playback)
- Buffer: Maintains 6-7 frames ahead for stutter-free rendering

## Files Structure

```
web_demo/
├── app.py                           # Flask API server (routes: / and /mixer)
├── model_manager.py                 # ModelManager + StreamAudioProcessor + StreamVQPAEProcessor
├── server.sh                        # start/stop/restart/status/log
├── start.sh                         # Foreground startup (for debugging)
├── static/
│   ├── audio/
│   │   ├── track1.wav               # Preset track 1 (mixer mode)
│   │   ├── track2.wav               # Preset track 2 (mixer mode)
│   │   └── track3.wav               # Preset track 3 (mixer mode)
│   ├── css/
│   │   ├── style.css                # Standard page styles
│   │   └── dj-style.css             # Mixer page styles
│   └── js/
│       ├── audio-capture.js         # Client-side audio capture (standard mode)
│       ├── main.js                  # Main app logic (standard mode)
│       ├── skeleton.js              # 3D skeleton rendering (shared)
│       ├── dj-mixer.js              # Mixer audio class (mixer mode)
│       └── dj-main.js               # Mixer app logic (mixer mode)
└── templates/
    ├── index.html                   # Standard web UI
    └── dj.html                      # Live mixer web UI
```

## Troubleshooting

### Issue: No microphone permission prompt
**Solution:**
- Ensure accessing via `http://localhost:5000` (not IP address)
- Check SSH tunnel is established
- Browser console should show no `getUserMedia` errors

### Issue: Port 5000 already in use
**Solution:**
```bash
bash server.sh stop
# or manually:
fuser -k 5000/tcp
```

### Issue: Server won't start
**Solution:**
```bash
bash server.sh log    # Check error messages
```

## Notes

- **Security:** HTTPS or localhost required for `getUserMedia()` API
- **Limitations:** System audio capture not supported in standard browsers (use microphone)
- **Model lazy loading:** Models are loaded on first generation request, not at server startup
- **Storage:** Audio buffers stored in memory only, not persisted to disk by default
