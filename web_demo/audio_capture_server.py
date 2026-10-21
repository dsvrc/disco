import asyncio
import websockets
import json
import queue
from datetime import datetime

class AudioServer:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=100)
        self.connected_clients = set()
        self.audio_buffers = []
        
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        self.connected_clients.add(websocket)
        print(f"🔌 Client connected. Total: {len(self.connected_clients)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data.get('type') == 'audio_data':
                    # Store audio data
                    audio_data = data.get('data')
                    self.audio_buffers.append(audio_data)
                    
                    # Keep only last 100 buffers
                    if len(self.audio_buffers) > 100:
                        self.audio_buffers.pop(0)
                    
                    print(f"📊 Received audio: {len(audio_data['data'])} samples, buffers: {len(self.audio_buffers)}")
                    
                elif data.get('command') == 'start':
                    print(f"📡 Client started capture")
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'status': 'started'
                    }))
                    
                elif data.get('command') == 'stop':
                    print(f"⏹️ Client stopped capture")
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'status': 'stopped'
                    }))
                    
                elif data.get('command') == 'clear':
                    self.audio_buffers.clear()
                    print(f"🗑️ Cleared audio buffers")
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'status': 'cleared'
                    }))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)
            print(f"🔌 Client disconnected. Total: {len(self.connected_clients)}")
                
    async def start_server(self, host='localhost', port=8765):
        """Start WebSocket server"""
        async with websockets.serve(self.handle_client, host, port):
            print(f"🚀 Audio server running on ws://{host}:{port}")
            await asyncio.Future()

def main():
    server = AudioServer()
    asyncio.run(server.start_server(host='0.0.0.0', port=8765))

if __name__ == '__main__':
    main()