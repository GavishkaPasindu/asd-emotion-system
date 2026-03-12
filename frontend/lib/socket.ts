import { io, Socket } from 'socket.io-client';

const SOCKET_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export interface EmotionUpdate {
    emotion: string;
    confidence: number;
    timestamp: string;
    frame_number: number;
}

export interface TrackingSession {
    session_id: string;
    start_time: string;
}

class SocketClient {
    private socket: Socket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;

    /**
     * Connect to the WebSocket server
     */
    connect(): Promise<Socket> {
        return new Promise((resolve, reject) => {
            if (this.socket?.connected) {
                resolve(this.socket);
                return;
            }

            this.socket = io(SOCKET_URL, {
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionDelayMax: 5000,
                reconnectionAttempts: this.maxReconnectAttempts,
            });

            this.socket.on('connect', () => {
                console.log('✅ WebSocket connected');
                this.reconnectAttempts = 0;
                if (this.socket) {
                    resolve(this.socket);
                }
            });

            this.socket.on('connect_error', (error) => {
                console.error('❌ WebSocket connection error:', error);
                this.reconnectAttempts++;

                if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    reject(new Error('Failed to connect to WebSocket server'));
                }
            });

            this.socket.on('disconnect', (reason) => {
                console.log('🔌 WebSocket disconnected:', reason);
            });

            this.socket.on('reconnect', (attemptNumber) => {
                console.log(`🔄 WebSocket reconnected after ${attemptNumber} attempts`);
            });
        });
    }

    /**
     * Disconnect from the WebSocket server
     */
    disconnect(): void {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
    }

    /**
     * Start emotion tracking session
     */
    startTracking(callback: (session: TrackingSession) => void): void {
        if (!this.socket) {
            throw new Error('Socket not connected');
        }

        this.socket.emit('start_tracking');
        this.socket.on('tracking_started', callback);
    }

    /**
     * Process a frame for emotion detection
     */
    processFrame(imageData: string, modelType?: string): void {
        if (!this.socket) {
            throw new Error('Socket not connected');
        }

        this.socket.emit('process_frame', {
            image: imageData,
            model_type: modelType
        });
    }

    /**
     * Listen for emotion updates
     */
    onEmotionUpdate(callback: (update: EmotionUpdate) => void): void {
        if (!this.socket) {
            throw new Error('Socket not connected');
        }

        this.socket.on('emotion_update', callback);
    }

    /**
     * Stop emotion tracking
     */
    stopTracking(): void {
        if (!this.socket) {
            throw new Error('Socket not connected');
        }

        this.socket.emit('stop_tracking');
        this.socket.off('emotion_update');
        this.socket.off('tracking_started');
    }

    /**
     * Check if socket is connected
     */
    isConnected(): boolean {
        return this.socket?.connected || false;
    }

    /**
     * Get the socket instance
     */
    getSocket(): Socket | null {
        return this.socket;
    }
}

// Export singleton instance
export const socketClient = new SocketClient();

export default socketClient;
