"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import { Camera, RefreshCw, CheckCircle, AlertCircle } from "lucide-react";

interface CameraCaptureProps {
    onCapture: (imageSrc: string) => void;
    isCapturing?: boolean;
    interval?: number; // Capture interval in ms
    isActive?: boolean;
}

const CameraCapture: React.FC<CameraCaptureProps> = ({
    onCapture,
    isCapturing = false,
    interval = 1000,
    isActive = true
}) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [hasPermission, setHasPermission] = useState<boolean>(false);

    // Initialize camera
    const startCamera = useCallback(async () => {
        try {
            if (stream) return;

            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                }
            });

            setStream(mediaStream);
            setHasPermission(true);
            setError(null);

            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
            }
        } catch (err) {
            console.error("Error accessing camera:", err);
            setError("Cannot access camera. Please allow permission.");
            setHasPermission(false);
        }
    }, [stream]);

    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
    }, [stream]);

    // Handle active state
    useEffect(() => {
        if (isActive) {
            startCamera();
        } else {
            stopCamera();
        }
        return () => {
            stopCamera(); // Cleanup on unmount
        };
    }, [isActive, startCamera, stopCamera]);

    // Handle automated capturing
    useEffect(() => {
        let captureInterval: NodeJS.Timeout;

        if (isCapturing && hasPermission && videoRef.current && canvasRef.current) {
            captureInterval = setInterval(() => {
                captureFrame();
            }, interval);
        }

        return () => {
            if (captureInterval) clearInterval(captureInterval);
        };
    }, [isCapturing, hasPermission, interval]);

    const captureFrame = () => {
        if (!videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");

        if (context && video.videoWidth && video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Draw video frame to canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert to base64
            const imageSrc = canvas.toDataURL("image/jpeg", 0.8);
            onCapture(imageSrc);
        }
    };

    return (
        <div className="relative rounded-2xl overflow-hidden shadow-2xl bg-black border-4 border-blue-400">
            {/* Camera View */}
            <div className="aspect-video relative">
                {!hasPermission && !error && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900 text-white z-20">
                        <RefreshCw className="w-10 h-10 animate-spin mb-2 text-blue-400" />
                        <p>Starting camera...</p>
                    </div>
                )}

                {error && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-900/80 text-white z-20 p-4 text-center">
                        <AlertCircle className="w-12 h-12 mb-2 text-red-200" />
                        <p className="font-bold">{error}</p>
                        <button
                            onClick={startCamera}
                            className="mt-4 px-4 py-2 bg-white text-red-900 rounded-full font-bold hover:bg-red-50"
                        >
                            Try Again
                        </button>
                    </div>
                )}

                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover transform scale-x-[-1] transition-opacity duration-500 ${hasPermission ? 'opacity-100' : 'opacity-0'}`}
                />

                {/* Face Guide Overlay */}
                {hasPermission && (
                    <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                        {/* Oval Face Guide */}
                        <div className={`w-48 h-64 border-2 border-dashed rounded-[50%] opacity-50 transition-colors duration-300 ${isCapturing ? 'border-green-400 opacity-80 animate-pulse' : 'border-white'}`}></div>

                        {/* Corner Indicators */}
                        <div className="absolute top-4 left-4 w-8 h-8 border-t-4 border-l-4 border-blue-400 rounded-tl-lg"></div>
                        <div className="absolute top-4 right-4 w-8 h-8 border-t-4 border-r-4 border-blue-400 rounded-tr-lg"></div>
                        <div className="absolute bottom-4 left-4 w-8 h-8 border-b-4 border-l-4 border-blue-400 rounded-bl-lg"></div>
                        <div className="absolute bottom-4 right-4 w-8 h-8 border-b-4 border-r-4 border-blue-400 rounded-br-lg"></div>
                    </div>
                )}

                {/* Status Indicator */}
                <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full flex items-center gap-2 z-10">
                    <div className={`w-3 h-3 rounded-full ${isCapturing ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
                    <span className="text-white text-xs font-bold tracking-wide">
                        {isCapturing ? 'RECORDING' : 'LIVE'}
                    </span>
                </div>
            </div>

            {/* Hidden Canvas for capture */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Decorative Robot/Toy Elements */}
            <div className="absolute -right-12 top-10 w-24 h-24 bg-blue-500 rounded-full blur-3xl opacity-20"></div>
            <div className="absolute -left-12 bottom-10 w-24 h-24 bg-purple-500 rounded-full blur-3xl opacity-20"></div>
        </div>
    );
};

export default CameraCapture;
