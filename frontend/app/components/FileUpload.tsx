'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Upload, X, Image as ImageIcon, Camera } from 'lucide-react';

interface FileUploadProps {
    onFileSelect: (file: File) => void;
    accept?: string;
    maxSizeMB?: number;
    disabled?: boolean;
}

export default function FileUpload({
    onFileSelect,
    accept = 'image/*',
    maxSizeMB = 16,
    disabled = false,
}: FileUploadProps) {
    const [dragActive, setDragActive] = useState(false);
    const [preview, setPreview] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isCameraOpen, setIsCameraOpen] = useState(false);
    const [stream, setStream] = useState<MediaStream | null>(null);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Cleanup camera stream on unmount
    useEffect(() => {
        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [stream]);

    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            setStream(mediaStream);
            setIsCameraOpen(true);
            setError(null);
            // Delay assignment to allow video element to render first
            setTimeout(() => {
                if (videoRef.current) {
                    videoRef.current.srcObject = mediaStream;
                }
            }, 100);
        } catch (err) {
            console.error("Error accessing camera:", err);
            setError('Could not access camera. Please ensure you have granted permission.');
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
        setIsCameraOpen(false);
    };

    const captureImage = () => {
        if (videoRef.current && canvasRef.current) {
            const context = canvasRef.current.getContext('2d');
            if (!context) return;

            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
            context.drawImage(videoRef.current, 0, 0);

            canvasRef.current.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
                    handleFile(file);
                    stopCamera();
                }
            }, 'image/jpeg', 0.9);
        }
    };

    const validateFile = (file: File): boolean => {
        // Check file type
        if (!file.type.startsWith('image/')) {
            setError('Please upload an image file');
            return false;
        }

        // Check file size
        const maxSize = maxSizeMB * 1024 * 1024;
        if (file.size > maxSize) {
            setError(`File size must be less than ${maxSizeMB}MB`);
            return false;
        }

        setError(null);
        return true;
    };

    const handleFile = useCallback((file: File) => {
        if (!validateFile(file)) return;

        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);

        // Call parent callback
        onFileSelect(file);
    }, [onFileSelect]);

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (disabled) return;

        const files = e.dataTransfer.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (disabled) return;

        const files = e.target.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    };

    const clearFile = () => {
        setPreview(null);
        setError(null);
    };

    return (
        <div className="w-full">
            {isCameraOpen ? (
                <div className="relative border-2 border-[rgb(var(--color-border))] rounded-[var(--radius-xl)] p-4 bg-black overflow-hidden flex flex-col items-center animate-scale-in">
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        className="w-full max-h-[60vh] object-contain bg-black rounded-lg transform scale-x-[-1]"
                    />
                    <canvas ref={canvasRef} className="hidden" />

                    <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-4">
                        <button
                            onClick={stopCamera}
                            className="p-3 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors shadow-lg flex items-center justify-center group"
                            title="Close Camera"
                        >
                            <X className="w-6 h-6 group-hover:scale-110 transition-transform" />
                        </button>
                        <button
                            onClick={captureImage}
                            className="p-4 bg-white text-blue-600 rounded-full hover:bg-gray-100 transition-colors shadow-xl flex items-center justify-center group border-4 border-blue-100"
                            title="Take Photo"
                        >
                            <Camera className="w-8 h-8 group-hover:scale-110 transition-transform" />
                        </button>
                    </div>
                </div>
            ) : !preview ? (
                <div
                    className={`
            relative border-2 border-dashed rounded-[var(--radius-xl)] p-8
            transition-all duration-300
            ${dragActive
                            ? 'border-[rgb(var(--color-primary))] bg-[rgb(var(--color-primary)/0.05)]'
                            : 'border-[rgb(var(--color-border))] hover:border-[rgb(var(--color-primary)/0.5)]'
                        }
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    <input
                        type="file"
                        accept={accept}
                        onChange={handleChange}
                        disabled={disabled}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed z-10"
                        id="file-upload"
                    />

                    <div className="flex flex-col items-center justify-center gap-4 text-center">
                        <div className="flex gap-4 mb-2">
                            <div className={`
                p-4 rounded-full transition-all duration-300
                ${dragActive
                                    ? 'bg-[rgb(var(--color-primary)/0.1)] scale-110'
                                    : 'bg-[rgb(var(--color-bg-secondary))]'
                                }
                `}>
                                <Upload
                                    className={`w-8 h-8 ${dragActive ? 'text-[rgb(var(--color-primary))]' : 'text-[rgb(var(--color-text-secondary))]'}`}
                                />
                            </div>

                            {!disabled && (
                                <button
                                    onClick={(e) => {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        startCamera();
                                    }}
                                    className="p-4 rounded-full bg-[rgb(var(--color-bg-secondary))] hover:bg-blue-50 hover:text-blue-600 transition-colors duration-300 z-20 relative"
                                    title="Use Camera"
                                >
                                    <Camera className="w-8 h-8 text-[rgb(var(--color-text-secondary))] hover:text-blue-600 transition-colors" />
                                </button>
                            )}
                        </div>

                        <div>
                            <p className="text-lg font-medium text-[rgb(var(--color-text))]">
                                {dragActive ? 'Drop your image here' : 'Drag & drop an image'}
                            </p>
                            <p className="text-sm text-[rgb(var(--color-text-secondary))] mt-1">
                                or click to browse files, or tap the camera icon to take a photo
                            </p>
                        </div>

                        <p className="text-xs text-[rgb(var(--color-text-secondary))]">
                            Supports: JPG, PNG, WebP (max {maxSizeMB}MB)
                        </p>
                    </div>
                </div>
            ) : (
                <div className="relative group animate-scale-in">
                    <div className="relative rounded-[var(--radius-xl)] overflow-hidden border-2 border-[rgb(var(--color-border))]">
                        <img
                            src={preview}
                            alt="Preview"
                            className="w-full h-auto max-h-96 object-contain bg-[rgb(var(--color-bg-secondary))]"
                        />

                        {!disabled && (
                            <button
                                onClick={clearFile}
                                className="absolute top-4 right-4 p-2 rounded-full bg-[rgb(var(--color-danger))] text-white
                         opacity-0 group-hover:opacity-100 transition-opacity duration-200
                         hover:scale-110 transform"
                                aria-label="Remove image"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        )}
                    </div>

                    <div className="mt-4 flex items-center gap-2 text-sm text-[rgb(var(--color-text-secondary))]">
                        <ImageIcon className="w-4 h-4" />
                        <span>Image ready for analysis</span>
                    </div>
                </div>
            )}

            {error && (
                <div className="mt-4 p-4 rounded-[var(--radius-md)] bg-[rgb(var(--color-danger)/0.1)] border border-[rgb(var(--color-danger)/0.3)]">
                    <p className="text-sm text-[rgb(var(--color-danger))] font-medium">
                        {error}
                    </p>
                </div>
            )}
        </div>
    );
}
