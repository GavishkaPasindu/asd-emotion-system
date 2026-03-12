"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, Volume2, VolumeX } from 'lucide-react';

interface YouTubePlayerProps {
    videoId: string;
    autoplay?: boolean;
    onEnded?: () => void;
    className?: string;
}

const YouTubePlayer: React.FC<YouTubePlayerProps> = ({
    videoId,
    autoplay = false,
    onEnded,
    className = ""
}) => {
    const [isPlaying, setIsPlaying] = useState(autoplay);
    const [isMuted, setIsMuted] = useState(false);
    const playerRef = useRef<HTMLIFrameElement>(null);

    // YouTube Embed URL construction
    const getEmbedUrl = () => {
        const params = new URLSearchParams({
            autoplay: autoplay ? '1' : '0',
            controls: '0', // Hide controls for cleaner UI for kids
            modestbranding: '1',
            rel: '0', // Don't show related videos
            showinfo: '0',
            iv_load_policy: '3', // Hide annotations
            enablejsapi: '1',
            origin: typeof window !== 'undefined' ? window.location.origin : '',
        });
        return `https://www.youtube.com/embed/${videoId}?${params.toString()}`;
    };

    const isFullscreen = className.includes('h-full');

    return (
        <div className={`relative overflow-hidden ${isFullscreen ? '' : 'rounded-2xl shadow-lg border-4 border-yellow-400'} ${className}`}>
            {/* TV decoration — only in normal mode */}
            {!isFullscreen && (
                <>
                    <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 w-20 h-3 bg-yellow-400 rounded-t-lg z-10"></div>
                    <div className="absolute -top-8 left-1/2 transform -translate-x-6 w-1 h-8 bg-gray-400 rotate-[-15deg] z-0"></div>
                    <div className="absolute -top-8 left-1/2 transform translate-x-6 w-1 h-8 bg-gray-400 rotate-[15deg] z-0"></div>
                </>
            )}

            <div className={isFullscreen ? 'absolute inset-0 bg-black' : 'aspect-video w-full bg-black relative'}>
                <iframe
                    ref={playerRef}
                    src={getEmbedUrl()}
                    title="YouTube video player"
                    className="w-full h-full"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                />

                {/* Child-friendly Overlay Controls */}
                <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/60 to-transparent flex justify-between items-center opacity-0 hover:opacity-100 transition-opacity duration-300">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="p-3 bg-white/20 hover:bg-white/40 rounded-full backdrop-blur-sm text-white transition-all transform hover:scale-110"
                    >
                        {isPlaying ? <Pause size={24} fill="white" /> : <Play size={24} fill="white" />}
                    </button>

                    <button
                        onClick={() => setIsMuted(!isMuted)}
                        className="p-3 bg-white/20 hover:bg-white/40 rounded-full backdrop-blur-sm text-white transition-all transform hover:scale-110"
                    >
                        {isMuted ? <VolumeX size={24} /> : <Volume2 size={24} />}
                    </button>
                </div>
            </div>

            {/* Fun border bottom — only in normal mode */}
            {!isFullscreen && (
                <div className="h-4 bg-yellow-400 w-full flex justify-center items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                    <div className="w-16 h-1 bg-black/20 rounded-full"></div>
                </div>
            )}
        </div>
    );
};

export default YouTubePlayer;
