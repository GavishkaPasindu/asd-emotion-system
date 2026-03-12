"use client";

import React from "react";
import { Star, Heart, Smile } from "lucide-react";

interface ASDFriendlyUIProps {
    type?: "loading" | "success" | "countdown";
    message?: string;
    count?: number;
}

const ASDFriendlyUI: React.FC<ASDFriendlyUIProps> = ({
    type = "loading",
    message,
    count
}) => {

    if (type === "countdown") {
        return (
            <div className="flex flex-col items-center justify-center animate-bounce-Gentle">
                <div className="relative">
                    <div className="w-32 h-32 rounded-full border-8 border-yellow-400 flex items-center justify-center bg-white shadow-xl">
                        <span className="text-6xl font-black text-blue-500">{count}</span>
                    </div>
                    {/* Decorative stars */}
                    <Star className="absolute -top-2 -right-2 text-yellow-400 w-10 h-10 animate-spin-slow" fill="currentColor" />
                    <Star className="absolute -bottom-2 -left-2 text-purple-400 w-8 h-8 animate-pulse" fill="currentColor" />
                </div>
                <p className="mt-4 text-2xl font-bold text-white drop-shadow-md">Get Ready!</p>
            </div>
        );
    }

    if (type === "success") {
        return (
            <div className="text-center p-8 bg-white/90 rounded-3xl backdrop-blur-sm shadow-2xl border-4 border-green-400 transform transition-all animate-pop-in">
                <div className="flex justify-center mb-4">
                    <div className="bg-green-100 p-4 rounded-full">
                        <Smile className="w-16 h-16 text-green-500" />
                    </div>
                </div>
                <h3 className="text-3xl font-black text-gray-800 mb-2">Great Job!</h3>
                <p className="text-xl text-gray-600 font-medium">You did amazing!</p>

                <div className="flex justify-center gap-2 mt-6">
                    <Star className="w-8 h-8 text-yellow-400" fill="currentColor" />
                    <Star className="w-10 h-10 text-yellow-400 -mt-2" fill="currentColor" />
                    <Star className="w-8 h-8 text-yellow-400" fill="currentColor" />
                </div>
            </div>
        );
    }

    // Default Loading Animation
    return (
        <div className="flex flex-col items-center">
            <div className="flex gap-3 mb-6">
                <div className="w-6 h-6 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                <div className="w-6 h-6 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-6 h-6 bg-yellow-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                <div className="w-6 h-6 bg-red-500 rounded-full animate-bounce" style={{ animationDelay: '0.6s' }}></div>
            </div>
            {message && (
                <p className="text-xl font-bold text-white tracking-wide animate-pulse">{message}</p>
            )}
        </div>
    );
};

export default ASDFriendlyUI;
