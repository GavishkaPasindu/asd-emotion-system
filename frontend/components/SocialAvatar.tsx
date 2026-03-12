'use client';

import { useState, useEffect } from 'react';

const PROMPTS = [
    { text: "Hello! 👋 Can you wave back at me?", emoji: "👋", delay: 0 },
    { text: "How are you feeling today? 😊", emoji: "😊", delay: 8000 },
    { text: "Can you show me a big smile? 😁", emoji: "😁", delay: 16000 },
    { text: "Do you like playing games? 🎮", emoji: "🎮", delay: 24000 },
    { text: "What's your favourite colour? 🌈", emoji: "🌈", delay: 32000 },
    { text: "Can you clap your hands? 👏👏", emoji: "👏", delay: 40000 },
    { text: "You are doing SO well! ⭐", emoji: "⭐", delay: 48000 },
    { text: "One more — give me a thumbs up! 👍", emoji: "👍", delay: 56000 },
];

export default function SocialAvatar() {
    const [promptIdx, setPromptIdx] = useState(0);
    const [bounce, setBounce] = useState(false);

    useEffect(() => {
        const timers = PROMPTS.map((p, i) =>
            setTimeout(() => {
                setPromptIdx(i);
                setBounce(true);
                setTimeout(() => setBounce(false), 600);
            }, p.delay)
        );
        return () => timers.forEach(clearTimeout);
    }, []);

    const current = PROMPTS[promptIdx];

    return (
        <div className="flex flex-col items-center justify-center h-full w-full bg-gradient-to-br from-sky-400 to-violet-500">
            {/* Title */}
            <p className="text-white/80 text-lg font-semibold mb-8 tracking-wide">
                👤 Social Interaction Mode
            </p>

            {/* Avatar face */}
            <div className={`relative transition-transform duration-300 ${bounce ? 'scale-110' : 'scale-100'}`}>
                {/* Body */}
                <div className="flex flex-col items-center">
                    {/* Head */}
                    <div className="w-48 h-48 md:w-56 md:h-56 rounded-full bg-yellow-300 border-8 border-yellow-400 shadow-2xl flex items-center justify-center relative overflow-hidden">
                        {/* Eyes */}
                        <div className="absolute top-12 flex gap-10">
                            <div className="w-8 h-8 rounded-full bg-gray-800 flex items-end justify-center pb-1">
                                <div className="w-3 h-3 rounded-full bg-white" />
                            </div>
                            <div className="w-8 h-8 rounded-full bg-gray-800 flex items-end justify-center pb-1">
                                <div className="w-3 h-3 rounded-full bg-white" />
                            </div>
                        </div>
                        {/* Cheeks */}
                        <div className="absolute top-20 flex gap-24">
                            <div className="w-8 h-5 rounded-full bg-pink-300 opacity-60" />
                            <div className="w-8 h-5 rounded-full bg-pink-300 opacity-60" />
                        </div>
                        {/* Mouth — big smile */}
                        <div className="absolute top-28 w-20 h-10 border-b-8 border-gray-800 rounded-b-full" />
                    </div>

                    {/* Waving hand */}
                    <div className={`mt-2 text-7xl transition-transform duration-500 ${bounce ? 'rotate-12' : '-rotate-6'}`} style={{ animation: 'wave 2s ease-in-out infinite' }}>
                        {current.emoji}
                    </div>
                </div>
            </div>

            {/* Speech bubble */}
            <div className="relative mt-8 max-w-sm mx-4">
                <div className="bg-white rounded-3xl px-8 py-5 shadow-2xl text-center">
                    {/* Triangle pointer */}
                    <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-0 h-0"
                        style={{ borderLeft: '12px solid transparent', borderRight: '12px solid transparent', borderBottom: '16px solid white' }} />
                    <p className="text-2xl font-bold text-gray-800 leading-snug">
                        {current.text}
                    </p>
                </div>
            </div>

            {/* Progress dots */}
            <div className="flex gap-2 mt-8">
                {PROMPTS.map((_, i) => (
                    <div
                        key={i}
                        className={`w-3 h-3 rounded-full transition-all duration-300 ${i === promptIdx ? 'bg-white scale-125' : 'bg-white/40'}`}
                    />
                ))}
            </div>

            <style jsx>{`
                @keyframes wave {
                    0%, 100% { transform: rotate(-6deg); }
                    50% { transform: rotate(12deg); }
                }
            `}</style>
        </div>
    );
}
