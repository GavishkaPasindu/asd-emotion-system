"use client";

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import YouTubePlayer from '../../components/YouTubePlayer';
import CameraCapture from '../../components/CameraCapture';
import ASDFriendlyUI from '../../components/ASDFriendlyUI';
import PuzzleGame from '../../components/PuzzleGame';
import SocialAvatar from '../../components/SocialAvatar';
import { Video, Award, ChevronLeft, Play } from 'lucide-react';
import { useModel } from '../context/ModelContext';

// --─ Types --------------------------------------------------------------------
interface ActivityItem {
    id: string;
    title: string;
    description?: string;
    category?: string;
}
interface Category {
    id: string;
    title: string;
    emoji: string;
    description: string;
    color: string;
    items: ActivityItem[];
}

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

// --─ Main page ----------------------------------------------------------------
export default function GamificationPage() {
    const router = useRouter();
    const { selectedModel } = useModel();

    type Step = 'intro' | 'category' | 'item' | 'capture' | 'completed';
    const [step, setStep] = useState<Step>('intro');
    const [categories, setCategories] = useState<Category[]>([]);
    const [selectedCategory, setSelectedCategory] = useState<Category | null>(null);
    const [selectedItem, setSelectedItem] = useState<ActivityItem | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const sessionIdRef = React.useRef<string | null>(null);
    const [countDown, setCountDown] = useState(3);
    const [isCapturing, setIsCapturing] = useState(false);
    const [capturedFrames, setCapturedFrames] = useState(0);
    const [loading, setLoading] = useState(false);
    const [lastAnalysis, setLastAnalysis] = useState('');

    // Fetch available activity categories on mount
    useEffect(() => {
        fetch(`${API}/api/gamification/activities`)
            .then(r => r.json())
            .then(d => { if (d.success) setCategories(d.categories); })
            .catch(console.error);
    }, []);

    // -- Start session ----------------------------------------------------------
    const handleStartSession = async (category: Category, item: ActivityItem) => {
        setLoading(true);
        try {
            const res = await fetch(`${API}/api/gamification/start-session`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    selected_video: item.id,
                    duration: 10,
                    session_type: 'screening',
                    category: category.id,
                }),
            });
            const data = await res.json();
            if (data.success) {
                setSessionId(data.session_id);
                sessionIdRef.current = data.session_id;
                setStep('capture');
                startCountdown();
            }
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    // Auto-start social sessions via effect (never call setState during render)
    useEffect(() => {
        if (step === 'item' && selectedCategory?.id === 'social') {
            const item = selectedCategory.items[0];
            if (item) handleStartSession(selectedCategory, item);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [step, selectedCategory?.id]);



    // -- Frame capture --------------------------------------------------------─
    const handleFrameCapture = async (imageSrc: string) => {
        if (!sessionIdRef.current || !isCapturing) return;
        try {
            const res = await fetch(`${API}/api/gamification/capture-frame`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionIdRef.current, frame: imageSrc, model_type: selectedModel }),
            });
            const data = await res.json();
            if (data.success) {
                setCapturedFrames(p => p + 1);
                if (data.emotion_result?.predicted_emotion) {
                    setLastAnalysis(`Emotion: ${data.emotion_result.predicted_emotion}`);
                }
            }
        } catch { /* silent */ }
    };

    // -- Finish session --------------------------------------------------------─
    const finishSession = async (solved: boolean = false) => {
        setIsCapturing(false);
        setLoading(true);
        const sid = sessionIdRef.current;
        if (!sid) { setLoading(false); return; }
        try {
            await fetch(`${API}/api/gamification/end-session`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sid }),
            });
            setStep('completed');
            setTimeout(() => router.push(`/gamification/results?session_id=${sid}&solved=${solved}`), 3000);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    // -- Pre-finish logic to clear intervals ----------------------------------
    // Used by Puzzle when solved early, or Social when finished early
    const handleEarlyFinish = () => {
        if (isCapturing) finishSession(true);
    };

    // -- Countdown ------------------------------------------------------------─
    const startCountdown = () => {
        setCountDown(3);
        const t = setInterval(() => {
            setCountDown(prev => {
                if (prev <= 1) {
                    clearInterval(t);
                    setIsCapturing(true);

                    // Determine max duration based on category
                    // Social takes ~60s (8 prompts * 8s)
                    // Cartoon, Music, Puzzle take max 30s
                    const durationMs = selectedCategory?.id === 'social' ? 65000 : 30000;
                    setTimeout(() => {
                        // Only finish if still capturing (might have finished early)
                        if (sessionIdRef.current) finishSession();
                    }, durationMs);

                    return 0;
                }
                return prev - 1;
            });
        }, 1000);
    };

    // ══════════════════════════════════════════════════════════════════════════
    // STEP: INTRO
    // ══════════════════════════════════════════════════════════════════════════
    if (step === 'intro') return (
        <div className="relative min-h-screen bg-gradient-to-br from-cyan-100 to-indigo-100 flex flex-col items-center justify-center p-4 overflow-hidden">
            <div className="text-center space-y-8 animate-fade-in">
                <div className="text-8xl animate-bounce">🎮</div>
                <h1 className="text-5xl md:text-6xl font-black text-blue-700 drop-shadow">Let's Play a Game!</h1>
                <p className="text-2xl text-gray-600 font-medium max-w-lg mx-auto">
                    Choose an activity and we'll watch your face to help you feel great!
                </p>
                <button
                    onClick={() => setStep('category')}
                    className="transform transition-all hover:scale-105 active:scale-95 bg-yellow-400 hover:bg-yellow-300 text-yellow-900 text-3xl font-bold py-6 px-12 rounded-full shadow-xl border-b-4 border-yellow-600 flex items-center gap-4 mx-auto"
                >
                    <Play className="w-10 h-10 fill-yellow-900" />
                    Start Playing
                </button>
            </div>
            {/* Background blobs */}
            <div className="absolute top-10 left-10 w-64 h-64 bg-purple-200 rounded-full blur-3xl opacity-50" />
            <div className="absolute top-10 right-10 w-64 h-64 bg-yellow-200 rounded-full blur-3xl opacity-50" />
            <div className="absolute bottom-10 left-20 w-64 h-64 bg-pink-200 rounded-full blur-3xl opacity-50" />
        </div>
    );

    // ══════════════════════════════════════════════════════════════════════════
    // STEP: CATEGORY PICKER
    // ══════════════════════════════════════════════════════════════════════════
    if (step === 'category') return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-50 p-4 md:p-12">
            <h2 className="text-3xl md:text-4xl font-black text-center text-indigo-800 mb-8 md:mb-12">
                What do you want to do? 🌟
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 md:gap-8 max-w-4xl mx-auto">
                {categories.map(cat => (
                    <button
                        key={cat.id}
                        onClick={() => { setSelectedCategory(cat); setStep('item'); }}
                        className={`group bg-gradient-to-br ${cat.color} rounded-3xl p-6 md:p-8 shadow-xl hover:shadow-2xl transition-all transform hover:-translate-y-2 text-white text-left flex flex-col items-center sm:items-start text-center sm:text-left`}
                    >
                        <div className="text-5xl md:text-6xl mb-3 md:mb-4 group-hover:scale-110 transition-transform inline-block">
                            {cat.emoji}
                        </div>
                        <h3 className="text-xl md:text-2xl font-black mb-2">{cat.title}</h3>
                        <p className="text-white/80 text-sm md:text-base">{cat.description}</p>
                        <div className="mt-4 md:mt-6 flex items-center justify-center sm:justify-start gap-2 text-white/90 font-semibold text-sm md:text-base">
                            <Play className="w-4 h-4 md:w-5 md:h-5 fill-white" />
                            <span>{cat.items.length} {cat.items.length === 1 ? 'option' : 'options'}</span>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );

    // ══════════════════════════════════════════════════════════════════════════
    // STEP: ITEM PICKER (video / puzzle theme / social)
    // ══════════════════════════════════════════════════════════════════════════
    if (step === 'item' && selectedCategory) {
        const cat = selectedCategory;
        // Social: auto-start handled by useEffect above -- show loader
        if (cat.id === 'social') {
            return (
                <div className="min-h-screen flex items-center justify-center bg-indigo-50 p-4">
                    <ASDFriendlyUI type="loading" message="Getting ready..." />
                </div>
            );
        }
        return (
            <div className="min-h-screen bg-white p-4 md:p-12">
                <button onClick={() => setStep('category')} className="flex items-center gap-2 text-gray-500 hover:text-gray-800 mb-6 md:mb-8 font-semibold transition-colors text-sm md:text-base">
                    <ChevronLeft className="w-4 h-4 md:w-5 md:h-5" /> Back to categories
                </button>

                <h2 className="text-2xl md:text-3xl font-black text-center text-gray-800 mb-8 md:mb-10">
                    {cat.emoji} Pick your {cat.title}!
                </h2>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6 max-w-5xl mx-auto">
                    {cat.items.map(item => (
                        <button
                            key={item.id}
                            onClick={() => { setSelectedItem(item); handleStartSession(cat, item); }}
                            className="group bg-white rounded-2xl overflow-hidden shadow-lg hover:shadow-2xl transition-all transform hover:-translate-y-1 border-2 border-transparent hover:border-indigo-400 text-left"
                        >
                            {/* Thumbnail for videos, gradient for puzzle/social */}
                            {(cat.id === 'cartoon' || cat.id === 'music') ? (
                                <div className="relative aspect-video overflow-hidden">
                                    <img
                                        src={`https://img.youtube.com/vi/${item.id}/mqdefault.jpg`}
                                        alt={item.title}
                                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                    />
                                    <div className="absolute inset-0 bg-black/10 group-hover:bg-transparent transition-colors flex items-center justify-center">
                                        <div className="bg-white/90 p-3 rounded-full shadow-lg group-hover:scale-110 transition-transform">
                                            <Video className="w-7 h-7 text-indigo-600" fill="currentColor" />
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className={`aspect-video bg-gradient-to-br ${cat.color} flex items-center justify-center`}>
                                    <span className="text-7xl">{cat.emoji}</span>
                                </div>
                            )}
                            <div className="p-5">
                                <h3 className="font-bold text-gray-800 text-lg truncate">{item.title}</h3>
                                {item.description && (
                                    <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                                )}
                            </div>
                        </button>
                    ))}
                </div>

                {loading && (
                    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center backdrop-blur-sm">
                        <ASDFriendlyUI type="loading" message="Starting session..." />
                    </div>
                )}
            </div>
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // STEP: CAPTURE -- full-screen stimulus + PIP camera bottom-left
    // ══════════════════════════════════════════════════════════════════════════
    if (step === 'capture') return (
        <div className="fixed inset-0 bg-black overflow-hidden">
            {/* Countdown overlay */}
            {countDown > 0 && (
                <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md">
                    <ASDFriendlyUI type="countdown" count={countDown} />
                </div>
            )}

            {/* -- Full-screen stimulus -- */}
            <div className="absolute inset-0 w-full h-full">
                {selectedCategory?.id === 'puzzle' && selectedItem && (
                    <PuzzleGame theme={selectedItem.id as 'animals' | 'faces' | 'fruits'} onSolve={handleEarlyFinish} />
                )}
                {selectedCategory?.id === 'social' && (
                    <SocialAvatar />
                )}
                {(selectedCategory?.id === 'cartoon' || selectedCategory?.id === 'music') && selectedItem && (
                    <YouTubePlayer videoId={selectedItem.id} autoplay={true} className="w-full h-full" />
                )}
            </div>

            {/* "Watch!" badge top-centre */}
            {isCapturing && (
                <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30 bg-yellow-400 text-yellow-900 px-6 py-2 rounded-full font-bold shadow-lg animate-bounce text-lg">
                    {selectedCategory?.id === 'puzzle' ? '🧩 Solve the Puzzle!'
                        : selectedCategory?.id === 'social' ? '👋 Interact with the friend!'
                            : selectedCategory?.id === 'music' ? '🎵 Sing Along!'
                                : '📺 Watch the Video!'}
                </div>
            )}

            {/* PIP Camera -- bottom-left */}
            <div className="absolute bottom-6 left-6 z-30 w-48 h-36 md:w-56 md:h-44 rounded-2xl overflow-hidden border-4 border-white shadow-2xl">
                <CameraCapture onCapture={handleFrameCapture} isCapturing={isCapturing} interval={500} />
                {lastAnalysis && (
                    <span className="absolute bottom-1 left-1 right-1 text-center text-[10px] bg-blue-600/90 text-white rounded-full px-2 py-0.5 truncate animate-pulse">
                        {lastAnalysis}
                    </span>
                )}
                {isCapturing && (
                    <span className="absolute top-2 right-2 w-3 h-3 rounded-full bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.8)] animate-pulse" />
                )}
            </div>

            {/* Progress bar -- bottom-centre */}
            {isCapturing && (
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30 w-72 md:w-96">
                    <div className="h-3 bg-white/20 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-green-400 to-blue-500 animate-progress origin-left" style={{ animationDuration: '10s' }} />
                    </div>
                    <p className="text-center text-white/60 mt-1 text-xs font-mono">Screening in progress...</p>
                </div>
            )}
        </div>
    );

    // ══════════════════════════════════════════════════════════════════════════
    // STEP: COMPLETED
    // ══════════════════════════════════════════════════════════════════════════
    if (step === 'completed') return (
        <div className="min-h-screen bg-gradient-to-b from-blue-400 to-purple-500 flex items-center justify-center">
            <ASDFriendlyUI type="success" />
        </div>
    );

    return null;
}
