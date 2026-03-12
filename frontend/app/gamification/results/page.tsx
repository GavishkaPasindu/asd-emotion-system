"use client";

import React, { useEffect, useState, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { CheckCircle, AlertTriangle, Home, RotateCcw, Activity, Star } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import LoadingSpinner from '../../components/LoadingSpinner';

interface EmotionStat {
    emotion: string;
    count: number;
    percentage: number;
}

interface SessionResult {
    id: string;
    status: string;
    total_frames: number;
    predictions: any[];
    emotion_stats?: EmotionStat[];
    metadata: {
        child_age: number;
        selected_video: string;
    };
}

function ResultsContent() {
    const searchParams = useSearchParams();
    const sessionId = searchParams.get('session_id');
    const solved = searchParams.get('solved');

    const [session, setSession] = useState<SessionResult | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!sessionId) {
            setError("No session ID provided");
            setLoading(false);
            return;
        }

        const fetchResults = async () => {
            try {
                const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'}/api/gamification/session/${sessionId}`);
                const data = await response.json();

                if (data.success) {
                    setSession(data.session as SessionResult);
                } else {
                    setError(data.error || "Failed to fetch results");
                }
            } catch (err) {
                setError("Error connecting to server");
            } finally {
                setLoading(false);
            }
        };

        fetchResults();
    }, [sessionId]);

    if (loading) {
        return <LoadingSpinner />;
    }

    if (error || !session) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-4">
                <AlertTriangle className="w-16 h-16 text-yellow-500 mb-4" />
                <h1 className="text-2xl font-bold text-gray-800 mb-2">Oops! Something went wrong</h1>
                <p className="text-gray-600 mb-6">{error || "Results not found"}</p>
                <Link href="/gamification" className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
                    Return to Screening
                </Link>
            </div>
        );
    }

    // Emotion colors map
    const EMOTION_COLORS: Record<string, string> = {
        'happy': '#4ade80',   // Green
        'joy': '#4ade80',     // Green
        'angry': '#f87171',   // Red
        'anger': '#f87171',   // Red
        'sad': '#60a5fa',     // Blue
        'sadness': '#60a5fa', // Blue
        'neutral': '#94a3b8', // Gray
        'fear': '#a78bfa',    // Purple
        'surprise': '#fbbf24' // Yellow
    };

    const pieData = session.emotion_stats && session.emotion_stats.length > 0
        ? session.emotion_stats.map((stat) => ({
            name: stat.emotion,
            value: stat.count,
            color: EMOTION_COLORS[stat.emotion.toLowerCase()] || '#cbd5e1'
        }))
        : [
            { name: 'No Data', value: 100, color: '#e2e8f0' }
        ];

    const resultMessage = "Screening Completed Successfully";
    const resultDescription = "Based on the preliminary analysis, the system has successfully captured and processed the screening session.";

    return (
        <div className="min-h-screen bg-gray-50 p-6 md:p-12">
            <div className="max-w-5xl mx-auto">
                <div className="bg-white rounded-3xl shadow-xl overflow-hidden mb-8 animate-slide-up">
                    {(solved === 'false' || ['animals', 'faces', 'fruits', 'puzzle'].includes(session.metadata.selected_video)) ? (
                        <div className="bg-gradient-to-r from-yellow-400 to-orange-500 p-8 text-white text-center">
                            <div className="inline-flex items-center justify-center w-16 h-16 bg-white/30 rounded-full backdrop-blur-sm mb-4 shadow-lg">
                                <Star className="w-10 h-10 text-white fill-white" />
                            </div>
                            <h1 className="text-3xl md:text-4xl font-black mb-2 tracking-tight">Game Completed!</h1>
                            <p className="text-yellow-50 text-xl font-bold max-w-2xl mx-auto mb-6">Great job playing! Here are 5 stars for your amazing effort!</p>
                            <div className="flex justify-center gap-2 mb-2 animate-bounce">
                                {[1, 2, 3, 4, 5].map(i => (
                                    <Star key={i} className="w-10 h-10 md:w-14 md:h-14 text-yellow-100 fill-white drop-shadow-md" />
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-8 text-white text-center">
                            <div className="inline-flex items-center justify-center w-16 h-16 bg-white/20 rounded-full backdrop-blur-sm mb-4">
                                <CheckCircle className="w-10 h-10 text-white" />
                            </div>
                            <h1 className="text-3xl md:text-4xl font-bold mb-2">{resultMessage}</h1>
                            <p className="text-blue-100 max-w-2xl mx-auto">{resultDescription}</p>
                        </div>
                    )}

                    <div className="p-8">
                        <div className="grid md:grid-cols-2 gap-8 mb-12">
                            {/* Summary Stats */}
                            <div className="bg-gray-50 p-6 rounded-2xl border border-gray-100">
                                <h3 className="text-lg font-semibold text-gray-700 mb-4 flex items-center gap-2">
                                    <Activity className="w-5 h-5 text-blue-500" />
                                    Session Summary
                                </h3>
                                <div className="space-y-4">
                                    <div className="flex justify-between items-center border-b pb-2">
                                        <span className="text-gray-500">Session ID</span>
                                        <span className="font-mono text-xs bg-gray-200 px-2 py-1 rounded text-gray-700">
                                            {session.id.substring(0, 8)}...
                                        </span>
                                    </div>
                                    <div className="flex justify-between items-center border-b pb-2">
                                        <span className="text-gray-500">Frames Captured</span>
                                        <span className="font-bold text-gray-800">{session.total_frames}</span>
                                    </div>
                                    <div className="flex justify-between items-center border-b pb-2">
                                        <span className="text-gray-500">Duration</span>
                                        <span className="font-bold text-gray-800">30 seconds</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-gray-500">Video Watched</span>
                                        <span className="font-medium text-gray-800 truncate max-w-[150px]">
                                            {session.metadata.selected_video || 'N/A'}
                                        </span>
                                    </div>

                                    {/* Emotion Breakdown List */}
                                    <div className="pt-4 border-t mt-4">
                                        <h4 className="text-sm font-bold text-gray-700 mb-3">Emotion Counts:</h4>
                                        <div className="space-y-2">
                                            {session.emotion_stats && session.emotion_stats.map((stat) => (
                                                <div key={stat.emotion} className="flex justify-between items-center text-sm p-2 bg-white rounded-lg border border-gray-100 shadow-sm">
                                                    <div className="flex items-center gap-2">
                                                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: EMOTION_COLORS[stat.emotion.toLowerCase()] || '#cbd5e1' }}></div>
                                                        <span className="capitalize font-medium text-gray-700">{stat.emotion}</span>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <span className="font-bold text-gray-800">{stat.count}</span>
                                                        <span className="text-xs text-gray-400">({stat.percentage}%)</span>
                                                    </div>
                                                </div>
                                            ))}
                                            {(!session.emotion_stats || session.emotion_stats.length === 0) && (
                                                <p className="text-sm text-gray-400 italic text-center py-2">No emotions detected</p>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Analysis Chart */}
                            <div className="h-64 relative">
                                <h3 className="text-lg font-semibold text-gray-700 mb-2 text-center">Emotion Distribution</h3>
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={pieData}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={80}
                                            paddingAngle={5}
                                            dataKey="value"
                                        >
                                            {pieData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Pie>
                                        <Tooltip />
                                        <Legend />
                                    </PieChart>
                                </ResponsiveContainer>
                                <div className="absolute inset-0 flex items-center justify-center pointer-events-none pt-6">
                                    <span className="text-sm font-medium text-gray-400">Results</span>
                                </div>
                            </div>
                        </div>

                        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 mb-8 flex gap-4 items-start">
                            <AlertTriangle className="w-6 h-6 text-yellow-600 flex-shrink-0 mt-1" />
                            <div>
                                <h4 className="font-bold text-yellow-800 mb-1">Important Note</h4>
                                <p className="text-yellow-700 text-sm">
                                    This screening tool is for educational and experimental purposes only.
                                    It does NOT provide a medical diagnosis. Please consult a healthcare professional for clinical advice.
                                </p>
                            </div>
                        </div>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Link
                                href="/gamification"
                                className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition font-medium"
                            >
                                <RotateCcw className="w-5 h-5" />
                                Start New Session
                            </Link>
                            <Link
                                href="/"
                                className="flex items-center justify-center gap-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-xl hover:bg-gray-200 transition font-medium"
                            >
                                <Home className="w-5 h-5" />
                                Back to Home
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function ResultsPage() {
    return (
        <Suspense fallback={<LoadingSpinner />}>
            <ResultsContent />
        </Suspense>
    );
}
