'use client';

import { useState, useEffect } from 'react';
import { Smile, Upload, Sparkles } from 'lucide-react';
import FileUpload from '../components/FileUpload';
import LoadingSpinner from '../components/LoadingSpinner';
import ResultsDisplay from '../components/ResultsDisplay';
import { api, PredictionResponse } from '@/lib/api';

export default function EmotionRecognitionPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<PredictionResponse | null>(null);
    const [availableLabels, setAvailableLabels] = useState<string[]>(['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Neutral']);

    useEffect(() => {
        const fetchLabels = async () => {
            try {
                const info = await api.getModelsInfo();
                const labels = info.models['resnet50v2']?.emotion_classes;
                if (labels && labels.length > 0) {
                    setAvailableLabels(labels);
                }
            } catch (err) {
                console.error("Failed to fetch model labels:", err);
            }
        };
        fetchLabels();
    }, []);

    const handleFileSelect = (file: File) => {
        setSelectedFile(file);
        setResult(null);
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        setLoading(true);
        try {
            const prediction = await api.predictEmotion(selectedFile, true);
            setResult(prediction);
        } catch (error) {
            console.error('Analysis error:', error);
            setResult({
                success: false,
                confidence: 0,
                error: error instanceof Error ? error.message : 'Analysis failed',
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="min-h-screen bg-[rgb(var(--color-bg))] py-12">
            <div className="container-custom">
                {/* Header */}
                <div className="text-center mb-12 animate-slide-down">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[rgb(var(--color-accent)/0.1)] mb-4">
                        <Smile className="w-5 h-5 text-[rgb(var(--color-accent))]" />
                        <span className="text-sm font-medium text-[rgb(var(--color-accent))]">
                            Emotion Recognition
                        </span>
                    </div>

                    <h1 className="text-4xl md:text-5xl font-bold mb-4 text-[rgb(var(--color-text))]">
                        Facial Emotion Recognition
                    </h1>

                    <p className="text-lg text-[rgb(var(--color-text-secondary))] max-w-2xl mx-auto">
                        Analyze facial expressions to detect emotions with AI-powered recognition
                    </p>
                </div>

                <div className="max-w-4xl mx-auto">
                    <div className="grid lg:grid-cols-2 gap-8">
                        {/* Upload Section */}
                        <div className="space-y-6">
                            <div className="card">
                                <h2 className="text-xl font-semibold mb-4 text-[rgb(var(--color-text))]">
                                    Upload Image
                                </h2>

                                <FileUpload
                                    onFileSelect={handleFileSelect}
                                    disabled={loading}
                                />

                                {selectedFile && !loading && (
                                    <button
                                        onClick={handleAnalyze}
                                        className="btn-gradient w-full mt-6"
                                    >
                                        <Sparkles className="w-5 h-5 inline mr-2" />
                                        Detect Emotion
                                    </button>
                                )}
                            </div>

                            {/* Info Card */}
                            <div className="card bg-[rgb(var(--color-bg-secondary))]">
                                <h3 className="font-semibold mb-3 text-[rgb(var(--color-text))]">
                                    Detected Emotions
                                </h3>
                                <div className="grid grid-cols-3 gap-2 text-center">
                                    {availableLabels.map((emotion) => (
                                        <div key={emotion} className="p-2 rounded-lg bg-[rgb(var(--color-bg))] text-sm capitalize">
                                            {emotion}
                                        </div>
                                    ))}
                                </div>

                                <div className="mt-4">
                                    <h4 className="font-semibold mb-2 text-sm text-[rgb(var(--color-text))]">
                                        Features
                                    </h4>
                                    <ul className="space-y-2 text-sm text-[rgb(var(--color-text-secondary))]">
                                        <li className="flex items-start gap-2">
                                            <span className="text-[rgb(var(--color-accent))]">•</span>
                                            <span>Multi-emotion probability distribution</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <span className="text-[rgb(var(--color-accent))]">•</span>
                                            <span>Attention heatmap visualization</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <span className="text-[rgb(var(--color-accent))]">•</span>
                                            <span>Real-time analysis</span>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>

                        {/* Results Section */}
                        <div>
                            {loading && (
                                <div className="card">
                                    <LoadingSpinner size="lg" text="Analyzing emotions..." />
                                </div>
                            )}

                            {!loading && result && (
                                <ResultsDisplay result={result} type="emotion" />
                            )}

                            {!loading && !result && !selectedFile && (
                                <div className="card text-center py-12">
                                    <Upload className="w-16 h-16 mx-auto mb-4 text-[rgb(var(--color-text-secondary))]" />
                                    <p className="text-[rgb(var(--color-text-secondary))]">
                                        Upload a facial image to detect emotions
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
