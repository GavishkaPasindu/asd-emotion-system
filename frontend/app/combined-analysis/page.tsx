'use client';

import { useState } from 'react';
import { Layers, Upload, Sparkles, Brain, Smile } from 'lucide-react';
import FileUpload from '../components/FileUpload';
import LoadingSpinner from '../components/LoadingSpinner';
import ResultsDisplay from '../components/ResultsDisplay';
import { api, CombinedPredictionResponse } from '@/lib/api';

export default function CombinedAnalysisPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<CombinedPredictionResponse | null>(null);

    const handleFileSelect = (file: File) => {
        setSelectedFile(file);
        setResult(null);
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        setLoading(true);
        try {
            const prediction = await api.predictCombined(selectedFile, true);
            setResult(prediction);
        } catch (error) {
            console.error('Analysis error:', error);
            setResult({
                success: false,
                asd: { success: false, confidence: 0, error: 'Analysis failed' },
                emotion: { success: false, confidence: 0, error: 'Analysis failed' },
                original_image: '',
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
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-[rgb(var(--color-primary)/0.1)] to-[rgb(var(--color-accent)/0.1)] mb-4">
                        <Layers className="w-5 h-5 text-[rgb(var(--color-primary))]" />
                        <span className="text-sm font-medium text-gradient">
                            Comprehensive Analysis
                        </span>
                    </div>

                    <h1 className="text-4xl md:text-5xl font-bold mb-4 text-[rgb(var(--color-text))]">
                        Comprehensive Analysis
                    </h1>

                    <p className="text-lg text-[rgb(var(--color-text-secondary))] max-w-2xl mx-auto">
                        Get both behavioral screening and emotion recognition in a single analysis
                    </p>
                </div>

                <div className="max-w-6xl mx-auto">
                    {/* Upload Section */}
                    <div className="card mb-8 max-w-2xl mx-auto">
                        <h2 className="text-xl font-semibold mb-4 text-[rgb(var(--color-text))]">
                            Upload Image for Analysis
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
                                Run Full Analysis
                            </button>
                        )}
                    </div>

                    {/* Loading State */}
                    {loading && (
                        <div className="card max-w-2xl mx-auto">
                            <LoadingSpinner size="lg" text="Running comprehensive analysis..." />
                        </div>
                    )}

                    {/* Results */}
                    {!loading && result && result.success && (
                        <div className="space-y-8">
                            {/* Original Image */}
                            {result.original_image && (
                                <div className="card max-w-md mx-auto">
                                    <h3 className="text-lg font-semibold mb-3 text-[rgb(var(--color-text))]">
                                        Original Image
                                    </h3>
                                    <img
                                        src={`data:image/png;base64,${result.original_image}`}
                                        alt="Original"
                                        className="w-full rounded-lg"
                                    />
                                </div>
                            )}

                            {/* Side by Side Results */}
                            <div className="grid lg:grid-cols-2 gap-8">
                                {/* ASD Results */}
                                <div>
                                    <div className="flex items-center gap-2 mb-4">
                                        <Brain className="w-6 h-6 text-[rgb(var(--color-primary))]" />
                                        <h3 className="text-2xl font-bold text-[rgb(var(--color-text))]">
                                            Behavioral Screening
                                        </h3>
                                    </div>
                                    <ResultsDisplay result={result.asd} type="asd" />
                                </div>

                                {/* Emotion Results */}
                                <div>
                                    <div className="flex items-center gap-2 mb-4">
                                        <Smile className="w-6 h-6 text-[rgb(var(--color-accent))]" />
                                        <h3 className="text-2xl font-bold text-[rgb(var(--color-text))]">
                                            Emotion Recognition
                                        </h3>
                                    </div>
                                    <ResultsDisplay result={result.emotion} type="emotion" />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Empty State */}
                    {!loading && !result && !selectedFile && (
                        <div className="card text-center py-16 max-w-2xl mx-auto">
                            <div className="flex justify-center gap-4 mb-6">
                                <div className="p-4 rounded-full bg-[rgb(var(--color-primary)/0.1)]">
                                    <Brain className="w-8 h-8 text-[rgb(var(--color-primary))]" />
                                </div>
                                <div className="p-4 rounded-full bg-[rgb(var(--color-accent)/0.1)]">
                                    <Smile className="w-8 h-8 text-[rgb(var(--color-accent))]" />
                                </div>
                            </div>
                            <h3 className="text-xl font-semibold mb-2 text-[rgb(var(--color-text))]">
                                Ready for Comprehensive Analysis
                            </h3>
                            <p className="text-[rgb(var(--color-text-secondary))]">
                                Upload an image to get both behavioral screening and emotion recognition results
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
}
