'use client';

import { useState } from 'react';
import { Brain, Upload, Sparkles } from 'lucide-react';
import FileUpload from '../components/FileUpload';
import LoadingSpinner from '../components/LoadingSpinner';
import ResultsDisplay from '../components/ResultsDisplay';
import { api, PredictionResponse } from '@/lib/api';

export default function ASDDetectionPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<PredictionResponse | null>(null);

    const handleFileSelect = (file: File) => {
        setSelectedFile(file);
        setResult(null);
    };
    const handleAnalyze = async () => {
        if (!selectedFile) return;

        setLoading(true);
        try {
            const prediction = await api.predictASD(selectedFile, true);
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
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[rgb(var(--color-primary)/0.1)] mb-4">
                        <Brain className="w-5 h-5 text-[rgb(var(--color-primary))]" />
                        <span className="text-sm font-medium text-[rgb(var(--color-primary))]">
                            Developmental Screening
                        </span>
                    </div>

                    <h1 className="text-4xl md:text-5xl font-bold mb-4 text-[rgb(var(--color-text))]">
                        Child Development Analysis
                    </h1>

                    <p className="text-lg text-[rgb(var(--color-text-secondary))] max-w-2xl mx-auto">
                        Upload an image for AI-powered developmental screening with explainable AI visualizations
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
                                        Run Screening
                                    </button>
                                )}
                            </div>

                            {/* Info Card */}
                            <div className="card bg-[rgb(var(--color-bg-secondary))]">
                                <h3 className="font-semibold mb-3 text-[rgb(var(--color-text))]">
                                    How it works
                                </h3>
                                <ul className="space-y-2 text-sm text-[rgb(var(--color-text-secondary))]">
                                    <li className="flex items-start gap-2">
                                        <span className="text-[rgb(var(--color-primary))]">•</span>
                                        <span>Upload a facial image for analysis</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-[rgb(var(--color-primary))]">•</span>
                                        <span>AI model analyzes visual patterns</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-[rgb(var(--color-primary))]">•</span>
                                        <span>Grad-CAM shows model attention areas</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-[rgb(var(--color-primary))]">•</span>
                                        <span>Results include confidence scores</span>
                                    </li>
                                </ul>

                                <div className="mt-4 p-3 rounded-lg bg-[rgb(var(--color-warning)/0.1)] border border-[rgb(var(--color-warning)/0.3)]">
                                    <p className="text-xs text-[rgb(var(--color-warning))]">
                                        <strong>Note:</strong> This is a screening tool only. Professional diagnosis is required.
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Results Section */}
                        <div>
                            {loading && (
                                <div className="card">
                                    <LoadingSpinner size="lg" text="Analyzing image..." />
                                </div>
                            )}

                            {!loading && result && (
                                <ResultsDisplay result={result} type="asd" />
                            )}

                            {!loading && !result && !selectedFile && (
                                <div className="card text-center py-12">
                                    <Upload className="w-16 h-16 mx-auto mb-4 text-[rgb(var(--color-text-secondary))]" />
                                    <p className="text-[rgb(var(--color-text-secondary))]">
                                        Upload an image to get started
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
