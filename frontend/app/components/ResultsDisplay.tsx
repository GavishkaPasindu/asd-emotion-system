'use client';

import { CheckCircle, AlertCircle, Brain, Smile, Activity, Info } from 'lucide-react';
import { PredictionResponse, api, ModelMetricsResponse } from '@/lib/api';
import { useState, useEffect } from 'react';
import { useModel } from '../context/ModelContext';

interface ResultsDisplayProps {
    result: PredictionResponse;
    type: 'asd' | 'emotion';
}

export default function ResultsDisplay({ result, type }: ResultsDisplayProps) {
    const { selectedModel } = useModel();
    const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
    const [showPerformance, setShowPerformance] = useState(false);

    useEffect(() => {
        // Fetch metrics for the selected model whenever it changes or a new result arrives
        api.getModelMetrics(selectedModel).then(setMetrics).catch(console.error);
    }, [selectedModel, result.success]);

    if (!result.success) {
        return (
            <div className="card bg-[rgb(var(--color-danger)/0.05)] border-[rgb(var(--color-danger)/0.3)] animate-slide-up">
                <div className="flex items-start gap-3">
                    <AlertCircle className="w-6 h-6 text-[rgb(var(--color-danger))] flex-shrink-0 mt-1" />
                    <div>
                        <h3 className="font-semibold text-[rgb(var(--color-danger))] mb-1">
                            Analysis Failed
                        </h3>
                        <p className="text-sm text-[rgb(var(--color-text-secondary))]">
                            {result.error || 'An error occurred during analysis'}
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    const confidence = Math.round(result.confidence * 100);
    const prediction = type === 'asd' ? result.predicted_class : result.predicted_emotion;

    return (
        <div className="space-y-6 animate-slide-up">
            {/* Main Result Card */}
            <div className="card-glass">
                <div className="flex items-start gap-4">
                    <div className={`
            p-3 rounded-full
            ${type === 'asd'
                            ? 'bg-[rgb(var(--color-primary)/0.1)]'
                            : 'bg-[rgb(var(--color-accent)/0.1)]'
                        }
          `}>
                        {type === 'asd' ? (
                            <Brain className={`w-8 h-8 text-[rgb(var(--color-primary))]`} />
                        ) : (
                            <Smile className={`w-8 h-8 text-[rgb(var(--color-accent))]`} />
                        )}
                    </div>

                    <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                            <h3 className="text-2xl font-bold text-[rgb(var(--color-text))]">
                                {prediction}
                            </h3>
                            <CheckCircle className="w-5 h-5 text-[rgb(var(--color-success))]" />
                        </div>

                        <div className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                                <span className="text-[rgb(var(--color-text-secondary))]">Confidence</span>
                                <span className="font-semibold text-[rgb(var(--color-text))]">{confidence}%</span>
                            </div>

                            <div className="progress-bar">
                                <div
                                    className="progress-fill"
                                    style={{ width: `${confidence}%` }}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Emotion Probabilities */}
            {type === 'emotion' && result.top_emotions && result.top_emotions.length > 0 && (
                <div className="card">
                    <h4 className="font-semibold text-[rgb(var(--color-text))] mb-4">
                        Top Emotions Detected
                    </h4>

                    <div className="space-y-3">
                        {result.top_emotions.slice(0, 5).map((item, index) => {
                            const prob = Math.round(item.probability * 100);
                            return (
                                <div key={index} className="space-y-1">
                                    <div className="flex items-center justify-between text-sm">
                                        <span className="capitalize text-[rgb(var(--color-text))]">
                                            {item.emotion}
                                        </span>
                                        <span className="font-medium text-[rgb(var(--color-text-secondary))]">
                                            {prob}%
                                        </span>
                                    </div>
                                    <div className="progress-bar h-1.5">
                                        <div
                                            className="h-full bg-[rgb(var(--color-accent))] rounded-full transition-all duration-500"
                                            style={{ width: `${prob}%` }}
                                        />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* XAI Visualizations */}
            {result.xai && (
                <div className="card">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold text-[rgb(var(--color-text))]">
                            Multi-XAI Explainability
                        </h4>
                        <div className="flex gap-2">
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-[rgb(var(--color-primary)/0.1)] text-[rgb(var(--color-primary))] font-medium uppercase">
                                Grad-CAM
                            </span>
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-[rgb(var(--color-success)/0.1)] text-[rgb(var(--color-success))] font-medium uppercase">
                                MediaPipe Regions
                            </span>
                        </div>
                    </div>

                    {/* 3-image grid */}
                    <div className="grid md:grid-cols-3 gap-4 mb-4">
                        {/* Col 1: Grad-CAM overlay */}
                        <div>
                            <p className="text-[10px] text-[rgb(var(--color-text-secondary))] mb-2 uppercase tracking-wide">
                                Grad-CAM (Heatmap)
                            </p>
                            <div className="relative rounded-[var(--radius-lg)] overflow-hidden border border-[rgb(var(--color-border))] aspect-square">
                                <img
                                    src={result.xai.overlay}
                                    alt="Grad-CAM Overlay"
                                    className="w-full h-full object-cover"
                                />
                                <div className="absolute inset-x-0 bottom-0 bg-black/40 backdrop-blur-sm py-1 text-[10px] text-white text-center">
                                    Feature Attention
                                </div>
                            </div>
                        </div>

                        {/* Col 2: MediaPipe face-region overlay */}
                        <div>
                            <p className="text-[10px] text-[rgb(var(--color-text-secondary))] mb-2 uppercase tracking-wide">
                                Face Region Attention
                            </p>
                            <div className="relative rounded-[var(--radius-lg)] overflow-hidden border border-[rgb(var(--color-border))] aspect-square bg-black/5">
                                {result.xai.face_regions?.face_overlay ? (
                                    <img
                                        src={result.xai.face_regions.face_overlay}
                                        alt="MediaPipe Face Region Attention"
                                        className="w-full h-full object-cover"
                                    />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-xs text-[rgb(var(--color-text-secondary))] px-4 text-center">
                                        Face region overlay not available
                                    </div>
                                )}
                                <div className="absolute inset-x-0 bottom-0 bg-black/40 backdrop-blur-sm py-1 text-[10px] text-white text-center">
                                    MediaPipe Landmark Regions
                                </div>
                            </div>
                        </div>

                        {/* Col 3: Raw heatmap */}
                        <div>
                            <p className="text-[10px] text-[rgb(var(--color-text-secondary))] mb-2 uppercase tracking-wide">
                                Heatmap Detail
                            </p>
                            <div className="rounded-[var(--radius-lg)] overflow-hidden border border-[rgb(var(--color-border))] aspect-square">
                                <img
                                    src={result.xai.heatmap}
                                    alt="Raw Heatmap"
                                    className="w-full h-full object-cover"
                                />
                            </div>
                        </div>
                    </div>

                    {/* XAI explanation */}
                    <div className="p-4 rounded-[var(--radius-md)] bg-[rgb(var(--color-bg-secondary))] border-l-4 border-[rgb(var(--color-primary))]">
                        <div className="flex gap-2 items-start">
                            <Info className="w-4 h-4 text-[rgb(var(--color-primary))] mt-0.5" />
                            <p className="text-sm text-[rgb(var(--color-text))] whitespace-pre-line leading-relaxed">
                                {result.xai.explanation}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Performance Tab Toggle */}
            <div className="flex justify-center">
                <button
                    onClick={() => setShowPerformance(!showPerformance)}
                    className="flex items-center gap-2 text-sm font-medium py-2 px-4 rounded-full border border-[rgb(var(--color-border))] hover:bg-[rgb(var(--color-bg-secondary))] transition-colors"
                >
                    <Activity className="w-4 h-4" />
                    {showPerformance ? 'Hide Model Performance' : 'Show Performance Matrix'}
                </button>
            </div>

            {/* Performance Matrix section */}
            {showPerformance && metrics && metrics.metrics[type] && (
                <div className="card border-t-4 border-[rgb(var(--color-accent))] animate-slide-up">
                    <h4 className="font-semibold text-[rgb(var(--color-text))] mb-2 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-[rgb(var(--color-accent))]" />
                        {metrics.metrics[type]?.title} -- Performance Matrix
                    </h4>
                    <p className="text-xs text-[rgb(var(--color-text-secondary))] mb-6">
                        Verified results from the training phase for model: <span className="font-mono text-[rgb(var(--color-primary))]">{metrics.model_type}</span>
                    </p>

                    <div className="grid md:grid-cols-2 gap-8">
                        {/* Confusion Matrix Visual */}
                        <div className="space-y-4">
                            <p className="text-xs font-medium uppercase text-[rgb(var(--color-text-secondary))]">Confusion Matrix (Heatmap)</p>
                            <div className="relative aspect-square bg-[rgb(var(--color-bg-secondary))] rounded-lg border border-[rgb(var(--color-border))] p-4 flex flex-col">
                                {metrics.metrics[type]?.confusion_matrix ? (
                                    <>
                                        <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-1">
                                            {(metrics.metrics[type]!.confusion_matrix as number[][]).flat().map((val, idx) => {
                                                const flatMatrix = (metrics.metrics[type]!.confusion_matrix as number[][]).flat();
                                                const maxVal = Math.max(...flatMatrix, 1);
                                                const opacity = 0.1 + (val / maxVal) * 0.9;
                                                return (
                                                    <div
                                                        key={idx}
                                                        className="flex flex-col items-center justify-center rounded transition-all hover:scale-[1.02]"
                                                        style={{ backgroundColor: `rgb(var(--color-primary) / ${opacity})` }}
                                                    >
                                                        <span className={`text-lg font-bold ${opacity > 0.5 ? 'text-white' : 'text-[rgb(var(--color-text))]'}`}>
                                                            {val}
                                                        </span>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                        <div className="flex justify-between mt-2 px-2">
                                            {metrics.metrics[type]?.labels.map(l => (
                                                <span key={l} className="text-[10px] font-medium uppercase text-[rgb(var(--color-text-secondary))]">{l}</span>
                                            ))}
                                        </div>
                                    </>
                                ) : (
                                    <div className="flex-1 flex flex-col items-center justify-center text-center gap-2">
                                        <Activity className="w-8 h-8 text-[rgb(var(--color-text-secondary))] opacity-30" />
                                        <p className="text-xs text-[rgb(var(--color-text-secondary))]">
                                            No confusion matrix available.<br />
                                            Run evaluation in Colab to generate metrics.
                                        </p>
                                        {metrics.metrics[type]?.labels && (
                                            <div className="flex gap-2 mt-1">
                                                {metrics.metrics[type]!.labels.map(l => (
                                                    <span key={l} className="text-[10px] px-2 py-0.5 rounded-full bg-[rgb(var(--color-bg))] border border-[rgb(var(--color-border))] text-[rgb(var(--color-text-secondary))]">{l}</span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Classification Metrics */}
                        <div className="space-y-4">
                            <p className="text-xs font-medium uppercase text-[rgb(var(--color-text-secondary))]">Accuracy Metrics</p>
                            <div className="space-y-4">
                                {Object.entries(metrics.metrics[type]?.classification_report || {}).map(([key, value]) => {
                                    if (typeof value !== 'object' || !('f1-score' in value)) return null;
                                    const accuracy = Math.round(value['f1-score'] * 100);
                                    return (
                                        <div key={key} className="space-y-1">
                                            <div className="flex justify-between items-center text-sm">
                                                <span className="capitalize font-medium text-[rgb(var(--color-text))]">{key}</span>
                                                <span className="text-[rgb(var(--color-text-secondary))]">F1: {accuracy}%</span>
                                            </div>
                                            <div className="progress-bar h-2">
                                                <div
                                                    className="progress-fill"
                                                    style={{
                                                        width: `${accuracy}%`,
                                                        backgroundColor: `rgb(var(${key === 'macro avg' ? '--color-primary' : '--color-accent'}))`
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                                <div className="mt-6 pt-4 border-t border-[rgb(var(--color-border))]">
                                    <div className="flex justify-between items-center bg-[rgb(var(--color-primary)/0.05)] p-3 rounded-lg">
                                        <span className="text-sm font-bold text-[rgb(var(--color-text))]">OVERALL ACCURACY</span>
                                        <span className="text-2xl font-black text-[rgb(var(--color-primary))]">
                                            {Math.round((metrics.metrics[type]?.classification_report as any)?.accuracy * 100)}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
