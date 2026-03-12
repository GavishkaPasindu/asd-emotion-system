// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

// API Response Types
export interface PredictionResponse {
  success: boolean;
  predicted_class?: string;
  predicted_emotion?: string;
  confidence: number;
  probabilities?: Record<string, number>;
  top_emotions?: Array<{ emotion: string; probability: number }>;
  xai?: {
    heatmap: string;
    overlay: string;
    face_regions?: {
      region_scores: Record<string, number>;
      face_overlay: string | null;
    };
    explanation: string;
    method?: string;
  };
  error?: string;
}

export interface ModelMetricsResponse {
  success: boolean;
  model_type: string;
  metrics: {
    asd: {
      title: string;
      confusion_matrix: number[][];
      classification_report: Record<string, any>;
      labels: string[];
    } | null;
    emotion: {
      title: string;
      confusion_matrix: number[][];
      classification_report: Record<string, any>;
      labels: string[];
    } | null;
  };
}

export interface CombinedPredictionResponse {
  success: boolean;
  asd: PredictionResponse;
  emotion: PredictionResponse;
  original_image: string;
  error?: string;
}

export interface AnalyticsSummary {
  total_predictions: number;
  asd_detections: number;
  emotion_distribution: Record<string, number>;
  average_confidence: number;
}

export interface EmotionTimeline {
  timestamps: string[];
  emotions: string[];
  confidences: number[];
}

export interface CalmingVideo {
  id: string;
  title: string;
  url: string;
  thumbnail: string;
  duration: number;
  emotion_trigger: string;
}

// API Client Functions
export const api = {
  /**
   * Predict ASD from an image
   */
  async predictASD(image: File, useEnsemble: boolean = true, modelType?: string): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('ensemble', useEnsemble.toString());

    const headers: Record<string, string> = {};
    if (modelType) headers['X-Model-Type'] = modelType;

    const response = await fetch(`${API_BASE_URL}/api/predict/asd`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  },

  /**
   * Predict emotion from an image
   */
  async predictEmotion(image: File, useEnsemble: boolean = true, modelType?: string): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('ensemble', useEnsemble.toString());

    const headers: Record<string, string> = {};
    if (modelType) headers['X-Model-Type'] = modelType;

    const response = await fetch(`${API_BASE_URL}/api/predict/emotion`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  },

  /**
   * Get combined ASD and emotion prediction
   */
  async predictCombined(image: File, useEnsemble: boolean = true, modelType?: string): Promise<CombinedPredictionResponse> {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('ensemble', useEnsemble.toString());

    const headers: Record<string, string> = {};
    if (modelType) headers['X-Model-Type'] = modelType;

    const response = await fetch(`${API_BASE_URL}/api/predict/combined`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  },

  /**
   * Get analytics summary
   */
  async getAnalyticsSummary(modelType?: string): Promise<AnalyticsSummary> {
    const url = new URL(`${API_BASE_URL}/api/analytics/summary`);
    if (modelType) url.searchParams.append('model_type', modelType);

    const response = await fetch(url.toString());

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  },

  /**
   * Get emotion timeline
   */
  async getEmotionTimeline(): Promise<EmotionTimeline> {
    const response = await fetch(`${API_BASE_URL}/api/analytics/emotion-timeline`);

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  },

  /**
   * Get calming videos
   */
  async getCalmingVideos(): Promise<CalmingVideo[]> {
    const response = await fetch(`${API_BASE_URL}/api/gamification/videos`);

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.videos || [];
  },

  /**
   * Get video suggestion based on emotion
   */
  async getVideoSuggestion(emotion: string): Promise<CalmingVideo> {
    const response = await fetch(`${API_BASE_URL}/api/gamification/suggest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ emotion }),
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.video;
  },

  /**
   * Get detailed info about loaded models
   */
  async getModelsInfo(): Promise<{ success: boolean; models: Record<string, any>; default: string }> {
    const response = await fetch(`${API_BASE_URL}/api/models/info`);
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Check health status
   */
  async healthCheck(): Promise<{ status: string; models: { asd_detector: boolean; emotion_detector: boolean } }> {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Get performance metrics for the active model
   */
  async getModelMetrics(modelType?: string): Promise<ModelMetricsResponse> {
    const url = new URL(`${API_BASE_URL}/api/models/metrics`);
    if (modelType) url.searchParams.append('model_type', modelType);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    return response.json();
  },
};

export default api;
