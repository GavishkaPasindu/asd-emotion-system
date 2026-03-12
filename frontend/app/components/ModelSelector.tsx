'use client';

import React from 'react';
import { useModel } from '../context/ModelContext';

export default function ModelSelector() {
    const { selectedModel, setSelectedModel, availableModels } = useModel();

    return (
        <div className="flex items-center space-x-2">
            <label htmlFor="model-select" className="text-sm font-medium text-gray-700 hidden md:block">
                Active Model:
            </label>
            <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as any)}
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md shadow-sm bg-blue-50 text-blue-900 border"
            >
                {availableModels.map((model) => (
                    <option key={model.id} value={model.id}>
                        {model.name}
                    </option>
                ))}
            </select>
        </div>
    );
}
