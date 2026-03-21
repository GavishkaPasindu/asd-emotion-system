'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type ModelType = 'resnet50v2';

interface ModelContextType {
    selectedModel: ModelType;
    setSelectedModel: (model: ModelType) => void;
    availableModels: { id: ModelType; name: string }[];
}

const availableModels: { id: ModelType; name: string }[] = [
    { id: 'resnet50v2', name: 'ResNet50V2 (Best Accuracy)' },
];


const ModelContext = createContext<ModelContextType | undefined>(undefined);

export const ModelProvider = ({ children }: { children: ReactNode }) => {
    const [selectedModel, setSelectedModelState] = useState<ModelType>('resnet50v2');

    // Load from localStorage on mount
    useEffect(() => {
        const saved = localStorage.getItem('selectedModel') as ModelType;
        if (saved && availableModels.some(m => m.id === saved)) {
            setSelectedModelState(saved);
        }
    }, []);

    const setSelectedModel = (model: ModelType) => {
        setSelectedModelState(model);
        localStorage.setItem('selectedModel', model);
    };

    return (
        <ModelContext.Provider value={{ selectedModel, setSelectedModel, availableModels }}>
            {children}
        </ModelContext.Provider>
    );
};

export const useModel = () => {
    const context = useContext(ModelContext);
    if (!context) {
        throw new Error('useModel must be used within a ModelProvider');
    }
    return context;
};
