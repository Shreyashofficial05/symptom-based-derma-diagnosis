import axios from 'axios';

const API_URL = 'http://localhost:5002';

export interface Prediction {
    disease: string;
    confidence: number;
    symptoms?: string[];
    originalConfidence?: number;
    change?: number;
}

export interface PredictResponse {
    predictions: Prediction[];
    topSymptoms: string[];
}

export interface EnsembleResponse {
    predictions: Prediction[];
}

export const predictImage = async (imageBase64: string): Promise<PredictResponse> => {
    try {
        const response = await axios.post(`${API_URL}/predict`, {
            image: imageBase64
        });
        return response.data;
    } catch (error) {
        console.error('Error predicting image:', error);
        throw error;
    }
};

export const fetchUniqueSymptoms = async (): Promise<string[]> => {
    try {
        const response = await axios.get(`${API_URL}/unique-symptoms`);
        return response.data.unique_symptoms;
    } catch (error) {
        console.error('Error fetching unique symptoms:', error);
        throw error;
    }
};

export const runEnsembleModel = async (
    selectedSymptoms: string[], 
    weight_image: number = 0.5,
    imageBase64?: string
): Promise<EnsembleResponse> => {
    try {
        const response = await axios.post(`${API_URL}/run-ensemble`, {
            selected_symptoms: selectedSymptoms,
            weight_image: weight_image,
            image: imageBase64
        });
        return response.data;
    } catch (error) {
        console.error('Error running ensemble model:', error);
        throw error;
    }
};
