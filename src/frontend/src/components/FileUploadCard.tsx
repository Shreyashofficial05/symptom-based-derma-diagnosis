import React, { useState } from 'react';
import {
    Button,
    Card,
    CardText,
    CardTitle,
    FormGroup,
    Input,
    InputGroup,
    Table,
    Spinner,
    Alert
} from "reactstrap";
import { predictImage, Prediction } from '../api/model_api/modelApiCalls';

interface FileUploadCardProps {
    onImageUpload: (base64Image: string) => void;
}

export const FileUploadCard: React.FC<FileUploadCardProps> = ({ onImageUpload }) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (!file.type.match('image.*')) {
                setError('Please select an image file');
                return;
            }
            setSelectedFile(file);
            setError(null);
            setPredictions([]);

            // Create preview URL
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewUrl(reader.result as string);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a file first');
            return;
        }

        try {
            setIsLoading(true);
            setError(null);

            // Convert file to base64
            const reader = new FileReader();
            reader.readAsDataURL(selectedFile);
            reader.onloadend = async () => {
                const base64String = reader.result as string;
                try {
                    const response = await predictImage(base64String);
                    setPredictions(response.predictions);
                    onImageUpload(base64String);
                } catch (err) {
                    setError('Error analyzing image. Please try again.');
                    console.error('Error:', err);
                } finally {
                    setIsLoading(false);
                }
            };
        } catch (err) {
            setError('Error uploading file. Please try again.');
            setIsLoading(false);
            console.error('Error:', err);
        }
    };

    const renderPredictionTable = () => {
        if (!predictions.length) return null;

        return (
            <div className="mt-4">
                <h5>Analysis Results</h5>
                <Table hover bordered responsive>
                    <thead>
                        <tr>
                            <th>Disease</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {predictions.map((pred, index) => (
                            <tr key={index}>
                                <td>{pred.disease}</td>
                                <td>
                                    <div className="d-flex align-items-center">
                                        <div 
                                            className="progress flex-grow-1" 
                                            style={{ height: '20px' }}
                                        >
                                            <div
                                                className="progress-bar"
                                                role="progressbar"
                                                style={{ width: `${pred.confidence}%` }}
                                                aria-valuenow={pred.confidence}
                                                aria-valuemin={0}
                                                aria-valuemax={100}
                                            />
                                        </div>
                                        <span className="ms-2">{pred.confidence.toFixed(1)}%</span>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </Table>
            </div>
        );
    };

    return (
        <Card body>
            <CardTitle tag="h5">Upload a photo</CardTitle>
            <CardText>
                <small>The photo should be cropped to the place of your potential skin disease</small>
            </CardText>
            
            {error && (
                <Alert color="danger" className="mb-3">
                    {error}
                </Alert>
            )}

            <FormGroup>
                <div className="mb-3">
                    <InputGroup>
                        <Input
                            onChange={handleFileChange}
                            id="uploadedFile"
                            name="file"
                            type="file"
                            accept="image/png,image/jpeg,image/jpg"
                        />
                        <Button 
                            color="primary"
                            onClick={handleUpload}
                            disabled={!selectedFile || isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <Spinner size="sm" className="me-2" />
                                    Analyzing...
                                </>
                            ) : (
                                'Analyze Image'
                            )}
                        </Button>
                    </InputGroup>
                    <small className="text-muted d-block mt-1">
                        Supported file extensions are: png, jpeg, jpg.
                    </small>
                </div>

                {/* Image Preview */}
                {previewUrl && (
                    <div className="mt-3 mb-3 text-center">
                        <img 
                            src={previewUrl} 
                            alt="Preview" 
                            style={{ 
                                maxWidth: '100%', 
                                maxHeight: '300px', 
                                objectFit: 'contain',
                                borderRadius: '8px',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                            }} 
                        />
                    </div>
                )}

                {/* Predictions Table */}
                {renderPredictionTable()}
            </FormGroup>
        </Card>
    );
};