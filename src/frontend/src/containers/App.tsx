import React, { useState } from 'react';
import { Container, Table } from 'reactstrap';
import NavigationBar from '../components/NavigationBar';
import { FileUploadCard } from '../components/FileUploadCard';
import { predictImage, runEnsembleModel, Prediction } from '../api/model_api/modelApiCalls';
import 'bootstrap/dist/css/bootstrap.min.css';

const App: React.FC = () => {
  const [availableSymptoms, setAvailableSymptoms] = useState<string[]>([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [imagePredictions, setImagePredictions] = useState<Prediction[]>([]);
  const [imageWeight, setImageWeight] = useState<number>(0.5);
  const [currentImageData, setCurrentImageData] = useState<string | null>(null);

  const handleImageUpload = async (base64Image: string) => {
    try {
      const response = await predictImage(base64Image);
      setImagePredictions(response.predictions);
      setAvailableSymptoms(response.topSymptoms);
      setCurrentImageData(base64Image); // Save the image data for ensemble
      // Reset selected symptoms when new image is uploaded
      setSelectedSymptoms([]);
    } catch (error) {
      console.error('Error predicting image:', error);
    }
  };

  const toggleSymptom = (symptom: string) => {
    setSelectedSymptoms(prev =>
      prev.includes(symptom)
        ? prev.filter(s => s !== symptom)
        : [...prev, symptom]
    );
  };

  const handleRunEnsemble = async () => {
    try {
      const response = await runEnsembleModel(
        selectedSymptoms, 
        imageWeight, 
        currentImageData || undefined
      );
      setPredictions(response.predictions);
    } catch (error) {
      console.error('Error running ensemble model:', error);
    }
  };

  return (
    <div>
      <NavigationBar />
      <Container className="mt-4">
        <h1 className="mb-4">Skin Disease Classification</h1>
        
        <div className="row">
          <div className="col-md-8 mx-auto mb-4">
            <FileUploadCard onImageUpload={handleImageUpload} />
          </div>
        </div>

        {/* Image Analysis Results */}
        {imagePredictions.length > 0 && (
          <div className="mt-4">
            <h3>Image Analysis Results</h3>
            <Table striped bordered>
              <thead>
                <tr>
                  <th>Disease</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {imagePredictions.slice(0, 5).map((pred, index) => (
                  <tr key={index}>
                    <td>{pred.disease}</td>
                    <td>{pred.confidence.toFixed(1)}%</td>  {/* Match backend format of 1 decimal place */}
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
        )}

        {/* Symptom Selection */}
        {availableSymptoms.length > 0 && (
          <div className="mt-4">
            <h3>Select Symptoms</h3>
            <div className="d-flex flex-wrap gap-2 mb-4">
              {[...availableSymptoms].sort().map((symptom) => (
                <button
                  key={symptom}
                  className={`btn ${selectedSymptoms.includes(symptom) ? 'btn-primary' : 'btn-outline-primary'}`}
                  onClick={() => toggleSymptom(symptom)}
                >
                  {symptom}
                </button>
              ))}
            </div>

            {/* Weight Slider */}
            <div className="mb-3">
              <label className="form-label d-block">Analysis Weights</label>
              <input
                type="range"
                className="form-range"
                min="0"
                max="1"
                step="0.1"
                value={imageWeight}
                onChange={(e) => setImageWeight(parseFloat(e.target.value))}
              />
              <div className="d-flex justify-content-between small text-muted">
                <span>Text: {(100 * (1 - imageWeight)).toFixed(0)}%</span>
                <span>Image: {(100 * imageWeight).toFixed(0)}%</span>
              </div>
            </div>

            <button 
              className="btn btn-primary"
              onClick={handleRunEnsemble}
              disabled={selectedSymptoms.length === 0}
            >
              Analyze with Selected Symptoms
            </button>
          </div>
        )}

        {/* Final Analysis Results */}
        {predictions.length > 0 && (
          <div className="mt-4">
            <h3>Combined Analysis Results</h3>
            <Table striped bordered>
              <thead>                  <tr>
                    <th>Disease</th>
                    <th>Confidence</th>
                    {predictions[0].originalConfidence !== undefined && predictions[0].change !== undefined && (
                      <th>Change</th>
                    )}
                  </tr>
              </thead>
              <tbody>
                {predictions.map((pred, index) => (
                  <tr key={index}>
                    <td>{pred.disease}</td>
                    <td>{pred.confidence.toFixed(1)}%</td>
                    {pred.originalConfidence !== undefined && pred.change !== undefined && (
                      <td>
                        <span className={pred.change >= 0 ? 'text-success' : 'text-danger'}>
                          {pred.change >= 0 ? '↑' : '↓'} {Math.abs(pred.change).toFixed(1)}%
                        </span>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
        )}
      </Container>
    </div>
  );
};

export default App;