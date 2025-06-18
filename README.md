# Skin Disease Classification Application

## Project Overview
This application provides a web-based interface for classifying skin diseases using machine learning models. It combines a React frontend with Python backend scripts for image analysis.

## 🔍 Features

- **Intuitive File Upload Interface**  
  Upload skin disease images easily through a clean and user-friendly interface.

- **Ensemble-Based Disease Classification**  
  Leverages a powerful ensemble of deep learning models for accurate and reliable classification of skin conditions.

- **Symptom-Aware Prediction Refinement**  
  Allows users to select relevant symptoms, enhancing model performance and delivering more precise results.


## Project Structure
```
skin-disease-classification-app/
├── .gitignore          # Repository-wide ignore rules
├── LICENCE             # MIT License
├── models/             # Machine learning model implementations
│   ├── disease_mappings.py
│   ├── ensemble.py
│   ├── image_classifier.py
│   ├── predict.py
│   └── text_classifier.py
└── src/
    └── python_scripts/
        └── app.py      # Main Python application
    └── frontend/       # React-based frontend application
```

## Setup Instructions
1. Clone the repository
2. Set up Python environment:
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
3. Install frontend dependencies:
   ```bash
   cd src/frontend
   npm install
   ```

## Usage
1. Start the Python backend:
   ```bash
   python src/python_scripts/app.py
   ```
2. In a separate terminal, start the React frontend:
   ```bash
   cd src/frontend
   npm start
   ```

## Dependencies
- Python: Flask, PyTorch
- JavaScript: React, TypeScript, Axios

## Contribution Guidelines
1. Fork the repository
2. Create a new branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details
