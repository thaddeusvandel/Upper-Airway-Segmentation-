# AI-Powered Prediction of Microvascular Complications in Diabetes Mellitus

A Django-based web application utilizing machine learning to predict the risk of diabetic microvascular complications including nephropathy, neuropathy, and retinopathy.

## Overview

This research project explores the application of artificial intelligence in predicting, diagnosing, and monitoring microvascular complications associated with diabetes mellitus. The system employs CatBoost machine learning models trained on clinical data to provide real-time risk assessments.

## Features

- **User Authentication System**: Secure login and registration for healthcare professionals
- **Clinical Data Input**: Comprehensive form capturing 15 clinical parameters
- **AI Risk Assessment**: Three separate CatBoost models for predicting:
  - Nephropathy (kidney damage)
  - Neuropathy (nerve damage)
  - Retinopathy (eye damage)
- **Risk Visualization**: Color-coded progress bars and risk stratification
- **Responsive Design**: Glassmorphism UI accessible across devices
- **Clinical Interpretation**: Evidence-based risk category guidelines

## Technical Stack

- **Backend**: Django 5.x
- **Frontend**: TailwindCSS, vanilla JavaScript
- **Machine Learning**: CatBoost, scikit-learn, pandas, numpy
- **Database**: SQLite (development), PostgreSQL-ready
- **Design**: Modern glassmorphism interface

## Clinical Parameters

The system analyzes 15 clinical parameters:

### Demographics
- Age (years)
- Sex (0: Female, 1: Male)
- Body Mass Index (BMI)

### Vital Signs
- Systolic Blood Pressure (mmHg)
- Diastolic Blood Pressure (mmHg)

### Laboratory Tests
- Glycated Hemoglobin (HbA1c, %)
- Fasting Plasma Sugar (mg/dL)
- Postprandial Sugar (mg/dL)

### Medical History
- Family History of Diabetes (0: No, 1: Yes)
- Onset Age (years)
- Duration of Diabetes (years or months)

### Lifestyle Factors
- Smoking Status (0: No, 1: Yes, 2: Ex-smoker)
- Physical Activity Level (0: Low, 1: Moderate, 2: High)

### Medication
- Medication Usage (0: No, 1: Yes)
- Medication Adherence (0: Poor, 1: Moderate, 2: Good)

## Installation

### Prerequisites

- Python 3.10+
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Wilworks/diabetes-ai-predictor.git
cd diabetes-ai-predictor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place trained model files in `ml_models/` directory:
   - `scaler_original.joblib`
   - `catboost_original_NEP.joblib`
   - `catboost_original_NEU.joblib`
   - `catboost_original_RET.joblib`

5. Run database migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

6. Create superuser (optional):
```bash
python manage.py createsuperuser
```

7. Start development server:
```bash
python manage.py runserver
```

8. Access the application at `http://127.0.0.1:8000/`

## Usage

1. **Register/Login**: Create an account or login with existing credentials
2. **Navigate to Dashboard**: Access the prediction interface
3. **Enter Patient Data**: Fill in all 15 clinical parameters
4. **Submit Prediction**: System processes data through ML models
5. **View Results**: Risk scores displayed with color-coded visualizations

## Risk Stratification

- **Low Risk (<30%)**: Continue regular monitoring
- **Medium Risk (30-70%)**: Increased monitoring frequency recommended
- **High Risk (>70%)**: Immediate clinical evaluation advised

## Project Structure

```
diabetes-ai-predictor/
├── diabetesai/          # Django project settings
├── predictor/           # Main prediction app
├── accounts/            # Authentication app
├── templates/           # HTML templates
├── static/              # Static files (CSS, JS, images)
├── ml_models/           # Trained ML models
└── manage.py            # Django management script
```

## Research Team

**Developers**:
- Osei Emmanuel Amponsah (Medical Laboratory Science)
- Asumboya Wilfred Ayine (Biomedical Engineering)

**Supervisor**: Dr. Kennedy Bentum

**Institution**: University of Ghana

**Departments**: Medical Laboratory Science & Biomedical Engineering

## Clinical Disclaimer

This system is designed for research and educational purposes. AI predictions should supplement, not replace, clinical judgment. Always consult qualified healthcare professionals for diagnosis and treatment decisions.

## Contributing

This is an academic research project. For inquiries or collaboration opportunities, please contact the development team through the University of Ghana.

## License

This project is developed as part of academic research at the University of Ghana. All rights reserved.

## Acknowledgments

- Department of Medical Laboratory Science, University of Ghana
- Department of Biomedical Engineering, University of Ghana
- Ghana Society of Biomedical Engineers

## References

World Health Organization (2024). Diabetes Statistics and Prevalence Data.

---

**Note**: Model files are not included in this repository due to size constraints. Contact the development team for access to trained models.
