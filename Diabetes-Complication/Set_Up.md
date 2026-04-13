# DiabetesAI - Setup and Deployment Guide

## Project Structure

Ensure your project follows this directory structure:

```
diabetes-ai-predictor/
├── venv/
├── diabetesai/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── predictor/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── admin.py
│   └── ml_predictor.py
├── accounts/
│   ├── __init__.py
│   ├── views.py
│   └── urls.py
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   └── results.html
├── ml_models/
│   ├── scaler_original.joblib
│   ├── catboost_original_NEP.joblib
│   ├── catboost_original_NEU.joblib
│   └── catboost_original_RET.joblib
├── static/
│   └── images/
│       └── lab-hero.jpg
└── manage.py
```

---

## Installation and Configuration

### Step 1: Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install required dependencies
pip install django pillow pandas numpy scikit-learn joblib catboost
```

### Step 2: Model Files

Copy the trained model files to the `ml_models/` directory:

```bash
cp /path/to/your/models/scaler_original.joblib ml_models/
cp /path/to/your/models/catboost_original_NEP.joblib ml_models/
cp /path/to/your/models/catboost_original_NEU.joblib ml_models/
cp /path/to/your/models/catboost_original_RET.joblib ml_models/
```

### Step 3: Database Configuration

```bash
# Create database migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create administrative user
python manage.py createsuperuser
```

---

## Running the Application

Start the development server:

```bash
python manage.py runserver
```

Access the application at: `http://127.0.0.1:8000/`

---

## Application Routes

| Route | Description |
|-------|-------------|
| `/` | Landing page with research overview |
| `/accounts/login/` | User authentication |
| `/accounts/register/` | New user registration |
| `/predictor/dashboard/` | Main dashboard (requires authentication) |
| `/predictor/predict/` | Prediction form interface |
| `/predictor/results/<id>/` | Prediction results display |
| `/admin/` | Administrative panel |

---

## Testing the System

### Verification Checklist

- Landing page loads with glassmorphism design
- Laboratory photograph displays in hero section
- User authentication system functions correctly
- Dashboard accessible after login
- Prediction form contains all 15 clinical parameters
- Models return risk scores for NEP, NEU, and RET
- Results page displays with animated progress bars
- Model files load successfully (check terminal output)

### Sample Test Data

Use the following patient data for system validation:

```
Age: 55
Sex: Male (1)
BMI: 28.5
Systolic BP: 145
Diastolic BP: 90
HbA1c: 8.2
Fasting Plasma Sugar: 145
Postprandial Sugar: 210
Family History: Yes (1)
Onset Age: 48
Diabetes Duration: 7
Smoking: No (0)
Physical Activity: Low (0)
Medication Use: Yes (1)
Medication Adherence: Moderate (1)
```

Expected output: Three risk probability scores between 0-100% for nephropathy, neuropathy, and retinopathy.

---

## Troubleshooting

### Model Loading Errors

**Issue**: `FileNotFoundError: scaler_original.joblib`

**Solution**:
```bash
ls -la ml_models/
# Verify all four .joblib files exist
```

### Dependency Errors

**Issue**: `ModuleNotFoundError: No module named 'joblib'`

**Solution**:
```bash
pip install joblib pandas numpy scikit-learn catboost
```

### Template Errors

**Issue**: `TemplateDoesNotExist at /`

**Solution**:
- Verify all HTML files are in the `templates/` directory
- Check `settings.py` contains: `'DIRS': [BASE_DIR / 'templates']`

### Database Migration Issues

**Issue**: `no such table: predictor_prediction`

**Solution**:
```bash
python manage.py makemigrations predictor
python manage.py migrate predictor
```

### Static File Configuration

**Issue**: CSS or images not loading

**Solution**:
```bash
python manage.py collectstatic --noinput
```

---

## Model Verification

Verify models load correctly using Django shell:

```python
python manage.py shell

# Execute in shell:
from predictor.ml_predictor import get_predictor
predictor = get_predictor()
print("Models loaded successfully!")
```

Successful output will display:
```
✓ Loaded scaler from .../ml_models/scaler_original.joblib
✓ Loaded CatBoost model for NEP from .../ml_models/catboost_original_NEP.joblib
✓ Loaded CatBoost model for NEU from .../ml_models/catboost_original_NEU.joblib
✓ Loaded CatBoost model for RET from .../ml_models/catboost_original_RET.joblib
```

---

## Production Deployment Considerations

Before production deployment:

1. **Security Configuration**
   - Change `SECRET_KEY` in `settings.py`
   - Set `DEBUG = False`
   - Configure appropriate `ALLOWED_HOSTS`

2. **Server Options**
   - Heroku
   - PythonAnywhere
   - Railway
   - Render

3. **Static Files**
   - Configure proper static file serving
   - Set up CDN if necessary

---

## System Architecture

### Technical Stack
- **Backend Framework**: Django 5.x
- **Frontend**: TailwindCSS with vanilla JavaScript
- **Machine Learning**: CatBoost, scikit-learn
- **Database**: SQLite (development), PostgreSQL recommended for production
- **Design Pattern**: Glassmorphism UI

### ML Pipeline
1. Input collection (15 clinical parameters)
2. Data preprocessing and scaling
3. CatBoost model inference (3 separate models)
4. Risk stratification and visualization

### Clinical Parameters
- Demographics: Age, Sex, BMI
- Vital Signs: Systolic BP, Diastolic BP
- Laboratory Tests: HbA1c, FPS, PPS
- Medical History: Family History, Onset Age, Diabetes Duration
- Lifestyle: Smoking Status, Physical Activity
- Medication: Usage, Adherence

---

## Research Team

**Developers**: Osei Emmanuel Amponsah & Asumboya Wilfred Ayine  
**Supervisor**: Dr. Kennedy Bentum  
**Institution**: University of Ghana  
**Department**: Medical Laboratory Science & Biomedical Engineering

---

## License and Usage

This system is designed for research and educational purposes. Clinical decisions should not be based solely on AI predictions. Always consult qualified healthcare professionals for diagnosis and treatment.