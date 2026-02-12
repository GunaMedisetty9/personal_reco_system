"""
Disease Prediction Model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class DiseasePredictionModel:
    """ML model for predicting diseases based on symptoms and vitals."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
        self.symptom_list = [
            'fever', 'cough', 'fatigue', 'headache', 'chest_pain',
            'shortness_of_breath', 'nausea', 'dizziness', 'joint_pain',
            'skin_rash', 'weight_loss', 'excessive_thirst', 'frequent_urination',
            'blurred_vision', 'numbness', 'muscle_weakness'
        ]
    
    def create_training_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'age': np.random.randint(18, 85, n_samples),
            'gender': np.random.choice([0, 1], n_samples),
            'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
            'blood_pressure_diastolic': np.random.randint(60, 120, n_samples),
            'glucose_level': np.random.uniform(70, 300, n_samples),
            'heart_rate': np.random.randint(50, 120, n_samples),
            'temperature': np.random.uniform(97, 104, n_samples),
            'bmi': np.random.uniform(15, 45, n_samples),
            'cholesterol': np.random.uniform(120, 300, n_samples),
            'hemoglobin': np.random.uniform(8, 18, n_samples),
        }
        
        for symptom in self.symptom_list:
            data[symptom] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        df = pd.DataFrame(data)
        
        # Generate diagnoses based on conditions
        conditions = []
        for i in range(n_samples):
            row = df.iloc[i]
            
            if row['glucose_level'] > 200 and (row['excessive_thirst'] == 1 or row['frequent_urination'] == 1):
                conditions.append('Diabetes')
            elif row['blood_pressure_systolic'] > 140 and row['blood_pressure_diastolic'] > 90:
                conditions.append('Hypertension')
            elif row['chest_pain'] == 1 and row['shortness_of_breath'] == 1 and row['age'] > 45:
                conditions.append('Heart Disease')
            elif row['fever'] == 1 and row['cough'] == 1 and row['temperature'] > 100:
                conditions.append('Flu')
            elif row['joint_pain'] == 1 and row['fatigue'] == 1 and row['age'] > 40:
                conditions.append('Arthritis')
            elif row['headache'] == 1 and (row['nausea'] == 1 or row['blurred_vision'] == 1):
                conditions.append('Migraine')
            elif row['skin_rash'] == 1:
                conditions.append('Allergy')
            elif row['fatigue'] == 1 and row['weight_loss'] == 1:
                conditions.append('Thyroid Disorder')
            elif row['fatigue'] == 1 and row['dizziness'] == 1 and row['hemoglobin'] < 12:
                conditions.append('Anemia')
            else:
                conditions.append('Healthy')
        
        df['diagnosis'] = conditions
        return df
    
    def train(self):
        """Train the disease prediction model."""
        data = self.create_training_data()
        
        self.feature_columns = [col for col in data.columns if col != 'diagnosis']
        X = data[self.feature_columns]
        y = data['diagnosis']
        
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_scaled, y_encoded)
        self.is_trained = True
        
        return {
            'classes': list(self.label_encoder.classes_),
            'n_features': len(self.feature_columns)
        }
    
    def predict(self, patient_data: dict):
        """Predict disease for patient."""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        input_df = pd.DataFrame([patient_data])
        
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[self.feature_columns]
        input_scaled = self.scaler.transform(input_df)
        
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        disease = self.label_encoder.inverse_transform([prediction])[0]
        
        all_predictions = []
        for i, prob in enumerate(probabilities):
            all_predictions.append({
                'disease': self.label_encoder.inverse_transform([i])[0],
                'probability': round(prob * 100, 2)
            })
        
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'primary_diagnosis': disease,
            'confidence': round(max(probabilities) * 100, 2),
            'all_predictions': all_predictions[:5],
            'risk_factors': self._get_risk_factors(patient_data)
        }
    
    def _get_risk_factors(self, data: dict):
        """Analyze risk factors."""
        risks = []
        
        if data.get('age', 0) > 60:
            risks.append("Age above 60 - increased risk for chronic conditions")
        if data.get('bmi', 0) > 30:
            risks.append("BMI indicates obesity - risk for diabetes and heart disease")
        if data.get('blood_pressure_systolic', 0) > 140:
            risks.append("High blood pressure detected")
        if data.get('glucose_level', 0) > 126:
            risks.append("Elevated glucose levels - diabetes risk")
        if data.get('cholesterol', 0) > 240:
            risks.append("High cholesterol - cardiovascular risk")
        
        return risks
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'symptom_list': self.symptom_list,
            'is_trained': self.is_trained
        }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.feature_columns = data['feature_columns']
            self.symptom_list = data.get('symptom_list', self.symptom_list)
            self.is_trained = data.get('is_trained', True)
            return True
        return False