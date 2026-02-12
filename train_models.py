"""
Train and save all ML models
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.disease_predictor import DiseasePredictionModel
from utils.medicine_recommender import MedicineRecommender
from utils.sentiment_analyzer import SentimentAnalyzer


def main():
    print("=" * 60)
    print("🏥 HEALTHCARE RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    
    # Train Disease Predictor
    print("\n🔍 Training Disease Prediction Model...")
    disease_model = DiseasePredictionModel()
    result = disease_model.train()
    disease_model.save('models/disease_model.pkl')
    print(f"   ✅ Saved! Classes: {result['classes']}")
    
    # Train Medicine Recommender
    print("\n💊 Training Medicine Recommendation Model...")
    medicine_model = MedicineRecommender()
    medicine_model.load_medicines()
    medicine_model.save('models/medicine_model.pkl')
    print(f"   ✅ Saved! Medicines: {len(medicine_model.medicines_df)}")
    
    # Train Sentiment Analyzer
    print("\n💬 Training Sentiment Analysis Model...")
    sentiment_model = SentimentAnalyzer()
    sentiment_model.train()
    sentiment_model.save('models/sentiment_model.pkl')
    print("   ✅ Saved!")
    
    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("   Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()