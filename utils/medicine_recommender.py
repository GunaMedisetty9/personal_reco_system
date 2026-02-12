"""
Medicine Recommendation Model
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MedicineRecommender:
    """Content-based medicine recommendation system."""

    def __init__(self):
        self.medicines_df = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self.medicine_vectors = None
        self.similarity_matrix = None

        # Disease-Medicine mapping
        self.disease_medicine_map = {
            "Diabetes": [
                {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily", "notes": "Take with meals"},
                {"name": "Glipizide", "dosage": "5mg", "frequency": "once daily", "notes": "Take before breakfast"},
                {"name": "Januvia", "dosage": "100mg", "frequency": "once daily", "notes": "Can take with or without food"},
            ],
            "Hypertension": [
                {"name": "Lisinopril", "dosage": "10mg", "frequency": "once daily", "notes": "ACE inhibitor"},
                {"name": "Amlodipine", "dosage": "5mg", "frequency": "once daily", "notes": "Calcium channel blocker"},
                {"name": "Losartan", "dosage": "50mg", "frequency": "once daily", "notes": "ARB medication"},
            ],
            "Heart Disease": [
                {"name": "Aspirin", "dosage": "81mg", "frequency": "once daily", "notes": "Blood thinner"},
                {"name": "Atorvastatin", "dosage": "20mg", "frequency": "once daily", "notes": "Take in evening"},
                {"name": "Metoprolol", "dosage": "25mg", "frequency": "twice daily", "notes": "Beta blocker"},
            ],
            "Flu": [
                {"name": "Tamiflu", "dosage": "75mg", "frequency": "twice daily", "notes": "Take within 48 hours of symptoms"},
                {"name": "Tylenol", "dosage": "500mg", "frequency": "every 6 hours", "notes": "For fever and pain"},
                {"name": "Robitussin", "dosage": "15mg", "frequency": "every 4 hours", "notes": "For cough relief"},
            ],
            "Arthritis": [
                {"name": "Ibuprofen", "dosage": "400mg", "frequency": "three times daily", "notes": "Take with food"},
                {"name": "Naproxen", "dosage": "250mg", "frequency": "twice daily", "notes": "Take with food"},
                {"name": "Celebrex", "dosage": "200mg", "frequency": "once daily", "notes": "COX-2 inhibitor"},
            ],
            "Migraine": [
                {"name": "Sumatriptan", "dosage": "50mg", "frequency": "as needed", "notes": "At first sign of migraine"},
                {"name": "Rizatriptan", "dosage": "10mg", "frequency": "as needed", "notes": "Fast dissolving"},
                {"name": "Topiramate", "dosage": "25mg", "frequency": "twice daily", "notes": "For prevention"},
            ],
            "Allergy": [
                {"name": "Cetirizine", "dosage": "10mg", "frequency": "once daily", "notes": "Non-drowsy"},
                {"name": "Loratadine", "dosage": "10mg", "frequency": "once daily", "notes": "Non-drowsy"},
                {"name": "Diphenhydramine", "dosage": "25mg", "frequency": "every 6 hours", "notes": "May cause drowsiness"},
            ],
            "Thyroid Disorder": [
                {"name": "Levothyroxine", "dosage": "50mcg", "frequency": "once daily", "notes": "Take on empty stomach"},
                {"name": "Liothyronine", "dosage": "25mcg", "frequency": "once daily", "notes": "T3 replacement"},
                {"name": "Methimazole", "dosage": "10mg", "frequency": "once daily", "notes": "For hyperthyroidism"},
            ],
            "Anemia": [
                {"name": "Ferrous Sulfate", "dosage": "325mg", "frequency": "once daily", "notes": "Take with vitamin C"},
                {"name": "Vitamin B12", "dosage": "1000mcg", "frequency": "once daily", "notes": "For B12 deficiency"},
                {"name": "Folic Acid", "dosage": "400mcg", "frequency": "once daily", "notes": "For folate deficiency"},
            ],
            "Healthy": [
                {"name": "Multivitamin", "dosage": "1 tablet", "frequency": "once daily", "notes": "General health"},
                {"name": "Vitamin D3", "dosage": "1000IU", "frequency": "once daily", "notes": "For bone health"},
            ],
        }

        # Drug interactions database
        self.interactions = {
            ("Aspirin", "Ibuprofen"): {"severity": "Moderate", "description": "Increased bleeding risk"},
            ("Aspirin", "Warfarin"): {"severity": "Severe", "description": "High bleeding risk"},
            ("Metformin", "Lisinopril"): {"severity": "Low", "description": "May enhance blood sugar lowering"},
            ("Lisinopril", "Losartan"): {"severity": "Moderate", "description": "Dual RAAS blockade risk"},
            ("Metoprolol", "Amlodipine"): {"severity": "Low", "description": "Additive BP lowering"},
        }

    def load_medicines(self, data_path: str | None = None) -> None:
        """Load medicine data (from CSV if provided, else synthetic mapping)."""
        if data_path and os.path.exists(data_path):
            self.medicines_df = pd.read_csv(data_path)
        else:
            data = []
            for disease, medicines in self.disease_medicine_map.items():
                for med in medicines:
                    data.append({
                        "name": med["name"],
                        "category": disease,
                        "dosage": med["dosage"],
                        "frequency": med["frequency"],
                        "description": f"Used for treating {disease}. {med.get('notes', '')}",
                    })
            self.medicines_df = pd.DataFrame(data)

        self._build_vectors()

    def _build_vectors(self) -> None:
        """Build TF-IDF vectors for content-based filtering."""
        if self.medicines_df is None or len(self.medicines_df) == 0:
            self.medicine_vectors = None
            self.similarity_matrix = None
            return

        self.medicines_df["combined"] = (
            self.medicines_df["name"].fillna("") + " " +
            self.medicines_df["category"].fillna("") + " " +
            self.medicines_df["description"].fillna("")
        )

        self.medicine_vectors = self.tfidf_vectorizer.fit_transform(self.medicines_df["combined"])
        self.similarity_matrix = cosine_similarity(self.medicine_vectors)

    def recommend_by_disease(self, disease: str, patient_data: dict | None = None, n: int = 5) -> list:
        """Recommend medicines for a disease."""
        recommendations = []

        if disease in self.disease_medicine_map:
            for med in self.disease_medicine_map[disease][:n]:
                rec = {
                    "medicine_name": med["name"],
                    "dosage": med["dosage"],
                    "frequency": med["frequency"],
                    "notes": med.get("notes", ""),
                    "reason": f"Commonly prescribed for {disease}",
                    "confidence": 0.85,
                    "warnings": [],
                }

                if patient_data:
                    warnings = self._check_contraindications(med["name"], patient_data)
                    rec["warnings"] = warnings
                    if warnings:
                        rec["confidence"] *= 0.7

                recommendations.append(rec)

        return recommendations

    def find_similar_medicines(self, medicine_name: str, n: int = 5) -> list:
        """Find similar medicines using content-based filtering."""
        if self.medicines_df is None or self.similarity_matrix is None:
            self.load_medicines()

        idx_matches = self.medicines_df[
            self.medicines_df["name"].str.lower() == medicine_name.lower()
        ].index

        if len(idx_matches) == 0:
            return []

        idx = int(idx_matches[0])
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        results = []
        for i, score in sim_scores[1:n + 1]:
            if score > 0:
                med = self.medicines_df.iloc[i]
                results.append({
                    "name": med["name"],
                    "category": med["category"],
                    "similarity": round(float(score), 3),
                })
        return results

    def check_interactions(self, medicines: list) -> list:
        """Check for drug interactions."""
        found = []
        for i, med1 in enumerate(medicines):
            for med2 in medicines[i + 1:]:
                key1 = (med1, med2)
                key2 = (med2, med1)
                if key1 in self.interactions:
                    found.append({"medicines": [med1, med2], **self.interactions[key1]})
                elif key2 in self.interactions:
                    found.append({"medicines": [med1, med2], **self.interactions[key2]})
        return found

    def _check_contraindications(self, medicine: str, patient_data: dict) -> list:
        """Check for contraindications."""
        warnings = []

        allergies = patient_data.get("allergies", [])
        conditions = patient_data.get("medical_conditions", [])
        age = patient_data.get("age", 0)

        if any(allergy.lower() in medicine.lower() for allergy in allergies):
            warnings.append(f"⚠️ Potential allergy to {medicine}")

        if "kidney disease" in [c.lower() for c in conditions]:
            if medicine in ["Metformin", "Ibuprofen", "Naproxen"]:
                warnings.append("⚠️ Caution: May affect kidney function")

        if age > 65 and medicine == "Diphenhydramine":
            warnings.append("⚠️ Caution: May cause confusion in elderly")

        return warnings

    def save(self, path: str) -> None:
        """Save model artifacts."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "medicines_df": self.medicines_df,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "medicine_vectors": self.medicine_vectors,
            "similarity_matrix": self.similarity_matrix,
        }, path)

    def load(self, path: str) -> bool:
        """Load model artifacts. Returns False if incompatible/corrupt pickle."""
        if not os.path.exists(path):
            return False
        try:
            data = joblib.load(path)
            self.medicines_df = data.get("medicines_df")
            self.tfidf_vectorizer = data.get("tfidf_vectorizer", self.tfidf_vectorizer)
            self.medicine_vectors = data.get("medicine_vectors")
            self.similarity_matrix = data.get("similarity_matrix")

            # If anything missing, rebuild from mapping
            if self.medicines_df is None or self.similarity_matrix is None:
                return False

            return True
        except Exception:
            return False