"""
SQLAlchemy table definitions.
Three tables: submissions, predictions, explanations.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    input_type = Column(String(10), nullable=False)   # "text" or "image"
    raw_input = Column(Text, nullable=False)           # original text or OCR output
    clean_text = Column(Text, nullable=True)           # after preprocessing
    submitted_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="submission", uselist=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)
    label = Column(String(20), nullable=False)         # "genuine" or "fraudulent"
    fraud_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_tier = Column(String(10), nullable=False)     # "low", "medium", "high"
    created_at = Column(DateTime, default=datetime.utcnow)

    submission = relationship("Submission", back_populates="prediction")
    explanation = relationship("Explanation", back_populates="prediction", uselist=False)


class Explanation(Base):
    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    method = Column(String(10), nullable=False)        # "lime"
    top_features_json = Column(Text, nullable=False)   # JSON list of {word, weight}
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="explanation")