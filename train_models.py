from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / 'data' / 'raw' / 'diabetes.csv'
OUTPUTS_DIR = ROOT / 'outputs'
MODELS_DIR = ROOT / 'models'

OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

NUMERIC_COLUMNS = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
]
CATEGORICAL_COLUMNS = ['gender', 'smoking_history']


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                'numeric',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                NUMERIC_COLUMNS,
            ),
            (
                'categorical',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=250, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'k-NN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
    }


def plot_confusion_matrix(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    ax.set_title(f'{model_name} - Confusion Matrix')
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300)
    plt.close(fig)


def plot_roc_curve(model_name: str, y_true: pd.Series, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_prob):.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Diabetes Prediction')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'diabetes_roc_curve.png', dpi=300)
    plt.close()


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    summary_rows = []
    for model_name, estimator in build_models().items():
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('model', estimator),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        summary_rows.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1_Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test, y_prob),
        })

        plot_confusion_matrix(model_name, y_test, y_pred)
        plot_roc_curve(model_name, y_test, y_prob)

    summary = pd.DataFrame(summary_rows).sort_values('F1_Score', ascending=False)
    summary.to_csv(OUTPUTS_DIR / 'diabetes_model_summary.csv', index=False)

    best_model_name = summary.iloc[0]['Model']
    best_model = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', build_models()[best_model_name]),
    ])
    best_model.fit(X, y)
    joblib.dump(best_model, MODELS_DIR / 'diabetes_best_model.pkl')

    print('\nDiabetes model summary:')
    print(summary.round(4).to_string(index=False))
    print(f'\nBest model: {best_model_name}')


if __name__ == '__main__':
    main()
