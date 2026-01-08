# District Prediction Workflow Guide

This guide shows what to add to each notebook in the district prediction project.

## Notebook 1: 01_exploration.ipynb

**Goal:** Load embeddings and add district labels

### Key Code to Add:

```python
# After setup, load the pre-computed embeddings
embeddings_file = project_root / 'data' / 'processed' / 'kerala_embeddings_YYYYMMDD_HHMM.npz'  # Use your actual file
data = np.load(embeddings_file)
embeddings = data['embeddings']
sampling_event_ids = data['sampling_event_ids']

print(f"Loaded {len(embeddings):,} embeddings")
print(f"Embedding shape: {embeddings.shape}")
```

```python
# Load original eBird data to get locality information
from src.data import load_ebird_data

data_path = project_root / 'data' / 'raw' / 'ebd_IN-KL_smp_relSep-2025.txt'
df = load_ebird_data(str(data_path))  # Load all data this time

# Keep only the sampling IDs we have embeddings for
df_with_embeddings = df[df['SAMPLING EVENT IDENTIFIER'].isin(sampling_event_ids)]
print(f"Matched {len(df_with_embeddings):,} observations to embeddings")
```

```python
# Extract district from locality names
KERALA_DISTRICTS = [
    'Thiruvananthapuram', 'Kollam', 'Pathanamthitta', 'Alappuzha',
    'Kottayam', 'Idukki', 'Ernakulam', 'Thrissur', 'Palakkad',
    'Malappuram', 'Kozhikode', 'Wayanad', 'Kannur', 'Kasaragod'
]

def extract_district(locality_text):
    """Extract district from locality string."""
    if pd.isna(locality_text):
        return 'Unknown'
    
    locality_lower = str(locality_text).lower()
    
    for district in KERALA_DISTRICTS:
        if district.lower() in locality_lower:
            return district
    
    return 'Unknown'

# Apply to dataframe
df_with_embeddings['district'] = df_with_embeddings['LOCALITY'].apply(extract_district)

# Check distribution
print("\nDistrict distribution:")
print(df_with_embeddings['district'].value_counts())
```

```python
# Create checklist-level dataframe (one row per checklist)
checklist_df = df_with_embeddings.groupby('SAMPLING EVENT IDENTIFIER').agg({
    'district': 'first',  # One district per checklist
    'LOCALITY': 'first',
    'OBSERVATION DATE': 'first',
    'LATITUDE': 'first',
    'LONGITUDE': 'first'
}).reset_index()

print(f"\nUnique checklists: {len(checklist_df):,}")
print(f"Districts represented: {checklist_df['district'].nunique()}")
```

```python
# Match embeddings to districts
# Create a mapping from sampling_event_id to index in embeddings array
id_to_idx = {sid: i for i, sid in enumerate(sampling_event_ids)}

# Get embedding indices for our checklists
checklist_df['embedding_idx'] = checklist_df['SAMPLING EVENT IDENTIFIER'].map(id_to_idx)

# Remove any that didn't match
checklist_df = checklist_df.dropna(subset=['embedding_idx'])
checklist_df['embedding_idx'] = checklist_df['embedding_idx'].astype(int)

# Extract corresponding embeddings
X = embeddings[checklist_df['embedding_idx'].values]
y = checklist_df['district'].values

print(f"\nFinal dataset:")
print(f"  Embeddings: {X.shape}")
print(f"  Labels: {len(y)}")
print(f"  Districts: {pd.Series(y).value_counts()}")
```

```python
# Remove 'Unknown' district
mask = y != 'Unknown'
X = X[mask]
y = y[mask]
checklist_df = checklist_df[mask].reset_index(drop=True)

print(f"\nAfter removing 'Unknown':")
print(f"  Samples: {len(X):,}")
print(f"  Districts: {len(np.unique(y))}")
```

```python
# Save for analysis
save_path = Path('../data/processed/embeddings_with_districts.npz')
np.savez(
    save_path,
    embeddings=X,
    districts=y,
    sampling_event_ids=checklist_df['SAMPLING EVENT IDENTIFIER'].values,
    locality=checklist_df['LOCALITY'].values,
    latitude=checklist_df['LATITUDE'].values,
    longitude=checklist_df['LONGITUDE'].values
)

print(f"\n✓ Saved to {save_path}")
```

---

## Notebook 2: 02_analysis.ipynb

**Goal:** Train district classifier

### Key Code to Add:

```python
# Load processed data
data = np.load('../data/processed/embeddings_with_districts.npz')
X = data['embeddings']
y = data['districts']

print(f"Loaded {len(X):,} samples")
print(f"Districts: {np.unique(y)}")
```

```python
# Train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
```

```python
# Train Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

print("Training classifier...")
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"\nTrain accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
```

```python
# Detailed evaluation
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

```python
# Save model
import joblib

model_path = Path('../models/district_classifier.pkl')
model_path.parent.mkdir(exist_ok=True)
joblib.dump(clf, model_path)

print(f"\n✓ Saved model to {model_path}")
```

```python
# Save predictions for analysis
import json

results = {
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'n_districts': len(np.unique(y)),
    'districts': list(np.unique(y))
}

results_path = Path('../results/tables/metrics.json')
results_path.parent.mkdir(parents=True, exist_ok=True)

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved metrics to {results_path}")
```

---

## Notebook 3: 03_results.ipynb

**Goal:** Visualize results and create plots

### Key Code to Add:

```python
# Load everything
import joblib

data = np.load('../data/processed/embeddings_with_districts.npz')
X = data['embeddings']
y = data['districts']

clf = joblib.load('../models/district_classifier.pkl')

# Recreate train/test split (same random_state)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = clf.predict(X_test)
```

```python
# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
districts = clf.classes_

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=districts,
    yticklabels=districts,
    cbar_kws={'label': 'Count'}
)
plt.title('District Prediction Confusion Matrix', fontsize=16, pad=20)
plt.ylabel('True District', fontsize=12)
plt.xlabel('Predicted District', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

```python
# Per-district accuracy
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, output_dict=True)

# Create bar plot
accuracies = {district: report[district]['precision'] 
              for district in districts}

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(accuracies)), list(accuracies.values()))
plt.xticks(range(len(accuracies)), list(accuracies.keys()), rotation=45, ha='right')
plt.ylabel('Precision')
plt.title('Per-District Prediction Precision')
plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Overall Accuracy: {test_acc:.3f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/per_district_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
```

```python
# Feature importance (embedding dimensions)
importances = clf.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.xlabel('Embedding Dimension')
plt.ylabel('Importance')
plt.title('Embedding Dimension Importance for District Prediction')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
```

```python
# Sample predictions table
results_df = pd.DataFrame({
    'True District': y_test[:20],
    'Predicted District': y_pred[:20],
    'Correct': y_test[:20] == y_pred[:20]
})

results_df.to_csv('../results/tables/sample_predictions.csv', index=False)
print("Sample predictions:")
print(results_df)
```

```python
# Summary statistics
summary_df = pd.DataFrame({
    'District': districts,
    'Precision': [report[d]['precision'] for d in districts],
    'Recall': [report[d]['recall'] for d in districts],
    'F1-Score': [report[d]['f1-score'] for d in districts],
    'Support': [report[d]['support'] for d in districts]
})

summary_df.to_csv('../results/tables/district_metrics.csv', index=False)
print("\nPer-district metrics:")
print(summary_df.to_string(index=False))
```

---

## Expected Results

After running all 3 notebooks, you should have:

1. **Data:** `projects/district_prediction/data/processed/embeddings_with_districts.npz`
2. **Model:** `projects/district_prediction/models/district_classifier.pkl`
3. **Figures:**
   - Confusion matrix
   - Per-district accuracy
   - Feature importance
4. **Tables:**
   - Metrics (JSON)
   - Sample predictions (CSV)
   - District-level metrics (CSV)

## Tips

- If some districts have very few samples, consider combining them or filtering them out
- Try different classifiers (SVM, Gradient Boosting, etc.)
- Use cross-validation for more robust evaluation
- Consider geographic features (lat/lon) in addition to embeddings
