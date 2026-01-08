# Wetland Prediction Via VAE Embeddings

Predict whether a checklist location is near a wetland using species composition embeddings from a Variational Autoencoder (VAE).

## Overview

**Goal:** Determine if VAE embeddings (trained on species lists) can predict wetland vs non-wetland habitats using two different labeling heuristics.

**Key Question:** Can learned embeddings from species composition alone capture habitat information?

## Three Labeling Approaches

We created labeled datasets using **three heuristics** to test if embeddings capture habitat information:

### **Approach 1: Bird Proportion Heuristic** 
**Notebook:** `00_preprocessing_proportion_heuristic.ipynb`

**Method:** Label based on proportion of wetland-indicator species in checklist
- **Hardcoded species list:** 50+ wetland-associated birds including:
  - Ducks, geese, grebes (e.g., Northern Pintail, Little Grebe)
  - Herons, egrets, storks (e.g., Grey Heron, Painted Stork)
  - Waders, plovers (e.g., Common Sandpiper, Red-wattled Lapwing)
  - Kingfishers, cormorants, rails (e.g., Common Kingfisher, Little Cormorant)
- **Labeling rule:** Checklist labeled "wetland" if ≥30% of species are wetland indicators
- **Data scope:** Uses ALL checklists (29,991 samples)

**Output:** `data/processed/wetland_proportion_labeled.npz`

---

### **Approach 2: Hotspot Name Heuristic**
**Notebook:** `00b_preprocessing_hotspot_heuristic.ipynb`

**Method:** Label based on eBird hotspot name keywords using word boundary matching
- **Wetland keywords:** lake, pond, river, reservoir, dam, backwater, wetland, marsh, creek, canal, kulam, kere, etc.
- **Non-wetland keywords:** forest, hill, grassland, garden, park, sanctuary, mala, vanam, aranya, etc.
- **Labeling rule:** Match hotspot names (via LOCALITY_ID) against keyword lists
- **Data scope:** Uses ONLY hotspot checklists with clear wetland/non-wetland names (3,760 samples)

**Output:** `data/processed/wetland_hotspot_labeled.npz`

---

### **Approach 3: Intersection (Hotspot + Proportion)**
**Notebook:** `01_exploration.ipynb` (created during exploration)

**Method:** Combine both heuristics for stricter labeling
- **Step 1:** Use hotspot name heuristic to identify potential wetland/non-wetland locations
- **Step 2:** Further filter wetland hotspots by requiring ≥30% wetland bird proportion
- **Rationale:** Remove false positives from hotspot names (e.g., "Lake Garden" that isn't actually a wetland)
- **Data scope:** Subset of hotspot-labeled data (2,889 samples after filtering)

**Output:** `data/processed/wetland_intersection_train.npz` & `wetland_intersection_test.npz`

---

## Results Summary

All three approaches use Random Forest classifiers trained on 16-dimensional VAE embeddings:

| Metric | Bird Proportion | Hotspot Name | Intersection |
|--------|-----------------|--------------|--------------|
| **Accuracy** | 94.6% | 87.9% | **97.0%** |
| **Precision** | 93.2% | 85.0% | **98.2%** |
| **Recall** | 96.3% | 79.5% | 95.7% |
| **F1 Score** | 94.7% | 82.1% | **96.9%** |
| **ROC AUC** | 98.9% | 94.2% | **99.7%** |
| **Training Samples** | 23,992 | 3,008 | 932 |
| **Test Samples** | 5,999 | 752 | 233 |

### Key Insights

1. **Bird Proportion (94.6% accuracy):** Higher accuracy is **expected** because:
   - VAE embeddings were trained on species composition
   - Wetland labels are derived from species presence
   - Strong correlation between input features and labels
   - Large dataset (30K samples) with balanced classes

2. **Hotspot Name (87.9% accuracy):** High accuracy is **surprising and interesting** because:
   - Labels are independent of species composition (based on location names)
   - VAE was never trained on location information
   - Shows embeddings learned meaningful habitat associations
   - Proves species composition implicitly encodes habitat type
   - Smaller dataset (3.7K samples) yet still strong performance

3. **Intersection (97.0% accuracy):** Best performance across all metrics:
   - **Highest precision (98.2%)** - Combining heuristics removes noisy labels
   - **High recall (95.7%)** - Still captures most true wetland cases
   - **Best ROC AUC (99.7%)** - Excellent discriminative ability
   - **Smallest but cleanest dataset** (1,165 samples)
   - Validates that stricter labeling improves model quality
   - Shows intersection of independent heuristics creates reliable ground truth


## Workflow

Both approaches follow the same analysis pipeline:

### Phase 0: Preprocessing & Labeling
**Notebooks:** 
- `00_preprocessing_proportion_heuristic.ipynb` (Bird-based labels)
- `00b_preprocessing_hotspot_heuristic.ipynb` (Location-based labels)

Create labeled datasets using one of the two heuristics described above.

---

### Phase 1: Exploration
**Notebook:** `01_exploration.ipynb`

**Tasks:**
- Load pre-computed VAE embeddings from backbone (`data/processed/vae_embeddings_1M.npz`)
- Load wetland labels from chosen preprocessing method
- Match embeddings to labels via sampling_event_id
- Explore label distribution and balance
- Create train/test datasets

**Outputs:** 
- `data/processed/bird_proportion_train.npz` & `bird_proportion_test.npz`
- `data/processed/hotspot_name_train.npz` & `hotspot_name_test.npz`
- `data/processed/wetland_intersection_train.npz` & `wetland_intersection_test.npz`

---

### Phase 2: Analysis & Training
**Notebook:** `02_analysis.ipynb`

**Tasks:**
- Train Random Forest classifiers for all three labeling approaches
- Hyperparameters: 100 trees, max depth 20, balanced class weights
- Evaluate: accuracy, precision, recall, F1, ROC AUC
- Compare models with ROC curves and confusion matrices
- Save models and predictions

**Outputs:** 
- Models: `models/rf_bird_proportion.pkl`, `rf_hotspot_name.pkl`, `rf_intersection.pkl`
- Metrics: `results/metrics_proportion.json`, `metrics_hotspot.json`, `metrics_intersection.json`
- Test predictions: `results/*_test_results.npz`
- Visualizations: `results/figures/roc_curves_comparison.png`, `confusion_matrices_comparison.png`

---
