# Topic Modelling of McDonald's Store Reviews

Using Latent Dirichlet Allocation (LDA) to uncover the main themes in 33,000+ McDonald's customer reviews, so management can understand what customers care about without reading every review by hand.

---

## Project Overview

McDonald's receives a constant stream of online reviews across its stores, covering everything from food quality and order accuracy to staff behaviour and cleanliness. Reading these manually at scale isn't practical, and manual review is prone to bias toward whatever a reader already expects to see.

This project applies **LDA topic modelling** to a dataset of **33,396 McDonald's store reviews** to automatically surface the recurring themes in customer feedback, validate which of those themes represent complaints versus positive experiences, and turn the findings into concrete business recommendations.

## Dataset

- **Source:** [McDonald's Store Reviews (Kaggle)](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews)
- **Size:** 33,396 reviews across 40 US store locations
- **Fields:** reviewer ID, store name, category, store address, latitude/longitude, rating count, review time, review text, star rating

## Methodology

1. **Exploratory analysis** — checked dataset size, missing values, duplicates, rating distribution, and review volume by store.
2. **Text preprocessing** — lowercasing, punctuation/number removal, tokenisation, stop-word removal, and lemmatisation with part-of-speech tagging.
3. **LDA topic modelling** — built a bag-of-words corpus, scanned topic coherence scores across k=2–10, then validated candidate topic counts qualitatively against the business question. Landed on **7 topics** as the smallest model where every topic was distinct and interpretable.
4. **Interpretation** — cross-checked each topic against average star rating (information the model never saw during training) to confirm which themes are complaints and which are positive experiences.
5. **Recommendations** — translated the findings into concrete, actionable steps for McDonald's management.

Full code, explanations, and reasoning are in [`McDonald_LDA_Analysis.ipynb`](./McDonald_LDA_Analysis.ipynb).

---

## Key Visualisations

### Rating distribution

Ratings are bimodal — customers tend to leave a review only when they are very satisfied or very dissatisfied, with fewer 2-4 star reviews in between.

![Rating Distribution](images/rating_distribution.png)

### Review volume by store

Review volume is concentrated in a handful of high-traffic locations, mostly near tourist areas.

![Top Stores by Review Count](images/top_stores.png)

### Choosing the number of topics

Coherence score alone favoured an uninformatively small number of topics (2-4). The final choice of **k=7** balanced coherence with topic interpretability, confirmed by manually reading topic content at each candidate value.

![Coherence Scores](images/coherence_scores.png)

### Top words per topic

Each of the 7 topics is defined by a distinct set of top-weighted words, used to name and interpret the theme.

![Top Words per Topic](images/topic_top_words.png)

### Word cloud of most frequent terms

A quick visual sense of the most common words across all cleaned reviews.

![Word Cloud](images/wordcloud_all_reviews.png)

### Average rating by topic

Cross-checking each topic against its average star rating validates the complaint/positive split, since the model never saw ratings during training.

![Average Rating by Topic](images/avg_rating_by_topic.png)

### Reviews per topic

Order Delays & Follow-Up and Fast/Clean/Friendly Service are by far the largest topics, together covering the majority of all reviews.

![Reviews per Topic](images/reviews_per_topic.png)

---

## Findings

| Topic | Avg. Rating | Type | Share of Reviews |
|---|---|---|---|
| Rude Staff & Poor Customer Service | 1.54 | Complaint | ~1% |
| Drive-Thru & Wait Times | 2.27 | Complaint | ~4% |
| Order Accuracy (Wrong/Missing Items) | 2.42 | Complaint | ~3% |
| Order Delays & Follow-Up | 2.58 | Complaint | ~51% |
| Food & Menu Items | 2.79 | Neutral/Mixed | ~4% |
| Cleanliness & Facility Complaints | 3.44 | Mixed | ~1% |
| Fast, Clean, Friendly Service | 4.18 | Positive | ~35% |

**Complaint themes** (below average rating): rude/unprofessional staff at specific locations, unreliable drive-thru hours and long waits, incorrect or missing items (especially from mobile/kiosk orders), and general order delays paired with poor complaint recovery — this last one is the single largest topic by volume, touching roughly half of all reviews.

**Positive themes**: fast, clean, and friendly service is the clearest positive signal and the second-largest topic overall. Cleanliness shows a mixed rating spread — some stores are praised for it, others criticised — pointing to inconsistency across locations rather than a uniform problem.

## Business Recommendations

1. **Fix order accuracy in mobile and kiosk ordering** — require a visible order-check step at pickup, with a printed checklist matching the digital order.
2. **Standardise drive-thru hours and staffing** — audit posted hours against real staffing schedules, and push real-time closure alerts to Maps and the app.
3. **Reduce order delays and improve complaint recovery** — train shift managers on a standard recovery script (acknowledge, fix, follow up); this addresses the single largest topic by review volume.
4. **Target customer service retraining at specific low-performing locations** — the rudeness topic is severe but low-volume, so focus intervention on the stores driving it rather than a blanket campaign.
5. **Enforce a consistent cleanliness standard** — introduce a scheduled cleaning checklist (hourly tables, bi-hourly restrooms) and audit compliance, since some stores already meet a high bar while others do not.

---

## Repository Structure

```
├── McDonald_LDA_Analysis.ipynb      # Full notebook: EDA, preprocessing, LDA modelling, interpretation
├── McDonald_Reviews_LDA_Report.docx # 2-page summary report (methodology, findings, recommendations)
├── lda_visualisation.html           # Interactive pyLDAvis topic visualisation
├── images/                          # All charts used in this README and the report
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn nltk gensim pyLDAvis wordcloud
jupyter notebook McDonald_LDA_Analysis.ipynb
```

The notebook downloads the required NLTK corpora (`stopwords`, `wordnet`, `punkt`, `averaged_perceptron_tagger`) automatically on first run.

## Deliverables

- **Notebook:** [`McDonald_LDA_Analysis.ipynb`](./McDonald_LDA_Analysis.ipynb)
- **Report:** [`McDonald_Reviews_LDA_Report.docx`](./McDonald_Reviews_LDA_Report.docx)
- **Interactive topic visualisation:** [`lda_visualisation.html`](./lda_visualisation.html)
