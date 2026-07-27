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

---

## Key Visualisations

### Rating distribution

Ratings are bimodal — customers tend to leave a review only when they are very satisfied or very dissatisfied, with fewer 2-4 star reviews in between.
<img width="900" height="500" alt="avg_rating_by_topic" src="https://github.com/user-attachments/assets/673e18d6-8128-461c-aca1-34cf27b70d78" />
<img width="700" height="450" alt="rating_distribution" src="https://github.com/user-attachments/assets/ba6f26ee-07a8-4982-9e19-ead6c4ec9ff4" />

### Review volume by store

Review volume is concentrated in a handful of high-traffic locations, mostly near tourist areas.

<img width="800" height="500" alt="top_stores" src="https://github.com/user-attachments/assets/6dbea8a6-a796-4825-8310-3706e9ed083a" />

### Choosing the number of topics

Coherence score alone favoured an uninformatively small number of topics (2-4). The final choice of **k=7** balanced coherence with topic interpretability, confirmed by manually reading topic content at each candidate value.

<img width="700" height="450" alt="review_length_distribution" src="https://github.com/user-attachments/assets/fc2efe0c-86be-4ee7-ac86-4976ae996023" />

### Word cloud of most frequent terms

A quick visual sense of the most common words across all cleaned reviews.

<img width="2400" height="1050" alt="wordcloud_positive_vs_negative" src="https://github.com/user-attachments/assets/cebfcb07-7591-4dad-a3b3-835ccdce1aec" />
<img width="1800" height="1050" alt="wordcloud_all_reviews" src="https://github.com/user-attachments/assets/4f57ae0a-4f05-4668-ac1f-173adb49ca04" />

### Reviews per topic

Order Delays & Follow-Up and Fast/Clean/Friendly Service are by far the largest topics, together covering the majority of all reviews.

<img width="1200" height="1400" alt="topic_top_words" src="https://github.com/user-attachments/assets/cf4dac64-4746-46f0-8ea0-aba89e25eab1" />

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

The notebook downloads the required NLTK corpora (`stopwords`, `wordnet`, `punkt`, `averaged_perceptron_tagger`) automatically on first run.
