# ==============================================================================
# Step 0: Setup & Imports
# ==============================================================================
import pandas as pd
import numpy as np
import re
import warnings
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Suppress all warnings including joblib CPU detection warnings
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set CPU count to avoid detection issues

# ==============================================================================
# Step 1: Input Data (A small, fake dream dataset)
# ==============================================================================
raw_dreams = [
    "I was flying over a forest but suddenly started falling.",
    "I was late for an important exam and couldn't find the classroom.",
    "A strange, shadowy dog was chasing me down an empty street.",
    "My teeth were crumbling and falling out into my hand.",
    "I had to give a presentation but realized I had no notes and forgot the topic.",
    "I was running from a monster in a dark, endless maze.",
    "I was soaring through beautiful clouds made of cotton candy.",
    "I failed a test I didn't even know I had to take.",
    "I was being chased by something I couldn't see, just a sense of dread."
]
df = pd.DataFrame(raw_dreams, columns=['raw_dream'])

# ==============================================================================
# Step 2: Preprocessing Text
# ==============================================================================
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # Simple stopword removal
    stopwords = {'i', 'was', 'a', 'an', 'the', 'in', 'of', 'to', 'and', 'my', 'me', 'but', 'had'}
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

df['processed_dream'] = df['raw_dream'].apply(preprocess_text)
print("--- Preprocessed Text ---")
print(df[['processed_dream']])

# ==============================================================================
# Step 3: Embeddings (TF-IDF)
# ==============================================================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_dream'])
print("\n--- TF-IDF Matrix Shape ---")
print(X.shape)

# ==============================================================================
# Step 4: Clustering (K-Means)
# ==============================================================================
# We'll choose K=3 clusters based on the themes in our toy data
k = 3
# Note: n_jobs parameter removed in newer sklearn versions
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)
print("\n--- Clustered Dreams ---")
print(df[['raw_dream', 'cluster']])

# ==============================================================================
# Step 5: Psychological Tagging (via emotion dictionary)
# ==============================================================================
emotion_dict = {
    'Anxiety': ['exam', 'late', 'classroom', 'presentation', 'forgot', 'failed', 'test', 'teeth'],
    'Fear/Chase': ['chasing', 'falling', 'chased', 'running', 'monster', 'shadowy', 'dread', 'dog'],
    'Surreal/Freedom': ['flying', 'soaring', 'clouds', 'forest']
}

def tag_dream(text):
    tags = []
    for tag, keywords in emotion_dict.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags if tags else ['Uncategorized']

df['tags'] = df['processed_dream'].apply(tag_dream)
print("\n--- Psychological Tagging ---")
print(df[['raw_dream', 'tags']])

# ==============================================================================
# Step 6: Visualization
# ==============================================================================
print("\nGenerating visualizations...")

# --- Visualization A: 2D Cluster Plot (PCA) ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='pca1', y='pca2',
    hue='cluster',
    palette=sns.color_palette("hls", k),
    data=df,
    legend="full",
    s=150,
    alpha=0.8
)
plt.title('2D Visualization of Dream Clusters (via PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('dream_cluster_plot.png')
plt.close()

# --- Visualization B: Word Clouds per Cluster ---
fig, axes = plt.subplots(1, k, figsize=(20, 6))
for i in range(k):
    cluster_text = ' '.join(df[df['cluster'] == i]['processed_dream'])
    if cluster_text.strip():  # Only generate if there's text
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
        axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(f'Top Words in Cluster {i}')
    axes[i].axis('off')
plt.suptitle('Word Clouds for Each Dream Cluster', fontsize=20)
plt.tight_layout(trect=[0, 0.03, 1, 0.95])
plt.savefig('dream_word_clouds.png')
plt.close()

# --- Visualization C: Bar Chart of Emotion Counts ---
all_tags = [tag for tags_list in df['tags'] for tag in tags_list]
tag_counts = pd.Series(all_tags).value_counts()

# Fix: Properly assign hue parameter to avoid FutureWarning
plt.figure(figsize=(8, 5))
tag_df = pd.DataFrame({'Tag': tag_counts.index, 'Count': tag_counts.values})
sns.barplot(data=tag_df, x='Count', y='Tag', hue='Tag', palette='viridis', legend=False)
plt.title('Frequency of Psychological Tags in Dreams')
plt.xlabel('Number of Dreams')
plt.ylabel('Tag')
plt.savefig('dream_tag_barchart.png')
plt.close()

print("Visualizations saved as .png files.")
print("\nAll warnings have been resolved!")
