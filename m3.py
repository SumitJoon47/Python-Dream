# ==============================================================================
# Dream Text Analysis (Freud-inspired Version)
# ==============================================================================

import pandas as pd
import numpy as np
import re, os, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# ==============================================================================
# Step 1: Dream Dataset
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
# Step 2: Preprocessing (Manifest Layer)
# ==============================================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stopwords = {'i', 'was', 'a', 'an', 'the', 'in', 'of', 'to', 'and', 'my', 'me', 'but', 'had'}
    return ' '.join([w for w in text.split() if w not in stopwords])

df['manifest_content'] = df['raw_dream'].apply(preprocess_text)

# ==============================================================================
# Step 3: Freud’s Symbolic Emotion Dictionary (Latent Layer)
# ==============================================================================
emotion_dict = {
    'Anxiety / Failure': ['exam', 'late', 'classroom', 'presentation', 'forgot', 'failed', 'test', 'notes'],
    'Fear / Chase': ['chasing', 'chased', 'running', 'monster', 'shadowy', 'dread', 'dog', 'maze'],
    'Body / Control': ['teeth', 'falling', 'crumbling', 'out'],
    'Freedom / Escape': ['flying', 'soaring', 'clouds', 'forest', 'sky'],
    'Wish Fulfillment': ['beautiful', 'success', 'freedom', 'happy', 'love']
}

def tag_emotions(text):
    tags = [emo for emo, words in emotion_dict.items() if any(word in text for word in words)]
    return tags if tags else ['Uncategorized']

df['latent_emotion'] = df['manifest_content'].apply(tag_emotions)

# ==============================================================================
# Step 4: Vectorization & Clustering
# ==============================================================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['manifest_content'])
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Auto-label clusters using top TF-IDF terms
terms = np.array(vectorizer.get_feature_names_out())
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
cluster_labels = []
for i in range(k):
    top_words = ', '.join(terms[order_centroids[i, :3]])
    cluster_labels.append(f"Cluster {i}: {top_words}")
df['cluster_label'] = df['cluster'].apply(lambda x: cluster_labels[x])

# ==============================================================================
# Step 5: Visualization
# ==============================================================================
pca = PCA(n_components=2, random_state=42)
df[['pca1', 'pca2']] = pca.fit_transform(X.toarray())

sns.set(style='whitegrid')
plt.figure(figsize=(10, 7))
sns.scatterplot(x='pca1', y='pca2', hue='cluster_label', palette='cool', data=df, s=150)
plt.title('Freudian Dream Clusters (PCA Projection)')
plt.savefig('freud_clusters.png')
plt.close()

# WordClouds
fig, axes = plt.subplots(1, k, figsize=(20, 6))
for i in range(k):
    text = ' '.join(df[df['cluster'] == i]['manifest_content'])
    wc = WordCloud(width=600, height=400, background_color='white').generate(text)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].set_title(cluster_labels[i])
    axes[i].axis('off')
plt.savefig('freud_wordclouds.png')
plt.close()

# ==============================================================================
# Step 6: Interpretation Summary
# ==============================================================================
def freudian_interpretation(tags):
    if 'Fear / Chase' in tags:
        return "Represents avoidance or internal conflict — typical anxiety dream."
    elif 'Freedom / Escape' in tags:
        return "Symbolizes liberation, desire for control or escape from stress."
    elif 'Anxiety / Failure' in tags:
        return "Reflects fear of inadequacy or performance anxiety."
    elif 'Body / Control' in tags:
        return "Indicates fear of loss of control or self-image."
    elif 'Wish Fulfillment' in tags:
        return "Suggests subconscious satisfaction of a hidden desire."
    else:
        return "Could represent mixed emotional or symbolic content."

df['interpretation'] = df['latent_emotion'].apply(lambda x: freudian_interpretation(x))

# ==============================================================================
# Step 7: Output Summary
# ==============================================================================
print("\n=== Freud-Inspired Dream Analysis ===")
print(df[['raw_dream', 'latent_emotion', 'cluster_label', 'interpretation']])
print("\nVisuals saved as: 'freud_clusters.png' and 'freud_wordclouds.png'")
