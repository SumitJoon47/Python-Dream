"""
Dream Analysis System - Production Ready
Author: Your Name
Version: 1.0.0
Dataset: DreamBank (20,000+ dreams)

BEFORE RUNNING:
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud textblob requests
python -m spacy download en_core_web_sm
"""

import pandas as pd
import numpy as np
import re
import warnings
import os
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# Configuration
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Create output directories
os.makedirs('outputs/visualizations', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'num_dreams': 200,  # Number of dreams to analyze
    'n_clusters': 5,  # Number of clusters
    'n_topics': 5,  # Number of topics
    'random_state': 42,
    'min_dream_length': 50,  # Minimum words per dream
    'max_tfidf_features': 500,
}


# ==============================================================================
# Step 1: Data Loading
# ==============================================================================
def load_dreambank_github(num_dreams=200):
    """Load dreams from DreamBank GitHub repository"""
    base_url = "https://raw.githubusercontent.com/mattbierner/DreamScrape/master/dreams/"

    dreamers = [
        'barb_sanders', 'emma', 'izzy', 'jasmine1', 'jeff',
        'kenneth', 'pegasus', 'vietnam_vet'
    ]

    all_dreams = []
    dreams_per_dreamer = num_dreams // len(dreamers)

    print("=" * 80)
    print("LOADING DREAM DATA FROM DREAMBANK")
    print("=" * 80)

    for dreamer in dreamers:
        try:
            url = f"{base_url}{dreamer}.json"
            print(f"Fetching dreams from {dreamer}...", end=" ")
            response = requests.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()
                dreams = data.get('dreams', [])

                for dream in dreams[:dreams_per_dreamer]:
                    content = dream.get('content', '')
                    if content and len(content.split()) >= CONFIG['min_dream_length']:
                        all_dreams.append({
                            'raw_dream': content,
                            'dreamer': data.get('dreamer', 'unknown'),
                            'date': dream.get('head', 'unknown')
                        })
                print(f"✓ Loaded {len(dreams[:dreams_per_dreamer])} dreams")
            else:
                print(f"✗ Failed (HTTP {response.status_code})")

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            continue

    print(f"\n✓ Total dreams loaded: {len(all_dreams)}")
    return all_dreams


# ==============================================================================
# Step 2: Text Preprocessing
# ==============================================================================
def preprocess_text(text):
    """Clean and preprocess dream text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)

    stopwords = {
        'i', 'was', 'were', 'is', 'am', 'are', 'a', 'an', 'the', 'in', 'of',
        'to', 'and', 'my', 'me', 'but', 'had', 'have', 'has', 'it', 'that',
        'with', 'for', 'on', 'at', 'be', 'been', 'being', 'we', 'they', 'them'
    }
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text


# ==============================================================================
# Step 3: Feature Engineering
# ==============================================================================
def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    try:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except:
        return {'polarity': 0, 'subjectivity': 0}


def tag_dream(text):
    """Tag dreams with psychological themes"""
    emotion_dict = {
        'Anxiety': ['exam', 'late', 'test', 'forgot', 'unprepared', 'worry', 'nervous'],
        'Fear/Chase': ['chasing', 'chased', 'running', 'escape', 'monster', 'danger'],
        'Flying/Freedom': ['flying', 'soaring', 'floating', 'gliding', 'sky', 'clouds'],
        'Water': ['water', 'ocean', 'sea', 'swimming', 'drowning', 'waves'],
        'Death': ['death', 'died', 'dying', 'funeral', 'grave', 'dead'],
        'Social': ['people', 'friends', 'family', 'talking', 'party', 'crowd'],
    }

    tags = []
    for tag, keywords in emotion_dict.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags if tags else ['Uncategorized']


def detect_symbols(text):
    """Detect recurring dream symbols"""
    symbol_keywords = {
        'Animals': ['dog', 'cat', 'bird', 'snake', 'spider', 'horse', 'fish'],
        'Family': ['mother', 'father', 'mom', 'dad', 'sister', 'brother'],
        'Vehicles': ['car', 'bus', 'train', 'plane', 'ship'],
        'Nature': ['tree', 'mountain', 'forest', 'field', 'garden'],
        'Body': ['hand', 'eye', 'face', 'hair', 'teeth', 'blood']
    }

    found = []
    text_lower = text.lower()
    for category, keywords in symbol_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found.append(category)
    return found


# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================
def main():
    print("\n" + "=" * 80)
    print("DREAM ANALYSIS SYSTEM - STARTING")
    print("=" * 80 + "\n")

    # Load data
    dreams = load_dreambank_github(num_dreams=CONFIG['num_dreams'])
    if len(dreams) == 0:
        print("ERROR: No dreams loaded. Check your internet connection.")
        return

    df = pd.DataFrame(dreams)

    # Preprocessing
    print("\n" + "=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    df['processed_dream'] = df['raw_dream'].apply(preprocess_text)
    print("✓ Text preprocessing complete")

    # Feature engineering
    print("✓ Extracting features...")
    df['sentiment'] = df['raw_dream'].apply(analyze_sentiment)
    df['polarity'] = df['sentiment'].apply(lambda x: x['polarity'])
    df['subjectivity'] = df['sentiment'].apply(lambda x: x['subjectivity'])
    df['dream_length'] = df['raw_dream'].apply(lambda x: len(str(x).split()))
    df['tags'] = df['processed_dream'].apply(tag_dream)
    df['symbols'] = df['processed_dream'].apply(detect_symbols)

    # TF-IDF Vectorization
    print("✓ Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=CONFIG['max_tfidf_features'],
        min_df=2,
        max_df=0.8
    )
    X = vectorizer.fit_transform(df['processed_dream'])

    # Clustering
    print("✓ Performing K-Means clustering...")
    kmeans = KMeans(
        n_clusters=CONFIG['n_clusters'],
        random_state=CONFIG['random_state'],
        n_init=10
    )
    df['cluster'] = kmeans.fit_predict(X)

    # Topic Modeling
    print("✓ Running LDA topic modeling...")
    lda = LatentDirichletAllocation(
        n_components=CONFIG['n_topics'],
        random_state=CONFIG['random_state'],
        max_iter=20
    )
    lda_matrix = lda.fit_transform(X)
    df['dominant_topic'] = lda_matrix.argmax(axis=1)

    # ==============================================================================
    # VISUALIZATION
    # ==============================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Set style
    sns.set_style("whitegrid")

    # 1. Cluster Plot (PCA)
    print("1/9 Creating cluster plot...", end=" ")
    pca = PCA(n_components=2, random_state=CONFIG['random_state'])
    X_pca = pca.fit_transform(X.toarray())
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', palette='husl',
                    data=df, s=100, alpha=0.6, edgecolor='black', linewidth=0.5)
    plt.title('Dream Clusters (PCA Visualization)', fontsize=16, fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig('outputs/visualizations/01_cluster_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")

    # 2. Word Clouds
    print("2/9 Creating word clouds...", end=" ")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(CONFIG['n_clusters']):
        cluster_text = ' '.join(df[df['cluster'] == i]['processed_dream'])
        if cluster_text.strip():
            wordcloud = WordCloud(width=600, height=400, background_color='white',
                                  colormap='viridis').generate(cluster_text)
            axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Cluster {i} ({len(df[df["cluster"] == i])} dreams)',
                          fontsize=12, fontweight='bold')
        axes[i].axis('off')
    axes[5].axis('off')

    plt.suptitle('Word Clouds per Cluster', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/visualizations/02_word_clouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")

    # 3. Psychological Tags
    print("3/9 Creating tag frequency chart...", end=" ")
    all_tags = [tag for tags_list in df['tags'] for tag in tags_list]
    tag_counts = pd.Series(all_tags).value_counts()

    plt.figure(figsize=(10, 6))
    tag_df = pd.DataFrame({'Tag': tag_counts.index, 'Count': tag_counts.values})
    sns.barplot(data=tag_df, x='Count', y='Tag', hue='Tag', palette='rocket', legend=False)
    plt.title('Psychological Themes in Dreams', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency')
    plt.ylabel('Theme')
    plt.tight_layout()
    plt.savefig('outputs/visualizations/03_psychological_tags.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")

    # 4. Sentiment Analysis
    print("4/9 Creating sentiment analysis...", end=" ")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    df.boxplot(column='polarity', by='cluster', ax=ax1)
    ax1.set_title('Sentiment Polarity by Cluster')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Polarity')

    df.boxplot(column='subjectivity', by='cluster', ax=ax2)
    ax2.set_title('Subjectivity by Cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Subjectivity')

    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('outputs/visualizations/04_sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")

    # 5. Dream Length Distribution
    print("5/9 Creating length distribution...", end=" ")
    plt.figure(figsize=(10, 6))
    plt.hist(df['dream_length'], bins=30, edgecolor='black', color='teal', alpha=0.7)
    plt.axvline(df['dream_length'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(df['dream_length'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
    plt.title('Dream Length Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/visualizations/05_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")

    # 6. Similarity Heatmap
    print("6/9 Creating similarity heatmap...", end=" ")
    similarity_matrix = cosine_similarity(X[:50])

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='YlOrRd', square=True)
    plt.title('Dream Similarity Heatmap (First 50 Dreams)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/visualizations/06_similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")

    # 7. Topic Distribution
    print("7/9 Creating topic distribution...", end=" ")
    topic_counts = df['dominant_topic'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(topic_counts.index, topic_counts.values, color='coral', edgecolor='black')
    plt.title('Topic Distribution Across Dreams', fontsize=16, fontweight='bold')
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Dreams')
    plt.xticks(range(CONFIG['n_topics']))
    plt.tight_layout()
    plt.savefig('outputs/visualizations/07_topic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")

    # 8. Symbol Frequency
    print("8/9 Creating symbol frequency chart...", end=" ")
    all_symbols = [sym for sym_list in df['symbols'] for sym in sym_list]
    symbol_counts = pd.Series(all_symbols).value_counts()

    if len(symbol_counts) > 0:
        plt.figure(figsize=(10, 6))
        symbol_df = pd.DataFrame({'Symbol': symbol_counts.index, 'Count': symbol_counts.values})
        sns.barplot(data=symbol_df, x='Count', y='Symbol', hue='Symbol', palette='mako', legend=False)
        plt.title('Dream Symbol Frequency', fontsize=16, fontweight='bold')
        plt.xlabel('Frequency')
        plt.ylabel('Symbol Category')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/08_symbol_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    print("✓")

    # 9. Cluster Distribution
    print("9/9 Creating cluster distribution...", end=" ")
    cluster_counts = df['cluster'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='black')
    plt.title('Dreams per Cluster', fontsize=16, fontweight='bold')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Dreams')
    plt.xticks(range(CONFIG['n_clusters']))
    plt.tight_layout()
    plt.savefig('outputs/visualizations/09_cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓")

    # ==============================================================================
    # GENERATE REPORT
    # ==============================================================================
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 80)

    # Top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topics_str = ""
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics_str += f"\nTopic {topic_idx}: {', '.join(top_words)}"

    report = f"""
{'=' * 80}
DREAM ANALYSIS COMPREHENSIVE REPORT
{'=' * 80}

DATASET OVERVIEW:
- Total dreams analyzed: {len(df):,}
- Total words processed: {df['dream_length'].sum():,}
- Unique dreamers: {df['dreamer'].nunique()}
- Date range: {df['date'].min()} to {df['date'].max()}

CLUSTERING RESULTS:
- Number of clusters: {CONFIG['n_clusters']}
- Cluster sizes: {dict(df['cluster'].value_counts().sort_index())}

SENTIMENT ANALYSIS:
- Average polarity: {df['polarity'].mean():.3f} (-1=negative, 0=neutral, +1=positive)
- Average subjectivity: {df['subjectivity'].mean():.3f} (0=objective, 1=subjective)
- Most positive cluster: Cluster {df.groupby('cluster')['polarity'].mean().idxmax()}
- Most negative cluster: Cluster {df.groupby('cluster')['polarity'].mean().idxmin()}

DREAM CHARACTERISTICS:
- Average length: {df['dream_length'].mean():.0f} words
- Shortest dream: {df['dream_length'].min()} words
- Longest dream: {df['dream_length'].max()} words

TOP PSYCHOLOGICAL THEMES:
{tag_counts.head(10).to_string()}

TOP DREAM SYMBOLS:
{symbol_counts.head(10).to_string() if len(symbol_counts) > 0 else 'No symbols detected'}

TOPIC MODELING (LDA):
{topics_str}

{'=' * 80}
Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""

    # Save report
    with open('outputs/reports/dream_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)

    # Save results CSV
    df_export = df.drop(columns=['sentiment'], errors='ignore')
    df_export.to_csv('outputs/results/dream_analysis_results.csv', index=False)

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 9 visualizations in: outputs/visualizations/")
    print("  📄 Report in: outputs/reports/dream_analysis_report.txt")
    print("  💾 Results CSV in: outputs/results/dream_analysis_results.csv")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()