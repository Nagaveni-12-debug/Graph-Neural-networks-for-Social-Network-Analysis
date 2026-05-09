import pandas as pd
import numpy as np
import re
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def load_fakenewsnet(base_path, subset='politifact', sample_size=500):
    real_df = pd.read_csv(os.path.join(base_path, f'{subset}_real.csv'))
    fake_df = pd.read_csv(os.path.join(base_path, f'{subset}_fake.csv'))
    
    real_df = real_df.sample(min(sample_size, len(real_df)), random_state=42)
    fake_df = fake_df.sample(min(sample_size, len(fake_df)), random_state=42)
    
    real_df['label'] = 1
    fake_df['label'] = 0
    
    df = pd.concat([real_df, fake_df]).reset_index(drop=True)
    df['title_clean'] = df['title'].apply(clean_text)
    
    # Feature Extraction for News
    vectorizer = TfidfVectorizer(max_features=128)
    news_features = vectorizer.fit_transform(df['title_clean']).toarray()
    
    # Extract edges (News -> Tweet)
    news_nodes = []
    tweet_nodes = []
    tweet_id_map = {} # Map global tweet_id to node index
    current_tweet_idx = 0
    
    edges = []
    for i, row in df.iterrows():
        t_ids = str(row['tweet_ids']).split('\t')
        for tid in t_ids:
            if tid.strip():
                if tid not in tweet_id_map:
                    tweet_id_map[tid] = current_tweet_idx
                    current_tweet_idx += 1
                edges.append([i, tweet_id_map[tid]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return news_features, df['label'].values, edge_index, current_tweet_idx

if __name__ == "__main__":
    path = r'c:\Users\Rohan\Desktop\GNN\dataset.csv\FakeNewsNet-master\FakeNewsNet-master\dataset'
    feats, labels, edges, num_tweets = load_fakenewsnet(path)
    print(f"News: {len(feats)}, Tweets: {num_tweets}, Edges: {edges.shape[1]}")
