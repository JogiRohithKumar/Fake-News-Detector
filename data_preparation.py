import pandas as pd

# Load datasets
fake_df = pd.read_csv('data/fake.csv')
true_df = pd.read_csv('data/true.csv')

# Label them
fake_df['label'] = 0
true_df['label'] = 1

# Combine
df = pd.concat([fake_df, true_df], ignore_index=True)

# Keep only title + text + label
df = df[['title', 'text', 'label']]
df['content'] = df['title'] + " " + df['text']

# Save combined dataset
df.to_csv('data/news_combined.csv', index=False)
print("Combined data saved to data/news_combined.csv")
