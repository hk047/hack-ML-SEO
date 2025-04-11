# Part 2: Sentiment Analysis with Machine Learning

Welcome back to the Nomura x SEO Hackathon! In Part 1, you built a machine learning model to classify news articles into subtopics like "AI" or "Mental Health" based on their titles and descriptions. Now, in Part 2, we’re diving into sentiment analysis—a way to uncover the mood or tone behind text. Imagine figuring out if a news article is cheerful, gloomy, or neutral just by analyzing its words. That’s what you’ll do here, and it’s a skill banks and companies use to understand customer feedback, track news trends, and more.

## What You’ll Learn

- What sentiment analysis is and why it matters.
- Real-world examples of sentiment analysis.
- Our mission: analyzing news article sentiment by subtopic.
- You’ll use three tools: TextBlob, Hugging Face Transformers, and VADER.

---

### What Is Sentiment Analysis?

Sentiment analysis is like giving a computer a mood ring to read text. It figures out if the text is positive, negative, or neutral. For example:

**Positive**: "The company’s profits soared beyond expectations."
**Negative**: "The economy is collapsing, and unemployment is rising."
**Neutral**: "The meeting will take place tomorrow at 10 AM."
Think of it as teaching the computer to spot the emotional vibe behind words—a bit like guessing how a friend feels from their message.

---

### Why It’s Cool

Sentiment analysis is everywhere in the real world:

- **Customer Reviews**: Companies check if people love or hate their products.
- **Social Media**: Brands track posts to see how people feel about them.
- **Financial News**: Investors analyze news to predict market moves—like whether positive headlines might boost stocks.
In this challenge, you’ll analyze news article descriptions to see which subtopics (like "AI" or "Mental Health") get more positive or negative coverage. This could help a bank understand public perception and make smarter decisions.

---

## Step 2: Load and Explore the Dataset

### What You’ll Learn

- How to load data with Python.
- How to explore and visualize it.
- Why data distribution matters.


### Step 1: Load the Data

We’ll use the same news article dataset from Part 1, with columns like "Title," "Description," and "Subtopic." Let’s load it using pandas.

```python
# Import pandas to handle tables of data
import pandas as pd

# Load the CSV file into a dataframe (like a spreadsheet)
df = pd.read_csv('news_articles.csv')  # Upload your file to Colab first!

# Show the first 5 rows to peek at the data
df.head()
```

**What’s Happening?**

- **`import pandas as pd`**: Brings in pandas and calls it "pd" for short.
- **`pd.read_csv('news_articles.csv')`**: Reads the file into a dataframe—a table where each row is an article.
- **`df.head()`**: Displays the first 5 rows. Look at the "Description" column—that’s our focus for sentiment.

---

### Step 2: Explore the Data

Let’s check how many articles each subtopic has to spot any imbalances.

```python
# Count articles per subtopic
subtopic_counts = df['Subtopic'].value_counts()

# Print the counts
print("Articles per Subtopic:")
print(subtopic_counts)
```

**What’s Happening?**

- **`df['Subtopic'].value_counts()`**: Counts articles per subtopic (e.g., "AI: 50, Mental Health: 30").
- Are some subtopics more common? That could affect our results.

---

### Step 3: Visualize the Distribution

A bar chart makes this easier to see—pictures beat numbers any day!

```python
# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean style for our plots
sns.set_style("whitegrid")

# Create a bar chart
plt.figure(figsize=(10, 6))  # Size: 10 inches wide, 6 tall
sns.barplot(x=subtopic_counts.index, y=subtopic_counts.values)
plt.title('Number of Articles per Subtopic')
plt.xlabel('Subtopic')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)  # Tilt labels to avoid overlap
plt.show()
```

**What’s Happening?**

- **`sns.barplot(...)`**: Draws bars with subtopics on the x-axis and counts on the y-axis.
- Check the chart—do some subtopics dominate? That’s data imbalance to watch for.

---

### Reflection Prompt

What issues might arise if one subtopic has way fewer articles?
(Hint: Could it make sentiment less reliable? Why?)

---

## Step 3: Sentiment Analysis with TextBlob (Beginner-Friendly)

### What You’ll Learn

- How to use TextBlob for sentiment analysis.
- What polarity and subjectivity mean.
- How to apply it to our dataset.


### Why TextBlob?

TextBlob is a simple tool that’s great for beginners. It gives us two scores:

- **Polarity**: From -1 (very negative) to 1 (very positive). 0 is neutral—like a mood scale.
- **Subjectivity**: From 0 (factual) to 1 (opinion-based)—how much personal feeling is in the text.
It’s like a quick emotional scanner!

---

### Step 1: Install and Import TextBlob

Let’s set it up.

```python
# Install TextBlob
!pip install textblob

# Import TextBlob
from textblob import TextBlob
```

---

### Step 2: Test TextBlob on Examples

Let’s try it on some sentences.

```python
# Example sentences
sentences = [
    "The company's profits soared beyond expectations.",  # Positive
    "The economy is collapsing, and unemployment is rising.",  # Negative
    "The meeting will take place tomorrow at 10 AM."  # Neutral
]

# Analyze each sentence
for sentence in sentences:
    blob = TextBlob(sentence)
    print(f"Sentence: {sentence}")
    print(f"Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}\n")
```

**What’s Happening?**

- **`TextBlob(sentence)`**: Turns the sentence into a TextBlob object.
- **`blob.sentiment.polarity`**: Gets the polarity score (-1 to 1).
- Do the scores match the moods you’d expect?

---

### Step 3: Apply TextBlob to the Dataset

Now, let’s analyze the "Description" column.

```python
# Function to get polarity
def get_textblob_polarity(text):
    return TextBlob(text).sentiment.polarity

# Add a new column with polarity scores
df['TextBlob_Polarity'] = df['Description'].apply(get_textblob_polarity)

# Show the first few rows
df[['Description', 'TextBlob_Polarity']].head()
```

**What’s Happening?**

- **`get_textblob_polarity(text)`**: A function that returns the polarity score for a given text.
- **`df['Description'].apply(get_textblob_polarity)`**: Applies this function to each description in the dataframe.

---

### Challenge Spot

We stored polarity in "TextBlob_Polarity." What about subjectivity?
Your Task: Add a column called "TextBlob_Subjectivity" with subjectivity scores.

(Hint: Define a function like `get_textblob_polarity` but for subjectivity, then apply it.)

---

### Reflection Prompt

Where might TextBlob fail?
(Think about sarcasm—like "Great, another Monday!"—or complex phrases. Could it misread the mood? Why? Share your thoughts and demonstrate your understanding of the model!)

---

## Step 4: Sentiment Analysis with Hugging Face Transformers (Advanced)

### What You’ll Learn

- How to use Transformers for sentiment analysis.
- How it differs from TextBlob.
- How to handle complex text.


### Why Transformers?

Transformers are like AI superheroes. They use models like BERT to understand context—like spotting sarcasm—better than TextBlob.

---

### Step 1: Install and Import Transformers

Let’s get it ready.

```python
# Install transformers
!pip install transformers

# Import the sentiment analysis pipeline
from transformers import pipeline
```

---

### Step 2: Set Up the Pipeline

We’ll use the default pipeline.

```python
# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
```

---

### Step 3: Test on Examples

Let’s compare it to TextBlob.

```python
# Same example sentences
sentences = [
    "The company's profits soared beyond expectations.",
    "The economy is collapsing, and unemployment is rising.",
    "The meeting will take place tomorrow at 10 AM."
]

# Analyze each sentence
for sentence in sentences:
    result = sentiment_pipeline(sentence)
    print(f"Sentence: {sentence}")
    print(f"Label: {result[^0]['label']}, Confidence: {result[^0]['score']:.2f}\n")
```

**What’s Happening?**

- **`result['label']`**: Gets POSITIVE or NEGATIVE.
- How do these differ from TextBlob?

---

### Step 4: Apply to the Dataset

Let’s analyze a sample (Transformers are slow on big data).

```python
# Sample 100 articles
sample_df = df.sample(100, random_state=42)

# Function to get sentiment and score
def get_transformer_sentiment(text):
    result = sentiment_pipeline(text)[^0]
    return result['label'], result['score']

# Add columns
sample_df['Transformer_Sentiment'], sample_df['Transformer_Score'] = zip(*sample_df['Description'].apply(get_transformer_sentiment))

# Show results
sample_df[['Description', 'Transformer_Sentiment', 'Transformer_Score']].head()
```

**What’s Happening?**

- **`get_transformer_sentiment(text)`**: Returns the sentiment label and confidence score.
- **`zip(*sample_df['Description'].apply(get_transformer_sentiment))`**: Unpacks the results into two columns.

---

### Extension Challenge

Test the Transformer on tricky text like "Oh great, another Monday!"
Your Task: Analyze two sarcastic or ambiguous sentences and note your observations in a markdown cell. This would be great to share later on and talk about.

Try this sarcastic analysis on TextBlob and try to create a graph to show how they compare. It would be excellent to show this analysis and why you think it is happening.

---

## Step 5: Bonus: Sentiment Analysis with VADER

### What You’ll Learn

- How VADER handles punctuation, capitalization, and emphasis.
- How it compares to TextBlob and Transformers.
- How to add a new model to your analysis.


### Why VADER?

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based tool designed for social media and short texts. It’s great at picking up on:

- **Punctuation**: "Great!!!" vs. "Great."
- **Capitalization**: "AWESOME" vs. "awesome"
- **Degree Modifiers**: "very good" vs. "good"
It’s like a detective for text emphasis!

---

### Step 1: Install and Import VADER

Let’s set it up.

```python
# Install VADER
!pip install vaderSentiment

# Import VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create a VADER analyzer object
vader_analyzer = SentimentIntensityAnalyzer()
```

---

### Step 2: Test VADER on Examples

Let’s see how it handles emphasis.

```python
# Test sentences
vader_sentences = [
    "The product is good.",
    "The product is GOOD!",
    "The product is very good!!",
    "The product is terrible..."
]

# Analyze each sentence
for sentence in vader_sentences:
    scores = vader_analyzer.polarity_scores(sentence)
    print(f"Sentence: {sentence}")
    print(f"Scores: {scores}\n")
```

**What’s Happening?**

- **`polarity_scores`**: Returns a dictionary with:
    - `pos`: Positive score (0 to 1)
    - `neg`: Negative score (0 to 1)
    - `neu`: Neutral score (0 to 1)
    - `compound`: Overall score (-1 to 1)
- Notice how "GOOD!" and "very good!!" get different scores than "good."

---

### Step 3: Apply VADER to the Dataset

Let’s add VADER to our sample.

```python
# Function to get VADER compound score
def get_vader_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

# Add a new column
sample_df['VADER_Sentiment'] = sample_df['Description'].apply(get_vader_sentiment)

# Show results
sample_df[['Description', 'TextBlob_Polarity', 'Transformer_Sentiment', 'VADER_Sentiment']].head()
```

**What’s Happening?**

- **`get_vader_sentiment(text)`**: Returns the compound score for a given text.
- **`apply(get_vader_sentiment)`**: Applies this function to each description.

---

### Challenge Spot

Your Task: Add VADER scores to the full dataset (`df`), not just the sample, and create a histogram of `VADER_Sentiment` scores.

(Hint: Use `sns.histplot` like we did earlier for TextBlob.)

---

### Reflection Prompt

How does VADER’s handling of emphasis change the results compared to TextBlob or Transformers?
(Think about news headlines with exclamation points or all caps—could VADER catch something the others miss?)

---

## Step 6: Compare and Visualize the Results

### What You’ll Learn

- How to compare three models.
- How to create clear visualizations.
- Why models might disagree.


### Step 1: Compare Scores

Let’s compare all three in our sample.

```python
# Map Transformer labels to numbers
sample_df['Transformer_Numerical'] = sample_df['Transformer_Sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1})

# Compare correlations
comparison_df = sample_df[['TextBlob_Polarity', 'Transformer_Numerical', 'VADER_Sentiment']]
correlation = comparison_df.corr()
print("Correlation between models:")
print(correlation)
```

**What’s Happening?**

- **`map({'POSITIVE': 1, 'NEGATIVE': -1})`**: Converts labels to numerical values for comparison.
- **`corr()`**: Calculates how closely the models agree.

---

### Step 2: Improved Visualization

The original scatter plot was tricky to read, so let’s use a bar chart to compare average sentiment per subtopic across models.

```python
# Average sentiment by subtopic for all models
subtopic_comparison = sample_df.groupby('Subtopic')[['TextBlob_Polarity', 'Transformer_Numerical', 'VADER_Sentiment']].mean()

# Plot side by side
plt.figure(figsize=(12, 8))
subtopic_comparison.plot(kind='bar', width=0.8)
plt.title('Average Sentiment by Subtopic Across Models')
plt.xlabel('Subtopic')
plt.ylabel('Sentiment Score (-1 to 1)')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.tight_layout()
plt.show()
```

**What’s Happening?**

- **`groupby('Subtopic')[...].mean()`**: Averages scores per subtopic for each model.
- This bar chart is clearer than a scatter plot—each subtopic gets three bars (one per model).

---

### Interactive Challenge

Your Task: Customize the bar chart by choosing the column name for one model’s scores. Replace `'TextBlob_Polarity'` with a variable you define (e.g., `my_column = 'VADER_Sentiment'`), then rerun the plot.

(Hint: Pay attention to how the column names match the dataframe!)

---

### Step 3: Sentiment Distribution

Let’s see the spread of TextBlob scores.

```python
# Histogram of TextBlob polarity
plt.figure(figsize=(10, 6))
sns.histplot(df['TextBlob_Polarity'], kde=True)
plt.title('Distribution of TextBlob Sentiment Polarity')
plt.xlabel('Polarity (-1 = Negative, +1 = Positive)')
plt.axvline(x=0, color='red', linestyle='--')
plt.show()
```

---

### Reflection Prompt

Why might the three models disagree on a subtopic’s sentiment?
(Consider context, emphasis, or how they interpret neutral text.)

---

## Step 7: Final Reflections and Discussion

### What You’ve Done

You’ve analyzed news article sentiment with:

1. **TextBlob**: Simple polarity and subjectivity scores.
2. **Hugging Face Transformers**: Context-aware labels.
3. **VADER**: Emphasis-sensitive scores.
You’ve visualized and compared them—amazing work!

### Think About It

Which model seemed most reliable? Why?
Did any subtopic’s sentiment surprise you?
How could sentiment analysis help in real life—like in finance or marketing?

#### Other Considerations

**Limitations**: What are the limitations of our analysis?

- Can we really reduce complex emotions in text to a single number?
- How reliable are these models for high-stakes decisions?

**Content Differences**: Could the differences in sentiment be due to inherent properties of the topics rather than media bias?

- Some topics (like disasters) are inherently negative
- Other topics (like technology innovations) might be inherently positive

---

## Final Deliverables

As you finalize your hackathon project, make sure you can clearly present:

### What Your Sentiment Model Does

Be ready to explain:

- How TextBlob and Hugging Face transformer models work
- The difference between polarity and subjectivity
- The strengths and limitations of each approach
- How you applied these models to news articles


### Visualizations Showing Insights

Create compelling visualizations that:

- Show sentiment differences across subtopics
- Compare results from different methods
- Tell a clear story about what you found

---

### Bonus Ideas

- Try a different Hugging Face model (e.g., one for financial news).
- Combine Title and Description for sentiment analysis—does it change the results?

---
