# What is Sentiment Analysis?

Sentiment analysis is about figuring out the mood or opinion in a piece of text—whether it's positive, negative, or neutral. It's essentially teaching a computer to be an emotion detector for written content, similar to how humans can read a sentence and get a feeling for whether it's happy, sad, angry, or neutral.

### Understanding Sentiment Through Examples

Let's look at some examples to understand this better:

- "The company's profits soared this quarter" → **Positive**
- "The service was adequate but nothing special" → **Neutral**
- "The economy is collapsing due to poor policies" → **Negative**

Sentiment analysis has numerous real-world applications that make it valuable beyond just an academic exercise. Companies analyze customer reviews and social media mentions to understand product reception. Financial analysts track news sentiment to inform investment decisions. Political campaigns monitor public opinion, and healthcare researchers analyze patient feedback to improve services.

In our project, we'll use sentiment analysis to understand how news articles portray different subtopics, revealing potential patterns in media coverage that might not be immediately obvious to human readers.

### Why Two Different Methods?

We'll explore two different approaches to sentiment analysis:

1. **TextBlob** - A simple, intuitive library perfect for beginners
2. **Hugging Face Transformers** - A more sophisticated approach using advanced AI models

Comparing these methods will give us a more comprehensive understanding of sentiment analysis techniques and their relative strengths and limitations for analyzing news content.

## Preparing Our Data

Before diving into sentiment analysis, we need to prepare our data and understand what we're working with:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load our dataset of news articles
# For this project, we'll assume the data is already processed and includes:
# - Title (of news article)
# - Description (text content) 
# - Subtopic (category of news)
df = pd.read_csv('news_articles.csv')

# Display the first few rows to understand our data
df.head()
```

This code loads our dataset of news articles using pandas, a powerful data analysis library. Each row represents a single article with columns for the title, description text, and subtopic category. By examining the first few rows, we can understand the structure of our data before proceeding with analysis[.

Let's also explore how our articles are distributed across different subtopics:

```python
# Count articles by subtopic and visualize the distribution
subtopic_counts = df['Subtopic'].value_counts()

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=subtopic_counts.values, y=subtopic_counts.index)
plt.title('Number of Articles per Subtopic')
plt.xlabel('Number of Articles')
plt.ylabel('Subtopic')
plt.show()

# Print the actual counts
print("Number of articles per subtopic:")
print(subtopic_counts)
```

Understanding the distribution of articles across subtopics is crucial for our analysis. If certain subtopics have very few articles, their sentiment averages might be less reliable. This visualization gives us context for interpreting our subsequent sentiment findings and helps identify potential sampling biases in our dataset.

## Option 1: Sentiment Analysis using TextBlob

TextBlob is a simple Python library that provides an easy-to-use interface for common natural language processing tasks, including sentiment analysis. It's a perfect starting point for beginners because of its simplicity and readability.

### Installing and Setting Up TextBlob

Let's start by installing and importing the library:

```python
# Install TextBlob if you don't have it already
!pip install textblob

# Import TextBlob
from textblob import TextBlob

# Download necessary language data (needed only the first time)
!python -m textblob.download_corpora
```

This installation process gives us access to the TextBlob library and downloads the necessary language data that TextBlob needs for its analysis. The download_corpora command ensures we have all the linguistic resources required for accurate sentiment analysis.

### Trying TextBlob on Individual Examples

Before applying TextBlob to our entire dataset, let's test it on some individual sentences to understand how it works:

```python
# Try TextBlob on a single sentence
test_sentence = "The market is doing terribly today."
blob = TextBlob(test_sentence)

# Get sentiment scores
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

print(f"Text: '{test_sentence}'")
print(f"Polarity: {polarity}")  # Range: -1 (very negative) to +1 (very positive)
print(f"Subjectivity: {subjectivity}")  # Range: 0 (very factual) to 1 (very opinionated)

# Try a few more examples to better understand the scoring
examples = [
    "The company reported amazing quarterly results, exceeding all expectations.",
    "The weather today is neither good nor bad.",
    "The new regulations have devastated small businesses across the country."
]

for example in examples:
    sentiment = TextBlob(example).sentiment
    print(f"\nText: '{example}'")
    print(f"Polarity: {sentiment.polarity}")
    print(f"Subjectivity: {sentiment.subjectivity}")
```

This code demonstrates how TextBlob analyzes sentiment by breaking down two key metrics:

1. **Polarity**: A float value ranging from -1 (very negative) to +1 (very positive), with 0 representing neutral sentiment. In our first example, "The market is doing terribly today," we expect a negative score because "terribly" carries negative connotation.
2. **Subjectivity**: A float value ranging from 0 (very objective/factual) to 1 (very subjective/opinionated). This helps distinguish between objective statements of fact and subjective expressions of opinion or emotion.

By testing multiple examples with varying sentiments, we can build intuition about how TextBlob assigns scores to different types of sentences. This "peek under the hood" helps us understand what the model is doing before we apply it to thousands of articles, similar to checking that a tool works properly before using it for a large project.

### Applying TextBlob to Our News Articles

Now that we understand how TextBlob works, let's apply it to all the articles in our dataset:

```python
# Apply sentiment analysis to each article using TextBlob
# We'll use the Description column which contains the article content
df['Sentiment_Polarity'] = df['Description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['Sentiment_Subjectivity'] = df['Description'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Look at the results
df[['Description', 'Sentiment_Polarity', 'Sentiment_Subjectivity']].head()
```

In this code, we're using pandas' `.apply()` function to run TextBlob analysis on every article description in our dataset. The lambda function creates a TextBlob object for each text and extracts the polarity and subjectivity scores, which we store in new columns called `Sentiment_Polarity` and `Sentiment_Subjectivity`.

This is a critical step because it transforms our raw text data into quantitative measures that we can analyze statistically. Adding these sentiment scores opens the door to investigating questions like:

- Which subtopics tend to have more negative coverage?
- Are economic news articles more negative than technology news?
- Which subtopics get the most emotional (subjective) coverage?

Let's visualize the distribution of these sentiment scores:

```python
# Get basic statistics about our sentiment scores
sentiment_stats = df[['Sentiment_Polarity', 'Sentiment_Subjectivity']].describe()
print(sentiment_stats)

# Create histograms to see the distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Sentiment_Polarity'], kde=True)
plt.title('Distribution of Sentiment Polarity')
plt.xlabel('Polarity (-1 = Negative, +1 = Positive)')
plt.axvline(x=0, color='red', linestyle='--')  # Add a line at zero

plt.subplot(1, 2, 2)
sns.histplot(df['Sentiment_Subjectivity'], kde=True)
plt.title('Distribution of Sentiment Subjectivity')
plt.xlabel('Subjectivity (0 = Factual, 1 = Opinionated)')

plt.tight_layout()
plt.show()
```

These histograms reveal the overall sentiment landscape of our news articles. The polarity histogram shows how many articles fall into different sentiment ranges, while the subjectivity histogram shows how factual versus opinionated the articles tend to be.

### Reflecting on TextBlob's Limitations

While TextBlob is straightforward to use, it has important limitations to consider:

- **Sarcasm Detection**: TextBlob struggles with sarcasm. For example, "Oh great, another market crash..." would likely be scored as positive (because of "great") when it clearly expresses negative sentiment.
- **Context Understanding**: It doesn't always grasp context. "The stock plummeted 50% after the announcement" contains negative financial news, but TextBlob might miss this nuance if it doesn't recognize "plummeted" as strongly negative.
- **Emotional Complexity**: Simple sentiment models like TextBlob have difficulty with complex emotions, mixed sentiments, or cultural references.

🧠 **Reflection Prompt**: When might TextBlob struggle to understand the true tone of a sentence? Can you think of examples from news articles where sentiment might be misinterpreted?

## Option 2: Sentiment Analysis using Hugging Face Transformers

Now we'll explore a more sophisticated approach using transformer models from Hugging Face, which represent the current state-of-the-art in natural language processing.

### Understanding Transformer Models

Transformers are advanced AI models trained on massive amounts of text data that can understand language in a much more sophisticated way than simpler algorithms like TextBlob.

Think of the difference this way:

- If TextBlob is like a dictionary with feelings attached to words, transformers are like English teachers who can pick up on tone, context, and hidden meaning.
- TextBlob looks at words somewhat independently, while transformers understand relationships between words and longer-range dependencies in a sentence.


### Setting Up Hugging Face Transformers

Let's install and configure the Hugging Face transformers library:

```python
# Install transformers library if needed
!pip install transformers

# Import the pipeline from transformers
from transformers import pipeline

# Load a pretrained sentiment analysis pipeline
# This uses a model called "distilbert-base-uncased-finetuned-sst-2-english" by default
sentiment_model = pipeline("sentiment-analysis")
```

This code installs the `transformers` library and imports the `pipeline` function, which provides a simple interface to use complex pre-trained models. We create a `sentiment_model` using the "sentiment-analysis" pipeline, which automatically loads a pre-trained model specifically designed for sentiment analysis.

The default model is based on DistilBERT, which is a smaller, faster version of BERT (Bidirectional Encoder Representations from Transformers). This model was trained on a massive text dataset and then fine-tuned specifically for sentiment classification tasks.

### Testing the Transformer Model

Let's try this model on some examples:

```python
# Try the model on a single example
test_text = "The new technology is breaking boundaries."
result = sentiment_model(test_text)
print(f"Text: '{test_text}'")
print(f"Result: {result}")

# Compare with TextBlob on our previous examples
examples = [
    "The company reported amazing quarterly results, exceeding all expectations.",
    "The weather today is neither good nor bad.",
    "The new regulations have devastated small businesses across the country."
]

# Try the model on a single example
test_text = "The new technology is breaking boundaries."
result = sentiment_model(test_text)
print(f"Text: '{test_text}'")
print(f"Result: {result}")

# Compare with TextBlob on our previous examples
examples = [
    "The company reported amazing quarterly results, exceeding all expectations.",
    "The weather today is neither good nor bad.",
    "The new regulations have devastated small businesses across the country."
]

for example in examples:
    # Get results from both models
    hf_result = sentiment_model(example)[0]  # Hugging Face result
    tb_polarity = TextBlob(example).sentiment.polarity  # TextBlob polarity
    
    print(f"\nText: '{example}'")
    print(f"Hugging Face: {hf_result['label']} (confidence: {hf_result['score']:.4f})")
    print(f"TextBlob: Polarity = {tb_polarity:.4f}")
    print(f"\nText: '{example}'")
    print(f"Hugging Face: {hf_result['label']} (confidence: {hf_result['score']:.4f})")
    print(f"TextBlob: Polarity = {tb_polarity:.4f}")
```

This code reveals how the Hugging Face model differs from TextBlob. The transformer model provides:

1. A categorical label ("POSITIVE" or "NEGATIVE")
2. A confidence score (between 0 and 1) indicating how certain the model is about its prediction

This confidence score is particularly valuable as it helps us gauge the reliability of each prediction, something TextBlob doesn't provide.

Let's also test how the models handle a challenging case like sarcasm:

```python
# Try with a sarcastic sentence
sarcastic_example = "Great, another market crash is exactly what we needed right now."

# TextBlob analysis
tb_result = TextBlob(sarcastic_example).sentiment
print(f"Text: '{sarcastic_example}'")
print(f"TextBlob: Polarity = {tb_result.polarity:.4f}, Subjectivity = {tb_result.subjectivity:.4f}")

# Hugging Face analysis
hf_result = sentiment_model(sarcastic_example)[0]
print(f"Hugging Face: {hf_result['label']} (confidence: {hf_result['score']:.4f})")
```

This example demonstrates how the two models may respond differently to sarcasm. TextBlob typically gives a positive score for this sentence because it sees words like "great" and "exactly what we needed" as positive, without understanding the sarcastic context. The transformer model is often better at detecting that "another market crash" combined with "exactly what we needed" likely indicates sarcasm and negative sentiment.

### Applying the Transformer Model to Our Dataset

Now let's apply the Hugging Face model to our dataset:

```python
# Apply the Hugging Face model to our dataset
# Note: This will take longer than TextBlob because transformer models are more complex

# For demonstration, we'll use a smaller sample to save time
sample_size = 100  # Adjust based on your computational resources
df_sample = df.sample(sample_size, random_state=42)

# Function to get sentiment label and score
def get_hf_sentiment(text):
    try:
        result = sentiment_model(str(text))[0]
        return result['label'], result['score']
    except Exception as e:
        print(f"Error processing text: {e}")
        return "ERROR", 0.0

# Apply to our sample
df_sample['HF_Sentiment_Label'], df_sample['HF_Sentiment_Score'] = zip(*df_sample['Description'].apply(get_hf_sentiment))

# Convert labels to numeric values for easier analysis (POSITIVE = 1, NEGATIVE = 0)
df_sample['HF_Sentiment_Numeric'] = df_sample['HF_Sentiment_Label'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

# Look at the results
df_sample[['Description', 'Sentiment_Polarity', 'HF_Sentiment_Label', 'HF_Sentiment_Score']].head()
```

In this code, we've created a helper function `get_hf_sentiment` that processes each article with our transformer model and returns both the sentiment label and confidence score. We then apply this to each article in our sample dataset and store the results in new columns.

The transformer model analysis gives us an alternative perspective on our data's sentiment. Having results from two different models provides a more well-rounded view; when both models agree, we can be more confident in our assessment, and when they disagree, we have an opportunity to investigate why.

🎁 **Bonus Idea**: Create a small test set of "tricky" sentences that contain sarcasm, metaphors, or industry jargon, and see how each model performs. Which one handles linguistic complexity better?

### Comparing the Two Models

Let's create a visualization to compare how the two models evaluate the same articles:

```python
# Create a scatter plot comparing TextBlob polarity vs. Hugging Face confidence
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Sentiment_Polarity',
    y='HF_Sentiment_Score',
    hue='HF_Sentiment_Label',
    data=df_sample
)

plt.title('Comparison of TextBlob vs. Hugging Face Sentiment Analysis')
plt.xlabel('TextBlob Polarity (-1 = Negative, +1 = Positive)')
plt.ylabel('Hugging Face Confidence Score')
plt.axvline(x=0, color='gray', linestyle='--')  # Add a vertical line at 0 polarity
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Calculate agreement percentage between the two models
agreement = (
    ((df_sample['Sentiment_Polarity'] > 0) & (df_sample['HF_Sentiment_Label'] == 'POSITIVE')) |
    ((df_sample['Sentiment_Polarity'] < 0) & (df_sample['HF_Sentiment_Label'] == 'NEGATIVE'))
).mean() * 100

print(f"Agreement between TextBlob and Hugging Face: {agreement:.2f}%")
```

This visualization compares how the two sentiment analysis models rate the same articles. The x-axis shows TextBlob's polarity score (-1 to +1), while the y-axis shows the Hugging Face model's confidence score (0 to 1). Points are colored based on whether Hugging Face classified the article as positive or negative.

Interesting patterns to look for include:

- Points in the upper right quadrant: Both models agree the text is positive
- Points in the lower left quadrant: Both models agree the text is negative
- Points in the upper left or lower right: The models disagree

The "agreement percentage" tells us how often the two methods reached the same sentiment classification (positive/negative), providing insight into model consistency.

🧠 **Reflection Prompt**: If the two models disagree on an article's sentiment, which one would you trust more and why? What factors might influence your decision?

## Analyze Sentiment by Subtopic

Now we come to our main research question: Which subtopics are reported on with more positive or negative sentiment?

### Analyzing TextBlob Sentiment by Subtopic

Let's group our articles by subtopic and calculate the average sentiment for each:

```python
# Group articles by subtopic and calculate average polarity
avg_sentiment = df.groupby('Subtopic')['Sentiment_Polarity'].mean().sort_values()

# Create a bar chart to visualize sentiment by subtopic
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_sentiment.values, y=avg_sentiment.index, palette="coolwarm")
plt.title("Average Sentiment by Subtopic")
plt.xlabel("Average Sentiment Polarity (-1 = Negative, +1 = Positive)")
plt.ylabel("Subtopic")
plt.axvline(x=0, color='gray', linestyle='--')  # Add a vertical line at 0
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Print the actual values
print("Average sentiment polarity by subtopic:")
for subtopic, avg_pol in avg_sentiment.items():
    print(f"{subtopic}: {avg_pol:.4f}")
```

This visualization ranks subtopics from most negative (bottom) to most positive (top) based on the average sentiment polarity of articles in each category. The color gradient from red to blue helps distinguish negative from positive sentiment, with the vertical line at 0 marking the boundary.

This analysis transforms our sentiment scores into actionable insights, revealing patterns in how different topics are portrayed in the news. These patterns might reflect:

- Inherent positive/negative aspects of certain topics
- Media bias in reporting
- Current events affecting particular topics during the period covered by our dataset


### Understanding Distribution Within Subtopics

Let's also look at the distribution of sentiment within each subtopic using box plots:

```python
# Create a box plot to show the distribution of sentiment within each subtopic
plt.figure(figsize=(12, 8))
sns.boxplot(x='Sentiment_Polarity', y='Subtopic', data=df, palette='coolwarm')
plt.title("Distribution of Sentiment by Subtopic")
plt.xlabel("Sentiment Polarity (-1 = Negative, +1 = Positive)")
plt.ylabel("Subtopic")
plt.axvline(x=0, color='gray', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

Box plots provide a more nuanced view of the sentiment distribution within each subtopic:

- The vertical line inside each box represents the median sentiment
- The box represents the middle 50% of articles (25th to 75th percentile)
- The "whiskers" extend to show the range of the data
- Dots represent outliers (articles with unusually high or low sentiment)

This visualization adds depth to our understanding by showing:

- Which subtopics have wide variation in sentiment (large boxes)
- Which have consistent sentiment across articles (small boxes)
- Whether there are outliers that might be interesting to investigate further


### Comparing Subtopic Sentiment Using Hugging Face

Let's perform a similar analysis using our Hugging Face results:

```python
# For this to work, we need to have applied the Hugging Face model to our dataset
# Group by subtopic and calculate percentage of positive articles
hf_sentiment_by_subtopic = df_sample.groupby('Subtopic')['HF_Sentiment_Numeric'].mean().sort_values() * 100

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=hf_sentiment_by_subtopic.values, y=hf_sentiment_by_subtopic.index, palette="coolwarm")
plt.title("Percentage of Positive Articles by Subtopic (Hugging Face)")
plt.xlabel("Percentage of Articles Classified as Positive")
plt.ylabel("Subtopic")
plt.axvline(x=50, color='gray', linestyle='--')  # Add a vertical line at 50%
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

This chart shows the percentage of articles classified as positive for each subtopic according to the Hugging Face model. Unlike TextBlob's polarity scores (-1 to +1), here we're showing percentages (0% to 100%), with the vertical line at 50% representing an equal number of positive and negative articles.

🧠 **Reflection Prompt**: Were you surprised by which topics had the most positive or negative coverage? What factors might explain why certain topics tend to be reported more positively or negatively?

## Visual Storytelling \& Interpretation

Data visualization is about more than just making charts—it's about telling a compelling story with your data. Let's create some more sophisticated visualizations that help us interpret our sentiment analysis results.

### Comparing TextBlob and Hugging Face Side by Side

Let's create a chart that directly compares how the two models evaluate each subtopic:

```python
# Prepare data for comparison
# Calculate average TextBlob polarity per subtopic for our sample
tb_by_subtopic = df_sample.groupby('Subtopic')['Sentiment_Polarity'].mean()

# Calculate percentage positive for Hugging Face per subtopic
hf_by_subtopic = df_sample.groupby('Subtopic')['HF_Sentiment_Numeric'].mean()

# Combine into a single dataframe for plotting
comparison_df = pd.DataFrame({
    'TextBlob': tb_by_subtopic,
    'HuggingFace': hf_by_subtopic
})

# Convert Hugging Face scores from 0-1 to -1 to 1 for better comparison
comparison_df['HuggingFace'] = (comparison_df['HuggingFace'] * 2) - 1

# Plot side by side
plt.figure(figsize=(12, 8))
comparison_df.plot(kind='barh', figsize=(12, 8))
plt.title('Comparison of Sentiment Analysis Methods by Subtopic')
plt.xlabel('Sentiment Score (-1 = Negative, +1 = Positive)')
plt.ylabel('Subtopic')
plt.axvline(x=0, color='gray', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Method')
plt.show()
```

This chart directly compares how the two sentiment analysis methods evaluate each subtopic. For each subtopic, we see two bars representing the sentiment score from each method. We converted the Hugging Face scores from a 0-1 range to a -1 to +1 range for easier comparison.

This visualization clearly shows where the methods agree and disagree on subtopic sentiment, potentially highlighting areas where one method might be more accurate than the other.

### Examining Sentiment vs. Subjectivity

Let's create a scatter plot that explores the relationship between sentiment polarity and subjectivity across subtopics:

```python
# Create a scatter plot of all articles with subtopics as colors
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Sentiment_Polarity',
    y='Sentiment_Subjectivity',
    hue='Subtopic',
    data=df,
    alpha=0.7
)
plt.title('Sentiment Polarity vs. Subjectivity by Subtopic')
plt.xlabel('Sentiment Polarity (-1 = Negative, +1 = Positive)')
plt.ylabel('Subjectivity (0 = Factual, 1 = Opinionated)')
plt.axvline(x=0, color='gray', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Subtopic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

This scatter plot shows the relationship between sentiment polarity and subjectivity for all articles, colored by subtopic. The x-axis shows sentiment from negative to positive, while the y-axis shows subjectivity from factual to opinionated.

This visualization helps us see interesting patterns such as:

- Whether certain subtopics tend to be more subjective than others
- If there's a relationship between subjectivity and polarity (e.g., are very positive or very negative articles more likely to be subjective?)
- How subtopics cluster in the sentiment-subjectivity space


### Creating Your Own Visual Story

Now it's your turn to create 1-2 new charts that tell a compelling story about your findings. Here are some ideas:

1. Create a chart comparing sentiment of different subtopics over time (if your data includes dates)
2. Design a visualization showing which subtopics have the most disagreement between TextBlob and Hugging Face
3. Create a chart showing how sentiment relates to other features in your dataset (like article length or source)

When creating your charts, remember these best practices:

- Label axes clearly with descriptive titles
- Add informative chart titles that explain what the visualization shows
- Choose readable, accessible colors (consider color blindness)
- Avoid misleading scales that might exaggerate differences
- Include legends when using multiple colors or shapes

🧠 **Reflection Prompt**: What story can you tell from your charts? If this was a presentation to your company, what would your headline be? What's the most interesting or surprising insight from your sentiment analysis?

## Final Discussion \& Extensions

Congratulations on completing your sentiment analysis of news articles by subtopic! Let's reflect on what we've learned and consider potential extensions to this project.

### Key Takeaways

Our analysis has revealed several important insights:

1. **Sentiment varies by subtopic**: We found that some news subtopics consistently have more positive or negative coverage than others
2. **Different methods yield different results**: TextBlob and Hugging Face transformers sometimes disagree, highlighting the complexity of sentiment analysis
3. **Context matters**: Simple models like TextBlob can miss nuances like sarcasm, while more advanced models might capture them better
4. **Visualization tells the story**: Charts help us identify patterns and communicate findings effectively

### Critical Evaluation

As data scientists, it's important to critically evaluate our methods and results:

**Model Bias**: Do you think the sentiment models are biased? How could we test that?

- Do they treat all topics fairly?
- Are certain types of language more likely to be misclassified?

**Limitations**: What are the limitations of our analysis?

- Can we really reduce complex emotions in text to a single number?
- How reliable are these models for high-stakes decisions?

**Content Differences**: Could the differences in sentiment be due to inherent properties of the topics rather than media bias?

- Some topics (like disasters) are inherently negative
- Other topics (like technology innovations) might be inherently positive

🎁 **Bonus Idea**: Try combining the Title + Description as one input for sentiment analysis. Does it produce different results? Why might title sentiment differ from description sentiment?

### Potential Extensions

If you had more time, here are some interesting ways to extend this project:

1. **Time-based analysis**: Track sentiment by subtopic over time to identify trends or changes
2. **Named entity recognition**: Identify specific people, companies, or organizations mentioned and analyze sentiment specifically about them
3. **Advanced models**: Try other pre-trained models from Hugging Face designed for sentiment analysis
4. **Multi-class sentiment**: Instead of just positive/negative, try classifying text into multiple emotion categories
5. **Topic modeling**: Use techniques like LDA to automatically discover topics in your articles, then analyze sentiment by topic

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
- Are properly labeled and easy to understand


### Reflections on Your Results

Consider:

- What worked well in your analysis?
- What challenges did you encounter?
- What surprised you about the results?
- How reliable do you think your sentiment analysis is?


### Improvements for the Future

Suggest at least one improvement you'd try with more time:

- A different model or approach
- Additional features to analyze
- Ways to address the limitations you identified
- How you might make the analysis more accurate

🧠 **Final Reflection**: What was the most valuable thing you learned from this sentiment analysis project? How might you apply these techniques to other problems or datasets that interest you?