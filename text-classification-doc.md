# Part 1: News Subtopic Classification with Machine Learning

Welcome to the Nomura x SEO Hackathon. You’re about to build a machine learning model that predicts the **subtopic** of a news article—like figuring out if a story about "Technology" is really about "AI" or "Cybersecurity"—using its **title**, **description**, and **topic**. This is the kind of thing banks use to spot trends in the news that might affect investments. Since you’re just starting with Python and machine learning, we’ll go step-by-step, breaking everything down with clear explanations, fun analogies, and chances to explore. By the end, you’ll have a working model and a ton of new skills—let’s get started!

---

## 1. 📘 Introduction: What Is Text Classification?

### What You’ll Learn
- What text classification is and why it’s awesome
- How it’s used in the real world
- Our mission: predicting news subtopics

### What Is Text Classification?
Text classification is like teaching a computer to sort text into boxes, just like you might sort emails into folders: "School," "Friends," or "Spam." Imagine picking up a book and guessing its genre—mystery, sci-fi, or romance—based on its title and back-cover blurb. That’s what we’re doing: training a model to read an article’s **title**, **description**, and **topic** and guess its **subtopic**, like "Mental Health" for a "Healthcare" story or "Startups" for "Technology."

This is called **supervised learning**—we give the computer examples (like articles with known subtopics) and teach it to spot patterns so it can predict subtopics for new articles. Think of it as training a super-smart librarian!

### Why It’s Cool
Text classification powers tons of real-world tools:
- **Spam Filters**: Deciding if an email is junk or worth reading.
- **News Apps**: Tagging articles as "Sports" or "Business."
- **Customer Service**: Sorting messages into "Complaints" or "Questions."

For our challenge, imagine a bank wanting to scan news fast—your model could flag articles about "AI breakthroughs" or "market crashes" to help them make smart moves.

### Our Goal
We’ll use:
- **Title**: The headline (e.g., "New AI Saves Lives").
- **Description**: A short summary (e.g., "A hospital uses AI to help doctors.").
- **Topic**: The big category (e.g., "Technology" or "Healthcare").
To predict **Subtopic**: The specific focus (e.g., "AI" or "Mental Health").

By the end, you’ll see how machines learn to "read" and "sort"—and you’ll be the one making it happen!

---

## 2. 📂 Load and Explore the Dataset

### What You’ll Learn
- How to load data with Python
- How to peek inside and spot patterns
- Why data balance matters

### Step 1: Load the Data
We’ll use a library called **pandas** to load our news articles from a CSV file (think of it as a spreadsheet). Each row is an article with columns like "Title," "Description," "Topic," and "Subtopic."

```python
# Import pandas to work with tables of data
import pandas as pd

# Load the CSV file into a "dataframe" (a fancy table)
df = pd.read_csv('news_articles.csv') # Change this to your file

# Show the first 5 rows to peek at our data
df.head()
```

#### What’s Happening?
- **`import pandas as pd`**: Brings in the pandas library and gives it a nickname, "pd," so we don’t have to type "pandas" every time.
- **`pd.read_csv('news_articles.csv')`**: Reads the file named "news_articles.csv" and turns it into a **dataframe**—a table where rows are articles and columns are details like "Title."
- **`df`**: Our dataframe’s name—short for "data frame."
- **`df.head()`**: Shows the first 5 rows. You’ll see columns like "Title," "Description," "Topic," and "Subtopic." Run this in Colab to check it out!

What do the titles look like? Any subtopics catch your eye?

### Step 2: Count Topics and Subtopics
Let’s see how many articles we have for each **topic** and **subtopic**:

```python
# Count articles per topic
topic_counts = df['Topic'].value_counts()

# Count articles per subtopic
subtopic_counts = df['Subtopic'].value_counts()

# Print both to see the numbers
print("Articles per Topic:")
print(topic_counts)
print("\nArticles per Subtopic:")
print(subtopic_counts)
```

#### What’s Happening?
- **`df['Topic']`**: Grabs the "Topic" column from our dataframe—like picking one column from a spreadsheet.
- **`.value_counts()`**: Counts how many times each unique value (e.g., "Technology," "Healthcare") appears. Returns a list like: "Technology: 50, Healthcare: 30."
- **`print(topic_counts)`**: Shows the counts for topics.
- **`print("\nArticles per Subtopic:")`**: The "\n" adds a blank line for readability, then shows subtopic counts.

Are some subtopics more common? Maybe "AI" has 60 articles, but "Cybersecurity" only has 10—keep that in mind!

### Step 3: Visualize Subtopics
Numbers are great, but a picture is worth a thousand words. Let’s make a bar chart:

```python
# Import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look nice
sns.set_style("whitegrid")

# Create a bar chart for subtopics
plt.figure(figsize=(10, 6))  # Make it 10 inches wide, 6 tall
sns.barplot(x=subtopic_counts.index, y=subtopic_counts.values)
plt.title('How Many Articles per Subtopic?')  # Add a title
plt.xlabel('Subtopic')  # Label the x-axis (subtopic names)
plt.ylabel('Number of Articles')  # Label the y-axis (counts)
plt.xticks(rotation=45)  # Tilt subtopic names so they fit
plt.show()  # Show the chart
```

#### What’s Happening?
- **`import matplotlib.pyplot as plt`**: Imports a basic plotting library, nicknamed "plt."
- **`import seaborn as sns`**: Imports Seaborn, a fancier plotting tool built on matplotlib, nicknamed "sns."
- **`sns.set_style("whitegrid")`**: Sets a clean style with a light grid—makes charts easier to read.
- **`plt.figure(figsize=(10, 6))`**: Creates a blank canvas 10 inches wide, 6 inches tall—big enough to see details.
- **`sns.barplot(x=subtopic_counts.index, y=subtopic_counts.values)`**:
  - `x=subtopic_counts.index`: The subtopic names (e.g., "AI," "Mental Health") go on the x-axis.
  - `y=subtopic_counts.values`: The counts (e.g., 60, 20) go on the y-axis as bar heights.
- **`plt.title('How Many Articles per Subtopic?')`**: Adds a title to explain what we’re looking at.
- **`plt.xlabel('Subtopic')`**: Labels the x-axis with "Subtopic."
- **`plt.ylabel('Number of Articles')`**: Labels the y-axis with "Number of Articles."
- **`plt.xticks(rotation=45)`**: Rotates x-axis labels 45 degrees so long subtopic names don’t overlap.
- **`plt.show()`**: Displays the chart in Colab.

Look at the bars—are some subtopics towering over others? That’s called **class imbalance**, and it might make our model favor the tall bars.

### 🧠 Reflection Prompt
- If "AI" has 100 articles and "Cybersecurity" has 10, will our model be better at predicting "AI"? Why might that happen? (Think about practicing for a test with tons of math problems but only a few history questions.)

---

## 3. 🧼 Simple Preprocessing (Beginner Mode)

### What You’ll Learn
- Why we clean text before feeding it to a model
- How to combine and tweak text with Python
- What "noise" means in text

### Why Preprocess?
Computers don’t read like us—they need numbers, not words. First, we’ll clean the text to make it simpler and more consistent. Think of it like tidying up a messy room so it’s easier to find stuff. For now, we’ll:
1. Combine **Title** and **Description** into one column: "Text."
2. Make everything **lowercase.**
3. Clean up extra **whitespace.**

### Step 1: Combine Title and Description
```python
# Combine Title and Description with a space between
df['Text'] = df['Title'] + " " + df['Description']

# Peek at the new column alongside the originals
df[['Title', 'Description', 'Text']].head()
```

#### What’s Happening?
- **`df['Title'] + " " + df['Description']`**: Takes each row’s "Title" (e.g., "AI Breakthrough") and "Description" (e.g., "New tech helps doctors"), adds a space (" "), and glues them together (e.g., "AI Breakthrough New tech helps doctors").
- **`df['Text'] = ...`**: Creates a new column called "Text" in our dataframe and fills it with these combined strings.
- **`df[['Title', 'Description', 'Text']]`**: Picks these three columns to show together.
- **`.head()`**: Displays the first 5 rows so we can check our work.

### Step 2: Make It Lowercase
```python
# Turn all text into lowercase
df['Text'] = df['Text'].str.lower()

# Check the updated Text column
df[['Text']].head()
```

#### What’s Happening?
- **`df['Text']`**: Grabs the "Text" column we just made.
- **`.str`**: Tells pandas to treat each entry in the column as a string (text) so we can use string methods.
- **`.lower()`**: Changes every letter to lowercase (e.g., "AI Breakthrough" becomes "ai breakthrough").
- **`df['Text'] = ...`**: Updates the "Text" column with these lowercase versions.
- **`df[['Text']].head()`**: Shows the first 5 rows of "Text" to confirm it’s all lowercase.

Why? So "AI" and "ai" look the same to the model—computers don’t know they’re equal otherwise!

### Step 3: Clean Whitespace
```python
# Replace multiple spaces with one space
df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex=True)

# Remove spaces at the start or end
df['Text'] = df['Text'].str.strip()

# Check again
df[['Text']].head()
```

#### What’s Happening?
- **`df['Text'].str.replace(r'\s+', ' ', regex=True)`**:
  - `.str`: Treats "Text" entries as strings.
  - `r'\s+'`: A pattern (regex) matching one or more whitespace characters (spaces, tabs, etc.).
  - `' '`: Replaces all those messy spaces with a single space (e.g., "ai    breakthrough" → "ai breakthrough").
  - `regex=True`: Tells pandas this is a regular expression (a fancy search pattern).
- **`df['Text'] = ...`**: Updates "Text" with cleaner spaces.
- **`df['Text'].str.strip()`**: Removes extra spaces at the start or end (e.g., " ai breakthrough " → "ai breakthrough").
- **`df[['Text']].head()`**: Shows the tidied-up text.

This cuts out clutter that might confuse the model.

### 💬 Mini Discussion
Our text is cleaner, but there’s still "noise"—stuff that might trip up the model:
- **Punctuation**: Does a comma or exclamation mark change the meaning?
- **Common words**: "The," "is," "and"—do they help or just take up space?
- **Typos**: What if "AI" is misspelled as "A1"?

### 📦 Extension Box: Supercharge Your Cleaning!
Want to level up? Try these tricks—don’t worry, we’ll explain every bit:

#### Remove Punctuation
```python
# Zap punctuation like commas, periods, and exclamation marks
df['Text'] = df['Text'].str.replace(r'[^\w\s]', '', regex=True)

# Check it
df[['Text']].head()
```
- **`r'[^\w\s]'`**: A regex pattern:
  - `\w`: Matches letters, numbers, and underscores.
  - `\s`: Matches spaces.
  - `^`: Means "not" when inside `[]`.
  - So `[^\w\s]`: Matches anything that’s *not* a letter, number, or space (e.g., `.,!?`).
- **`''`**: Replaces those characters with nothing (deletes them).
- **Why?** Punctuation might not matter for subtopics—does "AI!" mean something different from "AI"?

#### Remove Stopwords
"Stopwords" are super-common words like "the," "a," "is" that might not add much meaning.
```python
# Install and import NLTK (a text-processing library)
!pip install nltk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords from Text
df['Text'] = df['Text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Check it
df[['Text']].head()
```
- **`!pip install nltk`**: Installs the NLTK library in Colab.
- **`import nltk`**: Brings it in.
- **`nltk.download('stopwords')`**: Downloads a list of common English stopwords.
- **`from nltk.corpus import stopwords`**: Imports the stopwords list.
- **`stop_words = set(stopwords.words('english'))`**: Makes a set (fast lookup list) of words like "the," "and."
- **`df['Text'].apply(...)`**:
  - `.apply(lambda x: ...)`: Runs a mini-function on each row’s "Text."
  - `x.split()`: Splits the text into a list of words (e.g., "ai breakthrough" → ["ai", "breakthrough"]).
  - `word for word in ... if word not in stop_words`: Keeps only words not in the stopwords list.
  - `' '.join(...)`: Puts the words back together with spaces (e.g., ["ai", "breakthrough"] → "ai breakthrough").
- **Why?** Cuts out fluff so the model focuses on big-meaning words.

#### Stemming
Stemming chops words to their roots (e.g., "running" → "run").
```python
from nltk.stem import PorterStemmer

# Set up the stemmer
stemmer = PorterStemmer()

# Stem each word
df['Text'] = df['Text'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))

# Check it
df[['Text']].head()
```
- **`from nltk.stem import PorterStemmer`**: Imports a tool that stems words.
- **`stemmer = PorterStemmer()`**: Creates a stemmer object.
- **`stemmer.stem(word)`**: Cuts a word to its root (e.g., "helps" → "help").
- **`' '.join(...)`**: Rebuilds the text from stemmed words.
- **Why?** Treats "help," "helping," "helps" as the same—might make patterns clearer.

Try one or all—see how they change your "Text" column!

---

## 4. 🔢 Vectorization (Text to Numbers)

### What You’ll Learn
- How to turn words into numbers
- What **CountVectorizer** does
- Why this step is key for machine learning

### Why Vectorize?
Computers love numbers, not words. We’ll use **CountVectorizer** from scikit-learn to count how often each word appears in each article. It’s like making a scorecard: "This article has 2 ‘ai’s and 1 ‘breakthrough’." This is called a **bag-of-words**—order doesn’t matter, just counts.

### Step 1: Set Up the Vectorizer
```python
# Import the tool from scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

# Create a vectorizer
vectorizer = CountVectorizer()

# Turn Text into a matrix of word counts
X = vectorizer.fit_transform(df['Text'])

# Set our target (Subtopic) as y
y = df['Subtopic']
```

#### What’s Happening?
- **`from sklearn.feature_extraction.text import CountVectorizer`**: Imports the tool from scikit-learn, a machine learning library.
- **`vectorizer = CountVectorizer()`**: Makes a new vectorizer with default settings—it’ll find all unique words and count them.
- **`vectorizer.fit_transform(df['Text'])`**:
  - `.fit()`: Scans all "Text" entries and builds a **vocabulary**—a list of every unique word (e.g., "ai," "breakthrough").
  - `.transform()`: Turns each article into a row of numbers, counting how many times each vocabulary word appears.
  - Together, `fit_transform`: Does both in one go—learns the vocabulary and makes the counts.
- **`X = ...`**: Stores the result as `X`, a **sparse matrix**—a smart way to save space since most counts are 0 (e.g., "zebra" isn’t in most articles).
- **`y = df['Subtopic']`**: Sets `y` as our target—the subtopics we’re predicting (e.g., "AI," "Mental Health").

### Step 2: Peek Inside
Let’s see what the vectorizer learned:
```python
# Get the vocabulary (list of words)
feature_names = vectorizer.get_feature_names_out()

# Print the 10,000th - 10,020th words. 
print("10,000th - 10,020th in the vocabulary:")
print(feature_names[10000:10020])

# Look at the first article’s counts
print("\nWord counts for the first article:")
print(X[0].toarray())
```

#### What’s Happening?
- **`vectorizer.get_feature_names_out()`**: Gets the vocabulary as an array (e.g., ["ai", "and", "breakthrough", ...]).
- **`feature_names[:10]`**: Shows the 10,000th - 10,020th words — run this to see what’s there! We did these because the first words are likely just numbers it has found in the articles. Try changing the numbers to get a different range and explore the vocabulary.
- **`X[0]`**: Grabs the first article’s row from the sparse matrix.
- **`.toarray()`**: Turns it into a full array (not sparse) so we can see it—each number matches a word in `feature_names`.
- **`print(X[0].toarray())`**: Shows something like `[2, 0, 1, ...]`—2 "ai"s, 0 "and"s, 1 "breakthrough," etc.

So, `X` is our features (word counts), and `y` is our labels (subtopics)—ready for learning!

### 📘 Picture This
Imagine a giant table:
- Rows: Articles.
- Columns: Every unique word (the vocabulary).
- Cells: How many times that word appears in that article.
That’s what `X` is—a number version of our text!

### 📦 Extension Box: Try TF-IDF!
**CountVectorizer** just counts words, but **TF-IDF** (Term Frequency-Inverse Document Frequency) is smarter—it weighs words by how rare and important they are.

#### What’s TF-IDF?
- **Term Frequency (TF)**: How often a word appears in an article (like CountVectorizer).
- **Inverse Document Frequency (IDF)**: How rare a word is across all articles. Common words like "the" get a low score; rare ones like "cybersecurity" get a high score.
- **TF-IDF = TF × IDF**: Combines them—words that appear a lot in one article but rarely elsewhere shine brightest.

#### Code It
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Turn Text into TF-IDF scores
X_tfidf = tfidf_vectorizer.fit_transform(df['Text'])

# Peek at the vocabulary
tfidf_features = tfidf_vectorizer.get_feature_names_out()
print("10,000th - 10,020th in the vocabulary:")
print(feature_names[10000:10020])

# See the first article’s scores
print("\nTF-IDF scores for the first article:")
print(X_tfidf[0].toarray())
```
- **`TfidfVectorizer()`**: Sets up the tool.
- **`fit_transform()`**: Learns the vocabulary and calculates TF-IDF scores.
- **`X_tfidf`**: Your new features—numbers between 0 and 1, not just counts.
- **Why?** Highlights key words over common ones—might make subtopics pop out better!

Try replacing `X` with `X_tfidf` later and see if it boosts your model!

---

## 5. 🧪 Train-Test Split

### What You’ll Learn
- Why we split data
- How to do it with `train_test_split`
- Why reproducibility matters

### Why Split?
Imagine studying for a quiz: you practice with some questions (training), then test yourself with new ones (testing) to see if you really get it. We split our data so the model:
- **Trains** on most of it (learns patterns).
- **Tests** on the rest (proves it works on new stuff).

This avoids **overfitting**—where the model memorizes the training data but flops on new articles. Think of **overfitting** like a when you revise for a test by memorising the answers to past paper questions. This will cause you a problem if you get a question in your exam that you've not seen before!

### Step 1: Split the Data
```python
# Import the splitting tool
from sklearn.model_selection import train_test_split

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### What’s Happening?
- **`from sklearn.model_selection import train_test_split`**: Imports the splitting function from scikit-learn.
- **`train_test_split(X, y, test_size=0.2, random_state=42)`**:
  - `X`: Our features (word counts).
  - `y`: Our targets (subtopics).
  - `test_size=0.2`: 20% goes to testing, 80% to training.
  - `random_state=42`: A seed to shuffle the data the same way every time (like setting a random playlist order).
- **Returns four pieces**:
  - `X_train`: Training features (80% of `X`).
  - `X_test`: Testing features (20% of `X`).
  - `y_train`: Training subtopics (80% of `y`).
  - `y_test`: Testing subtopics (20% of `y`).

Now we’ve got a practice set and a quiz set!

### 🧠 Reflection Prompt
- Why might a model ace the training data but flop on the test data? (Hint: Think about memorizing answers vs. understanding concepts.)

---

## Section 6: 🤖 Train a Simple Model

In this section, we’ll train two models—**Multinomial Naive Bayes** and **Logistic Regression**—to predict news article topics. Using two models lets us compare how different approaches tackle the same problem, giving you a deeper understanding of machine learning.

---

### 6.1 Multinomial Naive Bayes

#### Why Use It?

Naive Bayes is like a **quick-thinking librarian** who sorts books based on word clues. It’s a probabilistic model, meaning it calculates the likelihood of a topic based on the words in the text. It assumes each word contributes **independently** to the topic (a simplification that’s not true for language but works well anyway!). It’s fast, simple, and great for text because it handles lots of words (features) efficiently.

---

#### Step 1: Import and Create the Model

```python
# Import the Naive Bayes model from scikit-learn
from sklearn.naive_bayes import MultinomialNB

# Create an instance of the model
naive_bayes_model = MultinomialNB()
```

- **What’s happening here?**
    - `from sklearn.naive_bayes import MultinomialNB`: Imports the Multinomial Naive Bayes classifier from scikit-learn. "Multinomial" means it’s designed for count-based data, like word frequencies.
    - `naive_bayes_model = MultinomialNB()`: Creates a new Naive Bayes model object with default settings. Think of this as setting up a blank slate ready to learn.

---

#### Step 2: Train the Model

```python
# Train the model using our training data
naive_bayes_model.fit(X_train, y_train)
```

- **Training the model:**
    - `naive_bayes_model.fit(X_train, y_train)`: Trains the model using:
        - `X_train`: The training feature matrix (word counts or TF-IDF scores).
        - `y_train`: The training labels (topics).
        - `.fit()` calculates probabilities like “How often does ‘tech’ appear in Technology articles?” and stores them in `naive_bayes_model`.

---

#### Step 3: Make Predictions

```python
# Predict topics for the test data
naive_bayes_predictions = naive_bayes_model.predict(X_test)
```

- **Making predictions:**
    - `naive_bayes_model.predict(X_test)`: Makes predictions on the test data:
        - `X_test`: The test feature matrix.
        - `.predict()` looks at each test article’s words, uses learned probabilities, and picks the most likely topic.
    - `naive_bayes_predictions`: An array of predicted topics (e.g., `["AI", "Crypto", ...]`).

---

#### Quick Check: View Predictions

```python
print("First 5 Naive Bayes predictions:", naive_bayes_predictions[:5])
print("First 5 actual topics:", y_test[:5].values)
```

This prints out predictions from the Naive Bayes model alongside the actual topics for comparison.

---

### 6.2 Logistic Regression

#### Why Use It?

Logistic Regression is like a **judge weighing evidence**. It assigns weights to each word, showing how much each one pushes the prediction toward a topic. It’s a linear model, meaning it combines these weights in a straight-line way to make decisions. It’s slower than Naive Bayes but can capture more nuanced patterns and is easy to interpret.

---

#### Step 1: Import and Create the Model

```python
# Import the Logistic Regression model from scikit-learn
from sklearn.linear_model import LogisticRegression

# Create an instance of the model with a high iteration limit
logistic_regression_model = LogisticRegression(max_iter=1000)
```

- **What’s happening here?**
    - `from sklearn.linear_model import LogisticRegression`: Imports the Logistic Regression classifier from scikit-learn’s linear model tools.
    - `logistic_regression_model = LogisticRegression(max_iter=1000)`: Creates a new Logistic Regression model with:
        - `max_iter=1000`: Sets the maximum number of iterations (steps) for learning. Text data has many features, so we give it extra time to settle on the best weights.

---

#### Step 2: Train the Model

```python
# Train the model using our training data
logistic_regression_model.fit(X_train, y_train)
```

- **Training the model:**
    - `logistic_regression_model.fit(X_train, y_train)`: Trains the model using:
        - `X_train`: The training feature matrix.
        - `y_train`: The training labels.
        - `.fit()` adjusts weights for each word (e.g., “tech” might get a high positive weight for Technology) to best match the training data.

---

#### Step 3: Make Predictions

```python
# Predict topics for the test data
logistic_regression_predictions = logistic_regression_model.predict(X_test)
```

- **Making predictions:**
    - `logistic_regression_model.predict(X_test)`: Makes predictions on:
        - `X_test`: The test feature matrix.
        - `.predict()` multiplies each word’s TF-IDF score by its learned weight, sums them up, and picks the topic with the highest score.
    - `logistic_regression_predictions`: An array of predicted topics.

---

#### Quick Check: View Predictions

```python
print("First 5 Logistic Regression predictions:", logistic_regression_predictions[:5])
print("First 5 actual topics:", y_test[:5].values)
```

This prints out predictions from Logistic Regression alongside actual topics for comparison.

---

### Reflection Prompt 💡

Run all code above and compare predictions from both models:

- Do they look similar?
- Why might they differ?

Think about how each model works:

- Naive Bayes assumes all words contribute independently.
- Logistic Regression assigns weights that can capture more nuanced relationships between words and topics.

---

## Section 7: 📊 Evaluate the Models

With two models trained, let’s evaluate their performance and dig deeper. We’ll check accuracy, detailed metrics, and even see where they’re overly confident but wrong. This helps us understand their strengths and weaknesses.

---

### 7.1 Accuracy and Classification Report

#### Step 1: Import Evaluation Tools and Calculate Accuracy

```python
# Import tools to measure model performance
from sklearn.metrics import accuracy_score, classification_report

# Calculate how many predictions Naive Bayes got right compared to actual answers
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
print(f"Naive Bayes Accuracy: {naive_bayes_accuracy:.4f}")

# Calculate how many predictions Logistic Regression got right compared to actual answers
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
print(f"Logistic Regression Accuracy: {logistic_regression_accuracy:.4f}")
```

- **What’s happening here?**
    - `from sklearn.metrics import accuracy_score, classification_report`: Imports tools to measure performance.
    - `accuracy_score(y_test, nb_pred)`: Compares `y_test` (actual topics) to `nb_pred` (predicted topics) and calculates the fraction of correct predictions (e.g., 80 correct out of 100 = 0.8).
    - `.4f`: Formats the accuracy score to 4 decimal places (e.g., `0.8234`).
    - The same process is repeated for Logistic Regression predictions (`lr_pred`).

---

#### Step 2: Pick the Better Model and Generate a Detailed Report

```python
# Determine which model performed better based on accuracy
if logistic_regression_accuracy > naive_bayes_accuracy:
    better_predictions = logistic_regression_predictions
    better_model_name = "Logistic Regression"
else:
    better_predictions = naive_bayes_predictions
    better_model_name = "Naive Bayes"

# Show detailed performance breakdown for the better model
print(f"\nDetailed Report for {better_model_name}:")
print(classification_report(y_test, better_predictions))
```

- **What’s happening here?**
    - `better_pred`: Chooses predictions from the model with higher accuracy.
    - `classification_report(y_test, better_pred)`: Breaks down performance by topic:
        - **Precision**: Fraction of predictions for a topic that are correct (e.g., 90% of “Technology” guesses were right).
        - **Recall**: Fraction of actual topic instances correctly identified (e.g., found 80% of “Technology” articles).
        - **F1-score**: Balances precision and recall (higher is better).

---

#### Reflection Prompt 💡

- Which model has higher accuracy?
- Does the detailed report show any topics where it struggles (low F1-scores)?
- Why might that happen?

---

### 7.2 Confusion Matrix

#### Step 1: Create and Plot the Confusion Matrix

```python
# Import tools for creating visualizations
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create a table showing correct and incorrect predictions
confusion_matrix_table = confusion_matrix(y_test, better_predictions)

# Create a heatmap visualization of the confusion matrix
plt.figure(figsize=(10, 8))  # Set figure size to make it easier to read
sns.heatmap(
    confusion_matrix_table,
    annot=True,  # Display numbers in each cell
    fmt='d',     # Use whole numbers (no decimals)
    cmap='Blues',  # Use blue color gradient
    xticklabels=sorted(y_test.unique()),  # Label x-axis with class names
    yticklabels=sorted(y_test.unique())   # Label y-axis with class names
)
plt.title(f'Confusion Matrix for {better_model_name}')
plt.xlabel('Predicted Topic')
plt.ylabel('Actual Topic')
plt.show()
```

- **What’s happening here?**
    - `confusion_matrix(y_test, better_pred)`: Builds a table:
        - Rows represent actual topics (`y_test`).
        - Columns represent predicted topics (`better_pred`).
        - Cells show how many articles fall into each category (e.g., 15 “Technology” articles predicted correctly).
    - `sns.heatmap(...)`: Turns the table into a colorful grid:
        - `annot=True`: Displays numbers in each cell.
        - `fmt='d'`: Uses whole numbers in cells (e.g., `15`, not `15.0`).
        - `cmap='Blues'`: Colors cells blue—darker shades represent larger numbers.
        - `xticklabels`/`yticklabels`: Labels rows and columns with sorted topic names.

---

#### Reflection Prompt 💡

- Which topics get confused most often (big numbers off the diagonal)?
- Could similar words between topics explain this?

---

### 7.3 Model Confidence Analysis

#### Step 1: Find Confident but Wrong Predictions

```python
import numpy as np

# Get prediction probabilities for the better performing model
if better_model_name == "Logistic Regression":
    prediction_probabilities = logistic_regression_model.predict_proba(X_test)
    current_model = logistic_regression_model
else:
    prediction_probabilities = naive_bayes_model.predict_proba(X_test)
    current_model = naive_bayes_model

# Find indices where the model made incorrect predictions
incorrect_indices = np.where(better_predictions != y_test)[0]

# Store high-confidence mistakes
high_confidence_errors = []

# Check each wrong prediction
for index in incorrect_indices:
    predicted_topic = better_predictions[index]
    actual_topic = y_test.iloc[index]
    
    # Find probability for the predicted class
    predicted_class_index = np.where(current_model.classes_ == predicted_topic)[0][0]
    prediction_confidence = prediction_probabilities[index, predicted_class_index]
    
    # Save if model was confident but wrong
    if prediction_confidence > 0.8:
        high_confidence_errors.append((index, actual_topic, predicted_topic, prediction_confidence))

# Display up to 3 examples
print("\nConfidently Wrong Predictions:")
for example_number in range(min(3, len(high_confidence_errors))):
    example_index, true_label, predicted_label, confidence = high_confidence_errors[example_number]
    print(f"Article Preview: {df['Text'].iloc[example_index][:100]}...")
    print(f"True Topic: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}")

```

- **What’s happening here?**
    - `predict_proba(X_test)`: Returns an array of probabilities for each topic per test article.
        - Shape: `(n_samples, n_classes)` (e.g., `100 articles × 5 topics`).
        - Each row sums to `1` (e.g., `[0.9, 0.05, 0.05]` for three topics).
    - `np.where(better_pred != y_test)`: Finds indices where predictions don’t match actual topics.
    - Loop through wrong predictions:
        - **`pred_class`**: The predicted topic.
        - **`true_class`**: The actual topic.
        - **Confidence** (`pred_prob`): Probability assigned to the predicted topic.
            - Keeps only high-confidence mistakes (`pred_prob &gt; 0.8`).

---

#### Reflection Prompt 💡

- Why might the model be so confident but wrong?
- Could overlapping words (e.g., “market” in Business and Economy) trick it?

---

### 7.4 Comparing Models

#### Step 1: Compare Accuracies Directly

```python
print(f"Naive Bayes Accuracy: {naive_bayes_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {logistic_regression_accuracy:.4f}")
```

---

#### Discussion 💬

- **Naive Bayes**: Fast and assumes words are independent. If “tech” and “innovation” often appear together, it might miss that connection but is great for quick results.
- **Logistic Regression**: Slower but learns weights that can capture subtle relationships (e.g., “tech” + “innovation” might strongly suggest Technology).
- **Why the Difference?** If Logistic Regression wins, it might better handle word overlaps. If Naive Bayes wins, simplicity might suit our data better.

---

#### Reflection Prompt 💡

- Which model did better?
- Hypothesize why—could it be data size, topic similarity, or something else?

---

## Section 8: 🔍 Interpret the Models

Let’s peek inside both models to see what they’ve learned about topics. This helps us trust their predictions and spot areas to improve.

---

### 8.1 Interpreting Naive Bayes

#### How It Works

Naive Bayes uses **log probabilities** to measure how likely each word is for a topic. High probabilities mean a word is a strong clue for that topic.

---

#### Step 1: Get Vocabulary from Vectorizer

```python
# Get the list of words (vocabulary) from the TF-IDF vectorizer
feature_names = vectorizer.get_feature_names_out()
```

- **What’s happening here?**
    - `vectorizer.get_feature_names_out()`: Retrieves the list of words the model uses, like a dictionary of all terms it knows.

---

#### Step 2: Loop Through Topics and Find Top Words

```python
# Loop through each topic the model has learned
for i, topic in enumerate(naive_bayes_model.classes_):
    # Get log probabilities for all words in this topic
    log_probs = naive_bayes_model.feature_log_prob_[i]
    
    # Find indices of the 10 words with highest probabilities
    top_indices = np.argsort(log_probs)[-10:][::-1]
    
    # Convert indices to actual words
    top_words = [feature_names[j] for j in top_indices]
    
    # Display results
    print(f"Top words for {topic}: {', '.join(top_words)}")
```

- **What’s happening here?**
    - `enumerate(naive_bayes_model.classes_)`: Loops through topics, keeping track of their index (`i`) and name (`topic`).
    - `feature_log_prob_[i]`: Array of log probabilities for each word in the current topic.
    - `np.argsort(log_probs)[-10:][::-1]`:

1. Sorts word indices by probability (low to high).
2. Takes the last 10 indices (highest probabilities).
3. Reverses them to show most important words first.
    - `[feature_names[j] for j in ...]`: Maps numerical indices back to actual words (e.g., index 42 → "technology").

---

### 8.2 Interpreting Logistic Regression

#### How It Works

Logistic Regression uses **coefficients (weights)** to show each word’s importance. Positive weights mean a word strongly suggests that topic; negative weights push away from it.

---

#### Step-by-Step Interpretation

```python
# Loop through each topic the model has learned
for i, topic in enumerate(logistic_regression_model.classes_):
    # Get coefficients (weights) for all words in this topic
    coefficients = logistic_regression_model.coef_[i]
    
    # Find indices of the 10 words with strongest positive weights
    top_indices = np.argsort(coefficients)[-10:][::-1]
    
    # Convert indices to actual words
    top_words = [feature_names[j] for j in top_indices]
    
    # Display results
    print(f"Top words for {topic}: {', '.join(top_words)}")
```

- **What’s happening here?**
    - `logistic_regression_model.coef_[i]`: Array of weights for each word in the current topic.
        - *Example*: A weight of +2.5 for "tech" in Technology means the word strongly indicates this topic.
    - `np.argsort(coefficients)[-10:][::-1]`:

1. Sorts word indices by weight (low to high).
2. Takes the last 10 indices (highest positive weights).
3. Reverses them to show most influential words first.
    - `feature_names[j]`: Converts numerical indices to human-readable words.

---

### Reflection Prompt 💡

Compare the top words for both models:

1. **Are they similar?**
2. **Do they match your intuition** about each topic?
3. **What differences stand out?**

*Example*:

- Naive Bayes might prioritize common words like "new" or "report".
- Logistic Regression could highlight more specific terms like "blockchain" or "quantum".

---

## 9. 🤖 Neural Networks

Neural networks might sound like something out of a robot movie, but they’re really just a cool way for computers to learn from data—like how you figure out patterns in a game after playing it a few times. Don’t stress about the fancy terms; we’ll make them as easy as pie with analogies and a hands-on approach.

### What You’ll Do

- Use an **MLP Classifier** to classify news topics (e.g., "Sports" or "Tech").
- Learn what hidden layers and activation functions are with fun analogies.
- Build your own neural network with Keras and tweak it.
- Compare it to other models you’ve tried (if you have any).
- Experiment and write down what you discover.

---

### Step 1: Try a Simple MLP Classifier

Let’s kick off with the **MLP Classifier** from scikit-learn. Think of it like a pre-built LEGO set—you can snap it together and start playing without designing every piece yourself.

#### Code Block 1: Preparing the Data

First, we need to prepare the data so the neural network can understand it. Computers don’t understand words—they need numbers!

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Combine 'Title' and 'Description' into one 'Text' column
df['Text'] = df['Title'] + " " + df['Description']

# Convert text into numbers using TF-IDF (like turning words into coordinates on a map)
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['Text']).toarray()

# Convert topic labels (e.g., "Sports") into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Topic'])

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


#### Explanation: What’s Happening?

- **TF-IDF Vectorizer**: It turns words into numbers by measuring how important each word is in the text. For example, "goal" might get a higher score if it's unique to sports articles.
- **LabelEncoder**: Converts categories like "Sports" or "Tech" into numeric labels (e.g., 0 for Sports, 1 for Tech).
- **Train-Test Split**: Splits the data so the model can learn from one part (training) and be tested on another part (testing).

---

#### Code Block 2: Training the MLP Classifier

Now let’s train our neural network using the prepared data.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Create and train the MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=10, random_state=42)
mlp.fit(X_train, y_train)

# Test it out by predicting topics for the test set
y_pred = mlp.predict(X_test)

# Print results
print("MLP Classifier Results:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```


#### Explanation: What’s Going On?

- **Hidden Layer (64)**: Think of this as a team of 64 detectives inside the network. Each one is looking for patterns in your news articles.
- **Activation (‘relu’)**: A rule that decides if those detectives should shout out their findings. We’ll explain this with a light switch analogy later!
- **Max Iterations (10)**: The network gets 10 rounds to practice and improve its guesses.
- **fit()**: This is where the network learns from training data.
- **predict()**: This is where it guesses topics for new data.

---

#### Try This:

Run the code and look at the accuracy in the report. Compare it to other models you’ve tried before (like logistic regression). Write down your observations in a markdown cell:

```markdown
MLP Accuracy: 0.82  
Compared to Logistic Regression (0.75): It’s better! Maybe the MLP is smarter at finding patterns.
```

---

### Step 2: What’s Inside a Neural Network?

Now that you’ve seen a neural network work, let’s peek under the hood. Don’t worry—it’s not as complicated as it sounds. Let’s use analogies to make it click.

#### Picture This: A Team Solving a Mystery

Imagine a group of friends trying to figure out the topic of a news article:

1. **Input Layer**: These friends grab raw info—like words in the article ("goal," "tech," "stocks").
2. **Hidden Layers**: These are detectives who look for clues ("goal" might mean Sports). They pass their findings to others.
3. **Output Layer**: The leader decides, “Yep, this is Sports!” based on what the detectives found.

In computer terms:

- Each “friend” is called a neuron.
- Neurons are connected in layers.
- Each neuron processes info and passes it along until an answer pops out.

---

#### Hidden Layers: The Detectives

Hidden layers are where magic happens! Think of them as teams of detectives:

- Each detective checks for something specific—like if “goal” or “player” appears often.
- With 64 detectives (like in our MLP), they can spot tons of tiny clues.
- Adding more hidden layers means they can look for bigger patterns—like phrases or ideas.


#### Real-Life Example:

If an article says “scored a goal,” one detective might notice “goal,” while another connects “scored” to guess it’s Sports.

Here are some examples of different hidden layer configurations:

**Single Layer (64 Neurons)**

```python  
mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', ...)  
```

*Analogy*: One team of 64 detectives working together.

**Two Layers (128 → 64 Neurons)**

```python  
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', ...)  
```

*Analogy*: Two teams: 128 detectives pass clues to 64 specialists.

**Deep Network (32 → 16 → 8 Neurons)**

```python  
mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8), activation='relu', ...)  
```

*Analogy*: A hierarchy: 32 → 16 → 8 detectives narrowing down clues.

**Overkill Layer (2000 Neurons)**

```python  
mlp = MLPClassifier(hidden_layer_sizes=(2000,), activation='relu', ...)  
```

*Analogy*: A massive team—might overcomplicate simple problems!

---

#### Activation Functions: The Light Switch

Detectives need rules to decide whether their clue is worth sharing. That’s where activation functions come in!

Here are some examples of different activation functions:

**ReLU**

```python  
mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(64,), ...)  
```

*Analogy*: A light switch that only turns on for strong clues.

**Sigmoid**

```python  
mlp = MLPClassifier(activation='sigmoid', hidden_layer_sizes=(64,), ...)  
```

*Analogy*: A dimmer switch for "maybe" answers (e.g., 80% sure).

**Tanh**

```python  
mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(64,), ...)  
```

*Analogy*: A mood meter (-1 to 1) for nuanced decisions.

**Identity**

```python  
mlp = MLPClassifier(activation='identity', hidden_layer_sizes=(64,), ...)  
```

*Analogy*: No filter—raw numbers pass through unchanged.

---

#### Quick Check:

Our MLP used ReLU because it lets detectives shout only when they’re sure—helping the network learn faster without confusion.

---

### Step 3: Build Your Own Neural Network with Keras

Ready to be the master builder? Let’s use Keras to make a neural network from scratch. It’s like moving from a LEGO set to designing your own creation.

#### Code Block 1: Prepare Labels for Keras

```python  
from tensorflow.keras.utils import to_categorical  

# Convert labels to one-hot encoding (like turning "Sports" into [1,0,0])  
y_train_onehot = to_categorical(y_train)  
y_test_onehot = to_categorical(y_test)  
```

**What’s Happening?**
`to_categorical` converts labels (e.g., 0,1,2) into a format Keras prefers. If you have 3 topics, "Sports" (label 0) becomes ``.

---

#### Code Block 2: Build the Network Architecture

```python  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  

model = Sequential()  
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Detective layer  
model.add(Dense(len(label_encoder.classes_), activation='softmax'))       # Decision layer  
```

**What’s Happening?**

- **`Dense(64)`**: Adds 64 neurons (detectives) with ReLU activation.
- **`input_shape=(X_train.shape,)`**: Tells the network how many word features to expect.
- **`softmax`**: Final layer splits confidence like a pie (e.g., 70% Sports, 20% Tech).

---

#### Code Block 3: Configure Learning Parameters

```python  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
```

**What’s Happening?**

- **`optimizer='adam'`**: A smart coach that adjusts learning speed.
- **`loss='categorical_crossentropy'`**: Measures how wrong the guesses are.
- **`metrics=['accuracy']`**: Tracks percentage of correct predictions.

---

#### Code Block 4: Train the Network

```python  
history = model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_split=0.2)  
```

**What’s Happening?**

- **`epochs=10`**: Trains for 10 rounds.
- **`batch_size=32`**: Processes 32 articles at a time.
- **`validation_split=0.2`**: Uses 20% of training data to check progress.

---

#### Code Block 5: Evaluate Performance

```python  
test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)  
print(f"Test Accuracy: {test_accuracy:.3f}")  
```

**Try This**: Compare this accuracy to your earlier MLP Classifier. Is Keras better? Worse? Why?

---

### Step 4: Visualize Learning Progress

Let’s draw how your network improves over time.

#### Code Block 6: Plot Accuracy Trends

```python  
import matplotlib.pyplot as plt  

plt.plot(history.history['accuracy'], label='Training', color='blue')  
plt.plot(history.history['val_accuracy'], label='Validation', color='orange')  
plt.title('Learning Progress')  
plt.xlabel('Epochs')  
plt.ylabel('Accuracy')  
plt.legend()  
plt.show()  
```

**What to Look For**

- **Blue Line**: Training accuracy (should rise steadily).
- **Orange Line**: Validation accuracy (if flat, network isn’t generalizing).
- **Gap Between Lines**: Large gaps mean overfitting (memorizing instead of learning).

---

### Step 5: Experiment Like Mad

Tweak one thing at a time and observe changes! (Be patient though...)

#### Code Block 7: More Detectives Experiment

```python  
model = Sequential()  
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))  # Double the detectives!  
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  
```


#### Code Block 8: Additional Layer Experiment

```python  
model.add(Dense(32, activation='relu'))  # Add this line before the output layer  
```


#### Code Block 9: Try Sigmoid Activation

```python  
model.add(Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)))  
```

**Try This**
After each tweak:

1. Recompile (`model.compile(...)`)
2. Retrain (`model.fit(...)`)
3. Check if accuracy improves

---

### Step 6: Document Your Findings

Keep a log of experiments in a markdown cell:

```markdown  
#### Report  

**Experiment 1: 128 Neurons**  
- **Change**: Increased detectives from 64 → 128  
- **Result**: Accuracy 0.82 → 0.85  
- **Why**: More detectives found subtle clues  

**Experiment 2: Added Layer**  
- **Change**: Added 32-neuron layer  
- **Result**: Accuracy dropped to 0.78  
- **Why**: Too many layers confused the network  
```

---

### Why This Matters

You’ve built a system that learns like a human brain! By tweaking layers/activations, you’re doing what engineers do to create AI for games, apps, and more. Keep experimenting – every failure teaches you something new! 🚀

---

## 10. 🧠 Final Reflections

### What You’ve Done
You’ve built a subtopic classifier! You:
- Loaded and explored news data
- Cleaned text
- Turned words into numbers
- Trained and tested a model
- Checked its smarts

### Think About It
- What worked well? Any high scores?
- What surprised you? Weird predictions?
- How could we improve it? More cleaning? A different model?

### Deliverables
- Your model code
- Accuracy, confusion matrix, and top words
- Reflections: What you did and learned
- Optional: Bonus experiments (like DistilBERT)