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
df = pd.read_csv('news_articles.csv')

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

# Print the first 10 words
print("First 10 words in the vocabulary:")
print(feature_names[:10])

# Look at the first article’s counts
print("\nWord counts for the first article:")
print(X[0].toarray())
```

#### What’s Happening?
- **`vectorizer.get_feature_names_out()`**: Gets the vocabulary as an array (e.g., ["ai", "and", "breakthrough", ...]).
- **`feature_names[:10]`**: Shows the first 10 words—run this to see what’s there!
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
print("First 10 TF-IDF features:")
print(tfidf_features[:10])

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

This avoids **overfitting**—where the model memorizes the training data but flops on new articles.

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

## 6. 🤖 Train a Simple Model

### What You’ll Learn
- How to use **Multinomial Naive Bayes**
- What training and predicting mean
- How to test it out

### Why Multinomial Naive Bayes?
It’s a simple, fast model perfect for beginners. It looks at word counts and guesses subtopics based on which words show up most with each one—like noticing "virus" often means "Public Health." Plus, we can peek at what it learns!

### Step 1: Train the Model
```python
# Import the model
from sklearn.naive_bayes import MultinomialNB

# Create the model
model = MultinomialNB()

# Train it
model.fit(X_train, y_train)  # Learn patterns from training data
```

#### What’s Happening?
- **`from sklearn.naive_bayes import MultinomialNB`**: Imports the Multinomial Naive Bayes classifier.
- **`model = MultinomialNB()`**: Sets up the model with default settings.
- **`model.fit(X_train, y_train)`**:
  - `.fit()`: Teaches the model by looking at `X_train` (word counts) and `y_train` (subtopics).
  - It calculates probabilities—like "80% of articles with ‘ai’ are about AI."
  - Stores what it learns inside `model`.

### Step 2: Predict Subtopics
```python
# Predict subtopics for the test data
y_pred = model.predict(X_test)  # Guess subtopics for X_test

# Print the first 5 predictions
print("First 5 predictions:", y_pred[:5])
print("First 5 actual subtopics:", y_test[:5].values)
```

#### What’s Happening?
- **`model.predict(X_test)`**:
  - `.predict()`: Uses what the model learned to guess subtopics for `X_test`.
  - Looks at each test article’s word counts and picks the most likely subtopic.
- **`y_pred`**: An array of predicted subtopics (e.g., ["AI", "Mental Health", ...]).
- **`y_pred[:5]`**: Shows the first 5 predictions.
- **`y_test[:5].values`**: Shows the first 5 real subtopics to compare.

### Step 3: Try a New Article
```python
# Make a fake article
new_article = ["AI robots help doctors in hospitals"]

# Turn it into numbers
new_vector = vectorizer.transform(new_article)  # Use the same vocabulary

# Predict its subtopic
new_pred = model.predict(new_vector)
print("Predicted subtopic for new article:", new_pred[0])
```

#### What’s Happening?
- **`new_article = [...]`**: A list with one string—our pretend article.
- **`vectorizer.transform(new_article)`**:
  - `.transform()`: Turns the text into numbers using the vocabulary from before (no `.fit`—we’re not learning new words).
  - Returns a sparse matrix matching `X`’s format.
- **`model.predict(new_vector)`**: Guesses the subtopic.
- **`new_pred[0]`**: Gets the prediction (it’s a one-item array).

### 📘 Analogy
The model’s like a detective: "I’ve seen ‘robots’ and ‘doctors’ together in AI articles before—this must be AI!"

---

## 7. 📊 Evaluate the Model

### What You’ll Learn
- How to check if the model’s good
- What accuracy, confusion matrix, and reports mean
- Where it messes up

### Step 1: Accuracy
```python
# Import the tool
from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)  # Compare real vs. predicted
print(f"Accuracy: {accuracy:.2f}")  # Show it with 2 decimals
```

#### What’s Happening?
- **`from sklearn.metrics import accuracy_score`**: Imports a function to measure correctness.
- **`accuracy_score(y_test, y_pred)`**: Counts how many predictions in `y_pred` match `y_test`, then divides by the total (e.g., 80/100 = 0.8).
- **`f"Accuracy: {accuracy:.2f}"`**: Formats the number (e.g., 0.823 → "Accuracy: 0.82").

### Step 2: Confusion Matrix
```python
# Import the tool
from sklearn.metrics import confusion_matrix

# Make the matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')  # Title
plt.xlabel('Predicted Subtopic')  # X-axis
plt.ylabel('Actual Subtopic')  # Y-axis
plt.show()
```

#### What’s Happening?
- **`confusion_matrix(y_test, y_pred)`**: Builds a table:
  - Rows: Real subtopics (from `y_test`).
  - Columns: Predicted subtopics (from `y_pred`).
  - Cells: How many articles (e.g., 15 "AI" articles predicted as "AI").
- **`sns.heatmap(...)`**:
  - `cm`: The matrix to plot.
  - `annot=True`: Shows numbers in each cell.
  - `fmt='d'`: Uses whole numbers (not decimals).
  - `cmap='Blues'`: Colors cells blue—darker means bigger numbers.
  - `xticklabels=model.classes_`: Labels columns with subtopics (e.g., "AI," "Cybersecurity").
  - `yticklabels=model.classes_`: Labels rows the same way.
- The diagonal (top-left to bottom-right) shows correct guesses—off-diagonal are mistakes.

### Step 3: Classification Report
```python
# Import the tool
from sklearn.metrics import classification_report

# Print the report
print(classification_report(y_test, y_pred))
```

#### What’s Happening?
- **`classification_report(y_test, y_pred)`**: Gives a detailed breakdown:
  - **Precision**: For each subtopic, how many predictions were right? (e.g., 0.9 = 90% of "AI" guesses were correct).
  - **Recall**: How many real instances did we catch? (e.g., 0.8 = caught 80% of "AI" articles).
  - **F1-score**: A mix of precision and recall—higher is better.
  - **Support**: How many test articles per subtopic.

### 📘 Plain English
- **Precision**: "When I say ‘AI,’ how often am I right?"
- **Recall**: "Did I find most of the ‘AI’ articles?"
- **F1**: "How balanced are my guesses?"

### Step 4: Spot Mistakes
```python
# Find misclassified articles
mistakes = y_test != y_pred
print("Misclassified examples:")
df_test = df.iloc[y_test.index]  # Match test rows
df_test[mistakes][['Text', 'Subtopic']].head()
```

#### What’s Happening?
- **`y_test != y_pred`**: True where predictions don’t match reality.
- **`df.iloc[y_test.index]`**: Gets the original rows matching our test set.
- **`df_test[mistakes]`**: Filters to just the wrong guesses.
- **`[['Text', 'Subtopic']].head()`**: Shows the text and real subtopic for the first 5 mistakes.

### 🧠 Reflection Prompt
- Which subtopics does the model mix up? Why might that be? (Look at the confusion matrix or mistake examples—any word overlap?)

---

## 8. 🔍 Interpret the Model

### What You’ll Learn
- What the model learned
- Which words tip it off for each subtopic

### Step 1: Top Words
```python
# Get vocabulary
feature_names = vectorizer.get_feature_names_out()

# Show top 10 words per subtopic
for i, subtopic in enumerate(model.classes_):
    probs = model.feature_log_prob_[i]  # Log probabilities for this subtopic
    top_indices = probs.argsort()[-10:][::-1]  # Top 10 word indices
    top_words = [feature_names[j] for j in top_indices]  # Match to words
    print(f"Top words for {subtopic}: {', '.join(top_words)}")
```

#### What’s Happening?
- **`model.classes_`**: List of subtopics (e.g., "AI," "Mental Health").
- **`enumerate(...)`**: Loops with an index (`i`) and name (`subtopic`).
- **`model.feature_log_prob_[i]`**: Gets log probabilities—higher means a word strongly hints at this subtopic.
- **`.argsort()`**: Sorts from low to high (returns indices).
- **`[-10:]`**: Takes the last 10 (highest).
- **`[::-1]`**: Reverses to highest-first.
- **`[feature_names[j] for j in ...]`**: Turns indices into words.
- **`', '.join(...)`**: Makes a list like "ai, robot, tech."

### 🧠 Reflection Prompt
- Do these words fit the subtopics? Any surprises? What might the model miss (like sarcasm or rare words)?

---

## 9. 🤖 Optional Advanced Section: DistilBERT

### What You’ll Learn
- How a powerful model like DistilBERT works
- How to fine-tune it for subtopics

### Step 1: Install Stuff
```python
!pip install transformers datasets --quiet
```

#### What’s Happening?
- **`!pip install ...`**: Installs two libraries in Colab:
  - `transformers`: For pre-trained models like DistilBERT.
  - `datasets`: For handling data easily.
- **`--quiet`**: Keeps the output short.

### Step 2: Load Tools
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch
```

#### What’s Happening?
- **`AutoTokenizer`**: Turns text into tokens (word pieces) DistilBERT understands.
- **`AutoModelForSequenceClassification`**: A pre-trained model for classifying text.
- **`Trainer, TrainingArguments`**: Tools to train the model.
- **`LabelEncoder`**: Turns subtopics into numbers (e.g., "AI" → 0).
- **`Dataset`**: A format for our data.
- **`torch`**: A library for deep learning math.

### Step 3: Prepare Data
```python
# Encode subtopics as numbers
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Subtopic'])

# Make a Dataset
dataset = Dataset.from_pandas(df[['Text', 'label']])
```

#### What’s Happening?
- **`label_encoder.fit_transform(df['Subtopic'])`**: Turns subtopics into numbers (e.g., "AI" → 0, "Mental Health" → 1).
- **`df['label']`**: Adds these numbers as a new column.
- **`Dataset.from_pandas(...)`**: Converts our dataframe to a `Dataset` object with just "Text" and "label."

### Step 4: Tokenize
```python
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['Text'], truncation=True, padding='max_length')

tokenized_dataset = dataset.map(tokenize, batched=True)
```

#### What’s Happening?
- **`tokenizer = ...`**: Loads a tokenizer pre-trained for DistilBERT (lowercase version).
- **`tokenize(batch)`**: A function that:
  - `tokenizer(...)`: Splits text into tokens, cuts long text (`truncation`), and pads short text (`padding`).
- **`dataset.map(...)`**: Applies `tokenize` to all rows in batches (faster!).

### Step 5: Split
```python
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
```

#### What’s Happening?
- **`train_test_split(test_size=0.2)`**: Splits into 80% train, 20% test—like before, but for this format.

### Step 6: Load Model
```python
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))
```

#### What’s Happening?
- **`from_pretrained(...)`**: Loads DistilBERT, pre-trained on tons of text.
- **`num_labels=...`**: Sets it up for our number of subtopics.

### Step 7: Train
```python
training_args = TrainingArguments(
    output_dir='./results',  # Where to save stuff
    per_device_train_batch_size=16,  # How many articles per batch
    evaluation_strategy='epoch',  # Check progress each epoch
    num_train_epochs=3,  # Train for 3 rounds
    logging_dir='./logs'  # Where to save logs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test']
)

trainer.train()
```

#### What’s Happening?
- **`TrainingArguments(...)`**: Sets training rules (batch size, epochs, etc.).
- **`Trainer(...)`**: Sets up the training process.
- **`trainer.train()`**: Fine-tunes DistilBERT on our data.

### Step 8: Evaluate
```python
predictions = trainer.predict(split_dataset['test'])
preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
print(classification_report(split_dataset['test']['label'], preds, target_names=label_encoder.classes_))
```

#### What’s Happening?
- **`trainer.predict(...)`**: Makes predictions on the test set.
- **`torch.argmax(...)`**: Picks the most likely subtopic per article.
- **`classification_report(...)`**: Shows how well it did.

### 📘 Why It’s Cool
DistilBERT already knows language from training on books, articles, and more—we’re just tweaking it for subtopics!

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