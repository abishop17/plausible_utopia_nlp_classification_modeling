# README
## Project 3: Plausible Utopia

### Classification Modeling on Subreddits to Classify Scientists and Futurists

## Problem statement

**Question:** Do futurist and scientist interest groups use vocabulary in a distinguishable way? Does an analysis of the words in their vocabulary and the frequency of use enable us to classify them using only text from subreddit titles?

**Data source:** I will use datasets collected via the Python Reddit API Wrapper (PRAW) API from the [`futurology`](https://www.reddit.com/r/Futurology) and [`science`](https://www.reddit.com/r/Science) subreddits on the social news platform [Reddit](https://www.reddit.com/).

**Project goal and evaluation:** I will use binary classification models (logistic regression and Random Forest) to predict which of these two subreddits a given post comes from. I will evaluate the models' performance with their accuracy scores.

### Background: The interest areas

Both scientists and futurists aim to create change in the world with research studies, contributing to the pool of knowledge in their fields, and communicating their findings.

Futurism in the popular imagination has connotations more like ***science fiction*** than ***science*** (such as robot uprisings, dystopian futures, etc.).

[Oxford Reference](https://www.oxfordreference.com/view/10.1093/oi/authority.20110803095839389) describes futurology as a ***pseudo-science***:
> The activity of predicting the state of the world at some future time, by extrapolating from present trends. Mainly a pseudo-science, given the complexities of social, political, economic, technological, and natural factors

I will explore this dichotomy through EDA and modeling with these groups' vocabulary.

## The data

### Data collection

I used the Python Reddit API Wrapper (PRAW) in the data collection process. You can find the steps I took in the `PRAW_data_collection` notebook, located in the code folder of this repository.

The East Coast local instructors were very generous with their starter code / walkthrough of the process, so I definitely credit them for the ease of the data collection. 

Using the [API documentation](https://praw.readthedocs.io/en/latest/code_overview/praw_models.html), I selected additional columns beyond the starter code, in order to do more EDA. See data dictionary below.

I made three sets of API pulls (one for each subreddit), one per selected submission type of `new`, `controversial` and `new`. I made sure to obtain fairly balanced classes in each pull.

### Data dimensions

The size of the dataframe before vectorizing the text was 3,137 rows and 2 columns (the `title` and `subreddit` columns). After vectorizing the text with CountVectorizer, the size of the dataframe was 2 rows (consisting of the binary classes) and 50,491 columns.

### Background: The data source

[Reddit](https://www.reddit.com/) is a social news platform founded in 2005 and calls itself the "front page of the internet". 

>Reddit is broken up into more than a million communities known as “subreddits,” each of which covers a different topic. The name of a subreddit begins with /r/, which is part of the URL that Reddit uses.

Subreddits are moderated by user ("Redditor") volunteers, and overall the site is administered by Reddit employees, who have the ultimate privileges over subreddits (including banning subreddits and their moderators).

A system of upvoting / downvoting by users increases / decreases (respectively) the visibility of posts.

[Source](https://www.digitaltrends.com/web/what-is-reddit/)

<h2>Data dictionary</h2>

|Column name| Description |
| :-: | :-: |
|**title**|Subreddit post title|
|**ups**|Net number of upvotes for post
|**upvote_ratio**|Proportion of upvotes
|**id**|User id who posted|
|**url**|URL of the linked article|
|**body**|The body text of the post|
|**permalink**|The URL route (starts with `r/futurology` or `r/science`)
|**subreddit**|The subreddit (***target column***)
|**submission_type**|The subreddit submission type (`new`, `controversial` and `top`)
|**title_word_count**|Word count in post title

## Methodology

**Deduplication**: A significant number (1,530) posts had duplicate titles. I decided to drop the posts with duplicate titles because my intention was to see whether a classification model can predict classes based on vocabulary. If I were looking at the popularity or frequency of identical posts, I would keep the duplicates.

**Handling of nulls**: The `body` column in the raw dataframe included 4,544 nulls (about 97% of rows). I decided to drop this column because of the low value of such an insubstantial proportion of `body` text.

**Other cleaning tasks included**:
* Remove punctuation

* Remove digits and any elements other than words with RegexpTokenizer

* Stem words with PorterStemmer 

* Drop the `controversial` `submission_type`: The balance of `submission_type` classes was basically even between `new` and `top`, and `controversial` was tiny.

* Drop the `score` column: Duplicate of `ups`

* Add `title_word_count` column for EDA purposes

* Binarize the target feature: `1` for Science, `0` for Futurology

## Preprocessing

Tasks included:

**Use CountVectorizer to do the following:**

1. Count number of times a token is observed in a given subreddit post

2. Create a vector to store those counts

I searched for unigrams (single words) and bigrams (pairs of words) and excluded English stop words along with the stemmed word `ha`.  A meaningless `ha` showed up in the vocabulary (probably for the words 'has', 'have', etc.). I'd prefer to ignore that.

After vectorizing and creating a document-term matrix, the shape of the dataframe was 2 rows (for the two classes) by 50,232 columns.

## EDA

### Bag of Words analysis

I used this analysis to find the top common words, see below:

![top_words](https://git.generalassemb.ly/abishop17/project_3/blob/master/images/top_words.png)

![top_words_science_only](https://git.generalassemb.ly/abishop17/project_3/blob/master/images/top_words_science_only.png)

![top_words_futurology_only](https://git.generalassemb.ly/abishop17/project_3/blob/master/images/top_words_futurology_only.png)

### Sentiment analysis using [VADER (Valence Aware Dictionary and sEntiment Reasoner)](https://pypi.org/project/vaderSentiment/)

* I expected the `futurology` subreddit to have more emotionally charged language (highly positive or negative) than `science`. It turns out both sets of vocabulary are overwhelmingly neutral. This tells me there are more similarities in the tone of the language used, and also that sentiment is not an appropriate feature to include for modeling.

![sentiment](https://git.generalassemb.ly/abishop17/project_3/blob/master/images/sentiment.png)

## Metrics

* 50.2% is the baseline accuracy percentage we compare with the model's accuracy.

* If the model does better than the baseline, then it is better than null model (predicting the majority class).

* I chose accuracy because I did not prioritize one class over another.

## Modeling and Results

* The features I used consisted of the entire set of vectorized text.

* I chose to work with logistic regression and Random Forest because I prioritized interpretability of results. With logistic regression, the ridge regularization selected (after GridSearch) took care of the correlation among the word vectors. With Random Forest, I hoped to reduce variance and decorrelate the word vectors.

* The best performing model was logistic regression, which achieved an accuracy score of 83.7% (an improvement over the baseline). Below is the confusion matrix associated with the predictions.
![confusion_matrix_logreg](https://git.generalassemb.ly/abishop17/project_3/blob/master/images/confusion_matrix_logreg.png)

**Below are details on the models and their results:**

| Model # | Accuracy (test) | Accuracy (train) |  GridSearch?  |      Estimator      |  Estimator hyperparameters |   Transformer   | Transformer hyperparameters (best) |
|:-------:|:---------------:|:----------------:|:-------------:|:-------------------:|:--------------------------:|:---------------:|:----------------------------------:|
|    1    |      83.5%      |       99.6%      |       N       | Logistic Regression |           Default          | CountVectorizer |            max_df = 0.4            |
|    1    |                 |                  |               |                     |                            |                 |        max_features = 6_000        |
|    1    |                 |                  |               |                     |                            |                 |             min_df = 1             |
|    1    |                 |                  |               |                     |                            |                 |        ngram_range = (1, 2)        |
|    1    |                 |                  |               |                     |                            |                 |       stop_words = 'english'       |
|    2    |      83.5%      |       99.6%      | Y: C, penalty | Logistic Regression |       penalty = 'l2'       | CountVectorizer |           Same as model 1          |
|    2    |                 |                  |               |                     |            C = 1           |                 |                                    |
|    3    |      83.7%      |       99.6%      | Y: C, penalty | Logistic Regression |       penalty = 'l2'       | CountVectorizer |           Same as model 1          |
|    3    |                 |                  |               |                     |            C = 1           |                 |                                    |
|    3    |                 |                  |               |                     |    'solver': 'liblinear'   |                 |                                    |
|    4    |      82.1%      |        1.0       |  Y (see code) |    Random Forest    |    'rf__max_depth': None   | CountVectorizer |             max_df=0.75            |
|    4    |                 |                  |               |                     | 'rf__max_features': 'auto' |                 |         max_features=5_000         |
|    4    |                 |                  |               |                     |   'rf__n_estimators': 100  |                 |              min_df=1              |
|    4    |                 |                  |               |                     |                            |                 |        ngram_range = (1, 1)        |
|    4    |                 |                  |               |                     |                            |                 |              min_df: 1             |
|    4    |                 |                  |               |                     |                            |                 |       stop_words = 'english'       |

## Conclusions

**Holding all else constant, the effect of the feature `studi` is a 1.9 times evidence that we can predict classes accurately using vocabulary.** `studi` is the stemmed word, so this conclusion can be extended to words like ‘studies’, ‘study’, ‘studied’.

![log_reg_coefficients](https://git.generalassemb.ly/abishop17/project_3/blob/master/images/log_reg_coefficients.jpg)

With an overall accuracy score (on test data) of 83.7%, there is plenty of room for improvement, which I would tackle with the next steps below.

## Next steps

**Experiment with:**

* Feature engineering
* Boosting models 
* Topic modeling
