from abc import ABC, abstractmethod
from collections import namedtuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import numpy as np
import praw
import pandas as pd
import logging
logging.getLogger().setLevel(logging.INFO)

ForumTopic = namedtuple('ForumTopic', 'headline comments')
Score = namedtuple('Score', 'positive neutral negative compound')
reddit = praw.Reddit(client_id='fCrnCZjL30pPxQ',
                     client_secret='qYiFNCI9n9oE9sQrWgGuf_dnnTc',
                     user_agent='Majestic_Algae')


class SentimentDataSource(ABC):

    def __init__(self):
        pass

    @property
    def topics(self):
        return self._topics

    @property
    def topic_headlines(self):
        return self._topic_headlines

    @abstractmethod
    def retrieve_data(self):
        pass


class Subreddit(SentimentDataSource):

    def __init__(self, reddit, subreddit, max_topics=None, get_top=False, time_filter='day'):
        self.reddit = reddit
        self.subreddit = subreddit
        self.max_topics = max_topics
        self.get_top = get_top
        self.time_filter = time_filter

    def retrieve_data(self):
        self._topics = []
        self._topic_headlines = []

        if self.get_top:
            submissions = self.reddit.subreddit(self.subreddit).top(time_filter=self.time_filter, limit=self.max_topics)
        else:
            submissions = self.reddit.subreddit(self.subreddit).new(limit=self.max_topics)

        for submission in submissions:
            logging.info(f'Processing submission {submission.title}...')
            comments = []
            submission.comments.replace_more(limit=0)
            for top_level_comment in submission.comments:
               comments.append(top_level_comment.body)
            self._topics.append(ForumTopic(headline=submission.title, comments=comments))
            self._topic_headlines.append(submission.title)


class SentimentAnalyzer(ABC):

    def __init__(self, sentiment_data_source):
        self.sentiment_data_source = sentiment_data_source
        self.update()

    @abstractmethod
    def _calculate_score(self, text):
        pass

    def calculate_score_stats(self, texts, score_function):
        scores = []
        for text in texts:
            scores.append(score_function(text))
        return Score(*np.nanmean(scores, axis=0))

    def calculate_headline_score(self, topic):
        return self._calculate_score(topic.headline)

    def calculate_aggregated_comment_score(self, topic):
        if len(topic.comments) == 0:
            return Score(np.nan, np.nan, np.nan, np.nan)
        return self.calculate_score_stats(topic.comments, self._calculate_score)

    def calculate_current_mean_headline_sentiment(self):
        return self.calculate_score_stats(self.sentiment_data_source.topic_headlines, self._calculate_score)

    def calculate_current_mean_topic_sentiment(self):
        return self.calculate_score_stats(self.sentiment_data_source.topics, self.calculate_aggregated_comment_score)

    def update(self):
        self.sentiment_data_source.retrieve_data()

    def to_dataframe(self):
        rows = []
        for topic in self.sentiment_data_source.topics:
            headline_score = self.calculate_headline_score(topic)
            comment_score = self.calculate_aggregated_comment_score(topic)
            rows.append({
                'headline': topic.headline,
                'headline_positive': headline_score.positive,
                'headline_neutral': headline_score.neutral,
                'headline_negative': headline_score.negative,
                'headline_compound': headline_score.compound,
                'comment_positive': comment_score.positive,
                'comment_neutral': comment_score.neutral,
                'comment_negative': comment_score.negative,
                'comment_compound': comment_score.compound
            })

        return pd.DataFrame.from_records(rows)




class VaderSentimentAnalyzer(SentimentAnalyzer):

    def __init__(self, sentiment_data_source):
        super(VaderSentimentAnalyzer, self).__init__(sentiment_data_source)
        self.vader = SIA()

    def _calculate_score(self, text):
        score = self.vader.polarity_scores(text)
        return Score(positive = score['pos'], neutral=score['neu'],
                     negative=score['neg'], compound=score['compound'])


if __name__ == '__main__':

    subreddit = Subreddit(reddit, 'CryptoCurrency', max_topics=10)
    analyzer = VaderSentimentAnalyzer(subreddit)
    print(analyzer.to_dataframe().head())
    print(analyzer.calculate_current_mean_headline_sentiment())
    print(analyzer.calculate_current_mean_topic_sentiment())



