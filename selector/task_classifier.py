"""NLP Task Classifier for CompressRoute.

Uses TF-IDF feature extraction and Logistic Regression to classify user text
into one of five primary NLP tasks:
1. intent_classification
2. sentiment_analysis
3. summarization
4. question_answering
5. translation
"""

from typing import Dict, Any, Tuple, List
import json
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Balanced dataset of task descriptions across 5 NLP domains
TRAIN_DATA: List[Tuple[str, str]] = [
    # Intent Classification
    ("classify banking queries for our chatbot", "intent_classification"),
    ("route customer service queries to correct bank departments", "intent_classification"),
    ("detect user intent from mobile banking app chat", "intent_classification"),
    ("intent detection for credit card dispute questions", "intent_classification"),
    ("categorize incoming account inquiry requests", "intent_classification"),
    ("identify intent in financial transactions support", "intent_classification"),
    ("determine whether customer wants balance check or transfer", "intent_classification"),
    ("recognize banking requests for pin changes and card block", "intent_classification"),
    ("classify user dialogue intents for automated ticketing", "intent_classification"),
    ("identify user intention behind account statement questions", "intent_classification"),
    ("financial query intent classification system", "intent_classification"),
    ("sort customer messages by request type", "intent_classification"),
    ("identify user purpose in banking mobile application", "intent_classification"),
    ("bank customer intent recognition model", "intent_classification"),
    ("categorize intent for wire transfer and payments", "intent_classification"),
    ("understand what the bank user wants to accomplish", "intent_classification"),
    ("intent classifier for virtual banking assistants", "intent_classification"),
    ("classify user queries into 77 financial intent classes", "intent_classification"),
    ("route user message based on identified purpose", "intent_classification"),
    ("predict caller intent for retail banking call centers", "intent_classification"),
    ("customer intent recognition for atm pin reset", "intent_classification"),
    ("classify requests regarding loan status and overdraft", "intent_classification"),
    ("recognize user intent for debit card renewal requests", "intent_classification"),
    ("predict user action intention from short banking utterance", "intent_classification"),
    ("intent parser for financial services helpdesk", "intent_classification"),
    ("intent extraction from online banking chat prompts", "intent_classification"),
    ("classify if user wants exchange rate or account balance", "intent_classification"),
    ("categorize support ticket intent for bank branches", "intent_classification"),
    ("routing intent detector for fraud report queries", "intent_classification"),
    ("classify transaction dispute intent from customer messages", "intent_classification"),
    ("intent classification on banking dialogues", "intent_classification"),
    ("detect user intention to freeze lost credit card", "intent_classification"),
    ("intent categorization for direct deposit setup", "intent_classification"),
    ("identify intent in mobile banking transfers", "intent_classification"),
    ("route queries about mortgage interest rates", "intent_classification"),
    ("classify banking complaints and service request intent", "intent_classification"),
    ("detect customer goal in banking app search bar", "intent_classification"),
    ("identify intent in banking chat transcripts", "intent_classification"),
    ("intent recognition for contactless payment troubleshooting", "intent_classification"),
    ("classify banking query intents with high accuracy", "intent_classification"),

    # Sentiment Analysis
    ("analyze sentiment in customer reviews of credit cards", "sentiment_analysis"),
    ("detect positive and negative sentiment in customer tweets", "sentiment_analysis"),
    ("score sentiment on banking customer feedback", "sentiment_analysis"),
    ("opinion mining on user satisfaction survey comments", "sentiment_analysis"),
    ("classify reviews into positive, neutral, or negative polarity", "sentiment_analysis"),
    ("identify emotional tone in client complaint emails", "sentiment_analysis"),
    ("sentiment classification for financial product ratings", "sentiment_analysis"),
    ("detect anger or frustration in user support interactions", "sentiment_analysis"),
    ("monitor brand sentiment on financial news forums", "sentiment_analysis"),
    ("sentiment score calculation for customer satisfaction surveys", "sentiment_analysis"),
    ("classify user emotion from feedback forms", "sentiment_analysis"),
    ("detect customer dissatisfaction in online reviews", "sentiment_analysis"),
    ("sentiment extraction from banking app store feedback", "sentiment_analysis"),
    ("rate whether customer feedback is delighted or unhappy", "sentiment_analysis"),
    ("sentiment analysis for investor reaction comments", "sentiment_analysis"),
    ("gauge public perception and sentiment on new loan policy", "sentiment_analysis"),
    ("sentiment polarities on social media banking mentions", "sentiment_analysis"),
    ("opinion polarity analysis on bank customer survey", "sentiment_analysis"),
    ("measure user frustration from support messages", "sentiment_analysis"),
    ("sentiment classification for financial testimonials", "sentiment_analysis"),
    ("detect negative sentiment in branch visit reviews", "sentiment_analysis"),
    ("sentiment classification of net promoter score feedback", "sentiment_analysis"),
    ("classify customer mood from email inquiries", "sentiment_analysis"),
    ("detect tone and sentiment in product review text", "sentiment_analysis"),
    ("evaluate positive vs negative reviews on trustpilot", "sentiment_analysis"),
    ("opinion mining for retail bank customer feedback", "sentiment_analysis"),
    ("identify angry customer comments needing immediate escalation", "sentiment_analysis"),
    ("sentiment scoring for banking application reviews", "sentiment_analysis"),
    ("sentiment polarity detection for online banking feedback", "sentiment_analysis"),
    ("classify customer satisfaction sentiment in chat logs", "sentiment_analysis"),
    ("sentiment analysis on customer reviews", "sentiment_analysis"),
    ("detect negative reviews on investment products", "sentiment_analysis"),
    ("sentiment evaluation of merchant services", "sentiment_analysis"),
    ("measure customer sentiment from support tickets", "sentiment_analysis"),
    ("sentiment analysis of credit card user comments", "sentiment_analysis"),
    ("detect happy or upset customer feedback", "sentiment_analysis"),
    ("social media sentiment tracking for banking brand", "sentiment_analysis"),
    ("sentiment prediction for customer review dataset", "sentiment_analysis"),
    ("determine positive or negative feedback score", "sentiment_analysis"),
    ("sentiment detector for customer complaints", "sentiment_analysis"),

    # Summarization
    ("summarize long earnings call transcript into key points", "summarization"),
    ("generate concise executive summary of financial reports", "summarization"),
    ("condense lengthy banking terms and conditions into bullet points", "summarization"),
    ("create meeting notes summary from dialogue recordings", "summarization"),
    ("summarize quarterly financial disclosure documents", "summarization"),
    ("abstractive text summarization for annual audit files", "summarization"),
    ("extract summary of credit policy changes for customer notice", "summarization"),
    ("generate a brief synopsis of banking research paper", "summarization"),
    ("shorten long legal agreements into simple summary", "summarization"),
    ("summarize customer service phone transcripts", "summarization"),
    ("condense news articles regarding inflation and federal reserve", "summarization"),
    ("create three-bullet summary of loan contracts", "summarization"),
    ("generate digest of daily banking transactions report", "summarization"),
    ("extractive summarization of financial regulations", "summarization"),
    ("summarize mortgage disclosure package for homebuyers", "summarization"),
    ("produce short overview of wealth management portfolio review", "summarization"),
    ("summarize lengthy customer dispute history for agent review", "summarization"),
    ("text condensation for market research reports", "summarization"),
    ("generate headline and summary of financial press release", "summarization"),
    ("summarize regulatory compliance documentation", "summarization"),
    ("condense multi-page audit report into single page", "summarization"),
    ("generate abstract from economic research manuscript", "summarization"),
    ("summarize minutes of board of directors meeting", "summarization"),
    ("summarize customer dispute conversation history", "summarization"),
    ("create executive summary of branch performance report", "summarization"),
    ("summarize credit agreement terms and obligations", "summarization"),
    ("condense central bank press conference transcript", "summarization"),
    ("summarization model for financial news feeds", "summarization"),
    ("generate quick summary of banking service agreement", "summarization"),
    ("summarize portfolio quarterly report for investors", "summarization"),
    ("create summary of macroeconomic forecast report", "summarization"),
    ("summarize long financial statement disclosures", "summarization"),
    ("condense internal banking audit findings into bullet points", "summarization"),
    ("summarize customer interview transcriptions", "summarization"),
    ("generate short abstract for banking policy document", "summarization"),
    ("summarize customer complaints across branch network", "summarization"),
    ("condense investment prospectus into summary sheet", "summarization"),
    ("summarize compliance risks from auditor review", "summarization"),
    ("generate summary of mutual fund performance", "summarization"),
    ("summarize long customer feedback reports", "summarization"),

    # Question Answering
    ("answer customer questions based on banking knowledge base", "question_answering"),
    ("retrieve answers to frequently asked credit card questions", "question_answering"),
    ("answer factual queries about loan eligibility criteria", "question_answering"),
    ("extractive question answering from bank policy documents", "question_answering"),
    ("answer user questions about savings account interest rates", "question_answering"),
    ("provide direct answers to atm fee questions from faq", "question_answering"),
    ("answer questions on wire transfer limits and cutoff times", "question_answering"),
    ("question answering bot for internal employee hr policy", "question_answering"),
    ("extract exact answer spans from financial compliance handbook", "question_answering"),
    ("answer user query based on provided product brochure paragraph", "question_answering"),
    ("faq question answering engine for online mortgage portal", "question_answering"),
    ("answer customer query regarding international travel card usage", "question_answering"),
    ("answer questions from context about overdraft protection", "question_answering"),
    ("reading comprehension system for banking documentation", "question_answering"),
    ("answer what is the routing number for new york branch", "question_answering"),
    ("answer how long does a check deposit take to clear", "question_answering"),
    ("contextual question answering for bank teller operations", "question_answering"),
    ("extract answer to user question from banking manual", "question_answering"),
    ("answer questions regarding mobile check deposit limits", "question_answering"),
    ("answer factual questions about retirement account rules", "question_answering"),
    ("answer questions based on credit card rewards catalog", "question_answering"),
    ("answer customer queries using bank knowledge base articles", "question_answering"),
    ("answer questions about fraud investigation timeline", "question_answering"),
    ("automated question answering for mortgage refinancing faq", "question_answering"),
    ("answer questions regarding foreign exchange transaction fees", "question_answering"),
    ("answer customer question from policy document excerpt", "question_answering"),
    ("question answering on financial terms definitions", "question_answering"),
    ("answer what documents are needed to open business account", "question_answering"),
    ("answer questions regarding debit card pin security", "question_answering"),
    ("retrieve precise answer to customer question from knowledge base", "question_answering"),
    ("question answering model for wealth management documentation", "question_answering"),
    ("answer user questions on certificate of deposit maturity", "question_answering"),
    ("answer questions about credit score requirements for loans", "question_answering"),
    ("question answering for student banking program details", "question_answering"),
    ("answer questions on how to dispute merchant charge", "question_answering"),
    ("extract answer to customer query from loan guidelines", "question_answering"),
    ("answer questions about automated bill pay scheduling", "question_answering"),
    ("question answering on identity verification requirements", "question_answering"),
    ("answer user questions about bank branch opening hours", "question_answering"),
    ("answer customer questions using banking documentation", "question_answering"),

    # Translation
    ("translate customer banking portal from english to spanish", "translation"),
    ("convert english financial disclosures to german and french", "translation"),
    ("translate user chat messages into english in real time", "translation"),
    ("machine translation for banking app multilingual support", "translation"),
    ("translate account statements into chinese and japanese", "translation"),
    ("translate credit card agreement into portuguese", "translation"),
    ("language translation of branch marketing brochures", "translation"),
    ("translate customer support messages from french to english", "translation"),
    ("multilingual translation for global wealth management website", "translation"),
    ("translate privacy notice into 10 european languages", "translation"),
    ("translate banking terms from english to italian", "translation"),
    ("convert foreign currency exchange notices to arabic", "translation"),
    ("translate mobile banking interface text to spanish", "translation"),
    ("machine translation of loan application into vietnamese", "translation"),
    ("translate customer inquiries from german to english", "translation"),
    ("multilingual translation for international banking customers", "translation"),
    ("translate investment advisory reports to korean", "translation"),
    ("translate fraud alert sms messages into spanish", "translation"),
    ("neural machine translation for financial documents", "translation"),
    ("translate banking terms into spanish for bilingual teller", "translation"),
    ("translate customer dispute letter from spanish to english", "translation"),
    ("machine translation for cross-border banking wires", "translation"),
    ("translate ATM on-screen prompts to spanish and mandarin", "translation"),
    ("translate quarterly financial reports to japanese", "translation"),
    ("multilingual translation of banking disclosure forms", "translation"),
    ("translate banking app localization strings", "translation"),
    ("translate terms of service into german", "translation"),
    ("machine translation for international remittances", "translation"),
    ("translate user banking complaint from italian to english", "translation"),
    ("translate wire transfer instructions into french", "translation"),
    ("translate account opening forms to spanish", "translation"),
    ("multilingual translation for customer support portal", "translation"),
    ("translate mortgage contracts to chinese", "translation"),
    ("neural machine translation for bank communications", "translation"),
    ("translate financial website pages into multiple languages", "translation"),
    ("translate customer feedback from german to english", "translation"),
    ("translate tax documents into spanish and french", "translation"),
    ("translate credit policy changes to multilingual clients", "translation"),
    ("translate bank notification emails into portuguese", "translation"),
    ("machine translation for foreign exchange portal", "translation"),
]

TEST_DATA: List[Tuple[str, str]] = [
    # Intent Classification Test
    ("classify bank chatbot queries", "intent_classification"),
    ("determine if user wants to report lost card or check balance", "intent_classification"),
    ("intent classification on customer banking requests", "intent_classification"),
    ("route customer support messages by intent", "intent_classification"),
    ("recognize user intent for bank transfer", "intent_classification"),
    ("financial query intent detection on edge device", "intent_classification"),
    ("classify banking queries under 20ms latency", "intent_classification"),
    ("identify user intention for card pin reset", "intent_classification"),

    # Sentiment Analysis Test
    ("analyze sentiment on credit card customer feedback", "sentiment_analysis"),
    ("determine positive or negative sentiment of bank reviews", "sentiment_analysis"),
    ("customer review sentiment detection", "sentiment_analysis"),
    ("sentiment score on social media banking feedback", "sentiment_analysis"),
    ("detect negative complaints in customer feedback", "sentiment_analysis"),
    ("classify review polarity for mobile banking app", "sentiment_analysis"),
    ("sentiment evaluation of retail branch service", "sentiment_analysis"),
    ("sentiment analysis of customer satisfaction score comments", "sentiment_analysis"),

    # Summarization Test
    ("summarize annual financial audit report", "summarization"),
    ("create short summary of long earnings call", "summarization"),
    ("condense mortgage loan terms into executive summary", "summarization"),
    ("generate bullet-point summary of bank board meeting", "summarization"),
    ("summarize regulatory compliance memorandum", "summarization"),
    ("summarize customer service dialogue transcript", "summarization"),
    ("condense financial press release into synopsis", "summarization"),
    ("abstractive summarization of banking news", "summarization"),

    # Question Answering Test
    ("answer questions based on bank knowledge base article", "question_answering"),
    ("what is the daily withdrawal limit for debit card", "question_answering"),
    ("retrieve answer from mortgage faq context", "question_answering"),
    ("reading comprehension question answering on loan policy", "question_answering"),
    ("answer customer query about international wire transfer fee", "question_answering"),
    ("extract answer span from financial regulatory guide", "question_answering"),
    ("answer questions about account opening requirements", "question_answering"),
    ("faq answering engine for online credit applications", "question_answering"),

    # Translation Test
    ("translate banking mobile app into spanish", "translation"),
    ("machine translation of financial disclosures to french", "translation"),
    ("convert customer dispute message from german to english", "translation"),
    ("translate account terms into mandarin chinese", "translation"),
    ("multilingual translation for global remittance interface", "translation"),
    ("translate wire transfer guidelines to japanese", "translation"),
    ("neural translation of banking portal into portuguese", "translation"),
    ("translate customer support emails from italian to english", "translation"),
]


class TaskClassifier:
    """TF-IDF + Logistic Regression pipeline for NLP task categorization."""

    def __init__(self, model_path: str = "selector/task_classifier.joblib"):
        self.model_path = model_path
        self.pipeline: Pipeline = None

    def train_and_evaluate(self) -> Dict[str, Any]:
        """Trains the pipeline on TRAIN_DATA and evaluates on TEST_DATA."""
        train_texts = [x[0] for x in TRAIN_DATA]
        train_labels = [x[1] for x in TRAIN_DATA]

        test_texts = [x[0] for x in TEST_DATA]
        test_labels = [x[1] for x in TEST_DATA]

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(C=2.0, max_iter=200, random_state=42))
        ])

        self.pipeline.fit(train_texts, train_labels)
        test_preds = self.pipeline.predict(test_texts)
        acc = float(accuracy_score(test_labels, test_preds))

        report = classification_report(test_labels, test_preds, output_dict=True)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

        metrics = {
            "train_samples": len(train_texts),
            "test_samples": len(test_texts),
            "test_accuracy": round(acc, 4),
            "classes": sorted(list(set(train_labels))),
            "detailed_report": report
        }

        results_dir = "benchmark/results"
        os.makedirs(results_dir, exist_ok=True)
        metrics_file = os.path.join(results_dir, "task_classifier_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[TASK CLASSIFIER] Trained on {len(train_texts)} samples across 5 tasks.")
        print(f"[TASK CLASSIFIER] Test Accuracy: {acc * 100:.2f}% (Saved to {self.model_path})")
        return metrics

    def load(self) -> None:
        """Loads the trained pipeline from disk."""
        if not os.path.exists(self.model_path):
            self.train_and_evaluate()
        else:
            self.pipeline = joblib.load(self.model_path)

    def predict(self, text: str) -> Tuple[str, float]:
        """Predicts the NLP task label and confidence probability for given query text."""
        if self.pipeline is None:
            self.load()

        probs = self.pipeline.predict_proba([text])[0]
        max_idx = np.argmax(probs)
        pred_label = self.pipeline.classes_[max_idx]
        confidence = float(probs[max_idx])

        return pred_label, round(confidence, 4)


if __name__ == "__main__":
    clf = TaskClassifier()
    clf.train_and_evaluate()

    # Test sample inference
    queries = [
        "classify banking queries, CPU only, under 20 ms",
        "find customer sentiment in card reviews",
        "summarize our quarterly financial disclosures",
        "answer customer queries from banking faq",
        "translate app text into spanish"
    ]
    print("\nSample Task Classifier Predictions:")
    for q in queries:
        task, conf = clf.predict(q)
        print(f"  '{q}' -> Task: {task} (Confidence: {conf * 100:.1f}%)")
