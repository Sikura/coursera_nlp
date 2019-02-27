import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from utils import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + '.pkl')
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_thread = pairwise_distances_argmin(question_vec.reshape(1, -1), thread_embeddings)
        return thread_ids[best_thread][0]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\n' \
                               'This thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        self.chitchat_bot = ChatBot('Lets talk', trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
        self.paths = paths

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        trainer = ChatterBotCorpusTrainer(self.chitchat_bot)
        trainer.train('chatterbot.corpus.english')

        cornel_trainer = ListTrainer(self.chitchat_bot)
        with open(self.paths['CORNEL_MOVIES_DIALOGS_DATASET']) as f:
            cornell_dataset = f.read().splitlines()
        cornel_trainer.train(cornell_dataset)

        simple_trainer = ListTrainer(self.chitchat_bot)
        simple_talks = ['How are you?',
                        'Im just fine. What about you?',
                        'I am good',
                        'What are you doing now?',
                        'I am reading some books',
                        'What is your name?',
                        'My name is ' + self.chitchat_bot.name]
        simple_trainer.train(simple_talks)

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chitchat_bot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question, tag[0])
            return self.ANSWER_TEMPLATE % (tag, thread_id)


if __name__ == '__main__':
    dialog_mgr = DialogueManager(RESOURCE_PATH)
    dialog_mgr.create_chitchat_bot()
    while True:
        try:
            user_input = input()
            bot_input = dialog_mgr.chitchat_bot.get_response(user_input)
            print(bot_input)

        except(KeyboardInterrupt, EOFError, SystemExit):
            break

