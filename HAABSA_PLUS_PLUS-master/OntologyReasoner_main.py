from owlready2 import *
import numpy as np
import nltk
import time
import os
from nltk import *

from config_ont import FLAGS


class OntReasoner():
    def __init__(self):
        onto_path.append("data/externalData")
        self.onto = get_ontology("ontology.owl")
        self.onto = self.onto.load()
        self.timeStart = time.time()
        self.classes = set(self.onto.classes())
        self.sencount = -1
        self.my_dict = {}

        self.remaining_sentence_vector = []
        self.remaining_target_vector = []
        self.remaining_polarity_vector = []
        self.remaining_pos_vector = []
        self.prediction_vector = []

        self.sentence_vector, self.target_vector, self.polarity_vector, self.polarity, self.posinfo = [], [], [], [], []
        self.majority_count = []

        for onto_class in self.classes:
            self.my_dict[onto_class] = onto_class.lex

    def predict_sentiment(self, sentence, target, onto, use_cabasc, use_svm, posinfo, types1, types2, types3):
        words_in_sentence = sentence.split()
        self.sencount += 1

        lemma_of_words_with_classes, words_with_classes, words_classes, target_class = self.get_class_of_words(
            words_in_sentence, target)

        positive_class = onto.search(iri='*Positive')[0]
        negative_class = onto.search(iri='*Negative')[0]

        found_positive_list = []
        found_negative_list = []

        for x in range(len(words_with_classes)):
            word = words_with_classes[x]
            lemma_of_word = lemma_of_words_with_classes[x]
            word_class = words_classes[x]
            negated = self.is_negated(word, words_in_sentence)

            if lemma_of_word in types1:
                found_positive, found_negative = self.get_sentiment_of_class(positive_class, negative_class,
                                                                             word_class, negated, False)
                found_positive_list.append(found_positive)
                found_negative_list.append(found_negative)

            if lemma_of_word in types2:
                if self.category_matches(target_class, word_class):
                    found_positive, found_negative = self.get_sentiment_of_class(positive_class, negative_class,
                                                                                 word_class, negated, False)
                    found_positive_list.append(found_positive)
                    found_negative_list.append(found_negative)

            if lemma_of_word in types3:
                if target_class is not None:
                    new_class = self.add_subclass(word_class, target_class)
                else:
                    new_class = word_class

                found_positive, found_negative = self.get_sentiment_of_class(positive_class, negative_class,
                                                                             new_class, negated, True)
                found_positive_list.append(found_positive)
                found_negative_list.append(found_negative)

        if True in found_positive_list and True not in found_negative_list:
            self.prediction_vector.append([1, 0, 0])
        elif True not in found_positive_list and True in found_negative_list:
            self.prediction_vector.append([0, 0, 1])
        elif use_cabasc:
            self.sencount += -1
            self.polarity_vector = np.delete(self.polarity_vector, self.sencount, 0)
            self.remaining_sentence_vector.append(sentence)
            self.remaining_target_vector.append(target)
            self.remaining_pos_vector.extend((posinfo, posinfo + 1, posinfo + 2))
        elif use_svm:
            self.sencount += -1
            self.polarity_vector = np.delete(self.polarity_vector, self.sencount, 0)
            self.remaining_sentence_vector.append(sentence)
            self.remaining_target_vector.append(target)
            self.remaining_pos_vector.extend(
                (int(posinfo / 3 * 4), int((posinfo / 3 * 4) + 1),
                 int((posinfo / 3 * 4) + 2), int((posinfo / 3 * 4) + 3))
            )
        else:
            self.prediction_vector.append(self.get_majority_class(self.polarity_vector))
            self.majority_count.append(1)

    def get_class_of_words(self, words_in_sentence, target):
        self.classes = []
        words_with_classes = []
        lemma_of_words_with_classes = []
        wordnet_lemmatizer = WordNetLemmatizer()
        target_class = None

        for word in words_in_sentence:
            word_as_list = word_tokenize(word)
            pos_tag = nltk.pos_tag(word_as_list)
            tag_only = pos_tag[0][1]

            if tag_only.startswith('V'):
                lemma_of_word = wordnet_lemmatizer.lemmatize(word, 'v')
            elif tag_only.startswith('J'):
                lemma_of_word = wordnet_lemmatizer.lemmatize(word, 'a')
            elif tag_only.startswith('R'):
                lemma_of_word = wordnet_lemmatizer.lemmatize(word, 'r')
            else:
                lemma_of_word = wordnet_lemmatizer.lemmatize(word)

            for value in list(self.my_dict.values()):
                if lemma_of_word in value:
                    lemma_of_word_class = list(self.my_dict.keys())[list(self.my_dict.values()).index(value)]
                    self.classes.append(lemma_of_word_class)
                    lemma_of_words_with_classes.append(lemma_of_word)
                    words_with_classes.append(word)
                    if word == target:
                        target_class = lemma_of_word_class
                    break
        return lemma_of_words_with_classes, words_with_classes, self.classes, target_class

    def is_negated(self, word, words_in_sentence):
        try:
            index = words_in_sentence.index(word)
        except ValueError:
            return False

        if index < 3:
            window = words_in_sentence[:index]
        else:
            window = words_in_sentence[index - 3:index]

        for temp in window:
            if "not" in temp or "n't" in temp or "never" in temp:
                return True
        return False

    def get_sentiment_of_class(self, positive_class, negative_class, onto_class, negated, type3):
        found_positive = False
        found_negative = False
        if type3:
            onto_ancestors = onto_class.ancestors()
        else:
            onto_ancestors = onto_class.ancestors()
        for an in onto_ancestors:
            if an == positive_class:
                if negated:
                    found_negative = True
                else:
                    found_positive = True
            if an == negative_class:
                if negated:
                    found_positive = True
                else:
                    found_negative = True
        return found_positive, found_negative

    def category_matches(self, target_class, onto_class):
        if target_class is None:
            return False
        target_ancestors = target_class.ancestors()
        target_mentions = []
        for target_an in target_ancestors:
            name = target_an.__name__
            if "Mention" in name:
                target_mentions.append(name.rsplit('Mention', 1)[0])
        onto_ancestors = onto_class.ancestors()
        onto_mentions = []
        for onto_an in onto_ancestors:
            name = onto_an.__name__
            if "Mention" in name:
                onto_mentions.append(name.rsplit('Mention', 1)[0])
        common_list = list(set(target_mentions).intersection(set(onto_mentions)))
        return len(common_list) > 2

    def add_subclass(self, onto_class, target_class):
        onto_name = onto_class.__name__
        target_name = target_class.__name__
        new_class = types.new_class(onto_name + target_name, (onto_class, target_class))
        return new_class

    def get_majority_class(self, polarity_vector):
        total = polarity_vector.sum(0)
        index = np.argmax(total)
        if index == 0:
            return [1, 0, 0]
        elif index == 1:
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    def create_types(self):
        types1 = set()
        types2 = set()
        types3 = set()
        classes = list(self.onto.classes())
        for c in classes:
            name_class = c.__name__
            remove_words = ['Property', "Mention", "Positive", "Neutral", "Negative"]
            if any(word in name_class for word in remove_words):
                continue
            ancestors = c.ancestors()
            boolean = 1
            ancestors_list = []
            for an in ancestors:
                ancestors_list.append(an.__name__)
            ancestors_list.sort()
            for name_ancestor in ancestors_list:
                if boolean == 1:
                    if "Generic" in name_ancestor:
                        types1.add(name_class.lower())
                        boolean = 0
                    elif "Positive" in name_ancestor or "Negative" in name_ancestor:
                        types2.add(name_class.lower())
                        boolean = 0
                    elif "PropertyMention" in name_ancestor:
                        types3.add(name_class.lower())
                        boolean = 0
        return types1, types2, types3

    def run(self, use_backup, path, use_svm, cross_val=False, j=0):
        types1, types2, types3 = self.create_types()

        punctuation_and_numbers = ['– ', '(', ')', '?', ':', ';', ',', '.', '!', '/', '"', '\'', '’', '*', '$',
                                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        with open(path, "r") as fd:
            lines = fd.read().splitlines()
            for i in range(0, len(lines), 3):
                if lines[i + 2].strip().split()[0] == '-1':
                    self.polarity_vector.append([0, 0, 1])
                elif lines[i + 2].strip().split()[0] == '0':
                    self.polarity_vector.append([0, 1, 0])
                elif lines[i + 2].strip().split()[0] == '1':
                    self.polarity_vector.append([1, 0, 0])

                words_tar = lines[i + 1].lower()
                words = lines[i].lower()
                words = words.replace('$t$', words_tar)
                for _ in punctuation_and_numbers:
                    words_tar = words_tar.replace(_, '')
                for _ in punctuation_and_numbers:
                    words = words.replace(_, '')

                self.target_vector.append(words_tar)
                self.sentence_vector.append(words)
                self.posinfo.append(i)

        self.sentence_vector = np.array(self.sentence_vector)
        self.target_vector = np.array(self.target_vector)
        self.polarity_vector = np.array(self.polarity_vector)
        self.posinfo = np.array(self.posinfo)

        for x in range(len(self.sentence_vector)):
            self.predict_sentiment(self.sentence_vector[x], self.target_vector[x], self.onto,
                                   use_backup, use_svm, self.posinfo[x], types1, types2, types3)

        self.prediction_vector = np.array(self.prediction_vector)

        argmax_pol = np.argmax(self.polarity_vector, axis=1)
        argmax_pred = np.argmax(self.prediction_vector, axis=1)
        bool_vector = np.equal(argmax_pol, argmax_pred)
        int_vec = bool_vector.astype(float)

        accuracy = sum(int_vec) / (self.sentence_vector.size - len(self.remaining_sentence_vector))
        timeEnd = time.time()

        print("Accuracy: ", accuracy)
        print('RunTime: ', (timeEnd - self.timeStart))
        print('majority', len(self.majority_count))

        self.remaining_sentence_vector = np.array(self.remaining_sentence_vector)
        self.remaining_target_vector = np.array(self.remaining_target_vector)
        self.remaining_polarity_vector = np.array(self.remaining_polarity_vector)
        self.remaining_pos_vector = np.array(self.remaining_pos_vector)

        if use_backup == True:
            outF = open(FLAGS.remaining_test_path, "w")
            with open(FLAGS.test_path, "r") as fd:
                for i, line in enumerate(fd):
                    if i in self.remaining_pos_vector:
                        outF.write(line)
            outF.close()

        return accuracy, len(self.remaining_pos_vector) / 3
