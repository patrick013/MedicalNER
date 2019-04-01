from copy import deepcopy
from collections import defaultdict


from MER.ner_eval import collect_named_entities
from MER.ner_eval import compute_metrics
from MER.ner_eval import compute_precision_recall_wrapper
from MER.fscore_eval import fscoreeval
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import random


def word2features(sent, i):

    word = sent[i][0]
    postag = sent[i][1]
    semantictag= sent[i][2]
    grouptag=sent[i][3]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'semantictag':semantictag,
        'grouptag':grouptag,
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        semantictag1 = sent[i-1][2]
        grouptag1 = sent[i-1][3]

        features.update({
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:semantictag':semantictag1,
            '-1:grouptag':grouptag1,

        })
    else:
        features['BOS'] = True

    if i>1:
        word2=sent[i-2][0]
        postag2=sent[i-2][1]
        semantictag2=sent[i-2][2]
        grouptag2=sent[i-2][3]

        features.update({
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:postag': postag2,
            '-2:semantictag':semantictag2,
            '-2:grouptag':grouptag2,
        })
    else:
        features['+1:BOS']=True

    if i < len(sent) - 2:
        word2 = sent[i + 2][0]
        postag2 = sent[i + 2][1]
        semantictag2 = sent[i + 2][2]
        grouptag2 = sent[i + 2][3]

        features.update({
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:postag': postag2,
            '+2:semantictag': semantictag2,
            '+2:grouptag': grouptag2,
        })
    else:
        features['-1:EOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        semantictag1 = sent[i+1][2]
        grouptag1 = sent[i+1][3]

        features.update({
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:semantictag': semantictag1,
            '+1:grouptag': grouptag1,
        })
    else:
        features['EOS'] = True



    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):

    return [label for token, postag, semantictag, grouptag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]

def grabDataset():
    corpus = [line.rstrip('\n') for line in open('..\Dataset_Sample.txt')]
    random.shuffle(corpus)
    dataset=[]
    for sentence in [line.split() for line in corpus]:
        sent = []
        for word in sentence:
            sent.append(word.split("_"))
        dataset.append(sent)

    trainingset=dataset[:int(len(dataset)*6/8)]
    testingset=dataset[int(len(dataset)*6/8):]

    return trainingset,testingset


if __name__== "__main__":
    trainingset,testingset= grabDataset()

    X_train = [sent2features(s) for s in trainingset]
    y_train = [sent2labels(s) for s in trainingset]
    X_test = [sent2features(s) for s in testingset]
    y_test = [sent2labels(s) for s in testingset]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)
    # print(y_test)
    # print(y_pred)
    labels = list(crf.classes_)
    # print(metrics.flat_f1_score(y_test, y_pred,
    #                       average='weighted', labels=labels))
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))

    test_sents_labels = y_test

    metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                       'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0}

    # overall results
    results = {'strict': deepcopy(metrics_results),
               'ent_type': deepcopy(metrics_results),
               'partial': deepcopy(metrics_results),
               'exact': deepcopy(metrics_results)
               }

    # results aggregated by entity type
    evaluation_agg_entities_type = {e: deepcopy(results) for e in ['MED']}

    for true_ents, pred_ents in zip(test_sents_labels, y_pred):

        # compute results for one message
        tmp_results, tmp_agg_results = compute_metrics(
            collect_named_entities(true_ents), collect_named_entities(pred_ents)
        )

        # print(tmp_results)

        # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

        # Calculate global precision and recall

        results = compute_precision_recall_wrapper(results)

        # aggregate results by entity type

        for e_type in ['MED']:

            for eval_schema in tmp_agg_results[e_type]:

                for metric in tmp_agg_results[e_type][eval_schema]:
                    evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][
                        metric]

            # Calculate precision recall at the individual entity level

            evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(
                evaluation_agg_entities_type[e_type])

    print(evaluation_agg_entities_type)
    pre,recall=fscoreeval(y_test, y_pred)
    print("Precision:", pre)
    print("Recall:", recall)
    print("F-1 Score:", 2*(pre*recall)/(pre+recall))

