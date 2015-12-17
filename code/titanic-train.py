''' Kaggle challenge: predict survivors on the titanic disaster '''
import os
from base import *
from hyperopt_search_spaces import *

# set working directory
os.chdir("/your-path/titanic-ml")

# load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# feature engineered data
train_engnd = pd.read_csv('./engineered-data/train_enc.csv')
test_engnd = pd.read_csv('./engineered-data/test_enc.csv')

train_engnd.loc[:, 'Survived'] = train_df.loc[:, 'Survived']
train_engnd.pop('pos_0.0')
test_engnd.pop('pos_0.0')


''' Defining a working framework '''


def titanic_framework(space):

    # load data
    global train_engnd
    df = copy.deepcopy(train_engnd)

    # copy the search space
    space_copy = copy.deepcopy(space)

    # preprocessing with some parameters
    steps = [scaling_wrapper(Imputer(missing_values='NaN', strategy=space['na_strat'], axis=0)),
             scaling_wrapper(space['algorithm'].pop('scaling')), space['outlier_rem']]

    # framework with preprocessing and algo
    fmwk = supervised_framework(steps, space['algorithm'].pop('type'), space['algorithm'])
    fmwk_copy = copy.deepcopy(fmwk)

    # evaluation function choice
    auc_fn = metrics.roc_auc_score
    acc_fn = metrics.accuracy_score
    conf_mat = metrics.confusion_matrix

    # number of repetitions and folds
    n = 5
    k = 10

    global eval_number
    eval_number += 1
    print 'eval_number:', eval_number
    print space_copy['algorithm']
    print 'scaling:', space_copy['algorithm']['scaling'], 'outlier:', space_copy['outlier_rem']

    # repeat n times a k-fold cross-val
    res = []
    for i in range(n):

        res.append(k_fold_cross_val(k, df, fmwk, 'Survived'))

    # evaluating...
    accuracy_list = []
    auc_list = []
    conf_mat_list = []
    for results in res:

        for key in results.keys():

            predictions = results[key]['out']['preds']
            y = results[key]['gtruth']
            accuracy_list.append(acc_fn(y, predictions))
            conf_mat_list.append(conf_mat(y, predictions))

            try:
                probs = results[key]['out']['probs']
                auc_list.append(auc_fn(y, probs[:, 1]))
            except KeyError:
                pass

    mean_mat = np.mean(conf_mat_list, 0)
    mean_conf_mat = mean_mat/np.sum(conf_mat_list[0])
    mean_acc = np.mean(accuracy_list)
    mean_auc = np.mean(auc_list)
    weighted_acc = mean_mat[0,0]/(mean_mat[0,0] + mean_mat[1,0]) + mean_mat[1,1]/(mean_mat[1,1] + mean_mat[0,1])

    print 'accuracy:', mean_acc, 'std:', np.std(accuracy_list)
    print 'weighted accuracy:', weighted_acc/2, 'std:', np.std(accuracy_list)
    print 'AUC:', mean_auc, 'std:', np.std(auc_list)
    print 'confusion matrix:'
    print mean_conf_mat

    return {'loss': 1 - weighted_acc/2,
            'accuracy': mean_acc,
            'acc_sd': np.std(accuracy_list),
            'auc': mean_auc,
            'conf_mat': mean_conf_mat,
            'auc_std': np.std(auc_list),
            'status': STATUS_OK,
            'parameters': space_copy,
            'framework': fmwk_copy}


''' Optimizing hyperparameters '''


eval_number = 0
best_sgd = optimize(titanic_framework,
                    merge_two_dicts(preproc_space, {'algorithm': sgd_space}),
                    10)

eval_number = 0
best_rf = optimize(titanic_framework,
                   merge_two_dicts(preproc_space, {'algorithm': rf_space}),
                   10)

eval_number = 0
best_adaboost_dt = optimize(titanic_framework,
                            merge_two_dicts(preproc_space, {'algorithm': adaboost_dt_space}),
                            10)

eval_number = 0
best_gradboost = optimize(titanic_framework,
                          merge_two_dicts(preproc_space, {'algorithm': gradboost_space}),
                          10)

eval_number = 0
best_svm = optimize(titanic_framework,
                    merge_two_dicts(preproc_space, {'algorithm': svm_space}),
                    10)

eval_number = 0
best_extratrees = optimize(titanic_framework,
                           merge_two_dicts(preproc_space, {'algorithm': extratrees_space}),
                           10)

eval_number = 0
best_bagging_dt = optimize(titanic_framework,
                           merge_two_dicts(preproc_space, {'algorithm': bagging_dt_space}),
                           10)

eval_number = 0
best_bagging_svm = optimize(titanic_framework,
                            merge_two_dicts(preproc_space, {'algorithm': bagging_svm_space}),
                            10)

eval_number = 0
best_logreg = optimize(titanic_framework,
                       merge_two_dicts(preproc_space, {'algorithm': logreg_space}),
                       10)

eval_number = 0
best_kneigh = optimize(titanic_framework,
                       merge_two_dicts(preproc_space, {'algorithm': kneigh_space}),
                       10)

eval_number = 0
best_radneigh = optimize(titanic_framework,
                         merge_two_dicts(preproc_space, {'algorithm': radneigh_space}),
                         10)

'''Use best model to submit predictions...'''


dmatrix = copy.deepcopy(train_engnd)

# Top 10 models:
for i in range(10):

    best_ind = find_trials_opt(trials.losses(), i)
    print 1 - trials.losses()[best_ind], trials.trials[best_ind]['result']['sd']
    print trials.trials[best_ind]['result']['parameters']

# Bottom 10 models:
for i in range(1, 10):

    best_ind = find_trials_opt(trials.losses(), -i)
    print 1 - trials.losses()[best_ind]
    print trials.trials[best_ind]['result']['parameters']

# Fit on whole dataset -- last result: 0.822
best_ind = find_trials_opt(trials.losses())
print 1 - trials.losses()[best_ind]
best_fmwk = trials.trials[best_ind]['result']['framework']

y = train_engnd.pop('Survived')

predictions = best_fmwk.fit_predict(train_engnd, test_engnd, y)['preds']

res_df = pd.DataFrame({'PassengerId': range(892, 1310),
                       'Survived': predictions})

res_df.to_csv('sub27.csv', index=False)
