# preprocessing data for the titanic competition
from base import *
from hyperopt_search_spaces import *

'''Idea1: Get positions on the boat'''


def map_decks(deck):

    mapping = {'A': 0, 'B': 1, 'C': 2,
               'D': 3, 'E': 4, 'F': 5, 'G': 6}

    return mapping[deck]


def get_deck(cabin):

    cabin = cabin.split(' ')
    cabin = [element for element in cabin if len(element) > 1]
    if len(cabin) > 0:
        deck = cabin[0][0]
        return map_decks(deck)
    else:
        return map_decks('D')  # halfway?


def get_room(cabin):

    cabin = cabin.split(' ')
    cabin = [element for element in cabin if len(element) > 1]
    if len(cabin) > 0:
        room = int(cabin[0][1:])
        return np.digitize([room], [135, 70, 0])[0]
    else:
        return np.digitize([100], [135, 70, 0])[0]  # halfway?


def decks_framework(space):

    # load data
    global decks_train
    df = copy.deepcopy(decks_train)

    # copy the search space
    space_copy = copy.deepcopy(space)

    # preprocessing with some parameters
    steps = [scaling_wrapper(Imputer(missing_values='NaN', strategy=space['na_strat'], axis=0)),
             scaling_wrapper(space['algorithm'].pop('scaling'))]

    # framework with preprocessing and algo
    fmwk = supervised_framework(steps, space['algorithm'].pop('type'), space['algorithm'])
    fmwk_copy = copy.deepcopy(fmwk)

    # evaluation function choice
    acc_fn = metrics.accuracy_score

    # number of repetitions and folds
    n = 5
    k = 10

    global eval_number
    eval_number += 1
    print 'eval_number:', eval_number
    print space_copy['algorithm']
    print 'scaling:', space_copy['algorithm']['scaling']

    # repeat n times a k-fold cross-val
    res = []
    for i in range(n):

        res.append(strat_k_fold_cross_val(k, df, fmwk, 'deck'))

    # evaluating...
    accuracy_list = []
    for results in res:

        for key in results.keys():

            predictions = results[key]['out']['preds']
            y = results[key]['gtruth']
            accuracy_list.append(acc_fn(y, predictions))

    mean_acc = np.mean(accuracy_list)

    print 'accuracy:', mean_acc, 'std:', np.std(accuracy_list)

    return {'loss': 1 - mean_acc,
            'accuracy': mean_acc,
            'acc_sd': np.std(accuracy_list),
            'status': STATUS_OK,
            'parameters': space_copy,
            'framework': fmwk_copy}


def rooms_framework(space):

    # load data
    global rooms_train
    df = copy.deepcopy(rooms_train)

    # copy the search space
    space_copy = copy.deepcopy(space)

    # preprocessing with some parameters
    steps = [scaling_wrapper(Imputer(missing_values='NaN', strategy=space['na_strat'], axis=0)),
             scaling_wrapper(space['algorithm'].pop('scaling'))]

    # framework with preprocessing and algo
    fmwk = supervised_framework(steps, space['algorithm'].pop('type'), space['algorithm'])
    fmwk_copy = copy.deepcopy(fmwk)

    # evaluation function choice
    acc_fn = metrics.accuracy_score

    # number of repetitions and folds
    n = 5
    k = 10

    global eval_number
    eval_number += 1
    print 'eval_number:', eval_number
    print space_copy['algorithm']
    print 'scaling:', space_copy['algorithm']['scaling']

    # repeat n times a k-fold cross-val
    res = []
    for i in range(n):

        res.append(strat_k_fold_cross_val(k, df, fmwk, 'room_pos'))

    # evaluating...
    accuracy_list = []
    for results in res:

        for key in results.keys():

            predictions = results[key]['out']['preds']
            y = results[key]['gtruth']
            accuracy_list.append(acc_fn(y, predictions))

    mean_acc = np.mean(accuracy_list)

    print 'accuracy:', mean_acc, 'std:', np.std(accuracy_list)

    return {'loss': 1 - mean_acc,
            'accuracy': mean_acc,
            'acc_sd': np.std(accuracy_list),
            'status': STATUS_OK,
            'parameters': space_copy,
            'framework': fmwk_copy}


def location_optimize(obj_fn, trials, max_evals):

    space = {'na_strat': hp.choice('na_strat', ['mean']),
             'algorithm': hp.choice('alg', [rf_space,
                                            svm_space,
                                            bagging_dt_space])}

    return fmin(obj_fn, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)


'''Idea2: Find if relatives died or survived and check if there is a link '''


def get_surname(name_and_fsize):

    if name_and_fsize[1] > 2:
        return name_and_fsize[0].split(',')[0] + str(name_and_fsize[1])
    elif name_and_fsize[1] == 2:
        return 'Couple'
    else:
        return 'Single'


def get_families(df):

    names_and_fsizes = np.array(pd.concat((df['Name'], df['family_size']), axis=1))
    snames = pd.Series(map(get_surname, names_and_fsizes))
    df['Surname'] = snames
    counts = df['Surname'].value_counts()
    counts = counts[~(counts > 2)]

    transf = {}
    for i, element in enumerate(counts.index):
        if counts[i] == 1:
            transf[element] = "NoisySingle"
        else:
            transf[element] = "NoisyCouple"

    df = df.replace({'Surname': transf})

    return df

'''Idea3: fill missing ages with regression'''


def ages_framework(space):

    # load data
    global ages_train
    df = copy.deepcopy(ages_train)

    # copy the search space
    space_copy = copy.deepcopy(space)

    # preprocessing with some parameters
    steps = [scaling_wrapper(Imputer(missing_values='NaN', strategy=space['na_strat'], axis=0)),
             scaling_wrapper(space.pop('scaling'))]

    # framework with preprocessing and algo
    fmwk = supervised_framework(steps, space['algorithm'].pop('type'), space['algorithm'])
    fmwk_copy = copy.deepcopy(fmwk)

    # evaluation function choice
    eval_fn = metrics.mean_squared_error

    n = 5
    k = 10  # train data has 891 rows, test has 418

    global eval_number
    eval_number += 1
    print "yet another cross val... eval_number:", eval_number
    print space_copy['algorithm'], space_copy['scaling']

    # repeat n times a k-fold cross-val
    res = []
    for i in range(n):

        res.append(k_fold_cross_val(k, df, fmwk, 'Age'))

    # evaluating...
    mse_list = []
    for results in res:

        for key in results.keys():

            predictions = results[key]['out']['preds']
            y = results[key]['gtruth']
            mse_list.append(eval_fn(y, predictions))

    print 'RMSE:', np.sqrt(np.mean(mse_list))

    return {'loss': np.mean(mse_list),
            'sd': np.std(mse_list),
            'status': STATUS_OK,
            'parameters': space_copy,
            'framework': fmwk_copy}


def ages_optimize(trials, max_evals):

    space = {'na_strat': hp.choice('na_strat', ['mean']),
             'scaling': hp.choice('scaling', [None, StandardScaler()]),
             'algorithm': hp.choice('regressorr', [{'type': SVR,
                                                    'kernel': hp.choice('kernel', ['rbf']),
                                                    'C': hp.uniform('C_svc', 0.01, 1)
                                                    },
                                                   {'type': LinearRegression
                                                    },
                                                   {'type': Ridge,
                                                    'alpha': hp.uniform('alpha_ridge', 0.01, 1),
                                                    'solver': hp.choice('solver_ridge', ['svd', 'cholesky', 'sparse_cg', 'lsqr'])
                                                    },
                                                   {'type': ElasticNet,
                                                    'alpha': hp.uniform('alpha_enet', 0.01, 1),
                                                    'l1_ratio': hp.uniform('enet_ratio', 0, 1)
                                                    },
                                                   {'type': Lars
                                                    },
                                                   {'type': OrthogonalMatchingPursuit
                                                    }])}

    return fmin(ages_framework, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

#{'type': ARDRegression
#}

'''Idea4: Get Titles from names'''


def get_title(name):

    name = name.replace(' ', '')
    title = re.split('[,. ]', name)[1]

    if title in ['Mme', 'Mlle', 'Ms']:
        title = 'Miss'
    elif title in ['Capt', 'Don', 'Major', 'Sir', 'Dr', 'Col']:
        title = 'Sir'
    elif title in ['Dona', 'Lady', 'theCountess', 'Jonkheer']:
        title = 'Mrs'

    return title


def get_titles(df):

    titles = pd.Series(map(get_title, df['Name']))
    df['Title'] = titles

    return df


# load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#train_df = pd.read_csv('./engineered-data/train_notenc.csv')
#test_df = pd.read_csv('./engineered-data/test_notenc.csv')

combined_df = pd.concat([train_df, test_df])
combined_df.pop('Survived')

# mapping features:
combined_df = combined_df.replace({"Sex": {'female': 0, 'male': 1},
                                   "Embarked": {None: 'S'},
                                   "Fare": {None: np.mean(test_df['Fare'])}})

# idea1: family surname feature
# total family size:
combined_df.loc[:, "family_size"] = combined_df.loc[:, "SibSp"] + combined_df.loc[:, "Parch"] + 1

# get family surnames
combined_df = get_families(combined_df)

# idea2: extract title from name:
combined_df = get_titles(combined_df)

# idea3: get age through a regression!
# finding nans
ages_df = titanic_remove_unwanted(combined_df, mode='test')
ages_df = one_hot_enc(ages_df, ['Pclass', 'Embarked', 'Title', 'Surname'], ['class_', 'embk_', 'title_', 'name_'])
ages_df.index = range(len(ages_df))
age_nans = ages_df.loc[:, 'Age'].notnull()

# getting validation set
ages_train = ages_df.loc[age_nans, :]

# optimizing parameters
eval_number = 0
age_trials = Trials()
best = ages_optimize(age_trials, 50)

# getting best framework
best_ind = find_trials_opt(age_trials.losses())
best_fmwk = age_trials.trials[best_ind]['result']['framework']

# existing ages
y = ages_train.pop('Age')

# -- filling missing ages
missing = ages_df.loc[~ages_df.loc[:, 'Age'].notnull(), :]
missing_ages = best_fmwk.fit_predict(ages_train, missing.drop(['Age'], 1), y)['preds']
combined_df.loc[~combined_df.loc[:, 'Age'].notnull(), 'Age'] = list(missing_ages)

# idea4: find cabin position (deck plans)
# process Cabin strings
ok = combined_df['Cabin'].notnull()
decks = pd.Series(map(get_deck, combined_df.loc[ok, 'Cabin']))
rooms = pd.Series(map(get_room, combined_df.loc[ok, 'Cabin']))

combined_df.loc[ok, 'deck'] = list(decks)
combined_df.loc[ok, 'room_pos'] = list(rooms)

# create train and test splits
ok = combined_df['Cabin'].notnull()
locs_df = titanic_remove_unwanted(combined_df, mode='test')
locs_df = one_hot_enc(locs_df, ['Pclass', 'Embarked', 'Title', 'Surname'], ['class_', 'embk_', 'title_', 'name_'])

decks_train = locs_df.loc[ok, :]
decks_train.pop('room_pos')

rooms_train = locs_df.loc[ok, :]
rooms_train.pop('deck')

# optimizing parameters -> finding deck
eval_number = 0
deck_trials = Trials()
best = location_optimize(decks_framework, deck_trials, 50)

# getting best framework
best_ind = find_trials_opt(deck_trials.losses())
best_fmwk = deck_trials.trials[best_ind]['result']['framework']

# existing decks
y = decks_train.pop('deck')

# -- filling missing decks:
missing = locs_df.loc[~locs_df.loc[:, 'deck'].notnull(), :]
missing_decks = best_fmwk.fit_predict(decks_train, missing.drop(['deck', 'room_pos'], 1), y)['preds']
combined_df.loc[~combined_df.loc[:, 'deck'].notnull(), 'deck'] = list(missing_decks)

# optimizing parameters -> finding room_pos
eval_number = 0
room_trials = Trials()
best = location_optimize(rooms_framework, room_trials, 25)

# getting best framework
best_ind = find_trials_opt(room_trials.losses())
best_fmwk = room_trials.trials[best_ind]['result']['framework']

# existing ages
y = rooms_train.pop('room_pos')

# -- filling missing room_pos:
missing = locs_df.loc[~locs_df.loc[:, 'room_pos'].notnull(), :]
missing_rooms = best_fmwk.fit_predict(rooms_train, missing.drop(['deck', 'room_pos'], 1), y)['preds']
combined_df.loc[~combined_df.loc[:, 'room_pos'].notnull(), 'room_pos'] = list(missing_rooms)

# save engineered training and test sets

# one-hot encoding
encoded_df = one_hot_enc(combined_df, ['Pclass', 'Embarked', 'Title', 'Surname', 'deck', 'room_pos'], ['class_', 'embk_', 'title_', 'name_', 'deck_', 'pos_'])

# remove unwanted features:
encoded_df = titanic_remove_unwanted(encoded_df, mode='test')

train_enc = encoded_df.iloc[range(len(train_df)), :]
train_enc.loc[:, 'Survived'] = pd.read_csv('train.csv').loc[:, 'Survived']
test_enc = encoded_df.iloc[range(len(train_df), len(combined_df)), :]

train_notenc = combined_df.iloc[range(len(train_df)), :]
train_notenc.loc[:, 'Survived'] = pd.read_csv('train.csv').loc[:, 'Survived']
test_notenc = combined_df.iloc[range(len(train_df), len(combined_df)), :]

train_enc.to_csv('./engineered-data/train_enc.csv', index=False)
test_enc.to_csv('./engineered-data/test_enc.csv', index=False)

train_notenc.to_csv('./engineered-data/train_notenc.csv', index=False)
test_notenc.to_csv('./engineered-data/test_notenc.csv', index=False)
