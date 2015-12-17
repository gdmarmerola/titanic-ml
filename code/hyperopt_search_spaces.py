''' Common algorithms search spaces: classification '''


n_estimators_rf = [10, 100]
n_estimators_abt = [10, 100]
n_estimators_gb = [10, 100]
n_estimators_et = [10, 100]
n_estimators_bg = [10, 100]

preproc_space = {'na_strat': hp.choice('na_strat', ['mean']),
                 'outlier_rem': hp.choice('out_rem', [None])} 

rf_space = {'type': RandomForestClassifier,
            'n_estimators': hp.choice('rf_n', n_estimators_rf),
            'criterion': hp.choice('rf_crit', ['gini', 'entropy']),
            'max_features': hp.choice('rf_maxfeat', ['sqrt', 'log2', None]),
            'class_weight': hp.choice('rf_cweight', ['auto', 'subsample', None]),
            'scaling': hp.choice('rf_scaling', [None])
            }

adaboost_dt_space = {'type': AdaBoostClassifier,
                     'n_estimators': hp.choice('abt_n', n_estimators_abt),
                     'scaling': hp.choice('abt_scaling', [None])
                     }

gradboost_space = {'type': GradientBoostingClassifier,
                   'n_estimators': hp.choice('gb_n', n_estimators_gb),
                   'loss': hp.choice('gb_loss', ['deviance']),
                   'max_features': hp.uniform('gb_maxfeat', 0.1, 1),
                   'max_depth': hp.quniform('gb_maxdep', 10, 1000, 50),
                   'subsample': hp.uniform('gb_ss', 0.1, 1),
                   'scaling': hp.choice('gb_scaling', [None])
                   }

xgboost_space = {'type': xgb.XGBClassifier,
                 'n_estimators' : hp.quniform('xgb_n_estimators', 10, 100, 1),
                 'learning_rate' : hp.quniform('xgb_eta', 0.025, 0.5, 0.025),
                 'max_depth' : hp.quniform('xgb_max_depth', 1, 13, 1),
                 'min_child_weight' : hp.quniform('xgb_min_child_weight', 1, 6, 1),
                 'subsample' : hp.quniform('xgb_subsample', 0.5, 1, 0.05),
                 'gamma' : hp.quniform('xgb_gamma', 0.5, 1, 0.05),
                 'colsample_bytree' : hp.quniform('xgb_colsample_bytree', 0.5, 1, 0.05),
                 #'num_class' : 20,
                 #'eval_metric': 'merror',
                 'objective': hp.choice('xgb_objective', ['multi:softprob', 'multi:softmax'])
                 }

svm_space = {'type': SVC,
             'kernel': hp.choice('svm_kernel', ['linear', 'rbf']),
             'C': hp.uniform('svm_C', 0.1, 1),
             'class_weight': hp.choice('svc_cweight', ['auto', None]),
#             'probability': True,
             'scaling': hp.choice('svc_scaling', [None, StandardScaler(),
                                                  MinMaxScaler(),
                                                  MinMaxScaler(feature_range=(-1, 1))])
             }

extratrees_space = {'type': ExtraTreesClassifier,
                    'n_estimators': hp.choice('ext_n', n_estimators_et),
                    'criterion': hp.choice('ext_crit', ['gini', 'entropy']),
                    'max_features': hp.choice('ext_maxfeat', ['sqrt', 'log2', None]),
                    'class_weight': hp.choice('ext_cweight', ['auto', 'subsample', None]),
                    'scaling': hp.choice('ext_scaling', [None])
                    }

bagging_dt_space = {'type': BaggingClassifier,
                    'n_estimators': hp.choice('bag_n', n_estimators_bg),
                    'max_features': hp.uniform('bag_maxfeat', 0.1, 1),
                    'max_samples': hp.uniform('bag_maxsamp', 0.1, 1),
                    'scaling': hp.choice('bag_scaling', [None]),
                    }

bagging_svm_space = {'type': BaggingClassifier,
                     'base_estimator': SVC(),
                     'n_estimators': hp.choice('bagsvm_n', n_estimators_bg),
                     'max_features': hp.uniform('bagsvm_maxfeat', 0.1, 1),
                     'max_samples': hp.uniform('bagsvm_maxsamp', 0.1, 1),
                     'scaling': hp.choice('bagsvm_scaling', [None, StandardScaler(),
                                                             MinMaxScaler(),
                                                             MinMaxScaler(feature_range=(-1, 1))]),

                     }

logreg_space = {'type': LogisticRegression,
                'penalty': hp.choice('logreg_penalty', ['l1', 'l2']),
                'C': hp.uniform('logreg_C', 0.1, 1),
                'class_weight': hp.choice('logreg_cweight', ['auto', None]),
                'solver': hp.choice('logreg_solver', ['newton-cg', 'lbfgs', 'liblinear']),
                'scaling': hp.choice('logreg_scaling', [None, StandardScaler(),
                                                        MinMaxScaler(),
                                                        MinMaxScaler(feature_range=(-1, 1))]),
                }

sgd_space = {'type': SGDClassifier,
             'loss': hp.choice('sgd_loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),
             'penalty': hp.choice('sgd_pen', ['none', 'l2', 'l1', 'elasticnet']),
             'alpha': hp.uniform('sgd_alpha', 0.01, 0.001),
             'n_iter': hp.quniform('sgd_iter', 5, 25, 2),
             'l1_ratio': hp.uniform('sgd_l1r', 0.01, 1),
             'eta0': hp.uniform('sgd_eta0', 0.001, 0.1),
             'learning_rate': hp.choice('sgd_lr', ['constant', 'optimal', 'invscaling']),
             'class_weight': hp.choice('sgd_cweight', ['auto', None]),
             'scaling': hp.choice('sgd_scaling', [None, StandardScaler(),
                                                  MinMaxScaler(),
                                                  MinMaxScaler(feature_range=(-1, 1))]),
             }

sgdsvm_space = {'type': SGDClassifier,
                'loss': hp.choice('sgdsvm_loss', ['hinge']),
                'penalty': hp.choice('sgdsvm_pen', ['l2']),
                'alpha': hp.uniform('sgdsvm_alpha', 0.01, 0.001),
                'n_iter': hp.quniform('sgdsvm_iter', 5, 25, 5),
                'l1_ratio': hp.uniform('sgdsvm_l1r', 0.01, 1),
                'eta0': hp.uniform('sgdsvm_eta0', 0.001, 0.1),
                'learning_rate': hp.choice('sgdsvm_lr', ['constant', 'optimal', 'invscaling']),
                'class_weight': hp.choice('sgdsvm_cw', ['auto', None]),
                'scaling': hp.choice('sgdsvm_scaling', [None, StandardScaler(),
                                                     MinMaxScaler(),
                                                     MinMaxScaler(feature_range=(-1, 1))]),
                }

sgdlog_space = {'type': SGDClassifier,
                'loss': hp.choice('sgdlog_loss', ['log']),
                'penalty': hp.choice('sgdlog_pen', ['l2']),
                'alpha': hp.uniform('sgdlog_alpha', 0.01, 0.001),
                'n_iter': hp.quniform('sgdlog_iter', 5, 25, 2),
                'l1_ratio': hp.uniform('sgdlog_l1r', 0.01, 1),
                'class_weight': hp.choice('sgdlog_cw', ['auto', None]),
                'scaling': hp.choice('sgdlog_scaling', [None, StandardScaler(),
                                                     MinMaxScaler(),
                                                     MinMaxScaler(feature_range=(-1, 1))]),
                }


kneigh_space = {'type': KNeighborsClassifier,
                'n_neighbors': hp.quniform('kneigh_k', 5, 50, 2),
                'weights': hp.choice('kneigh_w', ['uniform', 'distance']),
                'algorithm': hp.choice('kneigh_algo', ['ball_tree', 'kd_tree', 'brute']),
                'scaling': hp.choice('kneigh_scaling', [None, StandardScaler(),
                                                        MinMaxScaler(),
                                                        MinMaxScaler(feature_range=(-1, 1))]),
                }

radneigh_space = {'type': RadiusNeighborsClassifier,
                  'radius': hp.uniform('rneigh_k', 0.1, 5),
                  'weights': hp.choice('rneigh_w', ['uniform', 'distance']),
                  'algorithm': hp.choice('rneigh_algo', ['ball_tree', 'kd_tree', 'brute']),
                  'outlier_label': 0,
                  'scaling': hp.choice('rneigh_scaling', [None, StandardScaler(),
                                                          MinMaxScaler(),
                                                          MinMaxScaler(feature_range=(-1, 1))])
                  }


''' Common algorithms search spaces: regression '''


linreg_space = {'type': LinearRegression}

ridge_space = {'type': Ridge,
               'alpha': hp.uniform('ridge_alpha', 0.01, 1),
               'solver': hp.choice('ridge_solver', ['svd', 'cholesky', 'sparse_cg', 'lsqr'])
               }

svr_space = {'type': SVR,
             'kernel': hp.choice('kernel', ['rbf']),
             'C': hp.uniform('svr_C', 0.01, 1)
             }

elnet_space = {'type': ElasticNet,
               'alpha': hp.uniform('enet_alpha', 0.01, 1),
               'l1_ratio': hp.uniform('enet_ratio', 0, 1)
               }

lars_space = {'type': Lars}

omp_space = {'type': OrthogonalMatchingPursuit}
