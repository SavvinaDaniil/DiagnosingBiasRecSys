from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials

from lenskit.algorithms import als, user_knn
from lenskit import crossfold as xf
from lenskit import util, batch, topn, Recommender
from lenskit.metrics.predict import rmse
import lenskit_tf

from cornac.models import UserKNN, MF, BPR
import cornac
from cornac.eval_methods import RatioSplit
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import GridSearch, RandomSearch

import time
import pickle

def optimize_cornac(ratings, algorithm_name, args, max_evals, verbose=False):

    # similarity, weighting, amplify?
    cornac_rmse = cornac.metrics.RMSE()

    num_users = len(ratings.user.unique())
    ratio_split = RatioSplit(data=ratings.values, test_size=0.2, val_size=0.2, verbose=verbose, seed=0)

    if algorithm_name=="CornacUserKNN":

        mean_centered = args['center']
        # Instantiate model with fixed hyperparameters
        userknn = UserKNN(mean_centered=mean_centered, seed=0, verbose=verbose)


        # RandomSearch
        rs = RandomSearch(
            model=userknn,
            space=[
                Discrete("k", range(1, num_users)),
            ],
            metric=cornac_rmse,
            eval_method=ratio_split,
            n_trails=max_evals, # what's this
        )


    elif algorithm_name=="CornacBPR":
        use_bias = args['bias']
        bpr = BPR(use_bias=use_bias, seed=0, verbose=verbose)
        # RandomSearch
        rs = RandomSearch(
            model=bpr,
            space=[
                Discrete("k", [10, 50, 100]),
                Discrete("lambda_reg", [0, 0.001, 0.01, 0.1]),
                Discrete("learning_rate", [0.001, 0.005, 0.01])
            ],
            metric=cornac_rmse,
            eval_method=ratio_split,
            n_trails=max_evals, # what's this
        )


    elif algorithm_name=="CornacMF":
        use_bias = args['bias']
        mf = MF(use_bias=use_bias, seed=0, verbose=verbose)
        # RandomSearch
        rs = RandomSearch(
            model=mf,
            space=[
                Discrete("k", [10, 50, 100]),
                Discrete("lambda_reg", [0, 0.001, 0.01, 0.1]),
                Discrete("learning_rate", [0.001, 0.005, 0.01])
            ],
            metric=cornac_rmse,
            eval_method=ratio_split,
            n_trails=max_evals, # what's this
        )

    
        

    cornac.Experiment(
        eval_method=ratio_split,
        models=[rs],
        metrics=[cornac_rmse],
        user_based=False,
        verbose=verbose,
        save_dir="cornacLogs"
        ).run()

    best_params = rs.best_params
    print('Best result: ', best_params)

    return best_params


def optimize_lkpy(ratings, algorithm_name, args, partition_way, max_evals):


        
    def objective_mf(params):
        features, reg = params['features'], params['reg']
        #print(features, reg)
        
        algorithm = als.BiasedMF(features=features, reg=reg, iterations=iterations,  bias=bias, rng_spec=0)
        fittable = util.clone(algorithm)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train_df)
    
        preds = batch.predict(algo=fittable, pairs=test_df)
        loss = rmse(preds.prediction, preds.rating)
        
        return {
            'loss': loss,
            'status': STATUS_OK,
             }

    def objective_userknn(params):
    
        nnbrs = params['nnbrs']
        
        algorithm = user_knn.UserUser(nnbrs=nnbrs, min_sim=min_sim, min_nbrs=min_nbrs)
        fittable = util.clone(algorithm)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train_df)
    
        preds = batch.predict(algo=fittable, pairs=test_df)
        loss = rmse(preds.prediction, preds.rating)

        return {
            'loss': loss,
            'status': STATUS_OK,
            }

    def objective_bpr(params):
        features, reg = params['features'], params['reg']
        #print(features, reg)
        
        algorithm = lenskit_tf.BPR(features=features, reg=reg, epochs=epochs, rng_spec=0)
        fittable = util.clone(algorithm)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train_df)
    
        preds = batch.predict(algo=fittable, pairs=test_df)
        loss = rmse(preds.prediction, preds.rating)
        
        return {
            'loss': loss,
            'status': STATUS_OK,
             }
    
    # divide ratings into train and test

    if partition_way == "user":
        sample = xf.SampleFrac(0.2, rng_spec=0)
        sets = [i for i in enumerate(xf.partition_users(ratings,5, sample,rng_spec=0))]
    elif partition_way == "row":
        sets = [i for i in enumerate(xf.partition_rows(ratings,5, rng_spec=0))]

    train_df, test_df = sets[0][1]
    num_users = len(ratings.user.unique())

    # set evaluation space and fixed parameters
    if algorithm_name=='UserKNN':
        # i removed min nnbrs
        space = {'nnbrs': hp.randint('nnbrs', 1, num_users)}
        # center = args['center']
        min_nbrs = args['min_nbrs']
        min_sim = args['min_sim']
        obj_function = objective_userknn

    elif algorithm_name=='MF':
        space = {'features': hp.choice('features', [10, 50,100]),
                'reg': hp.choice('reg', [0, 0.001, 0.01, 0.1])}

        iterations = 100
        bias = args['bias']
        obj_function = objective_mf


    elif algorithm_name=='BPR':
        space = {'features': hp.choice('features', [10, 50,100]),
                'reg': hp.choice('reg', [0, 0.001, 0.01, 0.1]),
                 'neg_weight': hp.choice('neg_weight', [True, False])}
        epochs = 100
        
        obj_function = objective_bpr

    # optimize!
    trials = Trials()
    best_params = fmin(obj_function, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    print("Best result: ",best_params)
    return best_params
