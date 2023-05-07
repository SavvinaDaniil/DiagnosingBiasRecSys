# set environment variables
import os
import numpy as np
# basic functions
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from collections import defaultdict


# lenskit RS library
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, item_knn, user_knn, basic
from lenskit import util, batch, topn
from lenskit.metrics.predict import rmse

# cornac RS library
import cornac
from cornac.eval_methods import BaseMethod
from cornac.models import ItemKNN, UserKNN
from cornac.metrics import RMSE

from enrich import calculate_user_similarities
import analysis

def track_longtail_items(train_df, user_col, item_col, limit = 0.2):
    item_dist, _ = analysis.item_distribution(train_df, user_col, item_col, verbose = False)
    rev_list = np.flip(item_dist/len(train_df)).cumsum()
    longtail = rev_list[rev_list<limit].index.values
    return longtail

def track_shorttail_items(train_df, user_col, item_col, limit = 0.2):
    item_dist, _ = analysis.item_distribution(train_df, user_col, item_col, verbose = False)
    rev_list = np.flip(item_dist/len(train_df)).cumsum()
    shorttail = rev_list[rev_list>=limit].index.values
    return shorttail

def calculate_ACLT(train_df, recs, user_col, item_col, limit = 0.2):
    longtail_items_in_training_set = track_longtail_items(train_df, user_col, item_col, limit)
    aclt = len(recs[recs[item_col].isin(longtail_items_in_training_set)])/len(recs[user_col].unique())
    return aclt

def calculate_ACST(train_df, recs, user_col, item_col, limit = 0.2):
    shorttail_items_in_training_set = track_shorttail_items(train_df, user_col, item_col, limit)
    acst = len(recs[recs[item_col].isin(shorttail_items_in_training_set)])/len(recs[user_col].unique())
    return acst

def calculate_topn_metrics(recs, test_df):
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test_df)
    return results.precision.mean(), results.recall.mean(), results.ndcg.mean()

def calculate_pop_bias_per_item(all_items, item_col, user_col, predict_col, train_df, recs):
    df = pd.DataFrame(index = all_items)
    df["profile"] = 0.0
    df["recommendation"] = 0.0
    df["average_rating"] = 0.0
    prof = train_df[item_col].value_counts()/len(train_df[user_col].unique())
    rating = train_df.groupby(item_col).mean()[predict_col]
    rec = recs[item_col].value_counts()
    items_in_profile = prof.index.values
    items_in_recommendation = rec.index.values
    items_in_ratings = rating.index.values
    df.at[items_in_profile,"profile"] = prof.values
    df.at[items_in_recommendation,"recommendation"] = rec.values
    df.at[items_in_ratings,"average_rating"] = rating.values
    return df

def calculate_ave_pop_per_user(test_users, item_col, user_col, pop_bias, train_df, recs_grouped):
    # don't do test users because maybe not everyone manages to get recommendations I guess.
    index = recs_grouped.index.values
    df = pd.DataFrame(index = index)
    df["profile"] = 0.0
    df["recommendation"] = 0.0
    df["profile"] = [pop_bias.profile.loc[train_df[train_df[user_col]==x][item_col].values].mean() for x in df.index.values]
    df["recommendation"] = [pop_bias.profile.loc[recs_grouped[x]].mean() for x in df.index.values]
    
    return df

def evaluate_item_coverage(recs):
    return sum(recs>0)/len(recs)

def plot_results(dfs, GAP, algo_name, loss,precision, recall,ndcg, stdev_20, ACLT, ACST, cv, n):
    font = {'size'   : 15}
    matplotlib.rc('font', **font)
    
    fig, axs = plt.subplots(2, 2)
    item_coverage = evaluate_item_coverage(dfs[0]["recommendation"].values)
    suptitle = 'For '+str(n)+' recommendations \n Coverage: '+ str(round(item_coverage,2))+"\n RMSE: "+str(np.round(loss,2))+"\n #Unique Scores in the first "+str(n+10)+" items: "+str(np.round(stdev_20,2))+"\n Precision@10: "+str(np.round(precision,5))+"\n Recall@10: "+str(np.round(recall,5))+"\n NDCG@10: "+str(np.round(ndcg,5))
    plt.suptitle(suptitle, fontsize=25)
    
    # plot 1
    if cv:
        x = dfs[0]["profile"].values/5
    else:
        x = dfs[0]["profile"].values
    y = dfs[0]["recommendation"].values
    
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x,y])
    #print(xy)
    z = stats.gaussian_kde(xy)(xy)
    axs[0,0].plot(x, line)
    axs[0,0].set_xlabel("Item popularity",fontsize=20)
    axs[0,0].set_ylabel("Recommendation frequency",fontsize=20)
    axs[0,0].set_title('Correlation: ' + str(round(r_value,2)),fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0,0].scatter(x, y, c=z, s=50)
    
    
    
    
    if cv:
        x = dfs[0]["average_rating"].values/5
    else:
        x = dfs[0]["average_rating"].values
    y = dfs[0]["recommendation"].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x,y])
    z = stats.gaussian_kde(xy)(xy)
    axs[0,1].plot(x, line)
    axs[0,1].set_xlabel("Item average rating",fontsize=20)
    axs[0,1].set_ylabel("Recommendation frequency",fontsize=20)
    axs[0,1].set_title('Correlation: ' + str(round(r_value,2)),fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0,1].scatter(x, y, c=z, s=50)
    
    
    # plot 2
    
    x = ["Average Longtail", "Average Shortail"]
    y = [ACLT, ACST]
    
    
   
    axs[1,0].bar(x, y)
    axs[1,0].set_xlabel("Items",fontsize=20)
    axs[1,0].set_ylabel("Average longtail and shorttail items in 10 recommendations",fontsize=20)
    #axs[1,0].set_title('Correlation: ' + str(round(r_value,2)),fontsize=20)
    
    
    
    
    x1 = GAP["profile"].values
    y1 = GAP["recommendation"].values
    
    DGAP = (y1-x1)/x1*100
    x = x1
    y = DGAP
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x,y])
    z = stats.gaussian_kde(xy)(xy)
    #axs[1,1].plot(x, y)
    axs[1,1].set_xlabel("User profile AP",fontsize=20)
    axs[1,1].set_ylabel("User %ΔAP",fontsize=20)
    axs[1,1].set_title(' ',fontsize=20)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1,1].scatter(x, y, c=z, s=50)
    axs[1,1].axhline(y=np.nanmean(y), color='red', linestyle='--', linewidth=3, label='Average %ΔGAP')
    axs[1,1].legend()
    
    

    
    
    fig.set_figheight(20)
    fig.set_figwidth(20)
    fig.tight_layout(pad=3.0)
    plt.show()
def recommend_cornac(exp, all_items, user_col, item_col, n):
    
    eval_method = exp.eval_method
    model = exp.models[0]
    
    user_variances = []
    all_recs = []     
    for uid in eval_method.test_set.uid_map.values(): # every user in the test set
        uid_variance = []
        user_id = list(eval_method.train_set.user_ids)[uid]
        recs = pd.DataFrame(index=range(n))
        recs[user_col] = user_id
        user_items_in_the_train_set = set(eval_method.train_set.user_data[uid][0])
        user_items_not_in_the_train_set = list(all_items.difference(user_items_in_the_train_set))
        #print(user_id, len(user_items_not_in_the_train_set))

        item_rank = model.rank(user_idx=uid, item_indices = user_items_not_in_the_train_set)[0] # items the user has NOT rated in the TRAIN set

        item_rank_top = item_rank[:n]
        rec_items = []
        for iid in item_rank_top:
            item_id = list(model.train_set.item_ids)[iid]

            rec_items.append(item_id)
        recs[item_col] = rec_items
        all_recs.append(recs)
        
        item_rank_for_variance = item_rank[:n+10]
        for iid in item_rank_for_variance:
            score = model.score(uid, iid)
            uid_variance.append(score)
        uid_variance = len(np.unique(uid_variance))
        user_variances.append(uid_variance)
        
    all_recs = pd.concat(all_recs, ignore_index=True)
    mean_std_20 = np.mean(user_variances)
    
    return all_recs, mean_std_20


    
def train_algorithm_cornac(algorithm, algo_name, centering, nnbrs, ratings, evaluation_way, partition_way, sampling_strategy = None,  verbose = True, min_sim = 0, user_col="user", item_col="item", predict_col = "rating", n = 10):
    all_items=set(ratings.item.unique())
    
   
    
    if partition_way == "user":
        if sampling_strategy == "frac":
            sample = xf.SampleFrac(0.2, rng_spec=0)
        elif sampling_strategy == "N":
            sample = xf.SampleN(5, rng_spec=0)
        sets = [i for i in enumerate(xf.partition_users(ratings,5, sample,rng_spec=0))]
    elif partition_way == "row":
        sets = [i for i in enumerate(xf.partition_rows(ratings,5, rng_spec=0))]
        
    if evaluation_way == "simple_split":
        
        
        pop_biases = []
        
        
        train_df, test_df = sets[0][1]
        
        eval_method = BaseMethod.from_splits(train_data=list(train_df[["user","item","rating"]].to_records(index=False)), test_data=list(test_df[["user","item","rating"]].to_records(index=False)), exclude_unknowns=False, verbose=False)
        
        models = [algorithm(k=nnbrs, mean_centered=centering)]
        metrics = [RMSE()]
        exp = cornac.Experiment(eval_method=eval_method, models=models, metrics=metrics, user_based=False,save_dir="cornacLogs")
        exp.run()
        loss = exp.result[0].metric_avg_results["RMSE"]
        
        if verbose:
            print("Training done!")
            
        test_users = test_df.user.unique() # the users in the test set
        
        recs, stdev_20 = recommend_cornac(exp = exp, all_items=all_items, user_col=user_col, item_col=item_col, n=n)
        recs_grouped = recs.groupby([user_col])[item_col].apply(list)
        if verbose:
            print("Recommendation done!")
                
            
        precision, recall, ndcg = calculate_topn_metrics(recs, test_df)
            
        pop_bias = calculate_pop_bias_per_item(all_items, item_col, user_col, predict_col, train_df, recs)
        
        GAP_vs_GAP = calculate_ave_pop_per_user(test_users, item_col, user_col, pop_bias, train_df, recs_grouped)
        
        ACLT = calculate_ACLT(train_df, recs, user_col, item_col, limit = 0.2)
        ACST = calculate_ACST(train_df, recs, user_col, item_col, limit = 0.2)
        pop_biases.append(pop_bias)
        
        
        plot_results(pop_biases.copy(), GAP_vs_GAP.copy(), algo_name, loss,precision, recall,ndcg, stdev_20, ACLT,ACST,cv=False, n=n)
        return pop_biases
    
    elif evaluation_way == "cross_validation":
        
        total_loss = 0.0
        total_pop_bias_10 = 0.0
        total_stdev_20 = 0.0
        total_ACLT = 0.0
        total_ACST = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_ndcg = 0.0
        total_GAP_vs_GAP = []
        
        #sets_list = enumerate(xf.partition_users(ratings,5, sample, rng_spec=0))
        
            
        for i, tp in sets:
            train_df = tp[0]
            test_df = tp[1]
            eval_method = BaseMethod.from_splits(train_data=list(train_df[["user","item","rating"]].to_records(index=False)), test_data=list(test_df[["user","item","rating"]].to_records(index=False)), exclude_unknowns=False, verbose=False)
            models = [algorithm(k=nnbrs, mean_centered=centering)]
            metrics = [RMSE()]
            exp = cornac.Experiment(eval_method=eval_method, models=models, metrics=metrics, user_based=False,save_dir="cornacLogs")
            exp.run()
            loss = exp.result[0].metric_avg_results["RMSE"]
            total_loss+=loss
            if verbose:
                print("Training done!")
        
            test_users = test_df.user.unique() # the users in the test set
            recs, stdev_20 = recommend_cornac(exp = exp, all_items=all_items, user_col=user_col, item_col=item_col, n=n)
            recs_grouped = recs.groupby([user_col])[item_col].apply(list)
            total_stdev_20+=stdev_20
            
            if verbose:
                print("Recommendation done!")
            
            precision, recall, ndcg = calculate_topn_metrics(recs, test_df)
            pop_bias = calculate_pop_bias_per_item(all_items, item_col, user_col, predict_col, train_df, recs)
            GAP_vs_GAP = calculate_ave_pop_per_user(test_users, item_col, user_col, pop_bias, train_df, recs_grouped)
            ACLT = calculate_ACLT(train_df, recs, user_col, item_col, limit = 0.2)
            ACST = calculate_ACST(train_df, recs, user_col, item_col, limit = 0.2)
            
            total_ACST += ACST
            total_ACLT += ACLT
            total_GAP_vs_GAP.append(GAP_vs_GAP)
            total_pop_bias_10+=pop_bias
            total_precision+=precision
            total_recall+=recall
            total_ndcg+=ndcg

                    
            
            
        total_loss/=5
        total_precision/=5
        total_recall/=5
        total_ndcg/=5
        total_ACLT/=5
        total_ACST/=5

        total_stdev_20/=5
        total_GAP_vs_GAP = pd.concat(total_GAP_vs_GAP)
        pop_biases = [total_pop_bias_10]
        
        plot_results(pop_biases.copy(), total_GAP_vs_GAP.copy(), algo_name, total_loss,total_precision,total_recall, total_ndcg, total_stdev_20, total_ACLT,total_ACST,cv=True, n=n)
        
        return pop_biases
    

def train_algorithm(algorithm, algo_name, centering, nnbrs, ratings, evaluation_way, partition_way, sampling_strategy = None, verbose = True, min_sim = 0, user_col="user", item_col="item", predict_col = "rating", n=10, fallback = False):
    all_items=set(ratings.item.unique())
    if partition_way == "user":
        if sampling_strategy == "frac":
            sample = xf.SampleFrac(0.2, rng_spec=0)
        elif sampling_strategy == "N":
            sample = xf.SampleN(5, rng_spec=0)
        sets = [i for i in enumerate(xf.partition_users(ratings,5, sample,rng_spec=0))]
    elif partition_way == "row":
        sets = [i for i in enumerate(xf.partition_rows(ratings,5, rng_spec=0))]
        
    if evaluation_way == "simple_split":
        
        pop_biases = []
        train_df, test_df = sets[0][1]
        
            
        algo = algorithm(nnbrs=nnbrs, center=centering, min_sim=min_sim)
        
        
        
        if fallback:
            algo_w_f = basic.Fallback(algo, basic.Bias())
            fittable = util.clone(algo_w_f)
        else:
            fittable = util.clone(algo)
            
            
       
        fittable = Recommender.adapt(fittable)
        fittable.fit(train_df)
        
        if verbose:
            print("Training done!")
            
        preds = batch.predict(algo=fittable, pairs=test_df)
        loss = np.round(rmse(preds.prediction, preds.rating),2)
        
        if verbose:
            print("Prediction done!")
            
        test_users = test_df.user.unique() # the users in the test set
        # check variance 
        recs_20 = batch.recommend(algo = fittable, users = test_users, n = n+10, candidates = None)
        mean_std_first_20 = recs_20.groupby(user_col).nunique()["score"].mean()
        
        recs = recs_20[recs_20["rank"]<=n]
        recs_grouped = recs.groupby([user_col])[item_col].apply(list)
        
        
        if verbose:
            print("Recommendation done!")
            
        
        precision, recall, ndcg = calculate_topn_metrics(recs, test_df) 
        pop_bias = calculate_pop_bias_per_item(all_items, item_col, user_col, predict_col, train_df, recs)
        
        GAP_vs_GAP = calculate_ave_pop_per_user(test_users, item_col, user_col, pop_bias, train_df, recs_grouped)
        ACLT = calculate_ACLT(train_df, recs, user_col, item_col, limit = 0.2)
        ACST = calculate_ACST(train_df, recs, user_col, item_col, limit = 0.2)
        
        pop_biases.append(pop_bias)
        
        
        plot_results(pop_biases.copy(), GAP_vs_GAP.copy(), algo_name, loss, precision, recall,ndcg,mean_std_first_20,ACLT,ACST, cv=False, n=n)
        return pop_biases
    
    elif evaluation_way == "cross_validation":
        
        total_loss = 0.0
        total_st_dev_20 = 0.0
        total_pop_bias_10 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_ACLT = 0.0
        total_ACST = 0.0
        total_ndcg = 0.0
        total_GAP_vs_GAP = []
        
        
        for i, tp in sets:
            train_df = tp[0]
            test_df = tp[1]
            algo = algorithm(nnbrs=nnbrs, center=centering, min_sim=min_sim)
            
            
            if fallback:
                algo_w_f = basic.Fallback(algo, basic.Bias())
                fittable = util.clone(algo_w_f)
            else:
                fittable = util.clone(algo)
            
            
            fittable = Recommender.adapt(fittable)
            fittable.fit(train_df)

            if verbose:
                print(i,"Training done!")
                
            preds = batch.predict(algo=fittable, pairs=test_df)
            loss = np.round(rmse(preds.prediction, preds.rating),2)
            total_loss+=loss
            if verbose:
                print("Prediction done!")
            
                
            test_users = test_df.user.unique() # the users in the test set
            
            # check variance 
            recs_20 = batch.recommend(algo = fittable, users = test_users, n = n+10, candidates = None)
            mean_std_first_20 = recs_20.groupby(user_col).nunique()["score"].mean()
            total_st_dev_20 += mean_std_first_20
            

            recs = recs_20[recs_20["rank"]<=n]
            recs_grouped = recs.groupby([user_col])[item_col].apply(list)
            
            
            if verbose:
                print(i, "Recommendation done!")
            
            precision, recall, ndcg = calculate_topn_metrics(recs, test_df)
            
            pop_bias = calculate_pop_bias_per_item(all_items, item_col, user_col, predict_col, train_df, recs)    
            GAP_vs_GAP = calculate_ave_pop_per_user(test_users, item_col, user_col, pop_bias, train_df, recs_grouped)
            ACLT = calculate_ACLT(train_df, recs, user_col, item_col, limit = 0.2)
            ACST = calculate_ACST(train_df, recs, user_col, item_col, limit = 0.2)
            
            total_ACLT+=ACLT
            total_ACST+=ACST
            total_GAP_vs_GAP.append(GAP_vs_GAP)
            total_pop_bias_10+=pop_bias
            total_precision+=precision
            total_recall+=recall
            total_ndcg+=ndcg
                
        total_loss/=5
        total_precision/=5
        total_recall/=5
        total_ndcg/=5
        total_ACLT/=5
        total_ACST/=5
        
        total_st_dev_20/=5
        total_GAP_vs_GAP = pd.concat(total_GAP_vs_GAP)
        pop_biases = [total_pop_bias_10]
        
        plot_results(pop_biases.copy(), total_GAP_vs_GAP,algo_name, total_loss,total_precision,total_recall, total_ndcg, total_st_dev_20,total_ACLT,total_ACST, cv=True, n=n)
        
        return pop_biases
        
        

    
    
    
