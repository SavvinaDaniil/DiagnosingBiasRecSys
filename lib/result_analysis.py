
import pickle as pkl
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def highest_average(df_list, column_name = 'recommendation'):
    highest_average = -(10**6)
    highest_i = -1
    for i in range(len(df_list)):
        df = df_list[i]
        mean = np.mean(df[column_name].values)
        # print('mean', mean)
        if mean > highest_average:
            highest_average = mean
            highest_i = i
    # print(highest_average, highest_i)
    return highest_average, highest_i
    
def mannwhitneyu_test(df_list, alt = 'greater', column_name = 'recommendation'):
    # find the highest average 
    
    ha, hi = highest_average(df_list, column_name)
    print('Highest version: ', hi)
    inds_df_list = list(range(len(df_list)))
    to_test_inds = inds_df_list[:hi] + inds_df_list[hi+1:]
    # print(to_test_inds)
    df1 = df_list[hi]
    pvalues = []
    for ind in to_test_inds:
        df2 = df_list[ind]
        x = df1[column_name].values
        y = df2[column_name].values
        pvalue = mannwhitneyu(x,y, alternative = alt)[1]
        pvalues.append(pvalue)
    return [(to_test_inds[i],pvalues[i]) for i in range(len(pvalues))] # pvalues for all comparisons

def analyse(data,algorithm):
    if data == 'fairbook' or data=='ml1m' or data == 'epinion':
        analyse_with_real_dataset(data, algorithm)
    elif data == 'synthetic':
        analyse_with_synthetic_dataset(algorithm)
    else: 
        print('Unavailable dataset.')


def analyse_with_real_dataset(data_strategy, algo_name):

    algo_versions = {"UserKNN":[{'min_nbrs':1, 'min_sim':0},
                                {'min_nbrs':2, 'min_sim':0},
                                # {'min_nbrs':10, 'min_sim':0},
                                {'min_nbrs':1, 'min_sim':-1},
                                {'min_nbrs':2, 'min_sim':-1},
                                # {'min_nbrs':10, 'min_sim':-1}
                               ],
                    "MF": [{"bias": True}, {"bias": False}],
                    "CornacUserKNN":[{'center':True}],
                     "CornacMF": [{"bias": True}, {"bias": False}]}
    if data_strategy == 'ml1m':
        algo_versions["UserKNN"]=[{'min_nbrs':1, 'min_sim':0},
                                {'min_nbrs':2, 'min_sim':0},
                                {'min_nbrs':10, 'min_sim':0},
                                {'min_nbrs':1, 'min_sim':-1},
                                {'min_nbrs':2, 'min_sim':-1},
                                {'min_nbrs':10, 'min_sim':-1}
                               ]
    lkpy_versions = algo_versions[algo_name]
    file_location = "experimental_results/" + algo_name + "/"

    results = []
    for args in lkpy_versions:
        file = open(file_location + data_strategy + "_" + str(args) + ".pkl", "rb")
        result = pkl.load(file)
        results.append(result)
    
    stringed_versions = [str(args) for args in lkpy_versions]
    if algo_name == 'UserKNN':
        # Initialize empty lists for the two halves
        min_nbrs = []
        min_sim = []
        
        # Split the strings and populate the lists
        for s in stringed_versions:
            parts = s.split(", ")
        
            min_nbrs.append(parts[0].split(" ")[-1])
            min_sim.append(parts[1].split(" ")[-1].split("}")[0])
        
        over_common = ["False"]
        index = pd.MultiIndex.from_product(
            [min_sim, min_nbrs, over_common],
            names=["MinimumSimilarity", "MinimumNeighbours", "OverCommon"],
        ).drop_duplicates()

    elif algo_name == 'MF':
        # Initialize empty lists for the two halves
        bias = []
        # Split the strings and populate the lists
        for s in stringed_versions:
            parts = s.split(": ")
            bias.append(parts[-1].split("}")[0])
        library = ["Lenskit"]
        index = pd.MultiIndex.from_product(
            [bias, library], names=["Bias", "Library"]
        ).drop_duplicates()
    
    results_lkpy = pd.DataFrame(results, index=index)
    lkpy_detailed_results = []
    for args in lkpy_versions:
        file = open(file_location + 'correct_detailed_per_item_'+data_strategy + "_" + str(args) + ".pkl", "rb")
        result = pkl.load(file)
        lkpy_detailed_results.append(result)


# ## Cornac
    if algo_name == 'UserKNN':
        algo_name = 'CornacUserKNN'
    elif algo_name == 'MF':
        algo_name = "CornacMF"

    cornac_versions = algo_versions[algo_name]
    file_location = "experimental_results/" + algo_name + "/"


# In[135]:


    results = []
    for args in cornac_versions:
        file = open(file_location + data_strategy + "_" + str(args) + ".pkl", "rb")
        result = pkl.load(file)
        results.append(result)
    
    stringed_versions = [str(args) for args in cornac_versions]
    
    if algo_name == 'CornacUserKNN':
    
        # Initialize empty lists for the two halves
        min_nbrs = []
        min_sim = []
        # Split the strings and populate the lists
        for s in stringed_versions:
            parts = s.split(": ")
            min_nbrs.append(parts[-1].split("}")[0])
            min_sim.append("-1")
        
        over_common = ["True"]
        index = pd.MultiIndex.from_product(
            [min_sim, min_nbrs, over_common],
            names=["MinimumSimilarity", "MinimumNeighbours", "OverCommon"],
        ).drop_duplicates()
        results_cornac = pd.DataFrame(results, index=index)
        results_cornac.index = results_cornac.index.set_levels(
            results_cornac.index.levels[1].str.replace("True", "1"), level=1
        )

    elif algo_name == "CornacMF":
        # Initialize empty lists for the two halves
        bias = []
        
        # Split the strings and populate the lists
        for s in stringed_versions:
            parts = s.split(": ")
            bias.append(parts[-1].split("}")[0])
        library = ["Cornac"]
        index = pd.MultiIndex.from_product(
            [bias, library], names=["Bias", "Library"]
        ).drop_duplicates()    
        results_cornac = pd.DataFrame(results, index=index)
        
    
    
    
    cornac_detailed_results = []
    for args in cornac_versions:
        file = open(file_location + 'correct_detailed_per_item_'+data_strategy + "_" + str(args) + ".pkl", "rb")
        result = pkl.load(file)
        cornac_detailed_results.append(result)

    
    
    if algo_name == "CornacUserKNN":
        user_knn_metrics = (
            pd.concat([results_lkpy, results_cornac])
            .reset_index()
            .sort_values(["MinimumSimilarity", "OverCommon", "MinimumNeighbours"])
            .set_index(["MinimumSimilarity", "OverCommon", "MinimumNeighbours"])
        )
        metrics_order = ["pop_corr", "ARP", "ave_PL", "ACLT", "AggDiv", "RMSE", "NDCG"]
        user_knn_metrics = user_knn_metrics[metrics_order]
        user_knn_metrics = user_knn_metrics.rename(
            columns={"pop_corr": "PopCorr", "ave_PL": "PL", "ACLT": "APLT", "NDCG": "NDCG@10"}
        ).reindex(["1", "2", "5", "10"], level=2)
    
    
        user_knn_metrics['RealPopCorr'] = user_knn_metrics.PopCorr.apply(lambda x: x[0])
        user_knn_metrics['Significance'] = user_knn_metrics.PopCorr.apply(lambda x: True if x[1]<0.005 else False)
        user_knn_metrics['PopCorr'] = user_knn_metrics.RealPopCorr 
        user_knn_metrics = user_knn_metrics.drop('RealPopCorr', axis=1)
    
        with open("metrics_combined/"+data_strategy+"_all_user_knn.pkl", "wb") as f:
            pkl.dump(user_knn_metrics.round(3), f)

    elif algo_name == 'CornacMF':
        mf_metrics = (
            pd.concat([results_lkpy, results_cornac])
            .reset_index()
            .sort_values(["Library", "Bias"])
            .set_index(["Library", "Bias"])
        )
        metrics_order = ["pop_corr", "ARP", "ave_PL", "ACLT", "AggDiv", "RMSE", "NDCG"]
        mf_metrics = mf_metrics[metrics_order]
        mf_metrics = mf_metrics.rename(
            columns={"pop_corr": "PopCorr", "ave_PL": "PL", "ACLT": "APLT", "NDCG": "NDCG@10"}
        )
        mf_metrics['RealPopCorr'] = mf_metrics.PopCorr.apply(lambda x: x[0])
        mf_metrics['Significance'] = mf_metrics.PopCorr.apply(lambda x: True if x[1]<0.005 else False)
        mf_metrics['PopCorr'] = mf_metrics.RealPopCorr 
        mf_metrics = mf_metrics.drop('RealPopCorr', axis=1)

        with open("metrics_combined/"+data_strategy+"_all_mf.pkl", "wb") as f:
            pkl.dump(mf_metrics.round(3), f)
        
    
   
    
    
    
    # # Significance tests
    print(cornac_versions+lkpy_versions) 
    print('Use the above to figure out significance comparisons.')
    results = cornac_detailed_results+lkpy_detailed_results
    
    # ## 1. Average Recommendation Popularity

    print("ARP:")
    print(mannwhitneyu_test(results))
    
    
    # ## 2. Popularity Lift
    
    for df in results:
        df['popularity_lift'] = (df['recommendation']-df['profile'])/df['profile']*100
    print("PL:")
    print(mannwhitneyu_test(results, column_name = 'popularity_lift')) 
    
    
    
    
    
def analyse_with_synthetic_dataset(algo_name):

    algo_versions = {"UserKNN":[{'min_nbrs':1, 'min_sim':0},
                                {'min_nbrs':2, 'min_sim':0},
                                {'min_nbrs':1, 'min_sim':-1},
                                {'min_nbrs':2, 'min_sim':-1},
                               ],
                    "MF": [{"bias": True}, {"bias": False}],
                    "CornacUserKNN":[{'center':True}],
                     "CornacMF": [{"bias": True}, {"bias": False}]}



    
    lkpy_versions = algo_versions[algo_name]
    file_location = "experimental_results/" + algo_name + "/"


    data_strategies = [
        "uniformly_random",
        "popularity_good",
        "popularity_bad",
        "popularity_good_for_bp_ur",
        "popularity_bad_for_bp_ur",
    ]

    results = []
    for data_strategy in data_strategies:
        for args in lkpy_versions:
            file = open(file_location + data_strategy + "_" + str(args) + ".pkl", "rb")
            result = pkl.load(file)
            results.append(result)
    
    stringed_versions = [str(args) for args in lkpy_versions]
    
    if algo_name == 'UserKNN':
        # Initialize empty lists for the two halves
        min_nbrs = []
        min_sim = []
        
        # Split the strings and populate the lists
        for s in stringed_versions:
            parts = s.split(", ")
        
            min_nbrs.append(parts[0].split(" ")[-1])
            min_sim.append(parts[1].split(" ")[-1].split("}")[0])
        
        over_common = ["False"]
        index = pd.MultiIndex.from_product(
            [data_strategies, min_sim, min_nbrs, over_common],
            names=["DataStrategy", "MinimumSimilarity", "MinimumNeighbours", "OverCommon"],
        ).drop_duplicates()

    elif algo_name == 'MF':
        # Initialize empty lists for the two halves
        bias = []
        # Split the strings and populate the lists
        for s in stringed_versions:
            parts = s.split(": ")
            bias.append(parts[-1].split("}")[0])
        library = ["Lenskit"]
        index = pd.MultiIndex.from_product(
            [data_strategies, bias, library], names=["DataStrategy", "Bias", "Library"]
        ).drop_duplicates()
    
    results_lkpy = pd.DataFrame(results, index=index)
    lkpy_dict_detailed = {}
    for data_strategy in data_strategies:
        lkpy_detailed_results = []
        for args in lkpy_versions:
            file = open(file_location + 'correct_detailed_per_item_'+data_strategy + "_" + str(args) + ".pkl", "rb")
            result = pkl.load(file)
            lkpy_detailed_results.append(result)
        lkpy_dict_detailed[data_strategy] = lkpy_detailed_results


# ## Cornac
    if algo_name == 'UserKNN':
        algo_name = 'CornacUserKNN'
    elif algo_name == 'MF':
        algo_name = "CornacMF"

    cornac_versions = algo_versions[algo_name]
    file_location = "experimental_results/" + algo_name + "/"


# In[135]:


    results = []
    for data_strategy in data_strategies:
        for args in cornac_versions:
            file = open(file_location + data_strategy + "_" + str(args) + ".pkl", "rb")
            result = pkl.load(file)
            results.append(result)
    
    stringed_versions = [str(args) for args in cornac_versions]
    
    if algo_name == 'CornacUserKNN':
    
        # Initialize empty lists for the two halves
        min_nbrs = []
        min_sim = []
        # Split the strings and populate the lists
        for s in stringed_versions:
            parts = s.split(": ")
            min_nbrs.append(parts[-1].split("}")[0])
            min_sim.append("-1")
        
        over_common = ["True"]
        index = pd.MultiIndex.from_product(
            [data_strategies, min_sim, min_nbrs, over_common],
            names=["DataStrategy", "MinimumSimilarity", "MinimumNeighbours", "OverCommon"],
        ).drop_duplicates()
        results_cornac = pd.DataFrame(results, index=index)
        results_cornac.index = results_cornac.index.set_levels(
            results_cornac.index.levels[1].str.replace("-1", "1"), level=2
        )

    elif algo_name == "CornacMF":
        # Initialize empty lists for the two halves
        bias = []
        
        # Split the strings and populate the lists
        for s in stringed_versions:
            parts = s.split(": ")
            bias.append(parts[-1].split("}")[0])
        library = ["Cornac"]
        index = pd.MultiIndex.from_product(
            [data_strategies, bias, library], names=["DataStrategy", "Bias", "Library"]
        ).drop_duplicates()    
        results_cornac = pd.DataFrame(results, index=index)
        
    
    
    
    cornac_dict_detailed = {}
    for data_strategy in data_strategies:
        cornac_detailed_results = []
        for args in cornac_versions:
            file = open(file_location + 'correct_detailed_per_item_'+data_strategy + "_" + str(args) + ".pkl", "rb")
            result = pkl.load(file)
            cornac_detailed_results.append(result)
        cornac_dict_detailed[data_strategy] = cornac_detailed_results

    
    
    if algo_name == "CornacUserKNN":
        user_knn_metrics = (
            pd.concat([results_lkpy, results_cornac])
            .reset_index()
            .sort_values(
                ["DataStrategy", "MinimumSimilarity", "OverCommon", "MinimumNeighbours"]
            )
            .set_index(["DataStrategy", "MinimumSimilarity", "OverCommon", "MinimumNeighbours"])
            .reindex(data_strategies, level=0)
        )
        metrics_order = ["pop_corr", "ARP", "ave_PL", "ACLT", "AggDiv", "RMSE", "NDCG"]
        user_knn_metrics = user_knn_metrics[metrics_order]
        user_knn_metrics = user_knn_metrics.rename(
            columns={"pop_corr": "PopCorr", "ave_PL": "PL", "ACLT": "APLT", "NDCG": "NDCG@10"}
        )
        user_knn_metrics = user_knn_metrics.rename(
            index={
                "uniformly_random": "Scenario 1",
                "popularity_good": "Scenario 2",
                "popularity_bad": "Scenario 3",
                "popularity_good_for_bp_ur": "Scenario 4",
                "popularity_bad_for_bp_ur": "Scenario 5",
            }
        )
        user_knn_metrics = user_knn_metrics.reindex(["1", "2", "5", "10"], level=3)
    
    
        user_knn_metrics['RealPopCorr'] = user_knn_metrics.PopCorr.apply(lambda x: x[0])
        user_knn_metrics['Significance'] = user_knn_metrics.PopCorr.apply(lambda x: True if x[1]<0.005 else False)
        user_knn_metrics['PopCorr'] = user_knn_metrics.RealPopCorr 
        user_knn_metrics = user_knn_metrics.drop('RealPopCorr', axis=1)
    
        with open("metrics_combined/all_user_knn.pkl", "wb") as f:
            pkl.dump(user_knn_metrics.drop("APLT", axis=1).round(3), f)

    elif algo_name == 'CornacMF':
        mf_metrics = (
            pd.concat([results_lkpy, results_cornac])
            .reset_index()
            .sort_values(["DataStrategy", "Library", "Bias"])
            .set_index(["DataStrategy", "Library", "Bias"])
            .reindex(data_strategies, level=0)
        )
        metrics_order = ["pop_corr", "ARP", "ave_PL", "ACLT", "AggDiv", "RMSE", "NDCG"]
        mf_metrics = mf_metrics[metrics_order]
        mf_metrics = mf_metrics.rename(
            columns={"pop_corr": "PopCorr", "ave_PL": "PL", "ACLT": "APLT", "NDCG": "NDCG@10"}
        )
        mf_metrics = mf_metrics.rename(
            index={
                "uniformly_random": "Scenario 1",
                "popularity_good": "Scenario 2",
                "popularity_bad": "Scenario 3",
                "popularity_good_for_bp_ur": "Scenario 4",
                "popularity_bad_for_bp_ur": "Scenario 5",
            }
        )
        
        mf_metrics['RealPopCorr'] = mf_metrics.PopCorr.apply(lambda x: x[0])
        mf_metrics['Significance'] = mf_metrics.PopCorr.apply(lambda x: True if x[1]<0.005 else False)
        mf_metrics['PopCorr'] = mf_metrics.RealPopCorr 
        mf_metrics = mf_metrics.drop('RealPopCorr', axis=1)

        with open("metrics_combined/all_mf.pkl", "wb") as f:
            pkl.dump(mf_metrics.round(3), f)
        
    
   
    
    
    for data_strategy in data_strategies:
        results = cornac_dict_detailed[data_strategy]+lkpy_dict_detailed[data_strategy]
        print(data_strategy)
        # # Significance tests
        print(cornac_versions+lkpy_versions) 
        print('Use the above to figure out significance comparisons.')
        
        
        # ## 1. Average Recommendation Popularity
    
        print("ARP:")
        print(mannwhitneyu_test(results))
        
        
        # ## 2. Popularity Lift
        
        for df in results:
            df['popularity_lift'] = (df['recommendation']-df['profile'])/df['profile']*100
        print("PL:")
        print(mannwhitneyu_test(results, column_name = 'popularity_lift')) 
        print("--------------------------------------------------------------")
    
    
    
    
    
