#!/usr/bin/env python
# coding: utf-8

# # Steps
# 1. Make data choice.
# 5. Choose 'fixed' configuration.
# 6. For each 'fixed' configuration, optimize the other parameters based on RMSE.
# 7. Given optimal setting, run popularity bias analysis for every version of the 'fixed' configuration.

# ## Libraries

# In[1]:


# get_ipython().run_line_magic('env', 'MKL_THREADING_LAYER=tbb')
# get_ipython().run_line_magic('env', 'OPENBLAS_NUM_THREADS=24')
# get_ipython().run_line_magic('env', 'NUMBA_NUM_THREADS=96')
# get_ipython().run_line_magic('env', 'MKL_NUM_THREADS=96')
# get_ipython().run_line_magic('env', 'OMP_NUM_THREADS=1')


# In[2]:


import os

os.environ["MKL_THREADING_LAYER"] = "tbb"
os.environ["OPENBLAS_NUM_THREADS"] = "24"
os.environ["NUMBA_NUM_THREADS"] = "96"
os.environ["MKL_NUM_THREADS"] = "96"
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# for random generation


# basic functions
import pandas as pd

pd.options.display.max_rows = 100
import pickle
import scipy

# custom-made functions
import modelling_mf
from optimize_hp import optimize_lkpy, optimize_cornac

# lenskit RS library
from lenskit.algorithms import als


# cornac RS library
from cornac.models import MF


# ## Data and parameters

# In[3]:


data = "fairbook"
# user-item interactions
fairbook_ratings = pd.read_csv("data/" + data + "_events.csv")


# In[4]:


data = "ml1m"
# user-item interactions
ml1m_ratings = pd.read_csv(
    "data/" + data + "_events.dat", header=None, sep="::", engine="python"
).drop(3, axis=1)
ml1m_ratings.columns = ["user", "item", "rating"]


# In[5]:


data = "epinion"
mat = scipy.io.loadmat("data/" + data + "_events.mat")
mat_df = pd.DataFrame(mat["rating_with_timestamp"])
mat_df.columns = ["user", "item", ".", "rating", "..", "..."]
epinion_ratings = mat_df[["user", "item", "rating"]]


# Make data choice.

# In[6]:


ratings = ml1m_ratings.copy()
ratings = ratings.drop_duplicates(subset=["user", "item"], keep="last")
data_strategy = "ml1m"


# In[7]:


user_col = "user"  # the name of the column that includes the users
item_col = "item"  # the name of the column that includes the items
predict_col = "rating"  # the name of the column that includes the interaction


# In[8]:


evaluation_way = "cross_validation"
verbose = True
plot = True
save_plot = True  # save the plots
fallback = False
nr_recs = 10
sampling_strategy = "frac"
partition_way = "user"


# ## Optimize, train, evaluate LKPY
# - **Algorithm**
# - **Fixed parameters**
# - **To-optimize parameters**

# In[9]:


# algo_versions = {"MF": [{"bias": True}, {"bias": False}]}


# # In[ ]:


# # choose algorithm
# algorithm_lkpy = als.BiasedMF
# algo_name = "MF"
# versions = algo_versions[algo_name]


# # for every 'fixed' version of the algorithm
# for args in versions:
#     print(args)


#     p = "best_parameters/" + algo_name + "/" + data_strategy + "_" + str(args) + ".pkl"
#     if os.path.isfile(p):
#         print("We got them already")
#         with open(p, "rb") as f:
#             best_params = pickle.load(f)
#     else:
#         print("We have to compute them now")
#         # optimize for this fixed version
#         best_params = optimize_lkpy(
#             ratings=ratings, algorithm_name=algo_name, args=args, max_evals=20, partition_way = 'row'
#         )

#         # save the best parameters for this fixed version

#         with open(
#             "best_parameters/"
#             + algo_name
#             + "/"
#             + data_strategy
#             + "_"
#             + str(args)
#             + ".pkl",
#             "wb",
#         ) as f:
#             pickle.dump(best_params, f)
    
#     # # optimize for this fixed version
#     # best_params = optimize_lkpy(
#     #     ratings=ratings,
#     #     algorithm_name=algo_name,
#     #     args=args,
#     #     partition_way="row",
#     #     max_evals=20,
#     # )

#     # # save the best parameters for this fixed version

#     # with open(
#     #     "best_parameters/" + algo_name + "/" + data_strategy + "_" + str(args) + ".pkl",
#     #     "wb",
#     # ) as f:
#     #     pickle.dump(best_params, f)

#     reg_list = [0, 0.001, 0.01, 0.1]
#     features_list = [10, 50, 100]
#     optimal_reg = reg_list[best_params["reg"]]
#     optimal_features = features_list[best_params["features"]]

#     # run the training and evaluation for the fixed version + the best other parameters
#     pop_biases_lkpy, metrics_dict_lkpy = modelling_mf.train_algorithm(
#         algorithm=algorithm_lkpy(
#             features=optimal_features, reg=optimal_reg, bias=args["bias"]
#         ),
#         algo_name=algo_name,
#         ratings=ratings,
#         evaluation_way=evaluation_way,
#         verbose=verbose,
#         n=nr_recs,
#         sampling_strategy=sampling_strategy,
#         partition_way=partition_way,
#         plot=plot,
#         data_strategy=data_strategy,
#         args=args,
#         save_plot=save_plot,
#     )

#     # Save metrics!
#     with open(
#         "experimental_results/"
#         + algo_name
#         + "/"
#         + data_strategy
#         + "_"
#         + str(args)
#         + ".pkl",
#         "wb",
#     ) as f:
#         pickle.dump(metrics_dict_lkpy, f)
#     with open('experimental_results/'+algo_name+'/detailed_per_item_'+data_strategy+'_'+str(args)+'.pkl', 'wb') as f:
#         pickle.dump(pop_biases_lkpy, f)


# ## Optimize, train, evaluate Cornac
# - **Algorithm**
# - **Fixed parameters**
# - **To-optimize parameters**

# In[9]:


mapping_dict = {}  # Create a dictionary that maps each item to an integer - necessary for Cornac.
i = 0
for mov in ratings[item_col].unique():
    mapping_dict[mov] = i
    i += 1
ratings[item_col] = ratings[item_col].map(
    lambda x: mapping_dict.get(x, x)
)  # Map in the ratings file


# In[10]:


algo_versions = {"CornacMF": [{"bias": True}, {"bias": False}]}


# In[ ]:


algorithm_cornac = MF
algo_name = "CornacMF"
versions = algo_versions[algo_name]


# for every 'fixed' version of the algorithm
for args in versions:
    print(data_strategy, args)

    p = "best_parameters/" + algo_name + "/" + data_strategy + "_" + str(args) + ".pkl"
    if os.path.isfile(p):
        print("We got them already")
        with open(p, "rb") as f:
            best_params = pickle.load(f)
    else:
        print("We have to compute them now")
        # optimize for this fixed version
        best_params = optimize_cornac(
            ratings=ratings, algorithm_name=algo_name, args=args, max_evals=20
        )

        # save the best parameters for this fixed version

        with open(
            "best_parameters/"
            + algo_name
            + "/"
            + data_strategy
            + "_"
            + str(args)
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(best_params, f)

    optimal_k = best_params["k"]
    optimal_reg = best_params["lambda_reg"]
    optimal_lr = best_params["learning_rate"]

    pop_biases_cornac, metrics_dict_cornac = modelling_mf.train_algorithm_cornac(
        algorithm=algorithm_cornac(
            k=optimal_k,
            use_bias=args["bias"],
            lambda_reg=optimal_reg,
            learning_rate=optimal_lr,
        ),
        algo_name=algo_name,
        ratings=ratings,
        evaluation_way=evaluation_way,
        verbose=verbose,
        n=nr_recs,
        sampling_strategy=sampling_strategy,
        partition_way=partition_way,
        plot=plot,
        data_strategy=data_strategy,
        args=args,
        save_plot=save_plot,
    )

    # Save metrics!
    with open(
        "experimental_results/"
        + algo_name
        + "/"
        + data_strategy
        + "_"
        + str(args)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(metrics_dict_cornac, f)
    with open('experimental_results/'+algo_name+'/detailed_per_item_'+data_strategy+'_'+str(args)+'.pkl', 'wb') as f:
        pickle.dump(pop_biases_cornac, f)

