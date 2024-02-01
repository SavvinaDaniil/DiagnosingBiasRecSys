import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cs
import analysis


def calculate_userKNN_characteristics(ratings, user_dist, item_dist, user_col, item_col, predict_col,  df_item_dist, divide_by):
    # a pivot table version will be needed for later calculations
    num_users = len(user_dist)
    num_items = len(item_dist)
    # ratings_pivot = ratings.pivot_table(index=user_col, columns=item_col, values=predict_col, fill_value=0)
    num_top = int(0.2 * num_items)
    top_item_dist = item_dist[:num_top] # the "popular" items
    # for every user calculate: number of popular items, number of items, fraction of popular items, average popularity per item
    pop_count,user_hist,pop_fraq, pop_item_fraq = analysis.calculate_popularity(ratings, top_item_dist, item_dist, num_users, user_col, item_col)
    # merge the calculated characteristics in the user_dist
    all_users = analysis.sort_user_dist(user_dist,pop_count, user_hist,pop_fraq,pop_item_fraq, by = divide_by)
    # calculate average rating per user
    df_user_rating = ratings[[user_col, predict_col]].groupby(user_col).mean()
    # merge rating and other characteristics
    user_characteristics = pd.merge(all_users, df_user_rating, left_index=True, right_index=True).sort_index(ascending=True)
    # for every user, the items they've rated
    user_items_rated = ratings.groupby([user_col])[item_col].apply(list)
    # calculate average rating for every item
    df_item_rating = analysis.item_rating(ratings, item_col, predict_col)
    item_characteristics = pd.merge(df_item_rating, df_item_dist, left_index=True, right_index=True)
    # average global ratings of every user's consumed items
    user_characteristics["global_rating"] = global_rating(user_characteristics.index, user_items_rated, item_characteristics.rating)
    # standard deviation of every user's ratings
    user_characteristics["st_dev"] = st_dev(ratings, user_col,predict_col)
    # for every user, correlation between item ratings and item popularity
    user_characteristics["score_vs_pop"] = score_vs_pop(user_characteristics.index, ratings, item_characteristics)
    # user_similarities = calculate_user_similarities(ratings_pivot, mean_centered = False)
    # user_similarities_mc = calculate_user_similarities(ratings_pivot, mean_centered = True)
    # user_characteristics["nr_neighbours"] , user_characteristics["ave_similarity"], user_characteristics["ave_non_zero_similarity"] = similarities_per_user(user_characteristics.index, user_similarities)
    # user_characteristics["nr_neighbours_mc"] , user_characteristics["ave_similarity_mc"], user_characteristics["ave_non_zero_similarity_mc"] = similarities_per_user(user_characteristics.index, user_similarities_mc)
    # # for every user, correlation between item ratings and item global rating
    # user_characteristics["score_vs_global_score"] = score_vs_global_score(user_characteristics.index, ratings, item_characteristics)
    
    return user_characteristics, item_characteristics
    
    
def global_rating(users, user_items_rated, item_dist_rating):
    user_global_ratings = []
    for index in users:
        items_rated = user_items_rated[index]
        item_ratings = []
        for item in items_rated:
            average_item_rating = item_dist_rating.loc[item]
            item_ratings.append(average_item_rating)
        average_average = np.mean(item_ratings)
        user_global_ratings.append(average_average)
    return user_global_ratings

def st_dev(ratings, user_col, predict_col):
    user_variance = ratings.groupby(user_col).std().rename(columns={predict_col:"st_dev"})
    return user_variance.st_dev
    
def score_vs_pop(users, ratings, df_item_dist):
    user_score_vs_pop = []
    for index in users:
        personal_ratings = ratings[ratings.user==index]
        items_rated = personal_ratings.item.values
        item_ratings = personal_ratings.rating.values
        item_popularities = df_item_dist.loc[items_rated]["count"].values
        try:
            correlation = stats.pearsonr(item_ratings, item_popularities)[0]
        except:
            correlaiton = 0
        user_score_vs_pop.append(correlation)
    return user_score_vs_pop


def score_vs_global_score(users, ratings, df_item_dist_rating):
    user_score_vs_global_score = []
    for index in users:
        personal_ratings = ratings[ratings.user==index]
        items_rated = personal_ratings.item.values
        item_personal_ratings = personal_ratings.rating.values
        item_global_ratings = df_item_dist_rating.loc[items_rated]["rating"].values
        correlation = stats.pearsonr(item_personal_ratings, item_global_ratings)[0]
        user_score_vs_global_score.append(correlation)
    return user_score_vs_global_score


def similarities_per_user(users, user_similarities):
    nr_neighbours = []
    average_similarity = []
    average_non_zero_similarity = []
    
    for index in users:
        
        neighbours = np.count_nonzero(user_similarities.loc[index].values) - 1
        
        if neighbours>0:
            non_zero_similarities = user_similarities.loc[index][user_similarities.loc[index]!=0].drop(index).values
            ave_non_zero_sim = np.mean(non_zero_similarities)
        else:
            ave_non_zero_sim = float("NaN")
            
        ave_sim = np.mean(user_similarities.loc[index].drop(index).values)
        
        nr_neighbours.append(neighbours)
        average_similarity.append(ave_sim)
        average_non_zero_similarity.append(ave_non_zero_sim)
        
    return nr_neighbours, average_similarity, average_non_zero_similarity


def calculate_user_similarities(ratings_pivot, mean_centered):
    ratings_pivot = ratings_pivot.copy()
    if mean_centered:
        
        ratings_pivot["mean"] = ratings_pivot.replace(0, np.NaN).mean(axis=1, skipna=True)
        ratings_pivot = ratings_pivot.replace(0,np.NaN).sub(ratings_pivot["mean"],axis=0).fillna(0.0).drop("mean", axis=1)

    user_similarities = pd.DataFrame(sklearn_cs(ratings_pivot), index=ratings_pivot.index, columns=ratings_pivot.index)
    return user_similarities
    


# calculate the cosine similarity between two users
def cosine_similarity(u1,u2, over_common = False, mean_center = False):
    
    if mean_center:
        u1 = mean_center_vector(u1)
        u2 = mean_center_vector(u2)
    
        
    if over_common:
        u1,u2 = find_common_items(u1,u2) 
    if norm(u1)*norm(u2)==0:
        return float('NaN')
    
    return np.dot(u1,u2)/(norm(u1)*norm(u2))

def find_common_items(user_1,user_2):
    user_1_new = [user_1[i] for i in range(len(user_1)) if (user_1[i]!=0)&(user_2[i]!=0) ]
    user_2_new = [user_2[i] for i in range(len(user_2)) if (user_1[i]!=0)&(user_2[i]!=0) ]
    
    return user_1_new, user_2_new

# mean center a user vector
def mean_center_vector(user):
    mean=np.mean([x for x in user if x!=0])
    user_new = np.zeros(len(user))
    for i in range(len(user)):
        if user[i]!=0:
            user_new[i]=user[i]-mean
    
    return user_new


def generate_cumulative_sublists(lst):
    result = []
    for i in range(len(lst)):
        sublist = lst[:i+1]
        result.append(sublist)
    return result


def calculate_sparsity_cumulatively(ratings, df_item_dist, num_users_total):
    
    items = df_item_dist.index.values
    item_sets = generate_cumulative_sublists(items)
    
    sparsities = []
    for item_set in item_sets:
        item_ratings = ratings[ratings.item.isin(item_set)]
        num_items = len(item_set)
        
        sparsity = (num_users_total*num_items - len(item_ratings))/(num_items*num_users_total)
        sparsities.append(sparsity)
        
    return sparsities


def plot_relation(df, characteristic_x, characteristic_y, name_x, name_y, extra_title=""):
    x = df.dropna(axis=0)[characteristic_x].values
    y = df.dropna(axis=0)[characteristic_y].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept


    # Calculate the point density
    xy = np.vstack([x,y])
    z = stats.gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.plot(x, line)
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_title('Correlation: ' + str(round(r_value,2))+"\n"+extra_title)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]


    ax.scatter(x, y, c=z, s=50)
    plt.show()