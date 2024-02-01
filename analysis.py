import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import pandas as pd


def users_and_items(df_events, user_col, item_col):
    print('No. user events: ' + str(len(df_events)))
    print('No. items: ' + str(len(df_events[item_col].unique())))
    print('No. users: ' + str(len(df_events[user_col].unique())))
    print("\n")
def user_distribution(df_events, user_col, item_col, verbose=False):
    user_dist = df_events[user_col].value_counts() 
    num_users = len(user_dist)
    if verbose:
        print('Mean '+item_col+'s per user: ' + str(np.round(user_dist.mean(),1))) 
        print('Min '+item_col+'s per user: ' + str(np.round(user_dist.min(),1))) 
        print('Max '+item_col+'s per user: ' + str(np.round(user_dist.max(),1)))
        print("\n")
    return user_dist, num_users

def user_rating_distribution(df_events, user_col, item_col, predict_col = "rating", verbose=False):
    user_dist = df_events.groupby(user_col).mean()[predict_col].sort_values(ascending = False)
    #num_users = len(user_dist)
    if verbose:
        print('Mean '+item_col+'s per user: ' + str(np.round(user_dist.mean(),1))) 
        print('Min '+item_col+'s per user: ' + str(np.round(user_dist.min(),1))) 
        print('Max '+item_col+'s per user: ' + str(np.round(user_dist.max(),1)))
        print("\n")
    return user_dist
#, num_users



def item_distribution(df_events, user_col, item_col, verbose = False):
    item_dist = df_events[item_col].value_counts()
    num_items = len(item_dist)
    if verbose:
        print('Mean users per '+item_col+': ' + str(np.round(item_dist.mean(),1))) 
        print('Min users per '+item_col+': ' + str(np.round(item_dist.min(),1))) 
        print('Max users per '+item_col+': ' + str(np.round(item_dist.max(),1))) 
        print("\n")
    return item_dist, num_items

def item_rating(df_events,item_col, predict_col, verbose=False):
    item_rating = df_events[[item_col, predict_col]].groupby(item_col).mean()
    #num_items = len(item_dist)
    if verbose:
        print('Mean rating per '+item_col+': ' + str(np.round(item_rating[predict_col].mean(),1))) 
        print('Min rating per '+item_col+': ' + str(np.round(item_rating[predict_col].min(),1))) 
        print('Max rating per '+item_col+': ' + str(np.round(item_rating[predict_col].max(),1))) 
        print("\n")
    return item_rating

def plot_data_distribution(item_dist, user_dist, item_ratings, user_ratings, item_col, user_col):
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)
    fig.suptitle('Data and rating distribution')
    
    ax1.spines['bottom'].set_color('w')
    ax1.spines['top'].set_color('w')
    ax1.spines['right'].set_color('w')
    ax1.spines['left'].set_color('w')
    ax1.spines['left'].set_zorder(0)
    ax1.xaxis.set_ticks_position('none') 
    ax1.yaxis.set_ticks_position('none') 
    ax1.set_facecolor("aliceblue")
    ax1.grid(color = "w",linewidth = 2 )
    ax1.plot(item_dist.values)
    ax1.set_xlabel(item_col, fontsize='25')
    ax1.set_ylabel('Number of users', fontsize='25')
    
    
    ax2.spines['bottom'].set_color('w')
    ax2.spines['top'].set_color('w')
    ax2.spines['right'].set_color('w')
    ax2.spines['left'].set_color('w')
    ax2.spines['left'].set_zorder(0)
    ax2.xaxis.set_ticks_position('none') 
    ax2.yaxis.set_ticks_position('none') 
    ax2.set_facecolor("aliceblue")
    ax2.grid(color = "w",linewidth = 2 )
    ax2.plot(user_dist.values)
    ax2.set_xlabel(user_col, fontsize='25')
    ax2.set_ylabel('Number of items', fontsize='25')
    
    ax3.spines['bottom'].set_color('w')
    ax3.spines['top'].set_color('w')
    ax3.spines['right'].set_color('w')
    ax3.spines['left'].set_color('w')
    ax3.spines['left'].set_zorder(0)
    ax3.xaxis.set_ticks_position('none') 
    ax3.yaxis.set_ticks_position('none') 
    ax3.set_facecolor("aliceblue")
    ax3.grid(color = "w",linewidth = 2 )
    ax3.plot(np.sort(item_ratings))
    ax3.set_xlabel(item_col, fontsize='25')
    ax3.set_ylabel('Average rating', fontsize='25')
    
    
    ax4.spines['bottom'].set_color('w')
    ax4.spines['top'].set_color('w')
    ax4.spines['right'].set_color('w')
    ax4.spines['left'].set_color('w')
    ax4.spines['left'].set_zorder(0)
    ax4.xaxis.set_ticks_position('none') 
    ax4.yaxis.set_ticks_position('none') 
    ax4.set_facecolor("aliceblue")
    ax4.grid(color = "w",linewidth = 2 )
    ax4.plot(np.sort(user_ratings))
    ax4.set_xlabel(user_col, fontsize='25')
    ax4.set_ylabel('Average rating', fontsize='25')
    
    fig.set_figheight(15)
    fig.set_figwidth(20)
    plt.show(block=True)
    
    
def plot_rating_distribution(all_rating_values, user_rating_values, item_rating_values, rating_range):
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    fig, ([ax1, ax2, ax3]) = plt.subplots(1, 3)
    xa = all_rating_values
    xu = user_rating_values
    xi = item_rating_values
    
    
    n, bins, patches = ax1.hist(xa,  bins=np.arange(rating_range[1]+2) - 0.5,facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.7, alpha=0.8)

    #ax1.title('Rating distribution', fontsize=20)
    ax1.axvline(xa.mean(), color='k', linestyle='dashed', linewidth=1, label = "Mean rating")
    ax1.set_xticks(range(rating_range[0],rating_range[1]+1))
    ax1.legend(fontsize=15)
    ax1.set_xlabel('Rating', fontsize=20)
    ax1.set_ylabel('Frequency out of all ratings', fontsize=20)
    
    n, bins, patches = ax2.hist(xu,  bins=np.arange(rating_range[1]+2) - 0.5,facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.7, alpha=0.8)

    #ax1.title('Rating distribution', fontsize=20)
    ax2.axvline(xa.mean(), color='k', linestyle='dashed', linewidth=1, label = "Mean rating")
    ax2.set_xticks(range(rating_range[0],rating_range[1]+1))
    ax2.legend(fontsize=15)
    ax2.set_xlabel('User Average Rating', fontsize=20)
    ax2.set_ylabel('Frequency out of all users', fontsize=20)
    
    n, bins, patches = ax3.hist(xi,  bins=np.arange(rating_range[1]+2) - 0.5,facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.7, alpha=0.8)

    #ax1.title('Rating distribution', fontsize=20)
    ax3.axvline(xi.mean(), color='k', linestyle='dashed', linewidth=1, label = "Mean rating")
    ax3.set_xticks(range(rating_range[0],rating_range[1]+1))
    ax3.legend(fontsize=15)
    ax3.set_xlabel('Item Average Rating', fontsize=20)
    ax3.set_ylabel('Frequency out of all items', fontsize=20)
    
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.show()
    
    
    
def plot_popularity_distribution(pop_fraq, item_col, dividing = [False,0]):
    plt.figure()
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_facecolor("aliceblue")
    plt.grid(color = "w",linewidth = 2 )
    if dividing[0]:
        y = range(len(pop_fraq))
        x0 = int(len(y)*dividing[1]) 
        x1 = int(len(y)*(1-dividing[1]))
        x= sorted(pop_fraq)
        plt.plot(y[:x0+1],x[:x0+1], label="Niche users", linewidth = 5)
        plt.plot(y[x0:x1+1],x[x0:x1+1], label = "Diverse users", linewidth = 5)
        plt.plot(y[x1:],x[x1:], label = "BestSeller users", linewidth =5)
    else:
        plt.plot(sorted(pop_fraq))
    plt.xlabel('User', fontsize='15')
    plt.xticks(fontsize='13')
    plt.ylabel('Ratio of popular '+item_col+'s', fontsize='15')
    plt.yticks(fontsize='13')
    plt.axhline(y=0.8, color='black', linestyle='--', label='80% ratio of popular '+item_col+'s')
    plt.legend(fontsize='15')
    plt.show(block=True)
    
def calculate_popularity(df_events, top_item_dist, item_dist, num_users, user_col, item_col):
    pop_count = [] # number of top items per user
    user_hist = [] # user history sizes
    pop_fraq = [] # relative number of top items per user
    pop_item_fraq = [] # average popularity of items in user profiles
    for u, df in df_events.groupby(user_col):
        no_user_items = len(set(df[item_col]))
        no_user_pop_items = len(set(df[item_col]) & set(top_item_dist.index))
        pop_count.append(no_user_pop_items)
        user_hist.append(no_user_items) 
        pop_fraq.append(no_user_pop_items / no_user_items)
        user_pop_item_fraq = sum(item_dist[df[item_col]]) / no_user_items
        pop_item_fraq.append(user_pop_item_fraq)
    return pop_count,user_hist,pop_fraq, pop_item_fraq

def plot_popularity_distribution(pop_fraq, item_col, dividing = [False,0]):
    plt.figure()
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_facecolor("aliceblue")
    plt.grid(color = "w",linewidth = 2 )
    if dividing[0]:
        y = range(len(pop_fraq))
        x0 = int(len(y)*dividing[1]) 
        x1 = int(len(y)*(1-dividing[1]))
        x= sorted(pop_fraq)
        plt.plot(y[:x0+1],x[:x0+1], label="Niche users", linewidth = 5)
        plt.plot(y[x0:x1+1],x[x0:x1+1], label = "Diverse users", linewidth = 5)
        plt.plot(y[x1:],x[x1:], label = "BestSeller users", linewidth =5)
    else:
        plt.plot(sorted(pop_fraq))
    plt.xlabel('User', fontsize='15')
    plt.xticks(fontsize='13')
    plt.ylabel('Ratio of popular '+item_col+'s', fontsize='15')
    plt.yticks(fontsize='13')
    plt.axhline(y=0.8, color='black', linestyle='--', label='80% ratio of popular '+item_col+'s')
    plt.legend(fontsize='15')
    plt.show(block=True)

def plot_profile_size_vs_popularity(pop_metric, user_hist, way, item_col):
    plt.figure()
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_facecolor("aliceblue")
    plt.grid(color = "w",linewidth = 2 )
    slope, intercept, r_value, p_value, std_err = stats.linregress(user_hist, pop_metric)
    print('R-value: ' + str(r_value))
    line = slope * np.array(user_hist) + intercept
    plt.plot(user_hist, pop_metric, 'o', user_hist, line)
    plt.xlabel('User profile size', fontsize='15')
    plt.xticks(fontsize='13')
    if way == "count":
        ylabel = "Number of popular "+item_col+"s"
    elif way == "percentage":
        ylabel = 'Percentage of popular '+item_col+'s'
    else:
        ylabel = "Average popularity of "+item_col+"s"
    plt.ylabel(ylabel, fontsize='15')
    plt.yticks(fontsize='13')
    plt.show(block=True)
    
def sort_user_dist(user_dist,pop_count, user_hist,pop_fraq,pop_item_fraq, by = "pop_fraq"):
    user_dist = user_dist.sort_index()
    user_dist_sorted = pd.DataFrame(data = user_dist)
    
    user_dist_sorted.columns = ["count"]
    
    user_dist_sorted["pop_count"] = pop_count
    user_dist_sorted["user_hist"] = user_hist
    user_dist_sorted["pop_fraq"] = pop_fraq
    user_dist_sorted["pop_item_fraq"] = pop_item_fraq
    
    user_dist_sorted = user_dist_sorted.sort_values(by=[by])
    return user_dist_sorted

def split(user_dist_sorted, top_fraction):
    low, med, high = np.split(user_dist_sorted, [int(top_fraction*len(user_dist_sorted)), int((1-top_fraction)*len(user_dist_sorted))])
    return low, med, high

def calculate_group_characteristics(low, med, high):
    low_profile_size = low.user_hist.mean()
    med_profile_size = med.user_hist.mean()
    high_profile_size = high.user_hist.mean()
    
    low_nr_users = len(low)
    med_nr_users = len(med)
    high_nr_users = len(high)
    
    low_GAP = low.pop_item_fraq.mean()
    med_GAP = med.pop_item_fraq.mean()
    high_GAP = high.pop_item_fraq.mean()
    
    return low_profile_size, med_profile_size, high_profile_size, low_nr_users, med_nr_users, high_nr_users, low_GAP, med_GAP, high_GAP

def plot_group_characteristics(low_nr, med_nr, high_nr, way, item_col):
    plt.figure()
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_facecolor("aliceblue")
    plt.bar(np.arange(3), [low_nr, med_nr, high_nr])
    plt.xticks(np.arange(3), ['Niche', 'Diverse', 'BestSeller'])
    plt.xlabel('User group')
    if way=="size":
        ylabel = 'Average user profile size'
    else:
        ylabel = "Number of users per group"
    plt.ylabel(ylabel)
    
    print('Niche: ' + str(low_nr))
    print('Diverse: ' + str(med_nr))
    print('BestSeller: ' + str(high_nr))
    plt.show(block=True)