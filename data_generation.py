import numpy as np
from scipy import stats
import analysis


def truncated_poisson(mu, rating_range, length):
    lower_bound = rating_range[0]  # lower bound of the distribution
    upper_bound = rating_range[1]
    k_values = range(lower_bound, upper_bound + 1)
    probabilities = [stats.poisson(mu).pmf(k) for k in k_values]
    probabilities /= sum(probabilities)  # normalize the probabilities
    trunc_poisson = stats.rv_discrete(values=(k_values, probabilities))
    rates = trunc_poisson.rvs(size=length)
    return rates


def truncated_normal(mu, rating_range, length):
    lower_bound = rating_range[0]  # lower bound of the distribution
    upper_bound = rating_range[1]
    k_values = range(lower_bound, upper_bound + 1)
    probabilities = [stats.norm().pdf(k - mu) for k in k_values]
    probabilities /= sum(probabilities)  # normalize the probabilities
    trunc_poisson = stats.rv_discrete(values=(k_values, probabilities))
    rates = trunc_poisson.rvs(size=length)
    return rates


def popularity_to_rating(popularities, rating_range):
    mean_ratings = np.round(
        (popularities - np.min(popularities))
        / (np.max(popularities) - np.min(popularities))
        * (rating_range[1] - 1)
        + rating_range[0]
    )
    return mean_ratings


def generate_data(
    strategy,
    rating_range=[1, 10],
    random_seed=0,
    user_col="user",
    item_col="item",
    predict_col="rating",
    copying_dataset=None,
    average_rating=None,
    user_perc=None,
):
    ratings = copying_dataset.copy()
    num_ratings = len(ratings)
    num_users = len(ratings.user.unique())
    num_items = len(ratings.item.unique())
    lower_bound = rating_range[0]
    upper_bound = rating_range[-1] + 1
    np.random.seed(random_seed)

    if strategy == "uniformly_random":
        ratings[predict_col] = stats.randint(lower_bound, upper_bound).rvs(len(ratings))

    elif strategy == "poisson":
        mu = average_rating  # average rate of occurrence for each rating
        length = num_ratings
        ratings[predict_col] = truncated_poisson(mu, rating_range, length)

    elif strategy == "normal":
        mu = average_rating  # average rate of occurrence for each rating
        length = num_ratings
        ratings[predict_col] = truncated_normal(mu, rating_range, length)

    elif (
        strategy == "personal_normal"
    ):  # users rate random but each with their own distribution (around their own average)
        user_dist, num_items = analysis.user_distribution(ratings, user_col, item_col)

        i = 0
        for user in user_dist.index.values:
            mu = np.random.randint(lower_bound, upper_bound + 1)
            length = len(ratings[ratings.user == user])
            rates = truncated_normal(mu, rating_range, length)
            ratings.loc[ratings.user == user, "rating"] = rates
            i += 1

    elif strategy == "popularity_good":
        item_dist, num_items = analysis.item_distribution(ratings, "user", "item")
        item_dist_perc = item_dist / num_users
        mean_ratings = popularity_to_rating(item_dist_perc.values, rating_range)
        i = 0
        for item in item_dist.index.values:
            mu = mean_ratings[i]
            length = item_dist.loc[item]
            rates = truncated_normal(mu, rating_range, length)
            ratings.loc[ratings.item == item, "rating"] = rates
            i += 1

    elif strategy == "popularity_bad":
        item_dist, num_items = analysis.item_distribution(ratings, "user", "item")
        item_dist_perc = item_dist / num_users
        mean_ratings = popularity_to_rating(-item_dist_perc.values, rating_range)
        i = 0
        for item in item_dist.index.values:
            mu = mean_ratings[i]
            length = item_dist.loc[item]
            rates = truncated_normal(mu, rating_range, length)
            ratings.loc[ratings.item == item, "rating"] = rates
            i += 1

    elif strategy == "popularity_good_for_some_ur":  # the rest uniformly random
        ratings[predict_col] = stats.randint(lower_bound, upper_bound).rvs(len(ratings))

        item_dist, num_items = analysis.item_distribution(ratings, "user", "item")
        item_dist_perc = item_dist / num_users
        mean_ratings = popularity_to_rating(item_dist_perc.values, rating_range)

        indices_users = np.random.choice(
            num_users,
            int(user_perc * num_users),
            replace=False,
        )

        i = 0
        for item in item_dist.index.values:
            mu = mean_ratings[i]
            length = len(
                ratings[(ratings.item == item) & (ratings.user.isin(indices_users))]
            )
            rates = truncated_poisson(mu, rating_range, length)
            ratings.loc[
                (ratings.item == item) & (ratings.user.isin(indices_users)), "rating"
            ] = rates
            i += 1

    elif strategy == "popularity_good_for_bp_ur":  # the rest uniformly random
        ratings[predict_col] = stats.randint(lower_bound, upper_bound).rvs(len(ratings))

        item_dist, num_items = analysis.item_distribution(ratings, "user", "item")
        item_dist_perc = item_dist / num_users
        mean_ratings = popularity_to_rating(item_dist_perc.values, rating_range)

        user_dist, num_items = analysis.user_distribution(ratings, user_col, item_col)

        indices_users = user_dist[: int(user_perc * num_users + 1)].index.values

        i = 0
        for item in item_dist.index.values:
            mu = mean_ratings[i]
            length = len(
                ratings[(ratings.item == item) & (ratings.user.isin(indices_users))]
            )
            rates = truncated_poisson(mu, rating_range, length)
            ratings.loc[
                (ratings.item == item) & (ratings.user.isin(indices_users)), "rating"
            ] = rates
            i += 1

    elif strategy == "popularity_bad_for_bp_ur":  # the rest uniformly random
        ratings[predict_col] = stats.randint(lower_bound, upper_bound).rvs(len(ratings))

        item_dist, num_items = analysis.item_distribution(ratings, "user", "item")
        item_dist_perc = item_dist / num_users
        mean_ratings = popularity_to_rating(-item_dist_perc.values, rating_range)

        user_dist, num_items = analysis.user_distribution(ratings, user_col, item_col)

        indices_users = user_dist[: int(user_perc * num_users + 1)].index.values

        i = 0
        for item in item_dist.index.values:
            mu = mean_ratings[i]
            length = len(
                ratings[(ratings.item == item) & (ratings.user.isin(indices_users))]
            )
            rates = truncated_poisson(mu, rating_range, length)
            ratings.loc[
                (ratings.item == item) & (ratings.user.isin(indices_users)), "rating"
            ] = rates
            i += 1
    else:
        print("Error! You must choose one of the available data generation options.")

    return ratings
