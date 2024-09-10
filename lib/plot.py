import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib


def produce_complete_analysis(user_characteristics, item_characteristics, mc):
    font = {"size": 15}

    matplotlib.rc("font", **font)

    # plot 1: item rating vs popularity

    x = item_characteristics.dropna(axis=0)["count"].values
    y = item_characteristics.dropna(axis=0)["rating"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    fig, axs = plt.subplots(5, 2)
    axs[0, 0].plot(x, line)
    axs[0, 0].set_xlabel("Item popularity", fontsize=20)
    axs[0, 0].set_ylabel("Item average rating", fontsize=20)
    axs[0, 0].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0, 0].scatter(x, y, c=z, s=50)

    # plot 2: user liking for pop items vs influence

    x = user_characteristics.dropna(axis=0)["score_vs_pop"].values
    if mc:
        y = user_characteristics.dropna(axis=0)["ave_similarity_mc"].values
    else:
        y = user_characteristics.dropna(axis=0)["ave_similarity"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[0, 1].plot(x, line)
    axs[0, 1].set_xlabel("User liking for popular items", fontsize=20)
    axs[0, 1].set_ylabel("User average similarity to other users", fontsize=20)
    axs[0, 1].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0, 1].scatter(x, y, c=z, s=50)

    # plot 3: user tendency to consume popular items vs influence

    x = user_characteristics.dropna(axis=0)["pop_item_fraq"].values
    if mc:
        y = user_characteristics.dropna(axis=0)["ave_similarity_mc"].values
    else:
        y = user_characteristics.dropna(axis=0)["ave_similarity"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[1, 0].plot(x, line)
    axs[1, 0].set_xlabel("User tendency to consume popular items", fontsize=20)
    axs[1, 0].set_ylabel("User average similarity to other users", fontsize=20)
    axs[1, 0].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1, 0].scatter(x, y, c=z, s=50)

    # plot 4: user liking for quality items vs influence

    x = user_characteristics.dropna(axis=0)["score_vs_global_score"].values
    if mc:
        y = user_characteristics.dropna(axis=0)["ave_similarity_mc"].values
    else:
        y = user_characteristics.dropna(axis=0)["ave_similarity"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[1, 1].plot(x, line)
    axs[1, 1].set_xlabel("User liking for quality items", fontsize=20)
    axs[1, 1].set_ylabel("User average similarity to other users", fontsize=20)
    axs[1, 1].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1, 1].scatter(x, y, c=z, s=50)

    # plot 5 variance vs influence

    x = user_characteristics.dropna(axis=0)["st_dev"].values
    if mc:
        y = user_characteristics.dropna(axis=0)["ave_similarity_mc"].values
    else:
        y = user_characteristics.dropna(axis=0)["ave_similarity"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[2, 0].plot(x, line)
    axs[2, 0].set_xlabel("User rating variance", fontsize=20)
    axs[2, 0].set_ylabel("User average similarity to other users", fontsize=20)
    axs[2, 0].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[2, 0].scatter(x, y, c=z, s=50)

    # plot 6 # user hist vs influence

    x = user_characteristics.dropna(axis=0)["user_hist"].values
    if mc:
        y = user_characteristics.dropna(axis=0)["ave_non_zero_similarity_mc"].values
    else:
        y = user_characteristics.dropna(axis=0)["ave_non_zero_similarity"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[2, 1].plot(x, line)
    axs[2, 1].set_xlabel("User profile size", fontsize=20)
    axs[2, 1].set_ylabel("User average similarity to neighbours", fontsize=20)
    axs[2, 1].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[2, 1].scatter(x, y, c=z, s=50)

    # plot 7 profile size vs score_pop

    x = user_characteristics.dropna(axis=0)["user_hist"].values
    y = user_characteristics.dropna(axis=0)["score_vs_pop"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[3, 0].plot(x, line)
    axs[3, 0].set_xlabel("User profile size", fontsize=20)
    axs[3, 0].set_ylabel("User liking for popular items", fontsize=20)
    axs[3, 0].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[3, 0].scatter(x, y, c=z, s=50)

    # plot 8 # profile size vs quality liking

    x = user_characteristics.dropna(axis=0)["user_hist"].values
    y = user_characteristics.dropna(axis=0)["score_vs_global_score"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[3, 1].plot(x, line)
    axs[3, 1].set_xlabel("User profile size", fontsize=20)
    axs[3, 1].set_ylabel("User liking for quality items", fontsize=20)
    axs[3, 1].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[3, 1].scatter(x, y, c=z, s=50)

    # plot 9 user liking for quality items vs ave_non_zero_similarity

    x = user_characteristics.dropna(axis=0)["score_vs_global_score"].values
    if mc:
        y = user_characteristics.dropna(axis=0)["ave_non_zero_similarity_mc"].values
    else:
        y = user_characteristics.dropna(axis=0)["ave_non_zero_similarity"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[4, 0].plot(x, line)
    axs[4, 0].set_xlabel("User liking for quality items", fontsize=20)
    axs[4, 0].set_ylabel("User similarity to neighbours", fontsize=20)
    axs[4, 0].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[4, 0].scatter(x, y, c=z, s=50)

    # plot 10 # user tendency to consume popular items vs profile size

    x = user_characteristics.dropna(axis=0)["user_hist"].values
    y = user_characteristics.dropna(axis=0)["pop_item_fraq"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[4, 1].plot(x, line)
    axs[4, 1].set_xlabel("User profile size", fontsize=20)
    axs[4, 1].set_ylabel("User tendency to consume popular items", fontsize=20)
    axs[4, 1].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[4, 1].scatter(x, y, c=z, s=50)

    fig.set_figheight(45)
    fig.set_figwidth(20)

    # fig.tight_layout()


def produce_complete_analysis_item_knn(item_characteristics, mc):
    font = {"size": 15}

    matplotlib.rc("font", **font)

    # plot 1: item rating vs popularity

    x = item_characteristics.dropna(axis=0)["count"].values
    y = item_characteristics.dropna(axis=0)["rating"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, line)
    axs[0, 0].set_xlabel("Item popularity", fontsize=20)
    axs[0, 0].set_ylabel("Item average rating", fontsize=20)
    axs[0, 0].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0, 0].scatter(x, y, c=z, s=50)

    # plot 2: user liking for pop items vs influence

    x = item_characteristics.dropna(axis=0)["count"].values
    if mc:
        y = item_characteristics.dropna(axis=0)["ave_similarity_mc"].values
    else:
        y = item_characteristics.dropna(axis=0)["ave_similarity"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[0, 1].plot(x, line)
    axs[0, 1].set_xlabel("Item popularity", fontsize=20)
    axs[0, 1].set_ylabel("Item average similarity to other items", fontsize=20)
    axs[0, 1].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0, 1].scatter(x, y, c=z, s=50)

    # plot 3: user tendency to consume popular items vs influence

    x = item_characteristics.dropna(axis=0)["count"].values
    y = item_characteristics.dropna(axis=0)["st_dev"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[1, 0].plot(x, line)
    axs[1, 0].set_xlabel("Item popularity", fontsize=20)
    axs[1, 0].set_ylabel("Item rating standard deviation", fontsize=20)
    axs[1, 0].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1, 0].scatter(x, y, c=z, s=50)

    # plot 4: user liking for quality items vs influence

    x = item_characteristics.dropna(axis=0)["rating"].values
    if mc:
        y = item_characteristics.dropna(axis=0)["ave_similarity_mc"].values
    else:
        y = item_characteristics.dropna(axis=0)["ave_similarity"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    axs[1, 1].plot(x, line)
    axs[1, 1].set_xlabel("Item quality", fontsize=20)
    axs[1, 1].set_ylabel("Item average similarity to other items", fontsize=20)
    axs[1, 1].set_title("Correlation: " + str(round(r_value, 2)), fontsize=20)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1, 1].scatter(x, y, c=z, s=50)
    fig.set_figheight(15)
    fig.set_figwidth(20)

    # fig.tight_layout()


def compare_sparsity_evolution(sparsities_list, labels_list):
    for i in range(len(sparsities_list)):
        plt.plot(sparsities_list[i], label=labels_list[i])
    plt.xlabel("Number of items, sorted by popularity")
    plt.ylabel("Sparsity")
    plt.legend()
    plt.show()
