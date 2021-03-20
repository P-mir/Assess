import matplotlib. pyplot as plt


def sns_boxplot(cat, continuous, data, title= "continuous by cat", x_rotation = 90):
    """
    Beautiful Box plot for categorical variable vs continuous variable
    """
    plt.figure(figsize = (20, 6))
    sns.boxplot(x = cat, y = continuous, data = data);
    plt.title(title);
    plt.xticks(rotation = x_rotation)
    return plt

def ecdf_plot(data, var, title ='Empirical cumulative distribution of ' ):
    """
    CONTINUOUS VARIABLE
    Plot of empirical cumulative distribution function to show how skewed a variable is
    """
    
    def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

    x, y = ecdf(data.loc[data[var] > 0, var])
    plt.plot(x, y, marker = '.')
    plt.xlabel(var); plt.ylabel('Percentile'); plt.title(title)
    return plt

def var_distribution_vs_target_plot(data,col_x,col_y):
    """
    CLASSIFICATION
    Very useful plot to show the distribution of a continuous variable depending on a categorical target
    """

    g = sns.FacetGrid(data,
                      hue = col_y, size = 4, aspect = 3)
    g.map(sns.kdeplot, col_x)
    g.add_legend();
    plt.title('Distribution of Purchases Total by Label')
    return plt



