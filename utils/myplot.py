"""
===========================================================================
 Plot functions
===========================================================================
 Authors: Jui-Ting Lu 
 Date: 2025-10-02
 Description:
     This package contains all the plot functions required for our demonstration.
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 1. Plot rolling mean returns of all tickers
def plot_rolling_mean(rolling_mean,tickers, ax=None):
    """
    Plot the rolling mean of asset returns over time.

    Parameters
    ----------
    rolling_mean : pandas.DataFrame
        DataFrame containing the rolling mean values of asset returns,
        with time as the index and tickers as columns.
    tickers : list of str
        List of asset tickers to include in the plot.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, creates a new figure with a larger size.

    Returns
    -------
    None
        Displays a time series plot of the rolling mean for the given tickers.
    """
    if ax is None:
        # ax = plt.gca() # get currenct axis
        fig, ax = plt.subplots(figsize=(14, 6))
    for ticker in tickers:
        ax.plot(rolling_mean.index, rolling_mean[ticker], label=ticker, alpha=0.8)

    ax.set_title("Rolling Mean Returns (60-day window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Log Return")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.6)
    # plt.figure(figsize=(12, 6))
    # for ticker in tickers:
    #     plt.plot(rolling_mean.index, rolling_mean[ticker], label=ticker, alpha=0.8)
    # plt.title("Rolling Mean Returns (60-day window)")
    # plt.xlabel("Date")
    # plt.ylabel("Mean Log Return")
    # plt.legend(loc="upper left", ncol=2)
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.show()

# 2. Plot heatmap of today's covariance matrix
def plot_covariance_day(cov_day,day,tickers, ax=None):
    """    
    Plot a heatmap of the covariance matrix for a specific day.

    Parameters
    ----------
    cov_day : pandas.DataFrame
        Covariance matrix of asset returns for the given day.
    day : TimeStamp (pandas._libs.tslibs.timestamps.Timestamp)
        The date corresponding to the covariance matrix.
    tickers : list of str
        List of asset tickers corresponding to the rows/columns of the covariance matrix.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, creates a new figure with a larger size.

    Returns
    -------
    None
        Displays a heatmap plot of the covariance matrix.
    """
    if ax is None:
        ax = plt.gca() # get currenct axis
    plt.gca()
    im = ax.imshow(cov_day, cmap="Spectral") # viridis
    plt.colorbar(im, ax=ax, label="Covariance")  # Add colorbar: must use the figure, not the Axes
    
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers)
    ax.set_title(f"Covariance Matrix ({day.date()})")
        
# 3. Compare yesterday vs today expected returns
def plot_compare_expected_returns(mu_day1,mu_day2,day1,day2, tickers, ax=None):
    """
    Plot a comparison of expected returns between two dates.

    Parameters
    ----------
    mu_day1 : pandas.Series
        Expected returns for each asset on the first date.
    mu_day2 : pandas.Series
        Expected returns for each asset on the second date.
    day1 : TimeStamp (pandas._libs.tslibs.timestamps.Timestamp)
    day2 : TimeStamp (pandas._libs.tslibs.timestamps.Timestamp)
    tickers : list of str
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, creates a new figure with a larger size.
        
    Returns
    -------
    None
        Displays a bar chart (or similar) comparing expected returns
        between the two dates for the given tickers.
    """
    if ax is None:
        ax = plt.gca()
    bar_width = 0.35
    x = np.arange(len(tickers))
    
    ax.bar(x - bar_width/2, mu_day1, bar_width, label=f"{day1.date()}")
    ax.bar(x + bar_width/2, mu_day2, bar_width, label=f"{day2.date()}")
    
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_ylabel("Expected Return")
    ax.set_title(f"Expected Returns: {day1.date()} vs {day2.date()}")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)


def plot_x0_x1_energy_by_time(df, max_size=300, t=0, figsize=(8,6), cmap='Spectral', ax=None):
    """
    Plots (x0, x1) points from a dataframe colored by energy and sized by repeated occurrences.

    Parameters:
        df (pd.DataFrame): Must contain 'x0', 'x1', and 'energy_shifted' columns.
        max_size (float): Maximum point size for plotting.
        t (int): Which time step of x0/x1 arrays to use for plotting.
        figsize (tuple): Figure size.
        cmap (str): Colormap for energy.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Creates new figure if None.
    """
    
    # Extract coordinates for plotting
    df['x0_plot'] = df['x0'].apply(lambda x: x[t])
    df['x1_plot'] = df['x1'].apply(lambda x: x[t])

    # Count occurrences of each (x0_plot, x1_plot) pair
    counts = df.groupby(['x0_plot', 'x1_plot']).size().reset_index(name='pair_count')

    # Merge counts back to df to get size for each point
    df_plot = pd.merge(df, counts, on=['x0_plot', 'x1_plot'])

    # Scale sizes for plotting
    sizes = df_plot['pair_count'] / df_plot['pair_count'].max() * max_size

    # Scatter plot
    sc = ax.scatter(
        df_plot['x0_plot'],
        df_plot['x1_plot'],
        c=df_plot['energy_shifted'],  # color by energy
        s=sizes,                      # size by repeated occurrences
        cmap=cmap,
        edgecolor='k'
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Energy (shifted)')

    ax.set_xlabel(f'asset0')
    ax.set_ylabel(f'asset1')
    ax.set_title(f'Plot of (x0, x1) points colored by energy at time t={t}')
    ax.grid(True)
    return ax

def plot_x0_x1_energy(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    plot_x0_x1_energy_by_time(df, max_size=300, t=0, figsize=(8,6), cmap='Spectral', ax=axes[0])
    plot_x0_x1_energy_by_time(df, max_size=300, t=1, figsize=(8,6), cmap='Spectral', ax=axes[1])
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=figsize)

