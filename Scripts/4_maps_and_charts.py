"""
Title: Generate Maps and Charts
Version: 1.0
Date: 29 April 2024
Author: Logan Insinga
Depends:
    numpy           1.26.4
    openpyxl        3.1.2
    pandas          2.2.2
    matplotlib      3.8.4
    scikit-learn    1.4.2
    Cartopy         0.22.0

Description:
Generates maps and charts for the predictions of
a single element and model combination.
User specifies element, model, execution id, and 
number of predictions.
"""

import logging
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy.interpolate import griddata
from scipy.spatial import cKDTree as KDTree

import cartopy.crs as ccrs
import cartopy.feature as cfeature

EXECUTION_ID = "run1"
ELEMENT = "SE"
MODEL = "SVR"  # used for naming
FILTER_THRESHOLD = 0.3
MASTER_TABLE_ORIGINAL = r"Processing\_0_Master_table.xlsx"
MODEL_RESULTS_DIR = r"Processing\_3_SVR_Se"
OUTPUT_DIR = MODEL_RESULTS_DIR
NUM_PREDICTIONS = 100

LOGGER = logging.getLogger("my_logger")
LOGGER.setLevel(logging.DEBUG)

# define colors using RGB values for plots
d_brown = (0.451, 0.353, 0)
l_brown = (0.824, 0.706, 0.549)
tan = (1, 1, 0.784)
vl_green = (0.745, 0.902, 0.392)
l_green = (0.392, 0.588, 0)
d_green = (0.078, 0.235, 0)

red = (0.843, 0.196, 0.157)
really_red = (1, 0, 0)
orange = (0.961, 0.588, 0.392)
l_blue = (0.627, 0.706, 0.745)
d_blue = (0.275, 0.471, 0.706)

PLOT_INFO = {
    # "obs_v_pred_bins": [0, 100, 200, 300, 400, 500, 100000],  # S
    "obs_v_pred_bins": [0, 0.2, 0.3, 0.4, 0.5, 0.6, 100],  # Se
    "obs_v_pred_colors": [d_brown, l_brown, tan, vl_green, l_green, d_green],
    # "residuals_bins": [-1000000, -200, -100, 100, 200, 1000000],  # S
    "residuals_bins": [-100, -0.2, -0.1, 0.1, 0.2, 100],  # Se
    "residuals_colors": [d_blue, l_blue, tan, orange, red],
    "perc_res_bins": [-1, -0.2, -0.1, 0.1, 0.2, 1],
    "coeff_var_bins": [0, 0.1, 0.2, 0.3, 1],
    "coeff_var_colors": ["lightsteelblue", tan, "peachpuff", orange],
    "error_ratio_bins": [-1, -0.3, 0.3, 1],
    "error_ratio_colors": [l_blue, tan, orange],
    "perc_change_bins": [-150, -5, -3, -1, 1, 3, 5, 150],
    "perc_change_colors": [
        red,
        orange,
        "peachpuff",
        tan,
        "lightsteelblue",
        d_blue,
        "navy",
    ],
    # "perc_change_std_bins": [0, 0.25, 0.5, 0.75, 1, 10], # SVR
    "perc_change_std_bins": [0, 0.5, 1, 1.5, 2, 10],  # MLP
    "perc_change_std_colors": [d_blue, l_blue, tan, orange, red],
}


def setup_logging():
    """Sets up logging for documentation"""

    log_file_loc = os.path.dirname(MODEL_RESULTS_DIR)

    file_handler = logging.FileHandler(
        f"{log_file_loc}\\4_maps_{EXECUTION_ID}_{ELEMENT}_{MODEL}.log", "w"
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(logging.DEBUG)  # log file gets everything
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)


def import_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imports the model results. Gets the original measured data.
    Also gets each set of predictions."""

    # read in observed data from the original master table
    observed = pd.read_excel(
        MASTER_TABLE_ORIGINAL, usecols=["lon", "lat", f"Avg_{ELEMENT}"]
    )
    observed.rename(mapper={f"Avg_{ELEMENT}": "Observed"}, axis=1, inplace=True)
    observed.sort_values(by=["lon", "lat"], ascending=True, inplace=True)
    observed.set_index(keys=["lon", "lat"], drop=True, inplace=True)

    # read in the predictions
    predictions: dict[str, pd.DataFrame] = {}

    def import_prediction(predictions: dict[str, pd.DataFrame], descriptor: str):
        pred = pd.read_csv(
            os.path.join(MODEL_RESULTS_DIR, f"{EXECUTION_ID}_pred_{descriptor}.csv"),
            usecols=["lon", "lat", "avg", "std", "95th_centile", "5th_centile"],
        )
        pred.sort_values(by=["lon", "lat"], ascending=True, inplace=True)
        pred.set_index(keys=["lon", "lat"], drop=True, inplace=True)
        pred.columns = [f"{descriptor}_" + col for col in pred.columns]
        predictions[descriptor] = pred
        return predictions

    predictions = import_prediction(predictions, "current")
    predictions = import_prediction(predictions, "extreme")
    predictions = import_prediction(predictions, "extreme_AI")
    predictions = import_prediction(predictions, "extreme_ET")
    predictions = import_prediction(predictions, "extreme_precip")
    predictions = import_prediction(predictions, f"extreme_{ELEMENT}dep")
    predictions = import_prediction(predictions, "extreme_toc")

    predictions["extreme"]["has_future_point"] = 1

    # store predictions in single table
    predictions = pd.concat(list(predictions.values()), axis=1, join="outer")

    return observed, predictions


def get_lat_lon_values(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gets the lat lon values for current and future data. Used for creating spatial grid"""

    lat_lon_c = predictions.index
    lat_lon_f = predictions[predictions["has_future_point"] == 1].copy(deep=True).index

    return lat_lon_c, lat_lon_f


def calculate_percent_change(predictions: pd.DataFrame):
    """Calculates the percent change from current to future. Also calculates the
    standard deviation of percent change of all iterations of future predictions"""

    # calculate percent change for extreme predictions
    def calculate_pc(predictions: pd.DataFrame, descriptor: str):

        predictions[f"{descriptor}_percchange"] = (
            (predictions[descriptor] - predictions["current_avg"])
            / predictions["current_avg"]
        ) * 100

        return predictions

    predictions = calculate_pc(predictions, "extreme_avg")
    predictions = calculate_pc(predictions, "extreme_95th_centile")
    predictions = calculate_pc(predictions, "extreme_5th_centile")
    predictions = calculate_pc(predictions, "extreme_AI_avg")
    predictions = calculate_pc(predictions, "extreme_ET_avg")
    predictions = calculate_pc(predictions, "extreme_precip_avg")
    predictions = calculate_pc(predictions, f"extreme_{ELEMENT}dep_avg")
    predictions = calculate_pc(predictions, "extreme_toc_avg")

    # calculate std of percent change for standard extreme iterations
    # read in the full set of extreme predictions
    extreme_pred_temp = pd.read_csv(
        os.path.join(MODEL_RESULTS_DIR, f"{EXECUTION_ID}_pred_extreme.csv")
    )
    extreme_pred_temp.set_index(keys=["lon", "lat"], drop=True, inplace=True)
    extreme_pred_temp.drop(
        labels=["avg", "std", "95th_centile", "5th_centile"],
        axis=1,
        inplace=True,
    )
    extreme_pred_temp.columns = ["extreme_" + col for col in extreme_pred_temp.columns]
    extreme_pred_temp["has_future_point"] = 1

    # read in the full set of current predictions
    current_pred_temp = pd.read_csv(
        os.path.join(MODEL_RESULTS_DIR, f"{EXECUTION_ID}_pred_current.csv"),
    )
    current_pred_temp.set_index(keys=["lon", "lat"], drop=True, inplace=True)
    current_pred_temp.drop(
        labels=["avg", "std", "95th_centile", "5th_centile"],
        axis=1,
        inplace=True,
    )
    current_pred_temp.columns = ["current_" + col for col in current_pred_temp.columns]

    # join "has future point" col to current and trim current data to match future
    current_pred_temp = current_pred_temp.merge(
        right=extreme_pred_temp[["has_future_point"]].copy(deep=True),
        left_index=True,
        right_index=True,
    )
    current_pred_temp = current_pred_temp[
        current_pred_temp["has_future_point"] == 1
    ].copy(deep=True)

    extreme_pred_temp.drop("has_future_point", axis=1, inplace=True)
    current_pred_temp.drop("has_future_point", axis=1, inplace=True)

    # convert both future and current data to arrays
    extreme_pred_temp_np = extreme_pred_temp.to_numpy(dtype=float, copy=True)
    current_pred_temp_np = current_pred_temp.to_numpy(dtype=float, copy=True)

    # calcualte percent change for each model iteration
    perc_change_alliters = (
        (extreme_pred_temp_np - current_pred_temp_np) / current_pred_temp_np * 100
    )

    # convert back to dataframe
    perc_change_alliters_df = pd.DataFrame(
        data=perc_change_alliters,
        index=current_pred_temp.index,
        columns=[f"pc_{i}" for i in range(perc_change_alliters.shape[1])],
    )

    # calculate the std of the % changes for all predictions
    perc_change_alliters_df["extreme_percchange_std"] = perc_change_alliters_df.std(
        axis=1
    )

    extreme_pred_temp.to_csv(
        os.path.join(
            OUTPUT_DIR, f"{EXECUTION_ID}_context_percchange_future_alliters.csv"
        )
    )

    # merge the std of extreme pc for each iter into predictions table
    predictions = predictions.merge(
        right=perc_change_alliters_df[["extreme_percchange_std"]].copy(deep=True),
        right_index=True,
        left_index=True,
        how="outer",
    )

    return predictions


def calculate_residuals(predictions: pd.DataFrame, observed: pd.DataFrame):
    """Calculates residuals and residuals as percent"""

    # merge the observed data to the predictions
    predictions = predictions.merge(
        right=observed, how="outer", right_index=True, left_index=True
    )

    # calculates model residuals
    predictions["residuals"] = predictions["current_avg"] - predictions["Observed"]

    # calculate model residuals as a fraction of observed
    predictions["residuals_fracobs"] = (
        predictions["residuals"] / predictions["Observed"]
    )

    return predictions


def calculate_and_apply_filters(predictions: pd.DataFrame):
    """Calculates and applies the precision and accuracy filter"""

    # calculate the coefficient of variation (precision filter)
    predictions["coeff_var"] = predictions["current_std"] / predictions["current_avg"]

    # calculate the error ratio (accuracy filter)
    predictions["error_ratio"] = (
        predictions["current_avg"] - predictions["Observed"]
    ) / predictions["Observed"]

    # apply both filters individually
    predictions["current_avg_precfilter"] = np.where(
        predictions["coeff_var"] > FILTER_THRESHOLD, np.nan, predictions["current_avg"]
    )
    predictions["current_avg_accurfilter"] = np.where(
        np.abs(predictions["error_ratio"]) > FILTER_THRESHOLD,
        np.nan,
        predictions["current_avg"],
    )

    # and together
    predictions["current_avg_bothfilters"] = predictions.loc[
        ~pd.isna(predictions["current_avg_precfilter"])
        & ~pd.isna(predictions["current_avg_accurfilter"]),
        "current_avg",
    ]

    # apply current filter to future points (assign nan to future points where current points "failed")
    def apply_current_filter(
        predictions: pd.DataFrame, descriptor: str
    ) -> pd.DataFrame:
        predictions[f"{descriptor}_filtered"] = np.where(
            pd.isna(predictions["current_avg_bothfilters"]),
            np.nan,
            predictions[descriptor],
        )
        return predictions

    predictions = apply_current_filter(predictions, "extreme_avg")
    predictions = apply_current_filter(predictions, "extreme_95th_centile")
    predictions = apply_current_filter(predictions, "extreme_5th_centile")
    predictions = apply_current_filter(predictions, "extreme_AI_avg")
    predictions = apply_current_filter(predictions, "extreme_ET_avg")
    predictions = apply_current_filter(predictions, "extreme_precip_avg")
    predictions = apply_current_filter(predictions, f"extreme_{ELEMENT}dep_avg")
    predictions = apply_current_filter(predictions, "extreme_toc_avg")

    predictions = apply_current_filter(predictions, "extreme_avg_percchange")
    predictions = apply_current_filter(predictions, "extreme_95th_centile_percchange")
    predictions = apply_current_filter(predictions, "extreme_5th_centile_percchange")
    predictions = apply_current_filter(predictions, "extreme_percchange_std")
    predictions = apply_current_filter(predictions, "extreme_AI_avg_percchange")
    predictions = apply_current_filter(predictions, "extreme_ET_avg_percchange")
    predictions = apply_current_filter(predictions, "extreme_precip_avg_percchange")
    predictions = apply_current_filter(
        predictions, f"extreme_{ELEMENT}dep_avg_percchange"
    )
    predictions = apply_current_filter(predictions, "extreme_toc_avg_percchange")

    predictions = apply_current_filter(predictions, "residuals")
    predictions = apply_current_filter(predictions, "residuals_fracobs")

    return predictions


def plot_obs_vs_pred(predictions: pd.DataFrame, grid_info: dict):
    """Plots the predicted vs observed map"""

    predictions.to_csv(os.path.join(OUTPUT_DIR, f"{EXECUTION_ID}_map_info.csv"))

    obs = predictions["Observed"]
    pred = predictions["current_avg_bothfilters"]

    # interpolate obs and pred data to grid using griddata
    grid_data1 = griddata(
        points=(grid_info["lon_c"], grid_info["lat_c"]),
        values=obs,
        xi=(grid_info["Xi_c"], grid_info["Yi_c"]),
        method="nearest",
    )
    grid_data2 = griddata(
        points=(grid_info["lon_c"], grid_info["lat_c"]),
        values=pred,
        xi=(grid_info["Xi_c"], grid_info["Yi_c"]),
        method="nearest",
    )

    # griddat interpolates beyond the extend of the ungridded lat lon data which is undesirable
    # thankfully, scipy's KDTree can help us designate pixels a certain distance away from a point as nan
    # the distance is arbitrary, but will greatly impact the visual quality of the plot
    tree = KDTree(np.c_[grid_info["lon_c"], grid_info["lat_c"]])
    dist, _ = tree.query(
        np.c_[grid_info["Xi_c"].ravel(), grid_info["Yi_c"].ravel()], k=1
    )
    dist = dist.reshape(grid_info["Xi_c"].shape)
    grid_data1[dist > 0.6] = np.nan
    grid_data2[dist > 0.6] = np.nan

    # specify colorbar information
    bins = PLOT_INFO["obs_v_pred_bins"]  # define bin ranges based on input list
    colors = PLOT_INFO["obs_v_pred_colors"]
    # create a colormap
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        "pred_vs_obs", colors, N=len(bins)
    )
    norm = mpl.colors.BoundaryNorm(bins, len(bins))

    # plot the figure
    fig, ax = plt.subplots(
        ncols=2, figsize=((20, 8)), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax[0].set_extent(
        [
            grid_info["min_lon_c"],
            grid_info["max_lon_c"],
            grid_info["min_lat_c"],
            grid_info["max_lat_c"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax[0].coastlines("50m")
    ax[0].add_feature(cfeature.BORDERS, zorder=3)

    obs_mesh = ax[0].pcolormesh(
        grid_info["xi_shift_c"],
        grid_info["yi_shift_c"],
        grid_data1,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    ax[1].set_extent(
        [
            grid_info["min_lon_c"],
            grid_info["max_lon_c"],
            grid_info["min_lat_c"],
            grid_info["max_lat_c"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax[1].coastlines("50m")
    ax[1].add_feature(cfeature.BORDERS, zorder=3)
    pred_mesh = ax[1].pcolormesh(
        grid_info["xi_shift_c"],
        grid_info["yi_shift_c"],
        grid_data2,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )
    fig.subplots_adjust(wspace=0.05)  # adjust the heigh to
    cbar = fig.colorbar(
        obs_mesh, ax=ax.ravel().tolist(), pad=0.02, format="%0.3f", shrink=0.87
    )
    cbar.set_ticks([])

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{EXECUTION_ID}_{ELEMENT}_predvsobs_{MODEL}.png"),
        bbox_inches="tight",
    )

    # plot individual points

    # Create a Cartopy map
    fig, ax = plt.subplots(
        ncols=2, figsize=((20, 8)), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax[0].set_extent(
        [
            grid_info["min_lon_c"],
            grid_info["max_lon_c"],
            grid_info["min_lat_c"],
            grid_info["max_lat_c"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax[0].coastlines("50m")
    ax[0].add_feature(cfeature.BORDERS, zorder=3)

    obs_lon = obs.index.get_level_values(level="lon").to_numpy()
    obs_lat = obs.index.get_level_values(level="lat").to_numpy()
    obs_vals = obs.values

    # Plot individual points
    sp = ax[0].scatter(
        obs_lon,
        obs_lat,
        c=obs_vals,
        marker="o",
        s=30,
        cmap=cm,
        norm=norm,
        transform=ccrs.Geodetic(),
    )

    ax[1].set_extent(
        [
            grid_info["min_lon_c"],
            grid_info["max_lon_c"],
            grid_info["min_lat_c"],
            grid_info["max_lat_c"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax[1].coastlines("50m")
    ax[1].add_feature(cfeature.BORDERS, zorder=3)

    pred_lon = pred.index.get_level_values(level="lon").to_numpy()
    pred_lat = pred.index.get_level_values(level="lat").to_numpy()
    pred_vals = pred.values

    ax[1].scatter(
        pred_lon,
        pred_lat,
        c=pred_vals,
        marker="o",
        s=30,
        cmap=cm,
        norm=norm,
        transform=ccrs.Geodetic(),
    )

    fig.subplots_adjust(wspace=0.05)
    cbar = fig.colorbar(
        sp, ax=ax.ravel().tolist(), pad=0.02, format="%0.3f", shrink=0.87
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR, f"{EXECUTION_ID}_{ELEMENT}_predvsobs_indpoints_{MODEL}.png"
        ),
        bbox_inches="tight",
    )


def plot_obs_vs_pred_residuals(predictions: pd.DataFrame, grid_info: dict):
    """Plots the residuals"""

    res = predictions["residuals_filtered"]

    # interpolate
    grid_data = griddata(
        (grid_info["lon_c"], grid_info["lat_c"]),
        res,
        (grid_info["Xi_c"], grid_info["Yi_c"]),
        method="nearest",
    )

    tree = KDTree(np.c_[grid_info["lon_c"], grid_info["lat_c"]])
    dist, _ = tree.query(
        np.c_[grid_info["Xi_c"].ravel(), grid_info["Yi_c"].ravel()], k=1
    )
    dist = dist.reshape(grid_info["Xi_c"].shape)
    grid_data[dist > 0.6] = np.nan

    # specify colorbar information
    bins = PLOT_INFO["residuals_bins"]
    colors = PLOT_INFO["residuals_colors"]
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        "resid", colors, N=len(bins)
    )  # create a colormap
    norm = mpl.colors.BoundaryNorm(bins, len(bins))

    # plot the figure
    fig = plt.figure(figsize=(18, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [
            grid_info["min_lon_c"],
            grid_info["max_lon_c"],
            grid_info["min_lat_c"],
            grid_info["max_lat_c"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines("50m", zorder=1)  #'50m' means 1:50M resolution
    ax.add_feature(cfeature.BORDERS, zorder=3)

    resid = ax.pcolormesh(
        grid_info["xi_shift_c"],
        grid_info["yi_shift_c"],
        grid_data,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    cbar = fig.colorbar(resid, pad=0.02, format="%0.3f", shrink=1)

    cbar.set_ticks([])

    plt.savefig(
        os.path.join(
            OUTPUT_DIR, f"{EXECUTION_ID}_{ELEMENT}_predvsobs_residuals_{MODEL}.png"
        ),
        bbox_inches="tight",
    )


def plot_coeff_var(predictions, grid_info):
    """Plots the coefficient of variation (precision filter) for the current predictions"""

    coeff_var = predictions["coeff_var"]

    # interpolate
    grid_data = griddata(
        (grid_info["lon_c"], grid_info["lat_c"]),
        coeff_var,
        (grid_info["Xi_c"], grid_info["Yi_c"]),
        method="nearest",
    )

    tree = KDTree(np.c_[grid_info["lon_c"], grid_info["lat_c"]])
    dist, _ = tree.query(
        np.c_[grid_info["Xi_c"].ravel(), grid_info["Yi_c"].ravel()], k=1
    )
    dist = dist.reshape(grid_info["Xi_c"].shape)
    grid_data[dist > 0.6] = np.nan

    # specify colorbar information
    bins = PLOT_INFO["coeff_var_bins"]
    colors = PLOT_INFO["coeff_var_colors"]
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        "coeff_var", colors, N=len(bins)
    )  # create a colormap
    norm = mpl.colors.BoundaryNorm(bins, len(bins))

    # plot the figure
    fig = plt.figure(figsize=(18, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [
            grid_info["min_lon_c"],
            grid_info["max_lon_c"],
            grid_info["min_lat_c"],
            grid_info["max_lat_c"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines("50m", zorder=1)  #'50m' means 1:50M resolution
    ax.add_feature(cfeature.BORDERS, zorder=3)

    coeff_var = ax.pcolormesh(
        grid_info["xi_shift_c"],
        grid_info["yi_shift_c"],
        grid_data,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    cbar = fig.colorbar(coeff_var, pad=0.02, format="%0.3f", shrink=1)

    cbar.set_ticks([])

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{EXECUTION_ID}_{ELEMENT}_coeff_var_precisionfilter_{MODEL}.png",
        ),
        bbox_inches="tight",
    )


def plot_error_ratio(predictions, grid_info):
    """Plots the error ratio (accuracy filter) for the current predictions"""

    error_ratio = predictions["error_ratio"]

    # interpolate
    grid_data = griddata(
        (grid_info["lon_c"], grid_info["lat_c"]),
        error_ratio,
        (grid_info["Xi_c"], grid_info["Yi_c"]),
        method="nearest",
    )

    tree = KDTree(np.c_[grid_info["lon_c"], grid_info["lat_c"]])
    dist, _ = tree.query(
        np.c_[grid_info["Xi_c"].ravel(), grid_info["Yi_c"].ravel()], k=1
    )
    dist = dist.reshape(grid_info["Xi_c"].shape)
    grid_data[dist > 0.6] = np.nan

    # specify colorbar information
    bins = PLOT_INFO["error_ratio_bins"]
    colors = PLOT_INFO["error_ratio_colors"]
    cmap_name = "my_list"
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=len(bins)
    )  # create a colormap
    norm = mpl.colors.BoundaryNorm(bins, len(bins))

    # plot the figure
    fig = plt.figure(figsize=(18, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [
            grid_info["min_lon_c"],
            grid_info["max_lon_c"],
            grid_info["min_lat_c"],
            grid_info["max_lat_c"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines("50m", zorder=1)  #'50m' means 1:50M resolution
    ax.add_feature(cfeature.BORDERS, zorder=3)

    error_ratio = ax.pcolormesh(
        grid_info["xi_shift_c"],
        grid_info["yi_shift_c"],
        grid_data,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    cbar = fig.colorbar(error_ratio, pad=0.02, format="%0.3f", shrink=1)

    cbar.set_ticks([])

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{EXECUTION_ID}_{ELEMENT}_error_ratio_accuracyfilter_{MODEL}.png",
        ),
        bbox_inches="tight",
    )


def plot_perc_residuals(predictions: pd.DataFrame, grid_info: dict):
    """Plots the residuals as a fraction of the observed value"""

    res_perc = predictions["residuals_fracobs_filtered"]

    # interpolate
    grid_data = griddata(
        (grid_info["lon_c"], grid_info["lat_c"]),
        res_perc,
        (grid_info["Xi_c"], grid_info["Yi_c"]),
        method="nearest",
    )

    # griddat interpolates beyond the extend of the ungridded lat lon data which is undesirable
    # thankfully, scipy's KDTree can help us designate pixels a certain distance away from a point as nan
    # the distance is arbitrary, but will greatly impact the visual quality of the plot
    tree = KDTree(np.c_[grid_info["lon_c"], grid_info["lat_c"]])
    dist, _ = tree.query(
        np.c_[grid_info["Xi_c"].ravel(), grid_info["Yi_c"].ravel()], k=1
    )
    dist = dist.reshape(grid_info["Xi_c"].shape)
    grid_data[dist > 0.6] = np.nan

    # specify colorbar information
    res_bins = PLOT_INFO["perc_res_bins"]
    res_colors = PLOT_INFO["residuals_colors"]
    cmap_name = "my_list"
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        cmap_name, res_colors, N=len(res_bins)
    )  # create a colormap
    norm = mpl.colors.BoundaryNorm(res_bins, len(res_bins))

    # plot the figure
    fig = plt.figure(figsize=(18, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [
            grid_info["min_lon_c"],
            grid_info["max_lon_c"],
            grid_info["min_lat_c"],
            grid_info["max_lat_c"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines("50m", zorder=1)  #'50m' means 1:50M resolution
    ax.add_feature(cfeature.BORDERS, zorder=3)

    resid = ax.pcolormesh(
        grid_info["xi_shift_c"],
        grid_info["yi_shift_c"],
        grid_data,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    cbar = fig.colorbar(resid, pad=0.02, format="%0.3f", shrink=1)

    cbar.set_ticks([])

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{EXECUTION_ID}_{ELEMENT}_predvsobs_residuals_fracobs_{MODEL}.png",
        ),
        bbox_inches="tight",
    )


def plot_percent_change(predictions: pd.DataFrame, grid_info: dict):
    """Plots the percent change"""

    perc_change_avg = predictions.loc[
        predictions["has_future_point"] == 1, "extreme_avg_percchange_filtered"
    ].copy(deep=True)
    perc_change_95th = predictions.loc[
        predictions["has_future_point"] == 1, "extreme_95th_centile_percchange_filtered"
    ].copy(deep=True)
    perc_change_5th = predictions.loc[
        predictions["has_future_point"] == 1, "extreme_5th_centile_percchange_filtered"
    ].copy(deep=True)

    # interpolate to grid
    grid_data_avg = griddata(
        (grid_info["lon_f"], grid_info["lat_f"]),
        perc_change_avg,
        (grid_info["Xi_f"], grid_info["Yi_f"]),
        method="nearest",
    )
    grid_data_95th = griddata(
        (grid_info["lon_f"], grid_info["lat_f"]),
        perc_change_95th,
        (grid_info["Xi_f"], grid_info["Yi_f"]),
        method="nearest",
    )
    grid_data_5th = griddata(
        (grid_info["lon_f"], grid_info["lat_f"]),
        perc_change_5th,
        (grid_info["Xi_f"], grid_info["Yi_f"]),
        method="nearest",
    )

    # griddat interpolates beyond the extend of the ungridded lat lon data which is undesirable
    # thankfully, scipy's KDTree can help us designate pixels a certain distance away from a point as nan
    # the distance is arbitrary, but will greatly impact the visual quality of the plot
    tree = KDTree(np.c_[grid_info["lon_f"], grid_info["lat_f"]])
    dist, _ = tree.query(
        np.c_[grid_info["Xi_f"].ravel(), grid_info["Yi_f"].ravel()], k=1
    )
    dist = dist.reshape(grid_info["Xi_f"].shape)
    grid_data_avg[dist > 0.6] = np.nan
    grid_data_95th[dist > 0.6] = np.nan
    grid_data_5th[dist > 0.6] = np.nan

    # specify colorbar information
    perc_change_bins = PLOT_INFO["perc_change_bins"]
    perce_change_colors = PLOT_INFO["perc_change_colors"]
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        "cmap_name", perce_change_colors, N=len(perc_change_bins)
    )  # create a colormap
    norm = mpl.colors.BoundaryNorm(perc_change_bins, len(perc_change_bins))

    # plot the figure
    fig, ax = plt.subplots(
        ncols=3, figsize=((20, 5)), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax[0].set_extent(
        [
            grid_info["min_lon_f"],
            grid_info["max_lon_f"],
            grid_info["min_lat_f"],
            grid_info["max_lat_f"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax[0].coastlines("50m")
    ax[0].add_feature(cfeature.BORDERS, zorder=3)

    data1 = ax[0].pcolormesh(
        grid_info["xi_shift_f"],
        grid_info["yi_shift_f"],
        grid_data_avg,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    ax[1].set_extent(
        [
            grid_info["min_lon_f"],
            grid_info["max_lon_f"],
            grid_info["min_lat_f"],
            grid_info["max_lat_f"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax[1].coastlines("50m")
    ax[1].add_feature(cfeature.BORDERS, zorder=3)

    data2 = ax[1].pcolormesh(
        grid_info["xi_shift_f"],
        grid_info["yi_shift_f"],
        grid_data_95th,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    ax[2].set_extent(
        [
            grid_info["min_lon_f"],
            grid_info["max_lon_f"],
            grid_info["min_lat_f"],
            grid_info["max_lat_f"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax[2].coastlines("50m")
    ax[2].add_feature(cfeature.BORDERS, zorder=3)

    data3 = ax[2].pcolormesh(
        grid_info["xi_shift_f"],
        grid_info["yi_shift_f"],
        grid_data_5th,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    fig.subplots_adjust(wspace=0.05)  # adjust the heigh to
    cbar = fig.colorbar(
        data1, ax=ax.ravel().tolist(), pad=0.02, format="%0.3f", shrink=0.93
    )
    cbar.set_ticks([])
    plt.savefig(
        os.path.join(
            OUTPUT_DIR, f"{EXECUTION_ID}_{ELEMENT}_percent_change_{MODEL}.png"
        ),
        bbox_inches="tight",
    )


def plot_percchange_std_future(predictions: pd.DataFrame, grid_info: dict):
    """Plots the std of n percent change iterations"""

    future_stdev = predictions.loc[
        predictions["has_future_point"] == 1, "extreme_percchange_std_filtered"
    ].copy(deep=True)

    # interpolate obs and pred data to grid using griddata
    Z_perc_change_e = griddata(
        (grid_info["lon_f"], grid_info["lat_f"]),
        future_stdev,
        (grid_info["Xi_f"], grid_info["Yi_f"]),
        method="nearest",
    )

    tree = KDTree(
        np.c_[grid_info["lon_f"], grid_info["lat_f"]]
    )  # concatenate lon lat arrays, define tree
    dist, _ = tree.query(
        np.c_[grid_info["Xi_f"].ravel(), grid_info["Yi_f"].ravel()], k=1
    )  # concatentate raveled (flattened) grid, k is # of neighbors
    dist = dist.reshape(
        grid_info["Xi_f"].shape
    )  # reshape distance back to grid (2d array)
    Z_perc_change_e[dist > 0.6] = np.nan

    # specify colorbar information
    perc_change_bins = PLOT_INFO["perc_change_std_bins"]
    perce_change_colors = PLOT_INFO["perc_change_std_colors"]
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        "cmap_name", perce_change_colors, N=len(perc_change_bins)
    )  # create a colormap
    norm = mpl.colors.BoundaryNorm(perc_change_bins, len(perc_change_bins))

    # plot the figure
    fig = plt.figure(figsize=(18, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [
            grid_info["min_lon_f"],
            grid_info["max_lon_f"],
            grid_info["min_lat_f"],
            grid_info["max_lat_f"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines("50m", zorder=1)  #'50m' means 1:50M resolution
    ax.add_feature(cfeature.BORDERS, zorder=3)

    data = ax.pcolormesh(
        grid_info["xi_shift_f"],
        grid_info["yi_shift_f"],
        Z_perc_change_e,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    cbar = fig.colorbar(data, pad=0.02, format="%0.3f", shrink=0.93)
    cbar.set_ticks([])

    plt.savefig(
        os.path.join(
            OUTPUT_DIR, f"{EXECUTION_ID}_{ELEMENT}_percent_change_std_{MODEL}.png"
        ),
        bbox_inches="tight",
    )


def plot_percent_change_singlepredictor(
    predictions: pd.DataFrame, grid_info: dict, predictor: str
):
    """plots the percent change for the extreme scenario single predictor"""

    perc_change_avg = predictions.loc[
        predictions["has_future_point"] == 1,
        f"extreme_{predictor}_avg_percchange_filtered",
    ].copy(deep=True)

    # interpolate
    grid_data_avg = griddata(
        (grid_info["lon_f"], grid_info["lat_f"]),
        perc_change_avg,
        (grid_info["Xi_f"], grid_info["Yi_f"]),
        method="nearest",
    )

    # griddat interpolates beyond the extend of the ungridded lat lon data which is undesirable
    # thankfully, scipy's KDTree can help us designate pixels a certain distance away from a point as nan
    # the distance is arbitrary, but will greatly impact the visual quality of the plot
    tree = KDTree(np.c_[grid_info["lon_f"], grid_info["lat_f"]])
    dist, _ = tree.query(
        np.c_[grid_info["Xi_f"].ravel(), grid_info["Yi_f"].ravel()], k=1
    )
    dist = dist.reshape(grid_info["Xi_f"].shape)
    grid_data_avg[dist > 0.6] = np.nan

    # specify colorbar information
    perc_change_bins = PLOT_INFO["perc_change_bins"]
    perce_change_colors = PLOT_INFO["perc_change_colors"]
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        "cmap_name", perce_change_colors, N=len(perc_change_bins)
    )  # create a colormap
    norm = mpl.colors.BoundaryNorm(perc_change_bins, len(perc_change_bins))

    # plot the figure
    fig, ax = plt.subplots(
        ncols=1, figsize=((20, 5)), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.set_extent(
        [
            grid_info["min_lon_f"],
            grid_info["max_lon_f"],
            grid_info["min_lat_f"],
            grid_info["max_lat_f"],
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines("50m")
    ax.add_feature(cfeature.BORDERS, zorder=3)

    data = ax.pcolormesh(
        grid_info["xi_shift_f"],
        grid_info["yi_shift_f"],
        grid_data_avg,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    cbar = fig.colorbar(data, pad=0.02, format="%0.3f", shrink=1)
    cbar.set_ticks([])
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{EXECUTION_ID}_{ELEMENT}_percent_change_{predictor}_{MODEL}.png",
        ),
        bbox_inches="tight",
    )


def plot_obs_vs_pred_chart(predictions: pd.DataFrame):
    "Plots observed verses the avg. of the predicted current data"

    predictions = (
        predictions[["Observed", "current_avg"]].copy(deep=True).dropna(how="any")
    )

    obs = predictions["Observed"]
    pred = predictions["current_avg"]

    rmse = sqrt(mean_squared_error(predictions["Observed"], predictions["current_avg"]))
    print(rmse)

    fig, ax = plt.subplots()
    # ax.text(200, 1000, "RMSE = " + str(round(rmse, 0)), fontsize=12)
    ax.plot(obs, pred, ".b", markersize=12)  # plot data
    ax.plot([obs.min(), obs.max()], [obs.min(), obs.max()], "-", color="black")

    ax.set_xlabel("Observed", fontsize=20)
    ax.set_ylabel("Predicted", fontsize=20)
    ax.tick_params(which="both", length=4, width=1.5, pad=10)
    ax.tick_params(labelsize=20)

    # ax.loglog()

    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{EXECUTION_ID}_obs_vs_pred_chart.png"),
        bbox_inches="tight",
        dpi=1000,
    )


def main():

    # import results
    setup_logging()

    LOGGER.info("Generating maps")

    # preparations
    observed, predictions = import_results()
    lat_lon_c, lat_lon_f = get_lat_lon_values(predictions)
    predictions = calculate_percent_change(predictions)
    predictions = calculate_residuals(predictions, observed)
    predictions = calculate_and_apply_filters(predictions)
    grid_info = get_grid_info(lat_lon_c, lat_lon_f)

    # generate plots
    plot_obs_vs_pred(predictions, grid_info)
    plot_obs_vs_pred_residuals(predictions, grid_info)
    plot_coeff_var(predictions, grid_info)
    plot_error_ratio(predictions, grid_info)
    plot_perc_residuals(predictions, grid_info)
    plot_percent_change(predictions, grid_info)
    plot_percchange_std_future(predictions, grid_info)
    plot_percent_change_singlepredictor(predictions, grid_info, "AI")
    plot_percent_change_singlepredictor(predictions, grid_info, "ET")
    plot_percent_change_singlepredictor(predictions, grid_info, "precip")
    plot_percent_change_singlepredictor(predictions, grid_info, f"{ELEMENT}dep")
    plot_percent_change_singlepredictor(predictions, grid_info, "toc")

    # generate charst
    plot_obs_vs_pred_chart(predictions)


def get_grid_info(lat_lon_c: pd.MultiIndex, lat_lon_f: pd.MultiIndex):
    """Retrieves grid information for plotting"""

    lon_c = lat_lon_c.get_level_values(level="lon").to_numpy()
    lat_c = lat_lon_c.get_level_values(level="lat").to_numpy()

    lon_f = lat_lon_f.get_level_values(level="lon").to_numpy()
    lat_f = lat_lon_f.get_level_values(level="lat").to_numpy()

    Xi_c, Yi_c, xi_shift_c, yi_shift_c, min_lon_c, max_lon_c, min_lat_c, max_lat_c = (
        lat_lon_grid(lat_c, lon_c)
    )
    Xi_f, Yi_f, xi_shift_f, yi_shift_f, min_lon_f, max_lon_f, min_lat_f, max_lat_f = (
        lat_lon_grid(lat_f, lon_f)
    )

    grid_info = {
        "lon_c": lon_c,
        "lat_c": lat_c,
        "lon_f": lon_f,
        "lat_f": lat_f,
        "Xi_c": Xi_c,
        "Yi_c": Yi_c,
        "xi_shift_c": xi_shift_c,
        "yi_shift_c": yi_shift_c,
        "min_lon_c": min_lon_c,
        "max_lon_c": max_lon_c,
        "min_lat_c": min_lat_c,
        "max_lat_c": max_lat_c,
        "Xi_f": Xi_f,
        "Yi_f": Yi_f,
        "xi_shift_f": xi_shift_f,
        "yi_shift_f": yi_shift_f,
        "min_lon_f": min_lon_f,
        "max_lon_f": max_lon_f,
        "min_lat_f": min_lat_f,
        "max_lat_f": max_lat_f,
    }

    return grid_info


def lat_lon_grid(lat, lon):
    """Prepares the lat lon grid for plotting maps"""

    # define the limits of the map, with an extra few pixels for some space
    min_lon = math.floor(min(lon)) - 2
    max_lon = math.ceil(max(lon)) + 2
    min_lat = math.floor(min(lat)) - 2
    max_lat = math.ceil(max(lat)) + 2

    # define the pixel size in degrees
    cell_size = 1

    # how many cells are there for vertical and horizontal
    lon_num = int((max_lon - min_lon) / cell_size)
    lat_num = int((max_lat - min_lat) / cell_size)

    # create array defining limits for lat lon grid
    xi = np.linspace(min_lon, max_lon, lon_num)
    yi = np.linspace(min_lat, max_lat, lat_num)

    # use np.meshgrid create grid, Xi and Yi are now 2D arrays
    Xi, Yi = np.meshgrid(xi, yi)

    # Subtract 1/2 the grid size from both lon and lat arrays
    xi_shift = Xi - cell_size / 2
    yi_shift = Yi - cell_size / 2

    # Add 1 grid spacing to the right column of lon array and concatenate it as an additional column to the right
    xi_shift = np.c_[xi_shift, xi_shift[:, -1] + cell_size]

    # Duplicate the bottom row of the lon array and concatenate it to the bottom
    xi_shift = np.r_[xi_shift, [xi_shift[-1, :]]]

    # Duplicate the right-most column of lats array and concatenate it on the right
    yi_shift = np.c_[yi_shift, yi_shift[:, -1]]

    # Add 1 grid spacing to the bottom row of lat array and concatenate it as an additional row below
    yi_shift = np.r_[yi_shift, [yi_shift[-1, :] + cell_size]]

    return (
        Xi,
        Yi,
        xi_shift,
        yi_shift,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
    )  # return both the shifted grid and the normal grid


if __name__ == "__main__":
    main()
