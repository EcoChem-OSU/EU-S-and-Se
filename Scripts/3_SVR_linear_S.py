"""
Title: SVR Linear Sulfur - Model Predictions
Version: 1.0
Date: 29 April 2024
Author: Logan Insinga
Depends:
    numpy           1.26.4
    openpyxl        3.1.2
    pandas          2.2.2
    matplotlib      3.8.4
    scikit-learn    1.4.2

Description:
Tunes the SVR model for Sulfur. 
Evaluates the best SVR model performance.
Generates predictions using the optimized model for
    1. Current
    2. Future
    3. Future for each predictor
    4. Sensitivity analysis
Writes results to output dir.
"""

import os
import sys
import logging
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

EXECUTION_ID = "Run1"
MASTER_TABLE_PATH = r"Processing\_2_standardized_master_table.xlsx"
OUTPUT_DIR = r"Processing\_3_SVR_S"

LOGGER = logging.getLogger("my_logger")
LOGGER.setLevel(logging.DEBUG)


def setup_logging():
    """Sets up logging for documentation"""
    file_handler = logging.FileHandler(
        os.path.join(OUTPUT_DIR, f"_3_SVR_S_{EXECUTION_ID}.log"), "w"
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(logging.DEBUG)  # log file gets everything
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)


def get_predictors(master_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Gets predictors. Removes rows with toc_e = nan in future datasets"""

    LOGGER.info("Getting predictor datasets")

    predictors_current = master_table[
        [
            "Avg_CIA",
            "Avg_PH_CAC",
            "CLYPPT_M_s",
            "TOC_c",
            "AI_c_ext",
            "ET_c_ext",
            "Precip_c_e",
            "SDep_2005_2009",
        ]
    ].copy(deep=True)

    predictors_extreme = master_table[
        [
            "Avg_CIA",
            "Avg_PH_CAC",
            "CLYPPT_M_s",
            "TOC_e",
            "AI_e_ext",
            "ET_e_ext",
            "Precip_e_e",
            "SDep_SSP585",
        ]
    ].copy(deep=True)
    predictors_extreme.dropna(
        axis=0, inplace=True, how="any"
    )  # remove row if any col is nan
    extreme_valid_indxs = (
        predictors_extreme.index
    )  # get indexs associted with valid rows

    # get extreme predictors for each predictor
    # filter to get only extreme predictor rows that are valid
    predictors_extreme_AI = master_table[
        [
            "Avg_CIA",
            "Avg_PH_CAC",
            "CLYPPT_M_s",
            "TOC_c",
            "AI_e_ext",  # extreme
            "ET_c_ext",
            "Precip_c_e",
            "SDep_2005_2009",
        ]
    ].copy(deep=True)
    predictors_extreme_AI = predictors_extreme_AI.loc[extreme_valid_indxs, :]

    predictors_extreme_ET = master_table[
        [
            "Avg_CIA",
            "Avg_PH_CAC",
            "CLYPPT_M_s",
            "TOC_c",
            "AI_c_ext",
            "ET_e_ext",  # exteme
            "Precip_c_e",
            "SDep_2005_2009",
        ]
    ].copy(deep=True)
    predictors_extreme_ET = predictors_extreme_ET.loc[extreme_valid_indxs, :]

    predictors_extreme_precip = master_table[
        [
            "Avg_CIA",
            "Avg_PH_CAC",
            "CLYPPT_M_s",
            "TOC_c",
            "AI_c_ext",
            "ET_e_ext",
            "Precip_e_e",  # exteme
            "SDep_2005_2009",
        ]
    ].copy(deep=True)
    predictors_extreme_precip = predictors_extreme_precip.loc[extreme_valid_indxs, :]

    predictors_extreme_Sdep = master_table[
        [
            "Avg_CIA",
            "Avg_PH_CAC",
            "CLYPPT_M_s",
            "TOC_c",
            "AI_c_ext",
            "ET_e_ext",
            "Precip_c_e",
            "SDep_SSP585",  # exteme
        ]
    ].copy(deep=True)
    predictors_extreme_Sdep = predictors_extreme_Sdep.loc[extreme_valid_indxs, :]

    predictors_extreme_toc = master_table[
        [
            "Avg_CIA",
            "Avg_PH_CAC",
            "CLYPPT_M_s",
            "TOC_e",  # exteme
            "AI_c_ext",
            "ET_e_ext",
            "Precip_c_e",
            "SDep_2005_2009",
        ]
    ].copy(deep=True)
    predictors_extreme_toc = predictors_extreme_toc.loc[extreme_valid_indxs, :]

    predictors: dict[str, pd.DataFrame] = {
        "current": predictors_current,
        "extreme": predictors_extreme,
        "extreme_AI": predictors_extreme_AI,
        "extreme_ET": predictors_extreme_ET,
        "extreme_precip": predictors_extreme_precip,
        "extreme_Sdep": predictors_extreme_Sdep,
        "extreme_toc": predictors_extreme_toc,
    }

    return predictors


def tune_model(element_data: pd.Series, predictors_current: pd.DataFrame) -> dict:
    """Tune the model and get the best performing parameters"""

    LOGGER.info("Tuning the model to get the best combination of hyperparams.")

    num_combos = 1000
    num_random_seeds = 5
    n_params = 2

    # create hyperparameter sampling distributions
    C_dist = np.geomspace(0.0001, 10, num=1000, endpoint=True)
    epsilon_dist = np.geomspace(0.00001, 200, num=1000, endpoint=True)

    tuning_results_storage = np.zeros((num_combos, n_params + 1))

    for i in range(0, num_combos):

        # random parameter value from sampling distribution
        C = random.choice(C_dist)
        epsilon = random.choice(epsilon_dist)

        test_rmse_scores: np.array = np.zeros(num_random_seeds)

        for j in range(0, num_random_seeds):

            # split up the data into training and testing subsets
            x_train, x_test, y_train, y_test = train_test_split(
                predictors_current,
                element_data,
                test_size=0.2,
                random_state=None,
                shuffle=True,
            )

            # create a model using the selected parameters
            estimator = SVR(
                kernel="linear",
                gamma=1,
                tol=0.001,
                C=C,
                epsilon=epsilon,
                shrinking=True,
                cache_size=200,
                verbose=False,
                max_iter=-1,
            )

            # fit estimator and predict
            estimator.fit(x_train, y_train)
            # y_train_pred = estimator.predict(x_train)
            y_test_pred = estimator.predict(x_test)

            # calculate RMSE
            # train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))

            test_rmse_scores[j] = test_rmse

        avg_rmse = np.mean(test_rmse_scores)
        tuning_results_storage[i, :] = np.array([avg_rmse, C, epsilon])

    tuning_results: pd.DataFrame = pd.DataFrame(
        data=tuning_results_storage, columns=["RMSE", "C", "epsilon"]
    )
    tuning_results.sort_values(by="RMSE", ascending=True, inplace=True)
    tuning_results.to_csv(
        os.path.join(OUTPUT_DIR, f"{EXECUTION_ID}_tuning_results.csv"), index=False
    )
    LOGGER.info("Check output directory for tuning results table.")

    best_parameters: dict = dict(
        zip(tuning_results.columns.tolist(), tuning_results.iloc[0, :].tolist())
    )

    LOGGER.debug("\nBest hyper parameters:")
    LOGGER.debug(best_parameters)

    def plot_param_vs_score(param: str):
        fig, ax = plt.subplots()
        ax.plot(tuning_results[param], tuning_results["RMSE"], ".b")
        ax.plot(
            tuning_results[param].iloc[0:10], tuning_results["RMSE"].iloc[0:10], ".r"
        )
        ax.plot(tuning_results[param].iloc[0], tuning_results["RMSE"].iloc[0], ".c")
        ax.semilogx()
        ax.set_title(param)
        ax.set_xlabel(param)
        ax.set_ylabel("RMSE")
        fig.savefig(
            os.path.join(
                OUTPUT_DIR, f"{EXECUTION_ID}_tuning_results_{param}_vs_RMSE.png"
            )
        )

    plot_param_vs_score("C")
    plot_param_vs_score("epsilon")
    LOGGER.info("Check the output directory for tuning results charts.")

    return best_parameters


def evaluate_model(
    element_data: pd.Series, predictors_current: pd.DataFrame, best_parameters: dict
):
    """Evaluates model performance"""
    num_iterations = 100

    LOGGER.info("\nEvaluating model performance.")

    estimator = SVR(
        kernel="linear",
        gamma=1,
        tol=0.001,
        C=best_parameters["C"],
        epsilon=best_parameters["epsilon"],
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    )

    model_evalutation_results_storage: dict[str, list] = {
        "train_r2": [],
        "test_r2": [],
        "train_rmse": [],
        "test_rmse": [],
    }

    for _ in range(0, num_iterations):

        x_train, x_test, y_train, y_test = train_test_split(  # train test spit the data
            predictors_current,
            element_data,
            test_size=0.2,
            random_state=None,
            shuffle=True,
        )

        estimator.fit(x_train, y_train)
        y_train_pred = estimator.predict(x_train)
        y_test_pred = estimator.predict(x_test)

        training_r2 = estimator.score(x_train, y_train, sample_weight=None)
        testing_r2 = r2_score(y_test, y_test_pred, sample_weight=None)

        train_rmse = sqrt(mean_squared_error(10**y_train, 10**y_train_pred))
        test_rmse = sqrt(mean_squared_error(10**y_test, 10**y_test_pred))

        model_evalutation_results_storage["train_r2"].append(training_r2)
        model_evalutation_results_storage["test_r2"].append(testing_r2)
        model_evalutation_results_storage["train_rmse"].append(train_rmse)
        model_evalutation_results_storage["test_rmse"].append(test_rmse)

    model_evalutation_results = pd.DataFrame.from_dict(
        data=model_evalutation_results_storage, orient="columns"
    )
    model_evalutation_results.index.name = "Iteration"

    model_evalutation_results.to_csv(
        os.path.join(OUTPUT_DIR, f"{EXECUTION_ID}_model_evaluation_results.csv"),
        index=True,
    )
    LOGGER.info("Check the output directory for model performance results.")


def create_sensitivity_matrix(predictors_current: pd.DataFrame) -> np.array:
    """Creates a sensitivy matrix"""

    # create a sensitivity matrix with 31 rows for each predictor and columns = num predictors
    sens_matrix = np.zeros(
        (31 * len(predictors_current.columns), len(predictors_current.columns))
    )

    # for each column, create 31 rows+
    for i in range(0, len(predictors_current.columns), 1):
        sens_matrix[i * 31 + 0, i] = -6
        sens_matrix[i * 31 + 1, i] = -5.6
        sens_matrix[i * 31 + 2, i] = -5.2
        sens_matrix[i * 31 + 3, i] = -4.8
        sens_matrix[i * 31 + 4, i] = -4.4
        sens_matrix[i * 31 + 5, i] = -4
        sens_matrix[i * 31 + 6, i] = -3.6
        sens_matrix[i * 31 + 7, i] = -3.2
        sens_matrix[i * 31 + 8, i] = -2.8
        sens_matrix[i * 31 + 9, i] = -2.4
        sens_matrix[i * 31 + 10, i] = -2
        sens_matrix[i * 31 + 11, i] = -1.6
        sens_matrix[i * 31 + 12, i] = -1.2
        sens_matrix[i * 31 + 13, i] = -0.8
        sens_matrix[i * 31 + 14, i] = -0.4
        sens_matrix[i * 31 + 15, i] = 0
        sens_matrix[i * 31 + 16, i] = 0.4
        sens_matrix[i * 31 + 17, i] = 0.8
        sens_matrix[i * 31 + 18, i] = 1.2
        sens_matrix[i * 31 + 19, i] = 1.6
        sens_matrix[i * 31 + 20, i] = 2
        sens_matrix[i * 31 + 21, i] = 2.4
        sens_matrix[i * 31 + 22, i] = 2.8
        sens_matrix[i * 31 + 23, i] = 3.2
        sens_matrix[i * 31 + 24, i] = 3.6
        sens_matrix[i * 31 + 25, i] = 4
        sens_matrix[i * 31 + 26, i] = 4.4
        sens_matrix[i * 31 + 27, i] = 4.8
        sens_matrix[i * 31 + 28, i] = 5.2
        sens_matrix[i * 31 + 29, i] = 5.6
        sens_matrix[i * 31 + 30, i] = 6

    return sens_matrix


def model_predictions(
    element_data: pd.Series,
    predictors: dict[str, pd.DataFrame],
    sens_matrix: np.array,
    best_parameters: dict,
):
    """Generates model predictions"""

    LOGGER.info("\nGenerating model predictions.")
    num_iterations = 100

    lat_lon_current = (
        predictors["current"].reset_index()[["lon", "lat"]].copy(deep=True)
    )
    lat_lon_future = predictors["extreme"].reset_index()[["lon", "lat"]].copy(deep=True)

    estimator = SVR(
        kernel="linear",
        gamma=1,
        tol=0.001,
        C=best_parameters["C"],
        epsilon=best_parameters["epsilon"],
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    )

    model_prediction_storage: dict[str, dict[int, np.array]] = {
        "pred_current": {},
        "pred_extreme": {},
        "pred_extreme_AI": {},
        "pred_extreme_ET": {},
        "pred_extreme_precip": {},
        "pred_extreme_Sdep": {},
        "pred_extreme_toc": {},
        "pred_sens": {},
    }

    # each iteration is a key
    for i in range(0, num_iterations):

        # train the model on the current data
        x_train, x_test, y_train, y_test = train_test_split(
            predictors["current"].to_numpy(),
            element_data.to_numpy(),
            test_size=0.2,
            random_state=None,
            shuffle=True,
        )
        estimator.fit(x_train, y_train)  # fit the model to training data

        # make predictions for various datasets, undo log transformation
        # y_pred_test = 10 ** estimator.predict(x_test)
        y_pred_current = 10 ** estimator.predict(predictors["current"].to_numpy())
        y_pred_extreme = 10 ** estimator.predict(predictors["extreme"].to_numpy())
        y_pred_extreme_AI = 10 ** estimator.predict(predictors["extreme_AI"].to_numpy())
        y_pred_extreme_ET = 10 ** estimator.predict(predictors["extreme_ET"].to_numpy())
        y_pred_extreme_precip = 10 ** estimator.predict(
            predictors["extreme_precip"].to_numpy()
        )
        y_pred_extreme_Sdep = 10 ** estimator.predict(
            predictors["extreme_Sdep"].to_numpy()
        )
        y_pred_extreme_toc = 10 ** estimator.predict(
            predictors["extreme_toc"].to_numpy()
        )
        y_pred_sens = 10 ** estimator.predict(sens_matrix)

        # score model performance
        # training_r2 = estimator.score(x_train, y_train, sample_weight=None)
        # testing_r2 = r2_score(y_test, y_pred_test, sample_weight=None, multioutput="uniform_average")

        # store model predictions
        model_prediction_storage["pred_current"][i] = y_pred_current
        model_prediction_storage["pred_extreme"][i] = y_pred_extreme
        model_prediction_storage["pred_extreme_AI"][i] = y_pred_extreme_AI
        model_prediction_storage["pred_extreme_ET"][i] = y_pred_extreme_ET
        model_prediction_storage["pred_extreme_precip"][i] = y_pred_extreme_precip
        model_prediction_storage["pred_extreme_Sdep"][i] = y_pred_extreme_Sdep
        model_prediction_storage["pred_extreme_toc"][i] = y_pred_extreme_toc
        model_prediction_storage["pred_sens"][i] = y_pred_sens

    def post_process_predictions(table_name: str, lat_lon: pd.DataFrame):
        """Post-processes the predictions and writes output"""
        pred_df = pd.DataFrame.from_dict(
            data=model_prediction_storage[table_name],
            orient="columns",
        )
        avg_pred = pred_df.mean(axis=1)
        std_pred = pred_df.std(axis=1)
        _5th_centile = pred_df.quantile(q=0.05, axis=1, interpolation="linear")
        _95th_centile = pred_df.quantile(q=0.95, axis=1, interpolation="linear")
        pred_df["avg"] = avg_pred
        pred_df["std"] = std_pred
        pred_df["95th_centile"] = _95th_centile
        pred_df["5th_centile"] = _5th_centile

        output = lat_lon.join(other=pred_df, how="left")
        output.set_index(keys=["lon", "lat"], drop=True, inplace=True)
        output.to_csv(
            os.path.join(OUTPUT_DIR, f"{EXECUTION_ID}_{table_name}.csv"), index=True
        )

    post_process_predictions("pred_current", lat_lon_current)
    post_process_predictions("pred_extreme", lat_lon_future)
    post_process_predictions("pred_extreme_AI", lat_lon_future)
    post_process_predictions("pred_extreme_ET", lat_lon_future)
    post_process_predictions("pred_extreme_precip", lat_lon_future)
    post_process_predictions("pred_extreme_Sdep", lat_lon_future)
    post_process_predictions("pred_extreme_toc", lat_lon_future)

    pred_sens_output = pd.DataFrame.from_dict(
        data=model_prediction_storage["pred_sens"],
        orient="columns",
    )
    pred_sens_output.to_csv(os.path.join(OUTPUT_DIR, f"{EXECUTION_ID}_pred_sens.csv"))

    LOGGER.info("Check the output directory for model prediction output files")


def main():
    """Initiate script here"""

    setup_logging()

    LOGGER.info("Running machine learning - SVR Sulfur")

    master_table = pd.read_excel(MASTER_TABLE_PATH)
    master_table.set_index(keys=["lon", "lat"], drop=True, inplace=True)

    element_data = master_table["Avg_S"]

    predictors = get_predictors(master_table)

    best_parameters = tune_model(element_data, predictors["current"])
    evaluate_model(element_data, predictors["current"], best_parameters)
    sens_matrix = create_sensitivity_matrix(predictors["current"])
    model_predictions(element_data, predictors, sens_matrix, best_parameters)

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
