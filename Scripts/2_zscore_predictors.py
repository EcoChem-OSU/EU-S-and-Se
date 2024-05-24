"""
Title: Z Score Predictors
Version: 1.0
Date: 29 April 2024
Author: Logan Insinga
Depends:
    numpy           1.26.4
    openpyxl        3.1.2
    pandas          2.2.2

Description:
Z-scores the predictors.
This is known as standardizing the predictor
variables.
Standardized data is recommended for 
machine learning algorithms. Standard data is also
better for sensitivity analysis.

Predictors with a future component will be standardized
using the mean and std of their current data
components.

Elements will not be standardized.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

NORMALIZED_MASTER_TABLE_PATH = r"Processing\_1_normalized_master_table.xlsx"
OUTPUT_DIR = os.path.join(os.path.dirname(NORMALIZED_MASTER_TABLE_PATH))

PREDICTORS_WITH_NO_FUTURE_COMPONENT = [
    "Avg_CIA",
    "Avg_CEC",
    "Avg_PH_CAC",
    "BLDFIE_M_s",
    "CLYPPT_M_s",
    "SLTPPT_M_s",
    "SNDPPT_M_s",
]

LOGGER = logging.getLogger("my_logger")
LOGGER.setLevel(logging.DEBUG)


def setup_logging():
    """Sets up logging for documentation"""

    file_handler = logging.FileHandler(
        f"{OUTPUT_DIR}\\_2_normalized_master_table.log", "w"
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(logging.DEBUG)  # log file gets everything
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)


def z_score_current_predictors(master_table: pd.DataFrame) -> pd.DataFrame:
    """Z scores the predictors with no future component"""

    for predictor in PREDICTORS_WITH_NO_FUTURE_COMPONENT:

        pred_mean = np.mean(master_table[predictor])
        pred_std = np.std(master_table[predictor])

        master_table[predictor] = (master_table[predictor] - pred_mean) / pred_std

    return master_table


def z_score_future_predictors(master_table: pd.DataFrame) -> pd.DataFrame:
    """Z scores the predictors with a future component.
    Use the mean and std of the current component."""

    def future_z_score(col, mean, std):
        return (col - mean) / std

    # TOC
    mean_TOC_c = np.mean(master_table["TOC_c"])
    std_TOC_c = np.std(master_table["TOC_c"])
    master_table["TOC_c"] = future_z_score(master_table["TOC_c"], mean_TOC_c, std_TOC_c)
    master_table["TOC_e"] = future_z_score(master_table["TOC_e"], mean_TOC_c, std_TOC_c)

    # ET
    mean_ET_c = np.mean(master_table["ET_c_ext"])
    std_ET_c = np.std(master_table["ET_c_ext"])
    master_table["ET_c_ext"] = future_z_score(
        master_table["ET_c_ext"], mean_ET_c, std_ET_c
    )
    master_table["ET_e_ext"] = future_z_score(
        master_table["ET_e_ext"], mean_ET_c, std_ET_c
    )

    # AI
    mean_AI_c = np.mean(master_table["AI_c_ext"])
    std_AI_c = np.std(master_table["AI_c_ext"])
    master_table["AI_c_ext"] = future_z_score(
        master_table["AI_c_ext"], mean_AI_c, std_AI_c
    )
    master_table["AI_e_ext"] = future_z_score(
        master_table["AI_e_ext"], mean_AI_c, std_AI_c
    )

    # Precip
    mean_P_c = np.mean(master_table["Precip_c_e"])
    std_P_c = np.std(master_table["Precip_c_e"])
    master_table["Precip_c_e"] = future_z_score(
        master_table["Precip_c_e"], mean_P_c, std_P_c
    )
    master_table["Precip_e_e"] = future_z_score(
        master_table["Precip_e_e"], mean_P_c, std_P_c
    )

    # S deposition
    mean_SDep_c = np.mean(master_table["SDep_2005_2009"])
    std_SDep_c = np.std(master_table["SDep_2005_2009"])
    master_table["SDep_2005_2009"] = future_z_score(
        master_table["SDep_2005_2009"], mean_SDep_c, std_SDep_c
    )
    master_table["SDep_SSP585"] = future_z_score(
        master_table["SDep_SSP585"], mean_SDep_c, std_SDep_c
    )

    # Se deposition
    mean_SeDep_c = np.mean(master_table["SeDep_2005_2009"])
    std_SeDep_c = np.std(master_table["SeDep_2005_2009"])
    master_table["SeDep_2005_2009"] = future_z_score(
        master_table["SeDep_2005_2009"], mean_SeDep_c, std_SeDep_c
    )
    master_table["SeDep_SSP585"] = future_z_score(
        master_table["SeDep_SSP585"], mean_SeDep_c, std_SeDep_c
    )

    return master_table


def main():

    setup_logging()

    LOGGER.info("Standardizing the predictors")

    master_table = pd.read_excel(NORMALIZED_MASTER_TABLE_PATH)
    master_table.set_index(keys=["lon", "lat"], drop=True, inplace=True)

    master_table = z_score_current_predictors(master_table)
    master_table = z_score_future_predictors(master_table)

    master_table.to_excel(
        os.path.join(OUTPUT_DIR, "_2_standardized_master_table.xlsx"),
        index=True,
    )

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
