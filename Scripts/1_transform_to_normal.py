"""
Title: Transform to Normal
Version: 1.0
Date: 29 April 2024
Author: Logan Insinga
Depends:
    numpy           1.26.4
    openpyxl        3.1.2
    pandas          2.2.2
    scipy           1.13.0

Description:
The first script in the series for the soil trace elements work. 
Transforms the predictor variables to make distributions more normal.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy import stats
from datetime import date

# define the location of the master table and where the results should be written
MASTER_TABLE_PATH = r"Processing\_0_Master_table.xlsx"
OUTPUT_DIR = os.path.join(os.path.dirname(MASTER_TABLE_PATH))

PREDICTORS_WITH_NO_FUTURE_COMPONENT = [
    "Avg_S",
    "Avg_SE",
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
        f"{OUTPUT_DIR}\\_1_normalized_master_table.log", "w"
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(logging.DEBUG)  # log file gets everything
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)


def best_skew(data):
    """Calculates the skew after each transformation."""

    skew_nt = np.absolute(stats.skew(data))
    LOGGER.info("Skew no transformation: %s", round(skew_nt, 2))

    skew_sqrt = np.absolute(stats.skew(np.sqrt(data)))
    LOGGER.info("Skew sqrt transform: %s", round(skew_sqrt, 2))

    skew_log = np.absolute(stats.skew(np.log10(data)))
    LOGGER.info("Skew log transform: %s", round(skew_log, 2))


def transform_predictors_no_future_component(
    master_table: pd.DataFrame,
) -> pd.DataFrame:
    """Transforms the predictors with no future component.
    That is, the predictors with only current data.

    Args:
        master_table (pd.DataFrame)

    Returns:
        pd.DataFrame: updated master table
    """

    LOGGER.info("\nTransforming predictors with no future component.\n")
    LOGGER.debug(PREDICTORS_WITH_NO_FUTURE_COMPONENT)

    transform_info: dict[str, str] = {}

    for predictor in PREDICTORS_WITH_NO_FUTURE_COMPONENT:
        original_skew = np.absolute(stats.skew(master_table[predictor]))

        LOGGER.debug(predictor)
        LOGGER.debug("Original skew: %s", round(original_skew, 2))

        if original_skew <= 1:
            transform_info[predictor] = (
                f"No transformation needed: final skew = {round(original_skew,1)}"
            )
            continue
        else:
            sqrt_skew = np.absolute(stats.skew(np.sqrt(master_table[predictor])))
            LOGGER.debug("SQRT skew: %s", round(sqrt_skew, 2))

            if sqrt_skew <= 1:
                transform_info[predictor] = (
                    f"SQRT transformation needed: final skew = {round(sqrt_skew,1)}"
                )
                master_table[predictor] = np.sqrt(master_table[predictor])
                continue
            else:
                log_skew = np.absolute(stats.skew(np.log10(master_table[predictor])))
                LOGGER.debug("Log skew: %s", round(log_skew, 2))

                if log_skew <= 1:
                    transform_info[predictor] = (
                        f"Log transformation needed: final skew = {round(log_skew, 1)}"
                    )
                    master_table[predictor] = np.log10(master_table[predictor])
                    continue
                else:
                    transform_info[predictor] = (
                        "Neither sqrt nor log transform was sufficient."
                    )

    for predictor, result in transform_info.items():
        LOGGER.info("%s: %s", predictor, result)

    LOGGER.info("\nTransforming Avg_S manually:")
    best_skew(master_table["Avg_S"])
    LOGGER.info("Log transformation is best")

    master_table["Avg_S"] = np.log10(master_table["Avg_S"])

    return master_table


def transform_predictors_with_future_component(
    master_table: pd.DataFrame,
) -> pd.DataFrame:
    """Transforms predictors with a future component.
    Future predictors will be transformed the same way
    as their current data.

    Args:
        master_table (pd.DataFrame)

    Returns:
        pd.DataFrame: updated master_table
    """

    LOGGER.debug("\nTransforming data with a future component\n")

    # TOC
    LOGGER.info("\nTOC")
    best_skew(master_table["TOC_c"])
    LOGGER.info("Log transformation was best.")
    master_table["TOC_c"] = np.log10(master_table["TOC_c"])
    master_table["TOC_e"] = np.log10(master_table["TOC_e"])

    # AI
    LOGGER.info("\nAI")
    best_skew(master_table["AI_c_ext"])
    LOGGER.info("Log transformation was best.")
    master_table["AI_c_ext"] = np.log10(master_table["AI_c_ext"])
    master_table["AI_e_ext"] = np.log10(master_table["AI_e_ext"])

    # ET
    LOGGER.info("\nET")
    best_skew(master_table["ET_c_ext"])
    LOGGER.info("No transformation was best.")

    # Precip
    LOGGER.info("\nPrecip.")
    best_skew(master_table["Precip_c_e"])
    LOGGER.info("Log transformation was best.")
    master_table["Precip_c_e"] = np.log10(master_table["Precip_c_e"])
    master_table["Precip_e_e"] = np.log10(master_table["Precip_e_e"])

    # S Dep
    LOGGER.info("\nS Dep.")
    best_skew(master_table["SDep_2005_2009"])
    LOGGER.info("Log transformation was best.")
    master_table["SDep_2005_2009"] = np.log10(master_table["SDep_2005_2009"])
    master_table["SDep_SSP126"] = np.log10(master_table["SDep_SSP126"])
    master_table["SDep_SSP585"] = np.log10(master_table["SDep_SSP585"])

    # Se Dep
    LOGGER.info("\nSe Dep.")
    best_skew(master_table["SeDep_2005_2009"])
    LOGGER.info("Log transformation was best.")
    master_table["SeDep_2005_2009"] = np.log10(master_table["SeDep_2005_2009"])
    master_table["SeDep_SSP126"] = np.log10(master_table["SeDep_SSP126"])
    master_table["SeDep_SSP585"] = np.log10(master_table["SeDep_SSP585"])

    return master_table


def main():

    setup_logging()

    LOGGER.debug("Execution date: %s", date.today())

    master_table = pd.read_excel(MASTER_TABLE_PATH)
    master_table.set_index(keys=["lon", "lat"], drop=True, inplace=True)

    master_table = transform_predictors_no_future_component(master_table)
    master_table = transform_predictors_with_future_component(master_table)

    master_table.to_excel(
        os.path.join(OUTPUT_DIR, "_1_normalized_master_table.xlsx"),
        index=True,
    )

    LOGGER.info("Done. Please check log file.")


if __name__ == "__main__":
    main()
