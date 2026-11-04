from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


class MetadataFilter:
    """
    Apply structured filters to local ClinicalTrials.gov metadata.

    The filter does not rank trials.
    It only determines which trials satisfy the explicit
    structured constraints extracted from the user query.
    """

    REQUIRED_COLUMNS = {
        "nct_id",
        "phase",
        "status",
        "sex",
        "study_type",
        "min_age",
        "max_age",
    }

    def __init__(self, dataframe: pd.DataFrame):
        missing = self.REQUIRED_COLUMNS - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing required metadata columns: {sorted(missing)}"
            )

        # Keep a clean positional index because retrieval modules
        # use dataframe positions to identify documents.
        self.dataframe = dataframe.reset_index(drop=True)

    def apply(
        self,
        filters: Mapping[str, Any],
    ) -> list[int]:
        """
        Apply all supplied structured filters.

        Returns
        -------
        list[int]
            Positional indices of trials satisfying ALL filters.
        """

        mask = pd.Series(
            True,
            index=self.dataframe.index,
            dtype=bool,
        )

        # =========================================================
        # STATUS
        # =========================================================
        if "status" in filters:
            allowed = self._as_list(filters["status"])

            mask &= self.dataframe["status"].isin(allowed)

        # =========================================================
        # PHASE
        # =========================================================
        if "phase" in filters:
            allowed = self._as_list(filters["phase"])

            phase_values = (
                self.dataframe["phase"]
                .fillna("")
                .astype(str)
            )

            phase_mask = pd.Series(
                False,
                index=self.dataframe.index,
                dtype=bool,
            )

            for phase in allowed:
                # ClinicalTrials.gov can contain values such as:
                #
                # PHASE1
                # PHASE2
                # PHASE1; PHASE2
                #
                # Therefore we check whether the requested phase
                # is one of the phases listed for the trial.
                phase_pattern = rf"(?:^|;\s*){phase}(?:;|$)"

                phase_mask |= phase_values.str.contains(
                    phase_pattern,
                    regex=True,
                    na=False,
                )

            mask &= phase_mask

        # =========================================================
        # SEX
        # =========================================================
        if "sex" in filters:
            allowed = self._as_list(filters["sex"])

            mask &= self.dataframe["sex"].isin(allowed)

        # =========================================================
        # STUDY TYPE
        # =========================================================
        if "study_type" in filters:
            allowed = self._as_list(filters["study_type"])

            mask &= self.dataframe["study_type"].isin(allowed)

        # =========================================================
        # MINIMUM AGE
        # =========================================================
        if "min_age" in filters:
            requested_min_age = float(filters["min_age"])

            max_age_years = self._age_to_years(
                self.dataframe["max_age"]
            )

            # Keep trials where:
            #
            # max_age >= requested age
            #
            # OR max_age is unknown.
            #
            # This avoids incorrectly excluding trials whose
            # maximum age was not reported.
            mask &= (
                max_age_years.isna()
                | (max_age_years >= requested_min_age)
            )

        # =========================================================
        # MAXIMUM AGE (exclusive upper bound of the requested group,
        # e.g. 18 for children). Added for parser v1.2.
        # =========================================================
        if "max_age" in filters:
            requested_max_age = float(filters["max_age"])

            min_age_years = self._age_to_years(
                self.dataframe["min_age"]
            )

            # Keep trials that admit someone younger than the bound:
            # min_age < requested bound, OR min_age is unknown.
            mask &= (
                min_age_years.isna()
                | (min_age_years < requested_max_age)
            )

        return self.dataframe.index[mask].tolist()

    # =============================================================
    # HELPERS
    # =============================================================

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """
        Normalize scalar or sequence filter values to a list.

        Examples
        --------
        "RECRUITING"
            -> ["RECRUITING"]

        ["RECRUITING", "NOT_YET_RECRUITING"]
            -> unchanged
        """

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            return list(value)

        return [value]

    @staticmethod
    def _age_to_years(series: pd.Series) -> pd.Series:
        """
        Convert ClinicalTrials.gov age strings to numeric years.

        Examples
        --------
        "18 Years" -> 18.0
        "6 Months" -> 0.5
        "4 Weeks" -> approximately 0.077
        "10 Days" -> approximately 0.027

        Missing or unparseable values become NaN.
        """

        values = series.astype("string")

        # Extract numeric component.
        numbers = pd.to_numeric(
            values.str.extract(
                r"([\d.]+)",
                expand=False,
            ),
            errors="coerce",
        ).astype(float)

        # Extract unit.
        units = (
            values.str.extract(
                r"[\d.]+\s*(Years?|Months?|Weeks?|Days?|Hours?|Minutes?)",
                expand=False,
            )
            .str.lower()
        )

        result = numbers.copy()

        # Months -> years
        month_mask = units.str.startswith(
            "month",
            na=False,
        )

        result.loc[month_mask] = (
            result.loc[month_mask] / 12.0
        )

        # Weeks -> years
        week_mask = units.str.startswith(
            "week",
            na=False,
        )

        result.loc[week_mask] = (
            result.loc[week_mask] / 52.0
        )

        # Days -> years
        day_mask = units.str.startswith(
            "day",
            na=False,
        )

        result.loc[day_mask] = (
            result.loc[day_mask] / 365.0
        )

        # Hours / minutes -> years (previously left unconverted, so
        # "6 Hours" was read as 6 years).
        hour_mask = units.str.startswith(
            "hour",
            na=False,
        )

        result.loc[hour_mask] = (
            result.loc[hour_mask] / 8766.0
        )

        minute_mask = units.str.startswith(
            "minute",
            na=False,
        )

        result.loc[minute_mask] = (
            result.loc[minute_mask] / 525960.0
        )

        return result