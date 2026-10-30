import logging
import time
from typing import Optional

import requests


logger = logging.getLogger(__name__)


class ClinicalTrialsAPI:
    """Client for interacting with the ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, params: dict) -> requests.Response:
        """Make a GET request with retry and exponential backoff."""

        retryable_status_codes = {429, 500, 502, 503, 504}

        for attempt in range(self.max_retries + 1):

            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code in retryable_status_codes:

                    if attempt < self.max_retries:
                        wait_time = 2 ** attempt

                        logger.warning(
                            "Request returned %s. "
                            "Retrying in %s seconds.",
                            response.status_code,
                            wait_time,
                        )

                        time.sleep(wait_time)
                        continue

                response.raise_for_status()

                return response

            except requests.RequestException as error:

                if attempt == self.max_retries:
                    logger.error(
                        "Request failed after %s retries.",
                        self.max_retries,
                    )
                    raise

                wait_time = 2 ** attempt

                logger.warning(
                    "Request error: %s. "
                    "Retrying in %s seconds.",
                    error,
                    wait_time,
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Request failed after all retry attempts."
        )

    def fetch_studies(
        self,
        condition: str,
        page_size: int = 100,
        max_studies: Optional[int] = None,
    ) -> list:
        """
        Fetch clinical trials matching a condition.

        Parameters
        ----------
        condition : str
            Condition to search for.

        page_size : int
            Number of studies requested per API page.

        max_studies : int, optional
            Maximum number of studies to return.

        Returns
        -------
        list
            Raw ClinicalTrials.gov study records.
        """

        studies = []
        page_token = None
        page_number = 1

        logger.info(
            "Starting study fetch. Condition='%s', max_studies=%s",
            condition,
            max_studies,
        )

        while True:

            params = {
                "query.cond": condition,
                "pageSize": page_size,
            }

            if page_token:
                params["pageToken"] = page_token

            logger.info(
                "Fetching page %s for condition='%s'",
                page_number,
                condition,
            )

            response = self._get(params)

            data = response.json()

            current_studies = data.get("studies", [])

            studies.extend(current_studies)

            logger.info(
                "Fetched %s studies. Total collected: %s",
                len(current_studies),
                len(studies),
            )

            if (
                max_studies is not None
                and len(studies) >= max_studies
            ):
                studies = studies[:max_studies]

                logger.info(
                    "Reached max_studies=%s. Stopping.",
                    max_studies,
                )

                break

            page_token = data.get("nextPageToken")

            if not page_token:

                logger.info(
                    "No next page token. Fetch complete."
                )

                break

            page_number += 1

        logger.info(
            "Finished fetching. Returning %s studies.",
            len(studies),
        )

        return studies