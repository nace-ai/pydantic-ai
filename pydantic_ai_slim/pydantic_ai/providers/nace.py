from __future__ import annotations as _annotations

import os
from typing import Literal

import httpx2

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.decision import DecisionModelProfile, decision_model_profile
from pydantic_ai.providers.system_one import SystemOneProvider

NaceModelName = Literal['drex-max-finance', 'drex-max-v1.1', 'drex-v2.0'] | str
"""Drex model names, including open-ended names and aliases served by the API."""


class NaceProvider(SystemOneProvider):
    """Provider for Nace's Drex decision models over the System One API.

    Use with [`SystemOneModel`][pydantic_ai.models.system_one.SystemOneModel] for text decisions.
    This transport does not yet support Drex Max image input.
    """

    @property
    def name(self) -> str:
        return 'nace'

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile:
        return merge_profile(
            decision_model_profile(model_name),
            DecisionModelProfile(
                decision_max_choice_options=255,
                decision_max_score_levels=255,
                decision_requires_instructions=False,
            ),
        )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = 'https://console.nace.ai/v1',
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Create a Nace provider.

        Args:
            api_key: The Nace API key. Defaults to the `NACE_API_KEY` environment variable.
            base_url: The System One API URL, with or without a trailing `/v1`.
            http_client: An existing `httpx2.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('NACE_API_KEY')
        if not api_key:
            raise UserError('Set `NACE_API_KEY` or pass `NaceProvider(api_key=...)` to authenticate with Nace.')
        super().__init__(base_url=base_url, api_key=api_key, http_client=http_client)
