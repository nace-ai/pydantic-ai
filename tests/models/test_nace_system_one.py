"""Exercise provider registration and wire configuration without making paid provider calls.

Mock transport pins the request URL, credentials and provider identity, which cassette URI matching alone would miss.
System One's existing contract tests cover its shared answer mapping and error behavior.
"""

from __future__ import annotations as _annotations

import json
from typing import Literal

import httpx2
import pytest
from pydantic import BaseModel, Field

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model, known_model_names
from pydantic_ai.models.decision import DecisionRequest, NoulQuestion
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers import infer_provider, infer_provider_class
from pydantic_ai.providers.nace import NaceProvider

from ..conftest import TestEnv


def test_nace_registration_and_environment(env: TestEnv):
    env.set('NACE_API_KEY', 'nace-key')
    env.set('SYSTEM_ONE_API_KEY', 'another-provider-key')
    env.set('SYSTEM_ONE_BASE_URL', 'https://other.example/v1')
    assert infer_provider_class('nace') is NaceProvider
    provider = infer_provider('nace')
    assert isinstance(provider, NaceProvider)
    assert provider.api_key == 'nace-key'
    assert provider.base_url == 'https://console.nace.ai/v1'
    model = infer_model('nace:drex-max-v1.1')
    assert isinstance(model, SystemOneModel)
    assert model.system == 'nace'
    assert model.model_name == 'drex-max-v1.1'
    assert model.profile.get('decision_max_choice_options') == 255
    assert model.profile.get('decision_max_score_levels') == 255
    assert model.profile.get('decision_requires_instructions') is False
    assert 'nace:drex-max-v1.1' in known_model_names()


def test_nace_requires_its_own_key(env: TestEnv):
    env.remove('NACE_API_KEY')
    env.set('SYSTEM_ONE_API_KEY', 'another-provider-key')
    with pytest.raises(UserError, match='NACE_API_KEY'):
        NaceProvider()


class Triage(BaseModel):
    urgent: bool = Field(description='Does this need an immediate reply?')
    team: Literal['billing', 'support'] = Field(description='Which team owns the ticket?')


@pytest.mark.parametrize('base_url', ['https://console.nace.ai/v1', 'https://proxy.example/'])
@pytest.mark.parametrize('model_name', ['drex-max-v1.1', 'drex-v2.0', 'future-drex-version'])
async def test_nace_agent_uses_selected_endpoint_and_key(
    env: TestEnv, allow_model_requests: None, base_url: str, model_name: str
):
    env.set('NACE_API_KEY', 'ambient-key')
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                'model': model_name,
                'answers': {
                    'urgent': {'type': 'noul', 'noul': 0.9, 'confidence': 0.8},
                    'team': {
                        'type': 'choice',
                        'choice': 'billing',
                        'confidence': 0.8,
                        'probabilities': {'billing': 0.9, 'support': 0.1},
                    },
                },
                'usage': {'input_tokens': 42, 'output_tokens': 0},
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        provider = NaceProvider(api_key='explicit-key', base_url=base_url, http_client=client)
        assert provider.client is client
        model = infer_model(f'nace:{model_name}', provider_factory=lambda _: provider)
        result = await Agent(model, output_type=Triage).run('I was charged twice.')

    assert result.output == Triage(urgent=True, team='billing')
    assert result.response.provider_name == 'nace'
    assert result.response.model_name == model_name
    assert result.usage.input_tokens == 42
    assert len(requests) == 1
    expected_url = (
        'https://console.nace.ai/v1/systemone'
        if 'console.nace.ai' in base_url
        else 'https://proxy.example/v1/systemone'
    )
    assert str(requests[0].url) == expected_url
    assert requests[0].headers['authorization'] == 'Bearer explicit-key'
    body = json.loads(requests[0].content)
    assert body['model'] == model_name
    assert set(body['questions']) == {'urgent', 'team'}


async def test_nace_prefix_does_not_enable_images():
    async with httpx2.AsyncClient() as client:
        model = SystemOneModel('drex-max-v1.1', provider=NaceProvider(api_key='key', http_client=client))
        request = DecisionRequest(
            state='Image 1',
            questions={'visible': NoulQuestion(instructions='Is the item visible?')},
            images=(BinaryContent(data=b'image', media_type='image/png'),),
        )
        with pytest.raises(UserError, match='System One does not support image input'):
            await model.decide(request, {})
