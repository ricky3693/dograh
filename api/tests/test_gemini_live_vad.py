from types import SimpleNamespace

import pytest
from google.auth.credentials import AnonymousCredentials

from api.services.configuration.registry import (
    GoogleRealtimeLLMConfiguration,
    GoogleVertexRealtimeLLMConfiguration,
)
from api.services.pipecat.realtime.gemini_live import DograhGeminiLiveLLMService
from api.services.pipecat.realtime.gemini_live_vertex import (
    DograhGeminiLiveVertexLLMService,
)
from api.services.pipecat.service_factory import create_realtime_llm_service

VAD_FIELDS = {
    "start_of_speech_sensitivity",
    "end_of_speech_sensitivity",
    "prefix_padding_ms",
    "silence_duration_ms",
}


class DummyUserConfig:
    def __init__(self, realtime_config):
        self.realtime = realtime_config


def _audio_config():
    return SimpleNamespace(
        transport_out_sample_rate=24000,
        transport_in_sample_rate=16000,
    )


def _patch_vertex_credentials(monkeypatch):
    monkeypatch.setattr(
        DograhGeminiLiveVertexLLMService,
        "_get_credentials",
        staticmethod(lambda _credentials, _credentials_path: AnonymousCredentials()),
    )


def _vad_dump(service):
    assert service._settings.vad is not None
    return service._settings.vad.model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize(
    "config_cls",
    [GoogleRealtimeLLMConfiguration, GoogleVertexRealtimeLLMConfiguration],
)
def test_gemini_realtime_configs_expose_optional_vad_fields(config_cls):
    schema = config_cls.model_json_schema()
    assert VAD_FIELDS <= schema["properties"].keys()
    assert VAD_FIELDS.isdisjoint(schema.get("required", []))


def test_google_realtime_without_vad_preserves_provider_default():
    realtime_config = GoogleRealtimeLLMConfiguration(api_key="test-api-key")
    service = create_realtime_llm_service(
        DummyUserConfig(realtime_config), _audio_config()
    )

    assert isinstance(service, DograhGeminiLiveLLMService)
    assert service._settings.vad is None


def test_google_vertex_realtime_without_vad_preserves_provider_default(monkeypatch):
    _patch_vertex_credentials(monkeypatch)
    realtime_config = GoogleVertexRealtimeLLMConfiguration(
        project_id="test-proj",
        location="us-central1",
    )
    service = create_realtime_llm_service(
        DummyUserConfig(realtime_config), _audio_config()
    )

    assert isinstance(service, DograhGeminiLiveVertexLLMService)
    assert service._settings.vad is None


def test_google_realtime_passes_vad_settings_to_pipecat():
    realtime_config = GoogleRealtimeLLMConfiguration(
        api_key="test-api-key",
        start_of_speech_sensitivity="START_SENSITIVITY_HIGH",
        end_of_speech_sensitivity="END_SENSITIVITY_LOW",
        prefix_padding_ms=120,
        silence_duration_ms=350,
    )
    service = create_realtime_llm_service(
        DummyUserConfig(realtime_config), _audio_config()
    )

    assert _vad_dump(service) == {
        "start_sensitivity": "START_SENSITIVITY_HIGH",
        "end_sensitivity": "END_SENSITIVITY_LOW",
        "prefix_padding_ms": 120,
        "silence_duration_ms": 350,
    }


def test_google_vertex_realtime_passes_vad_settings_to_pipecat(monkeypatch):
    _patch_vertex_credentials(monkeypatch)
    realtime_config = GoogleVertexRealtimeLLMConfiguration(
        project_id="test-proj",
        location="us-central1",
        start_of_speech_sensitivity="START_SENSITIVITY_LOW",
        end_of_speech_sensitivity="END_SENSITIVITY_HIGH",
        prefix_padding_ms=160,
        silence_duration_ms=500,
    )
    service = create_realtime_llm_service(
        DummyUserConfig(realtime_config), _audio_config()
    )

    assert _vad_dump(service) == {
        "start_sensitivity": "START_SENSITIVITY_LOW",
        "end_sensitivity": "END_SENSITIVITY_HIGH",
        "prefix_padding_ms": 160,
        "silence_duration_ms": 500,
    }


def test_google_realtime_allows_partial_vad_configuration():
    realtime_config = GoogleRealtimeLLMConfiguration(
        api_key="test-api-key",
        silence_duration_ms=250,
    )
    service = create_realtime_llm_service(
        DummyUserConfig(realtime_config), _audio_config()
    )

    assert _vad_dump(service) == {"silence_duration_ms": 250}
