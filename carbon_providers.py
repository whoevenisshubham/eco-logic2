from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import requests


DEFAULT_MAHARASHTRA_ZONE = "IN-WE"
DEFAULT_OFFLINE_INTENSITY = 708.0


@dataclass
class CarbonReading:
    intensity_g_per_kwh: float
    source: str
    mode: str
    zone: str


class CarbonProviderError(RuntimeError):
    pass


class CarbonProvider:
    name = "base"

    def get_intensity(self, **kwargs) -> CarbonReading:
        raise NotImplementedError


class ElectricityMapsAdapter(CarbonProvider):
    name = "electricity_maps"

    def get_intensity(self, api_key: str, zone: str = DEFAULT_MAHARASHTRA_ZONE, timeout_s: int = 8, **kwargs) -> CarbonReading:
        if not api_key:
            raise CarbonProviderError("Electricity Maps API key missing")
        url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
        headers = {"auth-token": api_key}
        response = requests.get(url, headers=headers, params={"zone": zone}, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
        for key in ["carbonIntensity", "carbonIntensityAvg", "carbonIntensityForecast"]:
            value = payload.get(key)
            if value is not None:
                return CarbonReading(
                    intensity_g_per_kwh=float(value),
                    source=f"Electricity Maps ({zone})",
                    mode="live-api",
                    zone=zone,
                )
        raise CarbonProviderError("Electricity Maps response missing carbon intensity")


class WattTimeAdapter(CarbonProvider):
    name = "watttime"

    def get_intensity(self, token: str, ba: str, timeout_s: int = 8, **kwargs) -> CarbonReading:
        if not token:
            raise CarbonProviderError("WattTime token missing")
        if not ba:
            raise CarbonProviderError("WattTime BA code missing")

        url = "https://api2.watttime.org/v2/index"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers, params={"ba": ba}, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()

        possible_keys = ["moer", "value", "carbon_intensity"]
        for key in possible_keys:
            value = payload.get(key)
            if value is not None:
                return CarbonReading(
                    intensity_g_per_kwh=float(value),
                    source=f"WattTime ({ba})",
                    mode="live-api",
                    zone=ba,
                )
        raise CarbonProviderError("WattTime response missing intensity value")


class OfflineFallbackAdapter(CarbonProvider):
    name = "offline"

    def get_intensity(self, intensity_g_per_kwh: float = DEFAULT_OFFLINE_INTENSITY, zone: str = DEFAULT_MAHARASHTRA_ZONE, **kwargs) -> CarbonReading:
        return CarbonReading(
            intensity_g_per_kwh=float(intensity_g_per_kwh),
            source="Offline fallback constant",
            mode="fallback-constant",
            zone=zone,
        )


def resolve_carbon_reading(
    provider_name: str,
    electricity_maps_key: Optional[str],
    electricity_maps_zone: str,
    watttime_token: Optional[str],
    watttime_ba: str,
    offline_intensity: float,
) -> CarbonReading:
    provider_name = (provider_name or "electricity_maps").strip().lower()

    if provider_name == "electricity_maps":
        return ElectricityMapsAdapter().get_intensity(api_key=electricity_maps_key or "", zone=electricity_maps_zone)
    if provider_name == "watttime":
        return WattTimeAdapter().get_intensity(token=watttime_token or "", ba=watttime_ba)
    if provider_name == "offline":
        return OfflineFallbackAdapter().get_intensity(intensity_g_per_kwh=offline_intensity, zone=electricity_maps_zone)

    raise CarbonProviderError(f"Unsupported provider: {provider_name}")
