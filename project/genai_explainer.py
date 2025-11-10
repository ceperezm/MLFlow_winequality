import os
import json
from typing import Dict, Any

import requests


class WineExplainer:
    """Genera explicaciones del modelo.

    Soporta:
    - "gemini2.0 flash" (requiere GOOGLE_API_KEY)
    - "z-ai/glm-4.6" (requiere OPENROUTER_API_KEY)
    - Fallback local si la API no está disponible o falla
    """

    def __init__(self, model: str = "z-ai/glm-4.6") -> None:
        self.model = model.strip()

    def _fallback_explanation(self, features_dict: Dict[str, Any], prediction: float) -> str:
        alcohol = features_dict.get("alcohol")
        vol_acidity = features_dict.get("volatile acidity")
        sulphates = features_dict.get("sulphates")
        density = features_dict.get("density")

        parts = [
            f"Predicción {prediction:.2f}/10.",
            "Mayor alcohol suele asociarse a mejor calidad" if alcohol is not None else None,
            "Menor acidez volátil mejora el perfil aromático" if vol_acidity is not None else None,
            "Sulfatos adecuados contribuyen a la estructura" if sulphates is not None else None,
            "Densidad coherente con el equilibrio de azúcares y alcohol" if density is not None else None,
        ]
        return " ".join([p for p in parts if p])

    def _build_user_prompt(self, features_dict: Dict[str, Any], prediction: float) -> str:
        return (
            "Eres un experto enólogo y científico de datos. Analiza la siguiente predicción de calidad de vino (0-10)\n"
            f"Predicción: {prediction:.2f}\n"
            f"Características:\n{json.dumps(features_dict, indent=2)}\n\n"
            "Explica en español, brevemente (máximo 3 líneas), qué variables más influyen y por qué."
        )

    def _normalize_openrouter_model(self) -> str:
        # Permite pasar directamente el nombre de modelo de OpenRouter
        return self.model or "z-ai/glm-4.6"

    def _is_gemini(self) -> bool:
        normalized = self.model.replace(" ", "").lower()
        return normalized.startswith("gemini")

    def _is_openrouter(self) -> bool:
        return "/" in self.model or self.model.lower().startswith("deepseek")

    def _generate_explanation_openrouter(self, features_dict: Dict[str, Any], prediction: float) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY no está definido en el entorno. Configura la variable antes de ejecutar.")

        system_prompt = (
            "Eres un enólogo y científico de datos. Explica en 2-3 líneas en español, "
            "qué variables del vino más influyen en la predicción de calidad y por qué."
        )
        user_prompt = self._build_user_prompt(features_dict, prediction)

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "MLFlow Vinos",
        }
        payload = {
            "model": self._normalize_openrouter_model(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content = (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content")
            )
            text = content.strip() if content else ""
            if not text:
                raise RuntimeError("Respuesta vacía de OpenRouter.")
            return text
        except Exception as exc2:
            raise RuntimeError(f"Fallo llamando a OpenRouter: {exc2}") from exc2

    def _generate_explanation_gemini(self, features_dict: Dict[str, Any], prediction: float) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY no está definido en el entorno.")

        prompt = self._build_user_prompt(features_dict, prediction)

        try:
            url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content = (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content")
            )
            text = content.strip() if content else ""
            if not text:
                raise RuntimeError("Respuesta vacía de Gemini.")
            return text
        except Exception as exc:
            raise RuntimeError(f"Fallo llamando a Gemini: {exc}") from exc

    def explain(self, features_dict: Dict[str, Any], prediction: float) -> str:
        if self._is_gemini():
            return self._generate_explanation_gemini(features_dict, prediction)
        # Usar OpenRouter por defecto
        if self._is_openrouter():
            return self._generate_explanation_openrouter(features_dict, prediction)
        # Fallback si no coincide ningún proveedor conocido
        return self._fallback_explanation(features_dict, prediction)