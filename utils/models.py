from enum import StrEnum


class Model(StrEnum):
    # Ai2 (non-profit), US. Fully open. Apache 2.0.
    OLMO2_13B = "ollama/olmo2:13b"
    # Google DeepMind, US/UK. Open weights. Gemma terms.
    GEMMA3_12B = "ollama/gemma3:12b"
    # Microsoft, US. Open weights. MIT.
    PHI4_14B = "ollama/phi4:14b"
    # Apertus v1.1 4B Instruct, bf16 (unquantised). Swiss AI Initiative (EPFL, ETH Zurich, CSCS).
    # Fully open. Apache 2.0. Ollama packaging by Michel Rosselli, template checked against official HF metadata.
    APERTUS_V11_4B = "ollama/MichelRosselli/apertus-v1.1:4b-instruct-bf16"

    # Disabled:
    # OpenAI, US. Open weights. Apache 2.0.
    # GPT_OSS_20B = "ollama/gpt-oss:20b"

    @property
    def ollama_tag(self) -> str:
        """Tag as Ollama knows it, without Inspect's provider prefix."""
        return self.removeprefix("ollama/")