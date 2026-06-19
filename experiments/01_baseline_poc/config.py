"""
config.py — Configurações do pipeline Video RAG

Alinhado com o padrão do projeto openran/agents_openran.
Usa pydantic-settings para carregar de env vars com fallback para defaults.

Servidor avancos: 192.168.1.9:11434 (29 modelos disponíveis)
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configurações do pipeline Video RAG.

    Carregadas via env vars com prefixo VIDEORAG_.
    Exemplo: export VIDEORAG_VISION_MODEL=qwen3-vl
    """

    # --- Servidor avancos (Ollama remoto) ---
    # Catálogo completo de modelos disponíveis:
    #
    # 🎯 Vision/Multimodal:
    #   llava:latest ✅, llava:7b, llama3.2-vision:latest,
    #   minicpm-v:latest, moondream:latest, qwen3-vl:latest, glm-ocr:latest
    #
    # 📝 Texto:
    #   qwen2.5:7b-instruct, qwen2.5:14b, qwen2.5:32b,
    #   qwen3:8b, qwen3:14b, llama3:latest, llama3.1:latest, llama3.1:8b,
    #   llama3.3:latest, gemma3:12b
    #
    # 🧠 Raciocínio:
    #   deepseek-r1:7b, deepseek-r1:14b, deepseek-r1:32b, deepseek-r1:70b
    #
    # 💻 Código:
    #   codellama:7b, codellama:13b, codegemma:7b, qwen2.5-coder:7b
    #
    # 🔢 Embeddings:
    #   nomic-embed-text:latest, bge-m3:latest
    #
    avancos_base_url: str = "http://192.168.1.9:11434"
    fallback_ollama_url: str = "http://localhost:11434"

    # Modelos por tarefa
    vision_model: str = "llava:latest"       # descrição de frames (✅ testado)
    vision_model_fast: str = "moondream"     # pré-filtro rápido
    vision_model_ocr: str = "glm-ocr"       # slides/texto nos frames
    vision_model_detail: str = "qwen3-vl"   # análise detalhada

    rag_model: str = "qwen2.5:7b-instruct"  # resposta RAG padrão
    reasoning_model: str = "deepseek-r1:7b" # raciocínio complexo

    embed_model_local: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim CPU
    embed_model_remote: str = "nomic-embed-text"  # via avancos Ollama

    # --- Extração de frames ---
    frame_fps: float = 1.0          # frames por segundo (padrão)
    frame_max_fps: float = 4.0      # máximo em cenas com muito movimento
    frame_resize: str = "336x336"   # tamanho de entrada para vision models
    frame_format: str = "jpg"       # formato de saída (menor que PNG)

    # --- Descrição visual ---
    vision_batch_size: int = 5      # frames por batch (evitar sobrecarga)
    vision_max_tokens: int = 200    # tokens máximos por descrição
    vision_prompt: str = (
        "Descreva objetivamente o que está acontecendo nesta imagem em português. "
        "Foque em: pessoas e suas ações, objetos relevantes, texto visível, "
        "e contexto do ambiente. Seja conciso (2-3 frases)."
    )

    # --- ChromaDB (Vector Store) ---
    chroma_persist_dir: str = "/mnt/storage/videorag_index"
    collection_name: str = "video_chunks"

    # --- Chunking ---
    chunk_size: int = 512           # caracteres por chunk de texto
    chunk_overlap: int = 64         # sobreposição entre chunks

    # --- Retrieval ---
    retrieval_top_k: int = 5        # chunks recuperados por query
    context_window: int = 2         # chunks vizinhos ao recuperado

    # --- Geração ---
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # --- Avaliação ---
    eval_model: str = "deepseek-r1:70b"  # juiz de qualidade (sem custo)

    model_config = {"env_prefix": "VIDEORAG_"}


@lru_cache()
def get_settings() -> Settings:
    """Retorna instância única (cached) de Settings."""
    return Settings()
