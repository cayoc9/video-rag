# Arquitetura do Pipeline Video RAG — Laboratório

> **Revisão:** Junho/2026 — Servidor `avancos` (192.168.1.9:11434) integrado  
> Arquitetura 100% local: elimina necessidade de RunPod para experimentos

---

## Mudança de Estratégia: Cloud → Híbrido Local

| Componente | Plano Anterior (servidor fora) | **Plano Atual (avancos online)** |
|-----------|-------------------------------|----------------------------------|
| Embedding visual | CLIP via RunPod (~$5–80) | **llava / qwen3-vl via avancos ($0)** |
| Descrição de frames | GroundingDINO cloud | **llava:latest local (vision model)** |
| LLM de consulta | GPT-4o-mini (pago) | **qwen2.5:7b / deepseek-r1 local ($0)** |
| ASR (transcrição) | Whisper local (CPU) | Whisper local ou **agente_whisper existente** |
| Vector DB | FAISS CPU local | **ChromaDB local** (mesmo host) |

**Resultado:** Custo de experimentação = **R$ 0,00** (energia da rede já paga)

---

## Nova Arquitetura — 100% Local + avancos

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FASE 1: EXTRAÇÃO DE FEATURES                      │
│                       Máquina Local (CPU)                            │
│                                                                      │
│  Vídeo → ffmpeg (1fps) → Frames PNG                                  │
│       → Whisper (agente_whisper) → Transcript .txt                   │
│       → EasyOCR → Textos detectados nos frames                       │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ Frames + Textos
                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│                 FASE 2: DESCRIÇÃO VISUAL (avancos)                   │
│                   192.168.1.9:11434  via Ollama API                  │
│                                                                      │
│  Frame PNG → [llava:latest]  → "pessoa usando capacete..."           │
│  Frame PNG → [qwen3-vl]      → descrição mais detalhada (opcional)   │
│  Frame PNG → [minicpm-v]     → alternativa mais leve                 │
│                                                                      │
│  Modelos vision disponíveis:                                         │
│    llava:latest ✅  llava:7b  llama3.2-vision                        │
│    minicpm-v  moondream  qwen3-vl  glm-ocr                           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ Descrições textuais dos frames
                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    FASE 3: INDEXAÇÃO (Local)                         │
│                    /mnt/large-memory (364GB)                         │
│                                                                      │
│  [Transcript + OCR + Descrição Visual]                               │
│          → Embedding via nomic-embed / all-MiniLM (CPU)              │
│          → ChromaDB local                                            │
│          → Estrutura: {video_id, start_time, end_time,               │
│                        text_chunk, visual_desc, embedding}           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ Query do usuário
                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    FASE 4: CONSULTA RAG (avancos)                    │
│                   192.168.1.9:11434  via Ollama API                  │
│                                                                      │
│  Pergunta → Embedding → FAISS → Top-K segmentos                      │
│           → Contexto → [qwen2.5:7b ou deepseek-r1]                   │
│           → Resposta + timestamps das fontes                         │
│                                                                      │
│  Modelos de texto disponíveis:                                       │
│    qwen2.5:7b ✅  qwen3  gemma3  llama3.x                            │
│    deepseek-r1:7b→70b (raciocínio complexo)                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Componentes Técnicos (Revisados)

### Fase 1 — Extração Local

```python
# Stack de extração (CPU local)
EXTRACTION_STACK = {
    "audio_transcription": "agente_whisper (já configurado)",
    "frame_sampling": "ffmpeg",          # 1 frame/segundo padrão
    "ocr": "EasyOCR",                    # Texto detectado nos frames
    "scene_detection": "PySceneDetect"  # Segmentação por cena
}

FRAME_SAMPLING = {
    "base_fps": 1,           # 1 frame/segundo padrão
    "high_motion_fps": 4,    # se PySceneDetect detectar corte de cena
    "output_format": "jpg",  # menor que PNG
    "resize": "336x336"      # tamanho padrão de entrada para vision models
}
```

### Fase 2 — Descrição Visual via avancos

```python
# Cliente Ollama via API OpenAI-compatible
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.9:11434/v1",
    api_key="ollama"  # qualquer valor
)

def describe_frame(frame_path: str, model: str = "llava:latest") -> str:
    """Gera descrição textual de um frame via avancos."""
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Descreva objetivamente o que está acontecendo nesta imagem em português. Foque em pessoas, objetos, ações e texto visível."}
            ]
        }],
        max_tokens=200
    )
    return response.choices[0].message.content

# Seleção de modelo por necessidade:
VISION_MODELS = {
    "rápido":    "moondream",           # menor, mais veloz
    "padrão":    "llava:latest",        # testado e funcionando ✅
    "detalhado": "qwen3-vl",            # melhor para OCR e detalhes
    "ocr_puro":  "glm-ocr"             # especializado em texto
}
```

### Fase 3 — Indexação e Vector DB

```python
# ChromaDB local (sem custo, persiste em disco)
import chromadb
from sentence_transformers import SentenceTransformer

chroma = chromadb.PersistentClient(path="/mnt/large-memory/videorag_index")
collection = chroma.get_or_create_collection("video_chunks")

# Embedding de texto (CPU local)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, ~80MB

# Estrutura de chunk
CHUNK = {
    "id": "video123_t045_t090",        # video_id + timestamps
    "documents": "transcrição + descrição visual concatenadas",
    "metadatas": {
        "video_id": "aula_01.mp4",
        "start_time": 45.0,
        "end_time": 90.0,
        "transcript": "o professor explica...",
        "visual_desc": "slide com gráfico de barras...",
        "ocr_text": "Título: Redes Neurais"
    }
}
```

### Fase 4 — Consulta RAG via avancos

```python
def query_videorag(pergunta: str, top_k: int = 5) -> dict:
    """Pipeline completo de consulta."""
    # 1. Embedding da pergunta
    q_embedding = embed_model.encode(pergunta)

    # 2. Busca vetorial no ChromaDB
    results = collection.query(
        query_embeddings=[q_embedding.tolist()],
        n_results=top_k
    )

    # 3. Montar contexto com timestamps
    contexto = "\n---\n".join([
        f"[{m['video_id']} @ {m['start_time']:.0f}s–{m['end_time']:.0f}s]\n"
        f"Fala: {m['transcript']}\nVisual: {m['visual_desc']}"
        for m in results["metadatas"][0]
    ])

    # 4. Resposta via avancos (LLM de texto)
    resp = client.chat.completions.create(
        model="qwen2.5:7b",  # ou deepseek-r1 para raciocínio complexo
        messages=[
            {"role": "system", "content": "Você é um assistente que responde sobre vídeos com base nos trechos fornecidos. Sempre cite o timestamp da fonte."},
            {"role": "user", "content": f"Contexto dos vídeos:\n{contexto}\n\nPergunta: {pergunta}"}
        ]
    )

    return {
        "resposta": resp.choices[0].message.content,
        "fontes": results["metadatas"][0]
    }
```

---

## Mapa de Modelos do avancos para Video RAG

| Tarefa | Modelo Recomendado | Alternativa | Notas |
|--------|-------------------|-------------|-------|
| Descrição de frame geral | `llava:latest` ✅ | `minicpm-v` | Testado e funcionando |
| OCR / texto em slide | `glm-ocr` | `qwen3-vl` | Especializado em texto |
| Análise detalhada de cena | `qwen3-vl` | `llama3.2-vision` | Melhor compreensão multimodal |
| Frames rápidos (pré-filtro) | `moondream` | — | Mais leve e veloz |
| Resposta RAG (texto) | `qwen2.5:7b` | `gemma3` | Bom custo-benefício local |
| Raciocínio complexo | `deepseek-r1:7b` | `deepseek-r1:70b` | Para queries difíceis |
| Código / scripts | `qwen2.5-coder` | `codellama` | Geração de análise automática |

---

## Estimativa de Throughput (avancos local)

| Operação | Velocidade Estimada | Volume (ex: 10h vídeo) |
|----------|--------------------|-----------------------|
| Extração de frames (ffmpeg) | ~realtime | ~36.000 frames @ 1fps |
| Transcrição Whisper (CPU) | ~2–3x tempo real | ~20–30h de CPU |
| Descrição visual (llava:latest) | ~2–5s/frame | ~20–50h (gargalo) |
| Descrição visual (moondream) | ~0.5–1s/frame | ~5–10h |
| Indexação ChromaDB | muito rápido | < 10min |
| Query RAG | 3–8s/query | $0 |

> **Gargalo principal:** Descrição visual frame-a-frame. Estratégia: descrever apenas **1 frame por cena** (usar PySceneDetect para detectar cortes) → reduz de 36k para ~300–500 frames por hora de vídeo.

---

## Plano de Experimento Imediato (Proof of Concept)

```bash
# Sequência sugerida para primeiro teste

# 1. Preparar vídeo piloto (30min de aula)
VIDEO="aula_piloto.mp4"

# 2. Extrair frames @ 1fps
ffmpeg -i $VIDEO -vf fps=1 frames/frame_%04d.jpg

# 3. Transcrever áudio
# (usar agente_whisper já configurado)

# 4. Descrever frames via avancos
python extract_visual.py --model llava:latest --input frames/ --output descriptions.jsonl

# 5. Indexar no ChromaDB
python index.py --transcript transcript.txt --visual descriptions.jsonl --db /mnt/large-memory/videorag_index

# 6. Testar query
python query.py --question "Em que minuto o professor apresentou o gráfico de comparação?"
```

---

## Comparativo de Custos: Antes vs. Depois

| Fase | Antes (RunPod) | **Depois (avancos)** |
|------|---------------|----------------------|
| Descrição visual | ~$5–80 | **$0** |
| LLM de consulta | ~$0.15/1000 queries | **$0** |
| Embedding | ~$0.10/1000 | **$0** |
| **Total (100h vídeo)** | ~$15–80 | **$0** |

---

*Arquitetura revisada — Junho/2026 (avancos server integrado)*
