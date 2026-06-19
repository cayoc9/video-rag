# Roadmap de Implementação — Video RAG Laboratório

> **Revisado:** Junho/2026 — Servidor `avancos` (192.168.1.9:11434) integrado  
> **Mudança principal:** RunPod eliminado. Todo o pipeline roda localmente via avancos.

---

## Infraestrutura Disponível (Revisada)

### Servidor avancos — 192.168.1.9:11434

**Modelos Vision/Multimodal (usados para descrever frames):**
- `llava:latest` ✅ — testado e funcionando, modelo principal
- `llava:7b`, `llama3.2-vision`, `minicpm-v` — alternativas
- `qwen3-vl` — melhor para análise detalhada de cenas
- `glm-ocr` — especializado em OCR de slides/texto
- `moondream` — mais leve e rápido (pré-filtro)

**Modelos Texto (usados para resposta RAG):**
- `qwen2.5:7b-instruct` — principal para consulta
- `deepseek-r1:7b → 70b` — raciocínio complexo
- `qwen3:8b / 14b`, `gemma3:12b`, `llama3.x`

**Embeddings disponíveis no avancos:**
- `nomic-embed-text:latest` — embedding de texto via Ollama
- `bge-m3:latest` — multilíngue, alta qualidade

### Hardware Local
- **CPU:** Intel Xeon (ASR Whisper, ChromaDB queries)
- **Storage:** `/mnt/large-memory` (364GB) — índice vetorial e frames
- **RAM:** 35GB (12GB livres) — suficiente para ChromaDB + pipeline

---

## Fase 0 — Preparação do Ambiente (Semana 1)

- [ ] Criar ambiente Python no repo `video-rag`
  ```bash
  cd /home/cayo/repos/video-rag
  python -m venv .venv && source .venv/bin/activate
  pip install chromadb sentence-transformers openai ffmpeg-python easyocr pyscenedetect
  ```
- [ ] Testar conectividade com avancos
  ```bash
  curl http://192.168.1.9:11434/api/tags | python -m json.tool | grep name
  ```
- [ ] Verificar `agente_whisper` existente em `/home/cayo/repos/agentes/agente_whisper`
- [ ] Selecionar vídeo piloto (30min de aula como primeiro teste)
- [ ] Criar diretório de dados: `/mnt/large-memory/videorag_pilot/`

---

## Fase 1 — Proof of Concept (Semana 2–3)

**Objetivo:** Pipeline ponta-a-ponta funcionando com vídeo de 30min

### Etapa 1.1 — Extração de Features

```bash
# Extrair frames @ 1fps
ffmpeg -i aula_piloto.mp4 -vf fps=1 /mnt/large-memory/videorag_pilot/frames/frame_%04d.jpg

# Transcrever áudio (agente_whisper)
cd /home/cayo/repos/agentes/agente_whisper
python main.py --input aula_piloto.mp4 --output /mnt/large-memory/videorag_pilot/transcript.txt
```

### Etapa 1.2 — Descrição Visual via avancos

- [ ] Implementar `experiments/01_baseline_poc/extract_visual.py`
  - Processar 1 frame/cena (PySceneDetect para detectar cortes)
  - Enviar via API Ollama (http://192.168.1.9:11434/v1)
  - Modelo: `llava:latest` → `moondream` (se lento)
  - Saída: `descriptions.jsonl`

### Etapa 1.3 — Indexação ChromaDB

- [ ] Implementar `experiments/01_baseline_poc/index.py`
  - Concatenar: transcript + descrição visual + OCR
  - Embedding: `nomic-embed-text` via avancos (ou `all-MiniLM-L6-v2` local)
  - Persistência: `/mnt/large-memory/videorag_pilot/chroma_db/`

### Etapa 1.4 — Query RAG

- [ ] Implementar `experiments/01_baseline_poc/query.py`
  - Recuperação: ChromaDB top-5
  - Geração: `qwen2.5:7b` via avancos
  - Saída: resposta + timestamps das fontes

**Métricas mínimas para considerar PoC bem-sucedido:**
- Responde corretamente 3 de 5 perguntas conhecidas sobre o vídeo piloto
- Latência de query < 30s no total (busca + geração)

---

## Fase 2 — Comparativo de Modelos (Semana 4–5)

**Objetivo:** Benchmarkar modelos vision do avancos para descrição de frames

| Modelo | Métrica | Teste |
|--------|---------|-------|
| `llava:latest` | qualidade de descrição, latência | padrão |
| `qwen3-vl` | qualidade em slides/OCR | slides de aula |
| `moondream` | latência, recall em queries | escala (1000+ frames) |
| `glm-ocr` | precisão de OCR | frames com texto |

- [ ] Criar dataset de avaliação: 20 frames anotados manualmente
- [ ] Script de benchmark: `experiments/02_model_benchmark/`
- [ ] Registrar: latência/frame, qualidade de descrição (BLEU vs. anotação)

---

## Fase 3 — Experimentos das Lacunas (Semana 6–14)

### Experimento A: Citation Faithfulness (alta prioridade para publicação)

- [ ] Criar 50 QA pairs com timestamp ground-truth
- [ ] Implementar `CiteF1-Video`:
  1. Gerar resposta com Video RAG (inclui timestamps)
  2. Decompor em claims atômicos
  3. Para cada claim: verificar se frame citado suporta a afirmação via `llava:latest`
  4. Calcular F1 entre claims suportados e citados
- [ ] Comparar: sem verificação vs. com verificação de citação

### Experimento B: Benchmarking de Amostragem Adaptativa

- [ ] PySceneDetect para segmentar por cena vs. 1fps fixo
- [ ] Comparar: número de frames processados × qualidade de recuperação
- [ ] Hipótese: 1 frame/cena captura 95% da informação com 10–20% do custo

### Experimento C: Multi-modelo em cascata

- [ ] Usar `moondream` (rápido) para pré-filtrar frames irrelevantes
- [ ] `qwen3-vl` apenas nos frames selecionados pelo moondream
- [ ] Avaliar: qualidade total × custo computacional

---

## Fase 4 — Publicação (Semana 15–24)

**Foco primário:** Citation Faithfulness para Video RAG (Experimento A)

**Contribuições originais planejadas:**
1. CiteF1-Video — primeira métrica de verificação de citação por timestamp
2. Pipeline 100% local via Ollama (sem dependência de APIs pagas)
3. Benchmarking de modelos vision open-source para descrição de frames

**Conferências alvo:**

| Conferência | Deadline | Foco |
|-------------|----------|------|
| arXiv | A qualquer momento | Pré-print rápido para prioridade |
| EMNLP 2026 | ~Maio 2026 | NLP + Multimodal |
| ACL 2027 | ~Fevereiro 2027 | Principal NLP |
| CVPR Workshop 2027 | ~Novembro 2026 | Visão computacional |

---

## Stack Tecnológico Final

```python
STACK = {
    "extração": {
        "asr": "agente_whisper (já configurado)",
        "frames": "ffmpeg",
        "ocr": "EasyOCR (CPU local)",
        "scene_detection": "PySceneDetect"
    },
    "descrição_visual": {
        "servidor": "http://192.168.1.9:11434",
        "modelo_padrão": "llava:latest",
        "modelo_ocr": "glm-ocr",
        "modelo_detalhe": "qwen3-vl",
        "modelo_rápido": "moondream"
    },
    "indexação": {
        "vector_db": "ChromaDB (local, /mnt/large-memory/)",
        "embedding": "nomic-embed-text (avancos) ou all-MiniLM-L6 (CPU)"
    },
    "consulta": {
        "servidor": "http://192.168.1.9:11434",
        "modelo_padrão": "qwen2.5:7b",
        "modelo_raciocínio": "deepseek-r1:7b",
        "orquestração": "LangChain ou script Python puro"
    },
    "avaliação": {
        "juiz": "deepseek-r1:70b (via avancos, sem custo)",
        "métricas": "CiteF1-Video, mIoU, Acc@QA, Faithfulness"
    }
}
```

---

## Comparativo: Plano Original vs. Atual

| Aspecto | Plano Original | **Plano Atual (avancos)** |
|---------|---------------|--------------------------|
| GPU para indexação | RunPod ($5–80/experimento) | **$0 (avancos local)** |
| LLM de consulta | GPT-4o-mini (pago) | **qwen2.5:7b grátis** |
| Modelo vision | CLIP + GroundingDINO | **llava/qwen3-vl grátis** |
| Embedding | OpenAI ada-002 | **nomic-embed-text grátis** |
| Avaliação LLM | GPT-4.1 (pago) | **deepseek-r1:70b grátis** |
| **Custo total por experimento** | **$15–80** | **$0** |
| Dependência externa | Alta (RunPod + OpenAI) | **Zero** |
| Privacidade dos dados | Baixa (cloud) | **Alta (100% local)** |

---

*Roadmap revisado — Junho/2026 (servidor avancos reintegrado)*
