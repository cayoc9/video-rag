# 🎬 Video RAG — Pesquisa & Laboratório

> **Retrieval-Augmented Generation aplicado a vídeos de longa duração**  
> Um repositório de pesquisa, experimentos e implementações para busca semântica e raciocínio sobre vídeos.

[![Status](https://img.shields.io/badge/status-pesquisa_ativa-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](.)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

---

## 📋 Visão Geral

Este repositório documenta a pesquisa aprofundada sobre **Video RAG** — sistemas capazes de responder perguntas em linguagem natural sobre horas de vídeo, recuperando os trechos visuais e auditivos relevantes para gerar respostas precisas.

O projeto é voltado ao **contexto universitário/laboratorial**, mapeando o estado da arte (SOTA), lacunas abertas de pesquisa e uma estratégia de implementação viável com o hardware disponível.

---

## 📁 Estrutura do Repositório

```
video-rag/
├── README.md                        # Este arquivo
├── research/
│   ├── SOTA_VIDEORAG_RESEARCH.md    # Estado da arte: modelos e frameworks
│   ├── VIDEORAG_REPORT.md           # Viabilidade técnica e custos
│   ├── RESEARCH_GAPS_DEEP.md        # Lacunas aprofundadas (deep research)
│   └── BENCHMARKS.md                # Benchmarks e métricas de avaliação
├── docs/
│   ├── ARCHITECTURE.md              # Arquitetura do pipeline proposto
│   └── ROADMAP.md                   # Plano de implementação faseada
└── experiments/
    └── README.md                    # Guia para reprodução dos experimentos
```

---

## 🔬 Principais Documentos de Pesquisa

| Documento | Descrição |
|-----------|-----------|
| [Estado da Arte](research/SOTA_VIDEORAG_RESEARCH.md) | Mapeamento de soluções SOTA 2025–2026 |
| [Relatório de Viabilidade](research/VIDEORAG_REPORT.md) | Hardware, custos e estratégia híbrida |
| [Lacunas de Pesquisa (v1)](research/RESEARCH_GAPS_DEEP.md) | Deep research: 6 lacunas críticas identificadas |
| [Pesquisa Complementar (v2)](research/RESEARCH_COMPLEMENT_DEEP.md) | **🆕** Aprofundamento de cada lacuna com literatura 2025-2026 |
| [Benchmarks](research/BENCHMARKS.md) | Métricas e datasets de avaliação |
| [Arquitetura](docs/ARCHITECTURE.md) | Pipeline proposto para o laboratório |
| [Roadmap](docs/ROADMAP.md) | Fases de implementação |

---

## 🗺️ Estado da Arte Resumido

### Modelos Acadêmicos (2025–2026)

| Modelo | Conferência | Inovação |
|--------|------------|---------|
| **Video-RAG** | NeurIPS 2025 | Textos auxiliares visualmente alinhados (OCR+ASR) |
| **STAR-RAG** | ICLR 2026 | Temporal Graph Summarization |
| **E²RAG** | Jun 2025 | Dual-Graph (Entidade + Evento) |
| **TMRL** | Jan 2026 | Matryoshka Representation Learning |
| **LVAgent** | 2025 | Framework multi-agente colaborativo |
| **T\* (T-star)** | 2025 | Busca temporal como busca espacial |

### Soluções Comerciais

- **Twelve Labs (Pegasus 1.5)** → Indexação nativa, ideal para 10k+ horas
- **Gemini 1.5/2.0 Pro** → Janela de 2M tokens, raciocínio narrativo profundo

---

## 🔭 Lacunas Abertas de Pesquisa

> 6 lacunas críticas identificadas, com aprofundamento na [Pesquisa Complementar](research/RESEARCH_COMPLEMENT_DEEP.md):

| # | Lacuna | Novidade | Frameworks Emergentes |
|---|--------|---------|----------------------|
| 1 | **Streaming VideoRAG** — latência vs. qualidade em tempo real | ⭐⭐⭐ | StreamRAG, V-Rex |
| 2 | **Cross-Video Knowledge** — síntese entre múltiplos vídeos | ⭐⭐⭐ | MegaRAG, LightRAG v2 |
| 3 | **Citation Faithfulness** — verificação de timestamp-grounded claims | ⭐⭐⭐⭐ | MiRAGe/CiteF1, FaithfulRAG |
| 4 | **Raciocínio Espacial 3D** — integrar profundidade ao RAG | ⭐⭐⭐ | SLAM-RAG, Video Depth Anything |
| 5 | **Audio-Visual Grounding** — localizar fonte sonora na imagem | ⭐⭐ | SSL-SaN, JAEGER |
| 6 | **Edge Deployment** — VideoRAG em hardware < 4GB VRAM | ⭐⭐ | EdgeRAG, GGUF Q4_K_M |

> 💡 **Recomendação primária:** Implementar **CiteF1-Video** (verificação de citação por timestamp) — maior novidade, alto impacto prático, realizável com o modelo híbrido do laboratório.

---

## 💻 Estratégia de Implementação — 100% Local (avancos)

> **Atualização Junho/2026:** Servidor `avancos` (192.168.1.9:11434) reintegrado. RunPod eliminado.

```
Extração (CPU local):
  ffmpeg → frames | agente_whisper → transcript | EasyOCR → OCR
      ↓
Descrição Visual (avancos 192.168.1.9:11434):
  llava:latest → "pessoa usa capacete, slide com título..."
      ↓
Indexação (ChromaDB local /mnt/large-memory/):
  transcript + visual_desc → embedding → ChromaDB
      ↓
Consulta RAG (avancos):
  pergunta → retrieval → qwen2.5:7b → resposta + timestamps
```

**Modelos disponíveis no avancos:** `llava`, `qwen3-vl`, `glm-ocr`, `moondream`, `qwen2.5:7b`, `deepseek-r1:7b→70b`, `nomic-embed-text`

**Custo total de experimentação: R$ 0,00**

---

## 📚 Referências Principais

1. Luo et al. (2025). *Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension*. NeurIPS 2025.
2. Jeong et al. (2025). *VideoRAG: Retrieval-Augmented Generation over Video Corpus*. arXiv:2501.05874.
3. Zhu et al. (2026). *STAR-RAG: Right Answer at the Right Time*. ICLR 2026.
4. LV-HAYSTACK Benchmark (2025). 480 horas de vídeo, 15k+ instâncias anotadas.
5. VidHalluc (CVPR 2025). Benchmark de alucinação em vídeo.

---

## 🤝 Contribuindo

Este repositório está em fase de pesquisa ativa. Contribuições são bem-vindas via Pull Requests.

---

*Pesquisa iniciada em Junho/2026 — Laboratório Universitário de IA*
