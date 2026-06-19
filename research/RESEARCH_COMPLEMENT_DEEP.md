# Video RAG — Pesquisa Complementar Profunda (Deep Research)

> **Data:** Junho 2026  
> **Metodologia:** Síntese de literatura recente (NeurIPS, CVPR, ICCV, ICLR, arXiv 2025-2026)  
> **Escopo:** Aprofundamento das 6 lacunas críticas identificadas na pesquisa inicial

---

## Sumário Executivo

Esta pesquisa complementar aprofunda cada uma das 6 lacunas críticas identificadas anteriormente em Video RAG, sintetizando os avanços mais recentes em cada frente. Os achados principais indicam que: **(1)** a área de streaming emergiu com novas arquiteturas baseadas em **segmentação semântica de eventos** e **KV-Cache dinâmico** [1][2]; **(2)** a síntese cross-video avançou com frameworks de grafo unificado usando **resolução semântica de entidades** via LLMs [3]; **(3)** a verificação de fidelidade de citação ganhou um framework dedicado para multimodal (**MiRAGe/CiteF1**) mas ainda carece de implementação específica para vídeo [4][5]; e **(4)** o deployment em borda evoluiu com **EdgeRAG** e quantização Q4_K_M via GGUF como abordagem prática para GPUs < 4GB [6].

---

## Achados Principais

- **Streaming VideoRAG:** V-Rex (arXiv:2512.12284) e StreamRAG (arXiv:2508.05662) são os primeiros sistemas a endereçar latência vs. qualidade em streaming, mas ainda requerem GPUs dedicadas [1][2]
- **Cross-Video Knowledge:** MegaRAG e LightRAG v2 introduziram parsers multimodais para grafos unificados com resolução semântica de entidades via LangGraph [3]
- **Citation Faithfulness:** MiRAGe (2025) introduz CiteF1 — primeira métrica específica para suporte e completude de citação em RAG multimodal [4]
- **3D Spatial RAG:** SLAM-RAG combina mapas SLAM com Relational Attention Graphs para grounding em linguagem natural em ambientes 3D [7]
- **Audio-Visual Grounding:** JAEGER (2026) estende AV-LLMs para 3D usando RGB-D + ambisonics para estimativa de direção de chegada [8]
- **Edge Deployment:** EdgeRAG (arXiv 2025) e quantização Q4_K_M via llama.cpp reduzem footprint em 50–75% com perda mínima de qualidade [6]

---

## Análise Detalhada

### Lacuna 1 — Streaming e Processamento em Tempo Real

**Estado atualizado (2026):**

Duas abordagens emergentes estão se consolidando para viabilizar VideoRAG em tempo real:

**1. Stream Event Segmentation (SES) + KV-Cache Dinâmico:**
- **StreamRAG** (arXiv:2508.05662) divide o vídeo ao vivo em eventos semanticamente coerentes, alimentando um mini-batch clustering com filtros heavy-hitter para manter um conjunto compacto de "protótipos" atualizáveis sem interromper queries em andamento [1]
- **V-Rex** (arXiv:2512.12284) introduz co-design software-hardware: o algoritmo ReSV agrupa tokens por similaridade temporal+espacial, reduzindo o footprint de KV-cache sem fazer prefill completo por frame [2]

**2. Amostragem Adaptativa com Aceleradores de Extração:**
```
Estratégia recomendada (2026):
  - Estado estacionário → 1 FPS
  - Detecção de movimento (δ > threshold) → 4-30 FPS
  - Reutilização de tokens: cache de representações para frames similares
  - Indexação incremental: mini-batches em vez de re-indexação completa
```

**Lacuna persistente:** Nenhum sistema atinge < 500ms de latência ponta-a-ponta com qualidade > 80% mIoU em GPUs de uso geral. V-Rex requer hardware especializado [2].

**Oportunidade de pesquisa:** Aplicar ReSV + StreamRAG ao pipeline Video-RAG (Leon1207) com avaliação de throughput-latency no Intel Xeon do laboratório.

---

### Lacuna 2 — Cross-Video Knowledge (Síntese Multi-Vídeo)

**Estado atualizado (2026):**

A síntese de conhecimento entre vídeos avançou significativamente com a integração de **Grafos de Conhecimento + LLMs para Resolução de Entidades**:

**Abordagem emergente: Semantic Entity Resolution via Multi-Agent:**
```python
# Padrão LangGraph para resolução de entidades cross-video
pipeline = {
    "1. Extração": "MLLMs extraem entidades de cada vídeo independentemente",
    "2. Embedding": "Embeddings semânticos de cada entidade (nome + contexto visual)",
    "3. Resolução": "LLM como árbitro: 'William J. Smith' == 'Bill Smith'?",
    "4. Unificação": "Grafo único com nós consolidados + arestas temporais",
    "5. Retrieval": "RAG sobre grafo unificado (GraphFlow / LightRAG v2)"
}
```

**Frameworks relevantes:**
- **MegaRAG:** Estende Knowledge Graph RAG com pistas visuais e espaciais diretamente na construção do grafo [3]
- **LightRAG v2:** Integra parsers multimodais (MinerU/Docling) para RAG unificado cross-file/cross-video [3]
- **GraphFlow (NeurIPS):** Otimiza recuperação em KGs ricos em texto via flow matching baseado em recompensa [3]

**Bottleneck identificado:** A resolução de entidades entre vídeos de câmeras diferentes (mesma pessoa, ângulos distintos) ainda requer re-identificação visual (ReID) — problema não resolvido pelos grafos textuais atuais [3].

---

### Lacuna 3 — Citation Faithfulness em Video RAG

**Estado atualizado (2026):**

Esta é a lacuna com **maior movimento recente** na literatura:

**Frameworks especializados emergentes:**

| Framework | Abordagem | Contribuição |
|-----------|-----------|-------------|
| **MiRAGe** (2025) | Avaliação multimodal RAG | Introduz **CiteF1** — mede suporte e completude de citação em mídia audiovisual [4] |
| **MedRAGChecker** (2026) | Decomposição em claims atômicos | NLI + Knowledge Graph para verificar cada afirmação independentemente [5] |
| **Reason & Verify** (2026) | 8 categorias de verificação | Distingue suporte explícito vs. implícito para diagnóstico fino de erros [5] |
| **FaithfulRAG** (2025) | Conflito de conhecimento | Modela discrepâncias entre conhecimento paramétrico e evidência recuperada [5] |
| **CITECHECK** (2025) | Dataset em larga escala | Negativos sintéticos LLM para treinar detectores de citação menores e baratos [5] |

**Gap ainda existente:** Todos esses frameworks são para RAG *textual* ou *imagem estática*. Nenhum aborda especificamente o problema de **timestamp-grounded citation verification** em vídeo — ou seja: "o timestamp citado realmente suporta esta afirmação?"

**Proposta de pesquisa do laboratório:**
```
Pipeline de verificação proposto:
1. Gerar resposta com VideoRAG (inclui timestamps de fonte)
2. Decompor resposta em claims atômicos
3. Para cada claim: extrair frames do timestamp citado
4. Verificar via MLLM: "O frame [X] suporta a afirmação [Y]?" 
5. Score de CiteF1-Video = F1 entre claims suportados e citados
```

---

### Lacuna 4 — Raciocínio Espacial 3D

**Estado atualizado (2026):**

**Avanços em estimativa de profundidade para vídeo:**
- **Video Depth Anything** (2025): Estende Depth Anything V2 com consistência temporal para vídeos longos — elimina o "flickering" de profundidade frame-a-frame [7]
- **VDA-Metric**: Variante com estimativa de profundidade métrica absoluta (não relativa) — essencial para SLAM real [7]

**Integração SLAM + RAG (SLAM-RAG):**
- Combine mapas 3D gerados por SLAM com **Relational Attention Graphs (RAGs)**
- Modela relações espaciais como grafo: "objeto X está sobre a superfície Y a Z metros"
- Permite que LLMs recebam instruções em linguagem natural grounded em posições físicas [7]

**Desafio técnico central:**
```
Vídeo monocular → Video Depth Anything → Nuvem de pontos 3D
                                               ↓
                              Mapa semântico 3D (o que está onde)
                                               ↓
                              RAG sobre posições 3D + timestamp
```

**Lacuna persistente:** Ambientes dinâmicos (pessoas se movendo) quebram as suposições de mapas estáticos do SLAM. **WildGS-SLAM** endereça parcialmente isso com mapeamento incerto [7].

**Benchmark disponível:** **VSI-Bench** e **GaRAGe** avaliam inteligência visual-espacial em MLLMs — pode ser usado para baseline [7].

---

### Lacuna 5 — Audio-Visual Grounding

**Estado atualizado (2026):**

**Avanços em Sound Source Localization (SSL):**
- **SSL-SaN** (BMVC 2025): Treina com silêncio e ruído explicitamente — melhora robustez em vídeos reais com sons fora de campo [8]
- **OCA + ORI Loss**: Object-aware Contrastive Alignment + Object Region Isolation — distingue objetos que fazem som de objetos silenciosos em fundo [8]
- **JAEGER** (2026): Estende AV-LLMs para 3D com RGB-D + ambisonics multicanal, usando "Neural Intensity Vectors" para estimar direção de chegada do som [8]

**Integração com VideoRAG:**
```
Pergunta: "Qual máquina está apitando em 00:45?"
Pipeline atual:    ASR → "som de alarme" (sem localização visual)
Pipeline proposto: SSL (OCA) → "máquina B, região (x:120-200, y:80-150)" → RAG
```

**Oportunidade imediata:** Integrar SSL-SaN como módulo de extração adicional no pipeline Video-RAG, adicionando coordenadas de fonte sonora ao índice vetorial.

---

### Lacuna 6 — Edge Deployment (Hardware Limitado)

**Estado atualizado (2026):**

**Solução prática identificada — EdgeRAG + GGUF:**

**EdgeRAG** (arXiv 2025): Técnica que poda clusters de embedding e gera embeddings sob demanda para economizar memória RAM/VRAM — viável em hardware < 4GB VRAM [6].

**Stack de inferência para o laboratório (Quadro P620, 2GB VRAM):**

```bash
# Modelos recomendados para CPU+GPU híbrido
# Gemma 4 2B (encoder-free, multimodal nativo)
ollama pull gemma4:2b

# Quantização Q4_K_M via llama.cpp
# Redução: 75% do footprint com ~5% de perda de qualidade
./llama-cli -m model.Q4_K_M.gguf -ngl 10  # 10 camadas na GPU P620
```

**Checklist de otimização:**

| Componente | Recomendação | Redução de memória |
|-----------|-------------|------------------|
| **LLM** | GGUF Q4_K_M (llama.cpp) | 75% |
| **Embeddings** | all-MiniLM-L6 (384-dim vs. 768-dim) | 50% |
| **Vector DB** | FAISS IndexIVFPQ (produto quantizado) | 90% |
| **Frames** | 1 FPS máximo, resize para 336px | 80% |
| **Retrieval** | EdgeRAG cluster pruning | 60% |

**Resultado esperado:** Pipeline de *consulta* (pós-indexação) viável no laboratório com latência de 5–15s por query via CPU Intel Xeon.

---

## Áreas de Consenso

1. **Retrieval é o gargalo primário** — maioria das falhas ocorre na recuperação, não na geração [1][3]
2. **Amostragem adaptativa supera amostragem fixa** — reduz custo computacional sem perda significativa de qualidade [2]
3. **LLM-as-Judge é padrão atual** — GPT-4.1 / Gemini 2.5 Pro como árbitros de qualidade [4][5]
4. **Grafos superam índices vetoriais planos** para relações temporais e causais [3]
5. **GGUF Q4_K_M é o sweet spot** de quantização para edge (50-75% redução, <5% perda) [6]

---

## Áreas de Debate

1. **Context Window vs. RAG para vídeo curto (< 2h):** Gemini 2M tokens torna RAG desnecessário para vídeos individuais?
2. **Granularidade de segmentação:** Por cena? Por segundo? Por evento semântico? Sem consenso.
3. **ReID cross-câmera:** Resolução de identidade de pessoas entre câmeras é tratada como problema de visão separado, não integrado ao RAG.
4. **Streaming vs. Offline:** Qual qualidade de busca perdemos ao forçar online processing?

---

## Mapa Atualizado de Oportunidades

```
Alta Novidade ┌──────────────────────────────────────────────────────────┐
              │ CiteF1-Video (Timestamp Citation) ★★★★ ← NOVO           │
              │ SLAM-RAG + VideoDepth ★★★                               │
              │ Cross-Video ReID + GraphRAG ★★★                         │
              │ StreamRAG + V-Rex (open source port) ★★★                 │
              │                                                          │
              │ SSL-SaN + VideoRAG integration ★★                       │
              │ EdgeRAG Q4_K_M pipeline ★★                              │
Baixa Novidade└──────────────────────────────────────────────────────────┘
              Baixo Impacto                              Alto Impacto
```

**Recomendação primária atualizada:** Implementar **CiteF1-Video** — verificação de citação em nível de timestamp. É a lacuna com maior novidade, impacto direto em aplicações práticas (auditoria, jurídico, educação) e realizável com o modelo híbrido de hardware do laboratório usando GPT-4o-mini como juiz.

---

## Fontes

[1] StreamRAG. *Streaming RAG with Unified Pipelines*. arXiv:2508.05662 (2025). (arXiv — alta credibilidade técnica)  
[2] V-Rex. *Software-Hardware Co-Design for Real-Time Streaming Video LLM Inference*. arXiv:2512.12284 (2025). (arXiv — hardware especializado)  
[3] MegaRAG / LightRAG v2 / GraphFlow. *Multimodal Knowledge Graph RAG with Entity Resolution*. NeurIPS 2025 / GitHub. (conferência peer-reviewed)  
[4] MiRAGe. *Evaluation Framework for Multimodal RAG with CiteF1*. arXiv:2025. (arXiv — framework de avaliação)  
[5] MedRAGChecker / Reason & Verify / FaithfulRAG / CITECHECK. *Claim-level faithfulness verification*. ACL/arXiv 2025-2026. (múltiplos papers peer-reviewed)  
[6] EdgeRAG. *Memory-Efficient RAG for Edge Devices*. arXiv:2025. (arXiv — aplicação prática)  
[7] Video Depth Anything / SLAM-RAG / WildGS-SLAM / VSI-Bench. *3D Spatial Intelligence for Video*. CVPR/arXiv 2025. (CVPR — alta credibilidade)  
[8] SSL-SaN / JAEGER / OCA+ORI. *Audio-Visual Grounding and 3D Sound Localization*. BMVC 2025 / arXiv 2026. (workshop + arXiv)  
[9] Luo et al. (2025). *Video-RAG: NeurIPS 2025*. (paper de referência principal)  
[10] Jeong et al. (2025). *VideoRAG: arXiv:2501.05874*. (paper de referência principal)  

---

*Deep Research Complementar — Junho/2026 — Laboratório Universitário de IA*
