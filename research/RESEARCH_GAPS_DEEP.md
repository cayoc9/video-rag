# Lacunas de Pesquisa em Video RAG — Análise Profunda (Deep Research)

> **Data de compilação:** Junho 2026  
> **Metodologia:** Síntese de literatura acadêmica (NeurIPS, CVPR, ICCV, ICLR, arXiv) + análise de sistemas comerciais  
> **Objetivo:** Mapear fronteiras não resolvidas para orientar publicações e experimentos do laboratório

---

## Sumário Executivo

A pesquisa em Video RAG evoluiu rapidamente de sistemas de recuperação baseados em texto para frameworks multimodais sofisticados. No entanto, a transição para ambientes de streaming em tempo real, a precisão de recuperação temporal, e a fidelidade das respostas geradas continuam sendo os principais gargalos não resolvidos. Esta análise profunda identifica **seis lacunas críticas** com potencial direto para publicações e uma **lacuna transversal** de avaliação/benchmarking que afeta todo o campo [1][2].

---

## Principais Achados

- **Lacuna 1 (Streaming):** Nenhum sistema atual consegue processar vídeo ao vivo com latência < 500ms mantendo precisão > 80% [3][4]
- **Lacuna 2 (Espacial 3D):** RAG em vídeo ignora profundidade real — todos os sistemas operam em 2D [5]
- **Lacuna 3 (Cross-Video):** Síntese de conhecimento entre múltiplos vídeos concorrentes é virtualmente inexplorada [6]
- **Lacuna 4 (Fidelidade de Citação):** Modelos frequentemente "citam" segmentos que não sustentam a afirmação [7]
- **Lacuna 5 (Multimodalidade Pura):** Informação crítica puramente visual (sem texto ou áudio) ainda falha na recuperação [8]
- **Lacuna 6 (Edge/Borda):** Nenhum framework SOTA roda viávelmente em hardware <4GB VRAM [9]

---

## Análise Detalhada

### Lacuna 1 — Streaming e Processamento em Tempo Real

**Estado atual:** Todos os sistemas SOTA (Video-RAG, LVAgent, T*) operam em modo offline/batch [3].

**O problema:** A transição para análise de vídeo ao vivo (live) introduz um conflito fundamental entre:
- Necessidade de **latência baixa** (< 500ms para aplicações interativas)
- Qualidade da **extração de conhecimento** (que requer modelos pesados: 7B–70B parâmetros)

**Por que é difícil:**
```
Vídeo ao vivo → Segmentação semântica on-the-fly → Encodificação pesada → Indexação → Resposta
     ↑                                                                              ↓
  30 fps                                                                        < 500ms?  ← impossível hoje
```

**Direções de pesquisa abertas:**
- Poda dinâmica de grafo: algoritmos que decidem o que "esquecer" para economizar RAM sem perder contexto crítico
- Query-aware dynamic integration: ajustar o que indexar com base no tipo de pergunta esperada
- Token reuse: reutilizar representações visuais de frames similares sem re-encodar

**Potencial de publicação:** Alto — poucos papers tratam especificamente do trade-off streaming vs. qualidade [3][4]

---

### Lacuna 2 — Raciocínio Espacial 3D e Profundidade

**Estado atual:** Todos os sistemas atuais operam exclusivamente em espaço 2D (pixels, não posição real no mundo) [5].

**O problema:** Em aplicações industriais, de segurança e robótica, a pergunta não é "onde está o objeto na tela?" mas "onde ele está no espaço físico?". Um robô precisaria saber que a máquina com falha está a 3 metros à esquerda, não apenas "no canto inferior direito do frame".

**Gap específico:** Integrar SLAM (Simultaneous Localization and Mapping) ou estimativa de profundidade monocular ao pipeline de RAG.

**Exemplo de aplicação:**
```
Pergunta: "Em qual corredor da fábrica o operário parou de usar EPI?"
Resposta atual: "No corredor que aparece aos 1:23:45 do vídeo"
Resposta desejada: "No corredor B, posição GPS (X=-12.3, Y=5.7, Z=0), aos 1:23:45"
```

**Oportunidade de pesquisa:** Combinar:
- **DepthAnything V2** (estimativa monocular) + **VideoRAG** → índice com coordenadas 3D estimadas
- **SLAM** em câmeras de segurança com movimento known → índice geoespacial real

**Potencial de publicação:** Muito alto — campo praticamente virgem [5]

---

### Lacuna 3 — Persistência de Conhecimento Cross-Video

**Estado atual:** Todos os sistemas atuais tratam cada vídeo como universo isolado. Perguntas que cruzam múltiplos vídeos são praticamente impossíveis [6].

**O problema:** Casos de uso reais frequentemente exigem:
- "Compare como o procedimento X foi executado nos vídeos de Janeiro vs. Março"
- "Em qual das 50 aulas o professor explicou melhor o conceito Y?"
- "Quais câmeras do prédio mostram o suspeito Z?"

**Desafio técnico:** 
- Grafo de conhecimento unificado vs. grafos isolados por vídeo
- Resolução de entidades entre vídeos (mesmo objeto, câmeras diferentes)
- Reconciliação temporal (timestamps absolutos vs. relativos)

**Abordagens potenciais:**
```
Video A Index ─┐
Video B Index ─┼──→ Cross-Video Knowledge Graph ──→ RAG Multi-Source
Video C Index ─┘        (entidades unificadas)
```

**Potencial de publicação:** Alto — apenas um paper (arXiv:2501.05874) tangencia o problema [6]

---

### Lacuna 4 — Fidelidade de Citação (Citation Faithfulness)

**Estado atual:** Pesquisa recente (Wallat et al., 2025) demonstra que "fidelidade de citação" e "correção de citação" são frequentemente divergentes em sistemas multi-vídeo [7].

**O problema em detalhe:**
- **Fidelidade:** O modelo cita corretamente a fonte? ✓ (geralmente funciona)
- **Correção:** A fonte citada realmente sustenta a afirmação? ✗ (frequentemente falha)

**Exemplo:**
```
Pergunta: "O operário usou capacete durante o procedimento?"
Resposta: "Sim, o operário usou capacete [vídeo_A, 00:23:15]"
Realidade: No timestamp citado, o operário está de costas — impossível verificar
```

**Necessidade:** Um framework de verificação de citação em nível de **claim** (afirmação atômica), não de resposta completa.

**Ferramentas existentes relacionadas:**
- Ragas (métricas de faithfulness para RAG textual — não multimodal)
- DeepEval (parcialmente multimodal, 2025)
- FaithJudge (anotações humanas, não automatizado)

**Potencial de publicação:** Alto — primeiro benchmark end-to-end de citation faithfulness para Video RAG [7]

---

### Lacuna 5 — Recuperação de Informação Puramente Visual

**Estado atual:** Todos os sistemas SOTA dependem de "textos auxiliares visualmente alinhados" (OCR, ASR, object detection labels) para fazer a ponte visual↔texto [8]. Quando a informação crítica é **puramente visual** (sem palavras, sem áudio), a recuperação falha.

**O problema:**
- Uma expressão facial que muda sutilmente
- Uma postura corporal específica num vídeo de treino
- A cor exata de um indicador numa linha de produção
- Um gesto de linguagem de sinais

**Por que é difícil:** Embeddings CLIP/BLIP capturam categorias semânticas gerais, mas falham em discriminar variações visuais sutis sem âncora textual.

**Direções abertas:**
- Modelos de embedding visual *fine-tuned* para domínios específicos (medicina, indústria)
- Audio-visual grounding: vincular sons específicos a regiões da imagem (qual máquina está apitando?)
- Busca por similaridade gestual/postural

**Potencial de publicação:** Médio-alto — muito dependente do domínio de aplicação [8]

---

### Lacuna 6 — Execução em Hardware de Borda (Edge)

**Estado atual:** Todos os modelos SOTA requerem GPU com ≥ 16GB VRAM. TMRL é a exceção mais próxima de eficiência, mas ainda requer 8GB+ [9].

**O problema do laboratório:** Hardware disponível = Quadro P620 (2GB VRAM). Não existe framework SOTA que rode localmente para indexação.

**Oportunidade de pesquisa:** 
- Quantização agressiva (GGUF, AWQ) de modelos de embedding visual
- Destilação de conhecimento de modelos grandes → modelos tiny para edge
- Arquitetura split: embedding leve local + retrieval pesado em cloud

**Benchmarks de edge faltantes:**
- Nenhum benchmark avalia Video RAG em hardware < 8GB VRAM
- Nenhum paper propõe métricas de "pareto efficiency" (qualidade × recursos computacionais)

**Potencial de publicação:** Alto para laboratórios com restrição de hardware [9]

---

### Lacuna Transversal — Benchmarks e Avaliação

**Estado crítico:** O campo carece de benchmarks padronizados que avaliem o **pipeline completo** de Video RAG (retrieval + generation + faithfulness) [1][2].

**Benchmarks existentes e suas limitações:**

| Benchmark | O que avalia | O que falta |
|-----------|-------------|-------------|
| **LV-HAYSTACK** | Busca temporal | Não avalia geração de respostas |
| **LongerVideos** | RAG em vídeos extremamente longos | Pouca diversidade de domínio |
| **VidHalluc (CVPR 2025)** | Alucinação visual | Não avalia recuperação |
| **NExT-GQA** | Grounded QA | Vídeos curtos (< 3 min) |
| **OmniGround** | Grounding espácio-temporal | Não cobre cenários multi-vídeo |

**Métricas utilizadas:**
- mIoU — sobreposição temporal prevista vs. real
- R@k (IoU=α) — recall no top-k com threshold de sobreposição
- Acc@GQA — precisão combinada (grounding + resposta)
- Faithfulness Score (Ragas) — alinhamento entre resposta e contexto recuperado

**O que ainda falta:**
- Métrica de "Pareto Efficiency" (qualidade × custo computacional)
- Benchmark para cenários industriais/educacionais específicos
- Avaliação de streaming (latência + qualidade)
- Métricas para cross-video RAG

---

## Áreas de Consenso

1. **Retrieval é o gargalo primário** — A maioria dos falsos negativos vem da fase de recuperação, não da geração [1][3]
2. **Amostragem esparsa supera janela de contexto completo** — Processar todos os frames é ineficiente e reduz precisão por "dilution" [3][4]
3. **Grafos superam índices vetoriais planos** — Para relações temporais e causais, grafos oferecem recuperação mais precisa [1]
4. **Avaliação LLM-as-a-Judge é o novo padrão** — GPT-4.1/Gemini 2.5 Pro como juízes substituem BLEU/ROUGE [2][7]

---

## Áreas de Debate

1. **Context Window vs. RAG:** Quando um único vídeo cabe na janela de contexto (ex: Gemini 2M tokens), RAG ainda é necessário? Gemini Pro vs. VideoRAG mostra resultados mistos.
2. **End-to-End vs. Pipeline modular:** Treinar modelos end-to-end (Qwen3-VL) vs. pipeline de etapas especializadas (Video-RAG) — qual generaliza melhor?
3. **Granularidade de segmentação:** Segmentar por cena vs. por segundo vs. por evento semântico — sem consenso sobre optimal.

---

## Mapa de Oportunidades para o Laboratório

```
Alta Novidade │ Streaming VideoRAG ★★★    Cross-Video Knowledge ★★★
              │ 3D Spatial RAG ★★★       Citation Faithfulness ★★★
              │
              │ Audio-Visual Grounding ★★  Edge Deployment ★★
              │
Baixa Novidade│ Adaptive Sampling ★       Graph Pruning ★
              └─────────────────────────────────────────────────────
                Baixo Impacto                        Alto Impacto
```

**Recomendação principal:** Focar em **Citation Faithfulness para Video RAG** ou **Cross-Video Knowledge Graph** — ambos têm alta novidade, impacto prático imediato e são realizáveis com o modelo híbrido de hardware identificado no relatório de viabilidade.

---

## Fontes

[1] Luo et al. (2025). *Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension*. NeurIPS 2025.  
[2] Jeong et al. (2025). *VideoRAG: Retrieval-Augmented Generation over Video Corpus*. arXiv:2501.05874.  
[3] Revisão sistemática sobre streaming VideoRAG (thecvf.com/CVPR 2025 Workshop on Efficient Video Understanding).  
[4] Zhu et al. (2026). *STAR-RAG: Right Answer at the Right Time*. ICLR 2026.  
[5] Lacuna identificada via análise de literatura — nenhum paper publicado combina SLAM+VideoRAG até Jun/2026.  
[6] Wallat et al. (2025). *Citation Faithfulness in Multi-Source RAG*. ResearchGate / ACL 2025.  
[7] Jeong et al. (2025). arXiv:2501.05874 — seção 5.3 sobre cross-video limitations.  
[8] OmniGround Benchmark (2025). *Comprehensive Spatio-Temporal Grounding*. CVPR 2025.  
[9] Huynh et al. (2026). *Temporal-aware Matryoshka Representation Learning*. Jan 2026.  
[10] LV-HAYSTACK (2025). 480 horas, 15k+ instâncias. arXiv.  
[11] VidHalluc (CVPR 2025). Benchmark de alucinação em vídeo.  
[12] Alibaba Qwen Team (2026). *Qwen3-VL Technical Report*.  
[13] Microsoft Video RAG Pipeline. github.com/microsoft/Video_RAG_Pipeline.  

---

*Deep Research compilado em Junho/2026 — Laboratório Universitário de IA*
