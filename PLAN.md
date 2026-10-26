# inkear — plano

inkear é um app iOS que lê PDFs em voz alta com TTS on-device, preferencialmente no Apple Neural Engine (ANE), com reprodução em background e navegação pela estrutura do documento (páginas, capítulos, seções, parágrafos).

## Decisões

- **Motor inicial:** Supertonic 3 — ~99M parâmetros, 4 grafos ONNX (`text_encoder` → `duration_predictor` → `vector_estimator` × N passos → `vocoder`), 44,1 kHz, tokenização por caractere, `pt` suportado, 10 vozes fixas. O repositório oficial foi arquivado em set/2026; os pesos estão em `supertone-oss-archive/supertonic-3` (OpenRAIL-M).
- **Runtime:** Core ML com formas fixas, para ser elegível ao ANE. ONNX Runtime fica só como referência numérica no Mac.
- **Formas fixas:** o modelo recebe `text_mask` e `latent_mask`, então entradas são preenchidas até um bucket fixo. O vocoder não tem máscara; o áudio excedente é cortado. Cada bucket vira uma função de um modelo Core ML multifunction (iOS 18+), com pesos compartilhados.
- **Compute units:** `.cpuAndNeuralEngine`. A GPU não está disponível para apps em background no iPhone.
- **Plugabilidade:** protocolo `SpeechEngine` e um adaptador Swift por família de modelo. O Supertonic vem no bundle; os outros motores (`InstallableEngine`) são baixados sob demanda do Hugging Face para `Application Support/Models/<motor>`, a partir de conversões nossas publicadas em `itsmechico` (Fase 5).
- **PDFs:** prioridade para livros com texto selecionável; depois artigos acadêmicos (inclusive em duas colunas) e PDFs escaneados.
- **Plataforma:** iOS 26+ (macOS 26+ para os testes dos pacotes). Distribuição assumida como uso pessoal.
- **Idiomas da interface:** pt-BR e inglês, em String Catalog, seguindo o idioma do sistema (ou o idioma por app do iOS). Números, velocidades e nomes de idioma formatados pelo locale. O idioma da interface é independente do idioma do documento e da voz: títulos, capítulos e trechos do documento nunca são traduzidos.
- **Corpus de teste:** conteúdo aberto em pt-BR e inglês, baixado e rotulado por nós, em `corpus/`.

## Arquitetura

1. **`SpeechEngine`** — `languages`, `voices`, `usesQuality` (a qualidade só aparece na UI para os motores que a usam), `synthesize(text:voice:language:speed:quality:) -> SpeechStream`: a fala sai em partes, na ordem, assim que cada uma fica pronta (o Qwen entrega os primeiros ~2 s de uma frase antes do resto; os outros motores, uma parte por trecho que sintetizam). A unidade de síntese é a frase. Antes da síntese, normalização de texto (citações `[12]`, "Fig. 3", números, URLs, abreviações).
2. **`PDFStructure`** — extrai uma árvore `Documento → Seção (nível, título, páginas) → Bloco (título | parágrafo, texto, página, bounds) → Frase`, em camadas por confiabilidade:
   1. Outline do PDF (`PDFDocument.outlineRoot`).
   2. Rótulos de página (`PDFPage.label`): numeração romana vs. arábica separa pré-textual e miolo.
   3. Página de sumário: linhas terminadas em número de página, mapeadas para o índice real do PDF.
   4. Heurísticas de layout: tamanho/peso de fonte para títulos, espaçamento para parágrafos; remoção de cabeçalhos/rodapés repetidos, números de página e hifenização.
   5. PDFs escaneados: Vision `RecognizeDocumentsRequest`.
   6. Opcional: Foundation Models on-device para classificar casos ambíguos.
3. **Player** — `AVAudioEngine` + `AVAudioPlayerNode`, síntese adiantada com buffer, `MPNowPlayingInfoCenter` e `MPRemoteCommandCenter` (próximo/anterior = capítulo ou parágrafo), posição persistida como (bloco, frase).

## Fases

### Fase 0 — Spike de conversão no Mac (concluída)

Código em `tools/coreml/` (`uv run supertonic-convert`, `uv run supertonic-report`, `uv run pytest`). Medições num M5 Max.

- **Conversão:** ONNX com formas fixas → `onnxslim` → `onnx2torch` → `coremltools` 9.0 → ML Program. Seis operações do opset 19 (Cast, Equal, Pad, Reshape, Shape, Split) são registradas como aliases das versões que o `onnx2torch` implementa. Convertida em fp32, a saída reproduz o ONNX com erro relativo < 1e-4.
- **Semântica do padding:** as convoluções depthwise usam padding `edge`. Sem bucket, a borda replica o último caractere; com bucket, vê zeros mascarados, e a atenção espalha a diferença. Com padding, a saída é a mesma para qualquer tamanho de bucket (erro < 1e-4), e é o que o batch do upstream e o treino fazem. Adotado como semântica canônica.
- **Buckets:** pares (texto, latente) = (64, 96), (128, 192), (256, 384), com frames/id entre 0,8 e 1,45. Cobrem até ~27 s por utterance. Cada estágio é um `.mlpackage` multifunction com uma função por bucket e pesos compartilhados: 201 MB no total, contra 194 MB de um bucket só.
- **Precisão:** fp16 em `text_encoder`, `vector_estimator` e `vocoder`. O `duration_predictor` fica em fp32: ele roda na CPU de qualquer forma, e em fp16 a saída é quantizada (~1e-3).
- **ANE (`MLComputePlan`, `.cpuAndNeuralEngine`):** `vector_estimator` 100% do custo no ANE, `vocoder` 99%, `text_encoder` 75–87%, `duration_predictor` 0% (≤ 0,7 ms).
- **Latência no ANE:** o `vector_estimator` leva 3,3 / 7,0 / 14,2 ms por passo, por bucket. Uma frase de 8,5 s sai com RTF 0,007, contra 0,020 no Core ML só CPU e 0,226 no ONNX CPU.
- **Fidelidade fp16:** por estágio, erro relativo ≤ 5e-3, exceto o vocoder (~2e-2, SNR ~33 dB). No áudio final, LSD de ~2,3 dB em relação ao ONNX fp32, contra ~9,7 dB entre duas sementes de ruído. O erro vem quase todo do `vector_estimator` acumulado em 8 passos e não se concentra em nenhum tipo de operação (LayerNorm, Softmax e Sin/Cos testados).
- **Bug do coremltools 9.0:** as entradas numpy são liberadas por uma fila do Core ML sem o GIL, o que corrompe o heap do Python e aparece como crash em ORT ou pymalloc. Foi corrigido no upstream (apple/coremltools#2829), mas a correção não saiu em nenhuma release. Workaround em `coreml.stages`: manter referência a todas as entradas. Afeta só as ferramentas Python, não o app.

- **Qualidade fp16 (medida, sem escuta humana):** Whisper large-v3-turbo transcreve as amostras ANE fp16 como as ONNX fp32 (WER 0 em 3 de 4; 1 palavra em 28 na quarta). O UTMOS22 (MOS previsto, 1–5), sobre 2 textos × 5 sementes de ruído, dá ANE fp16 / CPU fp16 / ONNX fp32: F1 4,16 / 4,18 / 4,25; F3 4,06 / 4,07 / 4,05; M1 4,07 / 4,05 / 4,10, com desvio de 0,07 a 0,19 entre sementes. A diferença fica dentro da variação entre sementes; na F1, o fp16 varia mais (mínimo 3,75 contra 4,14).

### Fase 1 — `PDFStructure` (validada no corpus)

Target `PDFStructure` do pacote `InkearKit/` (`swift test`; `swift test --skip CorpusTests` para a rodada rápida). Pipeline:

1. **Extração:** linhas via PDFKit, com bounds, tamanho de fonte e negrito. Rótulos de página só quando o PDF define `/PageLabels`. Links internos de cada página.
2. **OCR (Vision `RecognizeDocumentsRequest`):** vale para páginas sem camada de texto e para documentos com camada inutilizável. Inutilizável é a maioria das páginas com menos letras que símbolos (fontes sem mapeamento Unicode, como no eBooksBrasil) ou com tamanhos medidos (OCR antigo, como no archive.org). As decisões são por documento, porque páginas de código ou tabelas também têm poucas letras.
3. **Tamanhos medidos:** um documento é "medido" quando tem mais de 40 tamanhos distintos (contra no máximo 25 em texto composto). Nesse caso, o corpo é a mediana e tamanhos a ±25% dele viram o corpo.
4. **Ordem de leitura:** o PDFKit não entrega as linhas em ordem de leitura. Linhas da mesma fileira são fundidas, com tabulação em lacunas largas. Duas colunas são detectadas por uma calha com cobertura mínima no terço central; linhas que a atravessam, como título e autores, dividem a página em faixas.
5. **Mobília de página:** números de página; linhas de margem que se repetem em 3 ou mais páginas (dígitos ignorados); e cabeçalhos cujo número na ponta acompanha a página com deslocamento constante. Linhas maiores que o corpo nunca são mobília.
6. **Blocos:** parágrafos por mudança de fonte, espaço vertical proporcional à fonte, recuo relativo (exceto recuo deslocado no meio da frase) e linha curta com fim de frase. Há continuação entre páginas e colunas, e a hifenização é desfeita. Títulos saem por tamanho, negrito ou padrão, com nível pelos tamanhos recorrentes.
7. **Seções:** vêm do outline do PDF; senão, do sumário impresso, lido por links ou por números de página (exigindo reticências ou lacuna larga), com resolução por rótulos ou voto de deslocamento; senão, dos títulos do layout, com títulos adjacentes formando uma seção.
8. **Início do texto principal:** página "1" após rótulos romanos, pulando seções pré-textuais; senão, a primeira seção com cara de capítulo depois do sumário; senão, a primeira não pré-textual.

Corpus (`corpus/manifest.json`): 13 PDFs abertos, 8 em pt-BR e 5 em inglês — livros com e sem outline, sumário por links, artigos em uma e duas colunas, camada de OCR antiga e escaneados sem texto. `CorpusTests` cobre 11 deles (início do texto, sumário, seções, frases intactas).

Desempenho no M5 Max: ~5 ms por página sem OCR (Pro Git, 524 páginas: ~4 s); ~0,2–0,26 s por página com OCR.

Limitações conhecidas:

- Notas de rodapé, legendas e tabelas são lidas como parágrafos.
- Em documentos com OCR, os níveis de título são ruidosos, e linhas de corpo com altura medida grande demais viram títulos.
- Em artigos sem outline, o texto principal começa no título do artigo.

### Fase 2 — `SpeechEngine` + adaptador Supertonic Core ML (concluída)

Target `Synthesis` do pacote `InkearKit/`.

- **Protocolo:** `SpeechEngine` define a interface por família de modelo.
- **Motor:** o ator `Supertonic` carrega o diretório gerado por `supertonic-convert` (pacotes, `tts.json`, indexador e vozes). Os modelos carregam sob demanda por estágio e bucket, e os buckets são lidos dos nomes das funções do pacote.
- **Seleção de bucket:** `duration_predictor` no menor bucket com `T > ids`; depois o menor bucket com `T > ids` e `L > frames`, sempre com pelo menos uma posição de padding.
- **Texto longo:** frases agrupadas até caber no maior bucket. Um trecho cujo latente não cabe é dividido ao meio numa vírgula ou espaço, com 0,3 s de silêncio entre os pedaços.
- **Paridade:** `supertonic-fixtures` grava 3 casos de referência, um por bucket, no CPU do Mac. No Mac, o Swift reproduz ids, bucket, duração, `text_emb`, latente e áudio do pipeline Python. Outros backends (ANE, CPU do simulador iOS) calculam o fp16 de outro jeito e a difusão amplia a diferença na forma de onda (erro relativo ~0,2–0,5 no áudio); neles o critério é o do `test_coreml.py`: distância espectral até a referência menor que 1/3 da distância entre duas sementes de ruído.
- **Velocidade:** `speed` no `SpeechEngine`; 1 é o ritmo natural (fator 1,05 do upstream).
- **Normalização (`SpeechText`):** antes da síntese, em pt e en: números (cardinais, decimais, milhares, anos em inglês, gênero em pt antes de substantivo feminino), ordinais, porcentagem, moeda, datas e horas (pt), intervalos de anos, abreviações, romanos depois de capítulo/parte/século, símbolos, citações `[12]` e URLs. Usa ICU (`NSRegularExpression`), porque o Swift Regex não tem lookbehind.

### Fase 3 — App iOS (concluída no simulador)

- **Nome:** o app se chama `inkear`; o pacote Swift é `InkearKit`.
- **Design:** https://claude.ai/artifact/3NxFN7B3dF1sBTdPbUsPwt
- **Pacote:** além de `PDFStructure` (agora `Codable`, com progresso e cancelamento da análise, frases por `NLTokenizer` e idioma por `NLLanguageRecognizer`) e `Synthesis`:
  - `Library`: ator dono dos documentos em disco (`index.json`, PDF e `structure.json` por documento), marcadores, preferências de leitura e `prepare` (análise + gravação).
  - `Playback`: `Script` (o documento como sequência de frases, com navegação por frase, seção e página), `AudioOutput` (`AVAudioEngine`, sessão `.playback`/`.spokenAudio`/`.longFormAudio`, interrupções e troca de rota), `NowPlaying` (tela de bloqueio e comandos remotos) e `Narrator` (≥ 3 frases ou 20 s à frente, segue a frase ouvida, temporizador de sono, log de RTF e signposts). O `Narrator` sintetiza juntas as frases de um parágrafo, até 300 caracteres, para o modelo ter contexto; com a fila vazia (início ou salto), sintetiza uma frase só e toca cada parte assim que chega, para o áudio começar logo. O áudio do grupo é cortado de volta em frases (`Speech.split`): cada corte parte da estimativa proporcional aos caracteres e vai para o ponto mais silencioso a até metade da frase mais curta. Modelos que emendam frases sem pausa (Pocket e Supertonic em pt) erram o corte por uma oração às vezes; o áudio sai contínuo e só o destaque fica aproximado. Antes de agendar o começo de um grupo, o `Narrator` acerta a pausa desde a frase anterior, contando o silêncio que o modelo já deixou no fim de uma e no começo da outra: 0,45 s depois de ponto final, exclamação ou interrogação, 0,6 s depois de reticências, 0,35 s depois de dois pontos ou ponto e vírgula, 0,8 s entre parágrafos e 1,2 s entre seções, divididos pela velocidade. Dentro de um grupo, as frases tocam com as pausas do próprio modelo, porque um corte errado com pausa no meio seria ouvido. Com a fila vazia, o silêncio inicial do modelo é cortado. Todo áudio toca sobre ruído rosa a −60 dBFS (`Speech.overRoomTone`): com fone de cancelamento de ruído, a queda para silêncio digital nas pausas é sentida como mudança de pressão.
- **App (`inkear/`, XcodeGen):** `xcodegen generate` gera `inkear.xcodeproj`. Uma fase de build compila os `.mlpackage` em `.mlmodelc` dentro do bundle (`supertonic-3/`, 211 MB no total). Telas do design em SwiftUI (biblioteca com mini player, preparando, pronto, player em modo texto e foco, sumário, voz e velocidade, modo escuro), String Catalog pt-BR/en, fontes Geist, Geist Mono e Newsreader (OFL). A análise continua em background com `BGContinuedProcessingTask`; análises interrompidas recomeçam ao abrir o app.
- **Verificação:** `swift test` (Mac) e `xcodebuild test -scheme InkearKit-Package` (simulador) passam. Os testes de UI (`inkearUITests`) importam um livro e um escaneado, esperam a análise, ouvem (a posição avança, inclusive com o app em background), abrem sumário e voz, alternam modo escuro e foco, e salvam capturas em `TEST_RUNNER_INKEAR_SHOTS`. Em DEBUG, `INKEAR_IMPORT=<caminho>` importa um PDF na abertura, porque os testes de UI não controlam o seletor de arquivos.
- **Simulador:** RTF de 0,08 a 0,18 só na CPU, em debug.

### Fase 4 — Verificação no device (pendente: precisa de um iPhone)

ANE em background (compute plan + Instruments), RTF, energia por hora de áudio, memória.

- **Pronto:** o build de Release para iPhone compila. Cada síntese registra caracteres, segundos de áudio, segundos de relógio e RTF em `os.Logger` (subsystem `inkear`, categoria `synthesis`), dentro de um intervalo de signpost `synthesize`, visível no Instruments junto do template Core ML (unidade de execução) e do Energy Log.
- **Aproximação no simulador (sem ANE, CPU do Mac):** lendo Dom Casmurro, o app ocupa de 48 a 72 MB (`footprint`; os pesos são mapeados do bundle e não entram nessa conta), o áudio continua em background e o RTF fica entre 0,08 e 0,18.
- **Falta:** instalar num iPhone (escolher o time de assinatura; o primeiro build registra o App ID `dev.gabfssilva.inkear`) e medir: ouvir com a tela bloqueada e conferir no Instruments se o `vector_estimator` continua no ANE; RTF por bucket; energia por hora; memória com os 3 estágios carregados.

### Fase 5 — Motores baixáveis (no Mac e no simulador)

Cada motor é um target do `InkearKit` (`KokoroEngine`, `PocketEngine`, `MossEngine`, `Qwen3Engine`, `MagpieEngine`, `OmniVoiceEngine`, `ChatterboxEngine`) e entra em `Engines/Catalog.swift`. `install` baixa só os `.mlmodelc` e os arquivos de que o app precisa (`HuggingFace.download`, pela API de árvore do repo) e grava um marcador `installed` com a revisão baixada: cada motor fixa o commit do repo (ou a versão do FluidAudio), e um marcador de outra revisão conta como não baixado. A tela Models também apaga os arquivos de um motor; um livro cujo motor não está baixado lê com o Supertonic. Cada motor carrega todas as suas funções Core ML no `load`, contando as carregadas (`ModelLoader`); o app mostra essa barra no Preparando e na folha de voz, e o player mostra um anel no lugar do play enquanto o livro espera o motor (`AppModel.awaitedEngine`): o player abre antes do motor carregar. A primeira carga depois de instalar ou atualizar o app recompila os modelos Core ML para o ANE: no M5 Max, ~50 s por bucket do flow do Chatterbox. O passo do OmniVoice é a exceção: roda com Core AI, de um `.aimodelc` compilado antes para cada arquitetura de iPhone (`xcrun coreai-build compile --preferred-compute neural-engine`, em `itsmechico/omnivoice-coreai`), e carrega em segundos; o decodificador continua no Core ML. No ANE, o Core AI rotula errado as saídas inteiras (o argmax `Int32` chega como int16 com o passo de linha dobrado), então o passo calcula o índice do token em Float16 (`Step(float_tokens=True)`). Cada função carregada guarda sua própria cópia dos pesos no ANE, e o iPhone encerra o app perto de 3,6 GB (o Chatterbox com as 20 funções carregadas foi encerrado assim). Por isso o OmniVoice usa só o bucket de 320 frames (~1,25 GB, RTF 0,19 contra 0,12 com os 12) e o Chatterbox carrega 7 funções: o maior prefill, as janelas de decode 384, 768 e 1024 e um tamanho de flow (640), ~1,4 GB e RTF 0,39 contra 0,33 com as 20. Trocar de função sob demanda custou mais (Chatterbox 0,51; OmniVoice até 2,2 numa troca de bucket). A tela Models do app baixa e depois carrega o modelo uma vez, para o ANE compilar ali; a folha de voz lista só os modelos baixados. Só o motor em uso fica na memória.

| Motor | Origem | Download | RTF no M5 Max | Whisper pt / en |
|---|---|---|---|---|
| Kokoro 82M | FluidAudio (`KokoroAneManager`) + G2P pt-BR nosso | 141 MB | 0,08 | 0 / 0 |
| Pocket TTS | FluidAudio, `.gpu` placement em `.cpuOnly` | 1,06 GB | 0,28 | 0,04 / 0 |
| MOSS-TTS-Nano 100M | `itsmechico/moss-tts-nano-coreml-ane` | 313 MB | 0,07 | 0,11 / 0,05 |
| Qwen3-TTS 0.6B Base | `itsmechico/qwen3-tts-0.6b-base-coreml-ane` | 1,16 GB | 0,21 | 0 / 0 |
| Magpie 357M v2607 | `itsmechico/magpie-tts-multilingual-357m-coreml-ane` | 801 MB | 0,125 | 0 / 0 |
| OmniVoice | `itsmechico/omnivoice-coreml-ane` + `itsmechico/omnivoice-coreai` | 1,32 GB | 0,11 (12 passos) | 0 / 0 |
| Chatterbox pt-BR | `itsmechico/chatterbox-multilingual-pt-br-coreml-ane` | 618 MB | 0,22 (6 passos) | 0,07 / 0 |
| Phonon 40M multilingual | Gradium (`ptts` SPM; flow LM Core ML no ANE, decoder Mimi na CPU) | `itsmechico/phonon-40m-multilingual-coreml` | 199 MB | 0,025 | 0,14 / 0,10 |

RTF medido pelos agentes de conversão no Mac; Whisper large-v3-turbo sobre a saída do motor Swift (frase de Dom Casmurro e uma em inglês). Os projetos de conversão estão em `tools/<modelo>-coreml/` (uv, `*-convert`, `*-report`).

- **Pocket:** os modelos `.ane` do checkpoint pt geram ruído, e os de rank 5 ficam mais lentos com o ANE do que só na CPU; roda com `.cpuOnly`.
- **MLState no macOS 27.0.1:** um programa que lê um estado depois de escrevê-lo, ou atualiza dois estados com `slice_update`, não monta o plano de execução sem a GPU (erro -14). As conversões ou passam o KV cache como entrada e saída (MOSS) ou leem cada estado uma vez e escrevem uma vez, no fim (Qwen3, Magpie, Chatterbox). As conversões publicadas por terceiros (aufklarer/speech-swift) só carregam com a GPU.
- **Vozes:** MOSS, Qwen3 e Chatterbox usam leitores do LibriVox (domínio público); OmniVoice, vozes desenhadas pelo próprio modelo e um leitor do LibriVox; Magpie, os 5 falantes do checkpoint. A voz padrão do Chatterbox (licença não declarada) não foi publicada.
- **Phonon:** roda com o runtime Swift oficial da Gradium (`ptts`/`PhononTTS`, SPM), que carrega um bundle Core ML exportado por ela (`export_coreml`): o flow LM em fp16 no ANE e o decoder Mimi em fp32 na CPU. O checkpoint original é gated; publicamos o bundle em `itsmechico/phonon-40m-multilingual-coreml` (CC BY-NC 4.0, mesma licença do OmniVoice). Uma voz só (`default`), com `setLanguage` por idioma e texto normalizado pelo runtime (o app não roda `SpeechText` para esse motor). Streaming de 80 ms por frame.
- **Licenças:** OmniVoice é CC BY-NC (uso não comercial), com o decodificador Higgs Audio 2 sob a licença da Boson; Magpie, NVIDIA Open Model License; os demais, MIT ou Apache-2.0.
- **Primeira carga:** o ANE compila cada função na primeira carga (até ~100 s por função no Mac) e o sistema guarda a compilação.

## Riscos

- **ANE em background:** sem documentação oficial; verificar na Fase 4.
- **fp16:** a saída diverge do fp32 (LSD ~2,3 dB), sem perda de inteligibilidade nem de MOS previsto além da variação entre sementes; uma escuta humana continua sendo o juízo final.
- **ANE do iPhone:** os números acima são do M5 Max; posicionamento e RTF precisam ser confirmados no device.
- **Modelo arquivado:** sem correções upstream.
- **Texto em português:** o pré-processamento oficial é mínimo (NFKD, aspas, travessões); números e abreviações precisam de normalização própria.
- **Kokoro:** o FluidAudio registra um crash no libBNNS depois de cerca de uma hora (issues #844 e #889), justamente na rota sem GPU.
- **RTF no iPhone:** Qwen3 (0,21 com pesos int8 no `CodeDecoder` e no `MultiCodeDecoder`; 0,32 em fp16), Pocket (0,28, só CPU) e Chatterbox (0,22) já são lentos no M5 Max; podem não acompanhar a leitura no iPhone.
- **Downloads grandes:** de 0,3 a 1,7 GB pela `URLSession` padrão, que não continua com o app suspenso; um arquivo que falha recomeça do zero.
