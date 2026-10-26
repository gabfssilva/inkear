# inkear — melhorias de desempenho

Levantamento de 20/10/2026, com o iPhone 16 Pro Max no iOS 27.0.1 e o app compilado com o SDK 27. Os números são do M5 Max, salvo indicação. Feitos: 3, 7 e 8; 2 em parte (entitlement). 1 em piloto.

## 1. Core AI no lugar do Core ML

O Core ML usa o caminho absoluto do `.mlmodelc` como chave do cache de compilação para o ANE (documentação do coremltools; no Mac, mover a pasta recompila e mudar a data do arquivo não). O caminho do container de dados muda a cada instalação ou atualização do app (TN2285). Por isso o iPhone recompila tudo depois de cada atualização: o cache do app tinha 533 entradas (0,9 GB), e o `decode` do Qwen foi compilado quatro vezes em nove horas. O Core ML não oferece como evitar isso.

O Core AI (iOS 27) guarda a especialização com chave pelo conteúdo do modelo e oferece compilação antecipada no Mac (`coreai-build`, um `.aimodelc` por arquitetura; o iPhone 16 Pro Max é `h17p`). Piloto com o OmniVoice, convertido com `coreai-torch` 0.4.3 (torch 2.14, `torch.export`):

| Medida | Core ML | Core AI |
|---|---|---|
| Decoder (f320), iPhone, primeira carga | 3,2 s | 14,9 s |
| Decoder, iPhone, ao reabrir | 0,24 s | 0,004 s |
| Decoder, iPhone, depois de reinstalar o app | 2,97 s (recompila) | 0,004 s (cache) |
| Passo (c576_t320), Mac, primeira carga | 62 s | 92 s; 85 s com compilação antecipada |
| Passo, Mac, por chamada | 95 ms | 93 ms |

- A reinstalação mudou o caminho do container de dados (`01236AC3…` → `B2F8960C…`), o que confirma a causa da recompilação no Core ML.
- A compilação antecipada não cobre o ANE: o `.aimodelc` leva a região do ANE como bitcode (`ANE_region_0.bc`), compilada no aparelho. Para modelos quase todos no ANE, a primeira carga continua longa, e no Core AI é mais longa que no Core ML.
- Os scores do passo batem com o Core ML (diferença típica de fp16). A saída `tokens` (int32) chega com os valores em int16, dois por elemento; devolver os tokens em fp16 contorna.
- Custo da migração: reconverter os modelos (no teste sintético, a atenção com `bmm` não rodou no ANE), reescrever a inferência (`MLMultiArray`/`MLModel` → `NDArray`/`InferenceFunction`, `run` assíncrono, estado como `MutableViews`) e exigir iOS 27. Kokoro e Pocket vêm do FluidAudio e continuam no Core ML.

Alternativa não testada: manter o Core ML e guardar os modelos num container de App Group, cujo caminho não deve mudar com as atualizações. Mantém a primeira carga mais rápida do Core ML. Requer a capability de App Group e é incompatível com o item 8, que apagaria um cache ainda válido.

## 2. Limite de memória maior

No iOS 27 a memória do ANE passou a contar no processo do app (nota 174796039), por isso o app é encerrado perto de 3,6 GB. Cada função carregada guarda sua própria cópia dos pesos. O entitlement `com.apple.developer.kernel.increased-memory-limit` subiu o limite para ~6 GB num 16 Pro Max (fórum da Apple, thread 834805). Mais memória devolve buckets e funções cortados: OmniVoice com os 12 buckets tem RTF 0,12 contra 0,19 com um; Chatterbox com as 20 funções, 0,33 contra 0,39. No Core ML, cada função a mais soma minutos de compilação no iPhone. O entitlement já está no app; carregar mais buckets e funções fica para depois da decisão do item 1.

## 3. Streaming da síntese

O `SpeechEngine` passa a entregar a fala em partes, na ordem, assim que cada uma fica pronta; o `Narrator` toca a primeira parte enquanto as outras são sintetizadas. Todo motor implementa a mesma interface; os que não geram frame a frame entregam uma parte por trecho que já sintetizam separadamente, ou uma só.

- Qwen: o decoder de áudio é causal, então os primeiros ~2 s de uma frase saem antes do resto, iguais ao início da frase inteira. Hoje uma frase de 10 s leva ~2,3 s no Mac (estimados 4–5 s no iPhone) antes de soar; com a parte antecipada, ~0,6 s no Mac. A normalização de pico pela fala inteira não cabe em streaming e vira um ganho fixo por voz (os ganhos por frase variavam ±25% em torno de 4,0 para thiago e 2,1 para peter).
- Chatterbox, OmniVoice e Supertonic: o flow do Chatterbox e os dois últimos processam a frase inteira de uma vez; entregam por trecho.
- A mudança de velocidade passa a ser feita em streaming, para que a fala esticada em partes soe igual à esticada inteira. O `AVAudioPlayerNode` alimentado por partes não reproduz isso (nem duas execuções iguais dele dão o mesmo resultado); um `AVAudioSourceNode` lendo as amostras recebidas, renderizado em blocos fixos de 4096, dá saída idêntica bit a bit.
- O decoder do Qwen olha ~2 frames à frente: o trecho antecipado decodifica 4 frames a mais e entrega 24 (1,92 s), idênticos ao início da fala inteira.
- Medido no Mac, frase de 9,4 s: o Qwen começa a soar em 0,61 s, contra 2,38 s antes; o RTF passou de ~0,23 para 0,25 pela decodificação a mais.

O ganho aparece com a fila vazia: ao abrir o livro e depois de um salto.

## 4. Menos ops nos modelos (se ficar no Core ML)

O tempo de compilação acompanha o número de ops: passo do OmniVoice com 21.893 ops (62 s por bucket a frio), `flow_step` do Chatterbox com 10.031 (~50–60 s), `MultiCodeDecoder` do Qwen com 3.775. A atenção é feita cabeça por cabeça (896 `softmax` e 1.792 `einsum` no OmniVoice). Atenção em lote reduziria as ops; o ganho de compilação não foi medido, e o comentário do conversor registra que esse formato é ~30% mais lento por camada.

## 5. Quantização

| Modelo | Passo | Tamanho |
|---|---|---|
| OmniVoice fp16 | 94 ms | 1,1 GB |
| OmniVoice int8 | 85,6 ms | 592 MB |
| OmniVoice 4 bits (paleta) | 85,2 ms | 546 MB |
| Qwen int8 (atual), decode + `MultiCodeDecoder` | 5,15 + 9,9 ms | 425 + 137 MB |
| Qwen 4 bits (paleta) | 5,05 + 9,45 ms | 214 + 69 MB |

A compilação não muda com a quantização (62–70 s nas três versões do OmniVoice). A memória do ANE no iPhone e a qualidade não foram medidas. O iOS 27 indica que fp8 roda no ANE pelo Core AI (nota 176210080); não testado.

## 6. Código Swift

Tempo de CPU sobre o tempo total da síntese, em Release: Magpie 21%, MOSS 16%, Qwen 9%, Chatterbox 8%, OmniVoice 1%. Os trechos apontados somam 1–2% ao RTF cada: ruído do Qwen 0,9 ms por frame, ruído do Magpie 1,0 ms por passo, amostragem do Chatterbox 0,18 ms por token, excitação do Chatterbox 14 ms por segundo de áudio, estado do Magpie 9 ms por frase. Prioridade baixa.

## 7. Entitlement de inferência em segundo plano

O iOS 27 documenta que o acesso ao ANE em segundo plano exige `com.apple.developer.background-tasks.continued-processing.inference`, também pelo Core ML. O entitlement já está no app. Num app de teste no 27.0.1, com a tela bloqueada e áudio tocando, o ANE funcionou sem ele em 234 de 234 rodadas, então a regra ainda não está sendo aplicada. A GPU falhou em todas as rodadas em segundo plano depois da primeira ("Insufficient Permission"), e `BGTaskScheduler.supportedResources` não inclui `.gpu` neste iPhone. Requer conta Apple no Xcode.

## 8. Cache antigo do ANE

As entradas do cache do Core ML de instalações anteriores nunca mais são usadas, porque os caminhos mudaram. Apagar o diretório `com.apple.e5rt.e5bundlecache` quando o app foi reinstalado ou atualizado libera esse espaço (0,9 GB no iPhone hoje).
