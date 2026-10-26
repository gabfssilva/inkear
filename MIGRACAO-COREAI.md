# Migração para Core AI (AOT, só Neural Engine)

Notas da migração de outubro de 2026: o que deu errado, como foi resolvido e o roteiro para os próximos
modelos. O objetivo é cada função rodar 100% no Neural Engine, a partir de pacotes compilados ahead of time
(`coreai-aot`) para as arquiteturas do A17 Pro em diante: h16p, h17g, h17p, h18p e h19p.

## Estado

| Modelo | Runtime | Repo | Observação |
|---|---|---|---|
| Qwen3-TTS 0.6B | Core AI | `itsmechico/qwen3-tts-0.6b-base-coreai` | |
| Chatterbox | Core AI | `itsmechico/chatterbox-multilingual-pt-br-coreai` | O repo ainda tem 4 pacotes de arquiteturas antigas, que não são baixados. Apagá-los depende de autorização. |
| Magpie TTS | Core AI, com o codec em Core ML | `itsmechico/magpie-tts-multilingual-357m-coreai` (`fc02b72`) | O AOT calcula o codec errado (ver abaixo). |
| MOSS-TTS-Nano | Core ML | `itsmechico/moss-tts-nano-coreml-ane` | Port Core AI parado: o passo global erra com o prompt real no cache. |
| OmniVoice | Core AI só no passo | | Falta o decoder (HiggsAudio2Decoder). |
| Supertonic | Core ML | | Pendente: Core AI no iPhone, Core ML no simulador. |
| Kokoro, Pocket | Core ML | | Não migram. |

O port Swift do MOSS para Core AI ficou no scratchpad da sessão de 22/10/2026, que é temporário.

## Problemas e soluções

### Escopo e processo

- **Arquiteturas demais.** Compilei para 9 arquiteturas e investiguei falhas de ANEs antigos, que o app não
  suporta: o limite de largura 16384 e o `reinterpret_type` int8 em h13g e h14g. `ARCHITECTURES` em `coreai-aot` e
  `NeuralEnginePackages.architectures` agora listam só as 5 suportadas. Falhas fora dessa lista não se investigam.
- **Processos em paralelo.** Vários conversores Python ao mesmo tempo passaram de 100 GB de RAM e o processo
  morreu. Só um processo pesado por vez (conversão, `coreai-build`, upload grande).
- **Bissecções longas sem reportar.** No Qwen3 e no MOSS passei horas numa pista falsa. Agora o limite é cerca de
  30 min por bloqueio; depois disso, reportar e propor a alternativa (manter a peça em Core ML).
- **Filtro de upload.** O glob `*.h1[6789][gp]` incluiria h16g. Filtrar por nome exato de arquitetura.
- **Modificadores do zsh nos argumentos do upload.** Em `"$O:heteronyms_en.txt"` e `"$O:audio_bos_embedding.f32"`, o
  zsh leu `:h` e `:a` como modificadores de variável (diretório pai, caminho absoluto). Os dois arquivos não subiram e
  o Magpie falhou no iPhone ao carregar. Usar sempre `"${O}:arquivo"`. Depois de publicar, comparar a lista de arquivos
  do repo com os prefixos que o Swift baixa, para cada arquitetura.

### O AOT gera resultado errado sem aviso

Os pacotes compilam, ficam 100% no ANE e mesmo assim calculam errado. Só a comparação com o PyTorch mostra o
problema.

- **Qwen3, speech decoder.** `ConvTranspose1d` com kernel igual ao stride sai errado. Foi reescrito como conv 1×1
  para s·C canais seguida do interleave (`_StrideUpsample`).
- **Magpie, codec (NanoCodec).** Cada correção movia o erro para outro lugar:
  - a conv agrupada do upsampling (432 grupos, kernel 2) seguida do interleave sai errada; calculada como
    produtos elemento a elemento, fica correta;
  - a soma de dois ramos residuais sai errada (31 dB), mesmo com cada ramo correto sozinho (59 e 50 dB);
    `torch.stack(...).sum(0)` corrige no 1º estágio;
  - no 2º estágio, até a concatenação dos três ramos sai errada (6 dB), também compilando o estágio sozinho.

  Sem uma forma estável, o codec ficou no pacote Core ML publicado, que é 100% ANE e tem 36 dB contra o NeMo.
- **MOSS, passo global.** Cada função bate com o PyTorch em entradas de teste, mas o passo erra 45 a 97% quando o
  cache tem o prompt real. Descartei, sem sucesso: atenção 3D/4D, cache de chaves transposto, softmax manual,
  LayerNorm segura, saturação, troca de camada ou cabeça e deslocamento de slot. O PyTorch em fp32 e em fp16 gera
  fala correta. Sem causa encontrada, o MOSS voltou para Core ML.

### Cache do Core AI contamina testes

O Core AI indexa o que compila pelo hash do programa, que não depende dos nomes dos atributos do módulo. Além
disso, Python e Swift têm caches separados. No Qwen3, um teste "funcionou" porque reaproveitou o resultado de
outro. Por isso, testes de peças usam nomes de saída únicos (`uuid`) e sempre comparam contra a referência.

### Falhas do compilador ANE sem mensagem

`ANECCompileOffline() failed ... CompilationFailure` sem nenhum `ane_validation_message`, e a região inteira vai
para a GPU. Causas encontradas:

- **Larguras ímpares.** O prefill do Magpie com 217 posições falha; com padding para 224 passa. A atenção
  causal mantém o contexto separado do padding.
- **Conv de kernel 3 com dilatação 3** falha. Com dilatação 5 e mais de 3 taps, a conv vai para a GPU. A soma de
  convs de 1 tap compila, mas no codec do Magpie isso não bastou.

O método que funciona é a bissecção por peças: converter cada pedaço com nome novo, compilar para h17p e procurar
`_GPU_region_` no `specialized_model_*.mpsgraph`. Os scripts estão em `tools/<modelo>-coreai`, e os do scratchpad
seguem o mesmo padrão.

### O que o MPSGraph manda para a GPU

| Construção | Alternativa |
|---|---|
| Lookup por índice inteiro (embedding, `index_select`) | Tabelas f16 lidas pelo app (`Float16Table`, mapeada da memória) |
| Escrita em estado numa posição conhecida só em tempo de execução (o compilador crasha) | Cache como entrada; o passo devolve o k/v novo e o app escreve |
| `torch.topk` | Limiar por bissecção e argmax via `amax` |
| Saída inteira, que o Core AI rotula errado | Floats com inteiros exatos |
| Ramo feito só de constantes | Pré-calcular como buffer |
| Saída que copia parte de uma entrada | Devolver só o novo; o app concatena |
| Máscara calculada de uma entrada `[1, 1]` | Passar a máscara como entrada |
| Mescla de dois softmax por log-sum-exp | Um softmax sobre a concatenação |
| Atenção do prefill que não é 4D | 4D, com a máscara broadcastável para `[1, heads, T, T]` |

Há também um problema de shape inference no coreai-torch: `reshape` → `transpose` → `reshape` antes de um
`linear` falha. Usar `unbind` + `stack`.

### Tipos e PyTorch

- **`.half()` não converte tensores fora de parâmetros e buffers.** O `Codec.weights` do Magpie é um dict e
  continuou em fp32. O `torch.export` aceitou os dtypes misturados e o pacote saiu errado. Antes de exportar, rodar
  o módulo `.half()` em eager com entradas fp16, que acusa o erro de dtype.
- **Mistura fp16/fp32.** `.half()` também não converte as constantes criadas no módulo, e `.float()` e
  `torch.where` com escalares produzem fp32. Converter com `x.dtype` explícito.
- **`.half()` em módulos compartilhados.** Ele altera o modelo NeMo original. Em testes que comparam fp32 com
  fp16, fazer `deepcopy` só do submódulo, porque o modelo NeMo inteiro não é copiável.

### Ambiente

- **Runtime Python.** `coreai.runtime` precisa de `USE_OS_COREAI=1`.
- **Dependência local.** Com `no-sources = true`, a dependência local é `"pkg @ file:///${PROJECT_ROOT}/../pkg"`.
  Ela é instalada sem modo editável: depois de mudar `magpie-coreml`, rodar
  `uv sync --reinstall-package magpie-coreml` em `magpie-coreai`.
- **Symlinks.** `AIModel.specialize` falha (`failedToSpecialize`) num `.aimodelc` que é symlink. Copiar o
  diretório.
- **Simulador.** Um arquivo de engine Core AI fora de `#if canImport(CoreAI)` quebra `make build` (aconteceu com o
  tokenizer e o bundle do Chatterbox). Ao tornar um engine Core AI-only, envolver todos os arquivos que dependem
  dele e tirar o engine da lista `#else` de `Catalog.swift`.
- **Whisper.** `uv run --no-project --python 3.13 --with mlx-whisper`, porque o padrão do uv (3.15) falha.
- **Primeiro load.** A primeira especialização no Mac leva cerca de 90 s; com `cachePolicy: .persistent`, as
  seguintes levam 0.2 s.

## Roteiro para o próximo modelo

1. **Converter.** Usar coreai-torch com o padrão de `tools/magpie-coreai` (`graphs.py` reorganiza os estágios e
   `convert.py` salva um pacote por grupo de funções). Rodar cada módulo `.half()` em eager antes de exportar.
2. **Placement.** `uv run coreai-aot <pkg>.aimodel <dir> --architecture h17p`. Se aparecer "partly off the Neural
   Engine", partir para a bissecção por peças. Larguras devem ser múltiplos de 8 (de preferência de 16).
3. **Numérica, ainda em Python.** Compilar para o Mac (`--platform macOS --architecture h17c`) e implementar o
   protocolo de estágios do modelo com `coreai.runtime`. Medir a paridade por função contra o PyTorch fp32, depois
   gerar fala e transcrever com Whisper. Uma peça só passa para o Swift quando estiver correta.
4. **Se o AOT errar uma peça e a correção não sair no limite de tempo**, deixar essa peça em Core ML, como o
   codec do Magpie, e reportar.
5. **Swift.** Seguir `Qwen3Bundle` e `MagpieBundle`: `NeuralEnginePackages.specialize`, `ModelLoader.load(_:of:)`,
   `Float16Table` e caches como `NDArray` escritos pelo app.
6. **EngineCheck e Whisper** no Mac, com cópias dos pacotes h17c.
7. **Publicar.** AOT nas 5 arquiteturas, upload para `itsmechico/<modelo>-coreai` e fixar `revision` e
   `downloadSize`. O `downloadSize` é o da maior arquitetura mais os arquivos comuns. Conferir, na revisão
   publicada, que cada prefixo do Swift existe para cada arquitetura, e calcular o `downloadSize` a partir do repo.
8. **Fechar.** `make build` (simulador), `swift test --skip CorpusTests` e `make install`.
