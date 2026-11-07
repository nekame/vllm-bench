# Прямой Pangu Ultra runtime

PoC сравнивает batch-one генерацию в обычном offline OmniInfer и в прямом цикле
target + MTP без scheduler, model runner и штатного proposer. Он предназначен
для измерения накладных расходов, а не заранее заданного результата ускорения.

**Статус:** реализация и CPU-проверки; исполнение модели, MXFP8 packing,
compile и ACL replay требуют проверки на сервере Ascend. Локальная машина
не содержит torch, vLLM, torch_npu и NPU. Наличие команды запуска не означает
подтверждённую аппаратную поддержку всей матрицы.

## Требования и границы

- Установленное рабочее окружение OmniInfer/vLLM/torch_npu с исходными операторами
  модели; интерфейсы адаптера рассчитаны на vLLM 0.25.1 и OmniInfer commit
  `4cd855b29b5eb0a78bca645fad39e3181c35d099`. Ничего устанавливать инструмент не будет.
- По умолчанию `architectures` checkpoint должен содержать только
  `PanguUltraMoEForCausalLM`. Флаг `--ignore-architecture` выбирает этот класс
  при любом исходном значении `architectures`, включая отсутствие поля.
  Поддерживаются MXFP8 и BF16 safetensors; конфигурация квантизации и scales
  не преобразуются.
- Полная модель либо сокращённый checkpoint. Для MTP нужны сохранённые predictor
  layers: `layer_profile.prepare_checkpoint --layers 12 --keep-mtp`. Без MTP
  подходит и checkpoint, подготовленный без `--keep-mtp`. Перенос индексов
  повторно не выполняется. Checkpoint только с MTP не подходит.
- Чистый TP 1/2/4/8, DP=PP=1, EP выключен. Operator config должен соответствовать
  выбранному TP и содержать `use_noncontiguous_kv=true` при включённом MoME.
  SP, CP, FlashComm2, DP/local LM head, CLA aliases и `fix_multi_mtp_kvcache=true`
  требуют других протоколов и явно отклоняются; автоматического отключения нет.
  Pass `merge_dynamic_quant` также отклоняется: он требует piecewise pass context.
- По умолчанию: контекст 8192, генерация 256, batch 1, sampling temperature=1,
  top-p=1, top-k выключен, seed=0. Greedy задаётся `--temperature 0`.
  EOS по умолчанию игнорируется для фиксированной длины benchmark;
  `--respect-eos` включает остановку.
  Если первый токен уже EOS, результат сохраняет token ID без TPOT;
  speedup для такого запроса не определяется.
- Depth 3/7 — число предложений. Как в текущем `patch_eagle.propose`, при
  `depth <= num_nextn_predict_layers` используются successive predictor heads;
  при большей глубине все шаги используют первый predictor. Это не циклический
  обход физических слоёв. Политика совпадает с default `fix_multi_mtp_kvcache=false`.
  Direct создаёт и загружает только реально используемые predictor-слои:
  при трёх checkpoint heads и depth 7 — первый, при depth 3 — все три.
  Неиспользуемые слои не занимают NPU-память и не читаются из safetensors.
  Выбранные физические индексы сохраняются в `details.draft_loaded_layers`.

Полная модель при TP1 может не поместиться в память. Такой запуск завершается
ошибкой и сохраняет log, без offload, случайных весов или смены checkpoint.
Перечень заимствований и причины оставшихся зависимостей — [UPSTREAM.md](UPSTREAM.md).

## Один запуск

Из корня `vllm-bench`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m vllm_bench.pangu_infer.run \
  --model /models/pangu-ultra-12-mtp \
  --operator-config /configs/pangu-tp2.json \
  --tp 2 --depth 3 --mode eager --temperature 0 \
  --output /results/direct-tp2-mtp3-eager
```

Без `--execute` команда только проверяет конфигурацию и печатает план, не
импортируя ML-пакеты и не создавая output. Добавление `--execute` запускает
прямой worker через `torchrun`; вручную окружать launcher в torchrun не нужно.
Output должен быть новым каталогом. Явные относительные пути отсчитываются
от текущего рабочего каталога.

Вывод worker-процессов (stdout и stderr) одновременно показывается на экране
и сохраняется в `runtime.log`, с немедленным сбросом каждой строки. Это работает
для direct, offline baseline и каждого запуска матрицы.

Для baseline замените `--engine direct` на `--engine omni` и выберите другой
output. По умолчанию используется direct. Baseline запускает stock offline
LLMEngine, без HTTP; в его время входят engine scheduling и IPC.

Для асинхронного планирования Omni добавьте `--async-scheduling` и новый
`--output`. Флаг совместим с MTP и `--prefill-chunk-size 2048`; без флага
планирование синхронное. В direct и target trace/replay флаг отклоняется.
После прогрева инструмент проверяет фактический `async_scheduling` всех workers
и выводит его в лог; значение сохраняется в `effective.async_scheduling`.

В async-режиме граница TPOT — получение первого и последнего результата
`engine.step()` на CPU. Синхронизации после первого токена нет: следующий
decode уже может исполняться. Финальная синхронизация выполняется после
остановки таймера. Перед ней инструмент через штатный `engine.step()`
завершает очередь batch и работу scheduler, включая обработку завершённых
запросов. Получение последнего frontend output само по себе не означает,
что in-process EngineCore уже пуст; синхронизация NPU эту очередь не обрабатывает.
Такой цикл повторяет условие `EngineCoreProc.has_work` для DP=1 и завершает
оставшуюся работу перед следующим запросом. Prefix cache при этом не сбрасывается,
модель и scheduler не подменяются.
Формула TPOT остаётся `decode_seconds / (output_tokens - 1)`; первый результат
должен содержать один токен. Acceptance по-прежнему восстанавливается из
прироста выходных токенов и не заменяет внутреннюю статистику Omni.
Без флага сохраняется прежняя граница с синхронизацией после первого и
последнего токенов. Граница записывается в `details.timing_boundary`.
По умолчанию сохраняется `VLLM_ENABLE_V1_MULTIPROCESSING=0`, в том числе
при async scheduling. Это offline LLMEngine с in-process EngineCore.

Для проверки различий со штатным HTTP-запуском добавьте
`--omni-multiprocessing`: launcher устанавливает
`VLLM_ENABLE_V1_MULTIPROCESSING=1`, а `LLMEngine.from_engine_args()` вызывается
с `enable_multiprocessing=True`. EngineCore работает в штатном фоновом
процессе через SyncMPClient. HTTP использует AsyncMPClient, поэтому полный
frontend-путь всё ещё отличается, но цикл EngineCoreProc становится общим.
Модель, metadata builders, operators и prefix cache не подменяются.
Флаг не включает async scheduling автоматически: для него сохраняется
отдельный `--async-scheduling`.

Для синхронизации NPU и чтения worker state process-режим регистрирует
`OmniBaselineWorkerExtension` через штатный `worker_extension_cls`.
По RPC передаются строковые имена `pangu_bench_synchronize` и
`pangu_bench_worker_state`: Python callables не проходят стандартный
msgpack transport SyncMPClient при выключенной insecure serialization.
Существующие worker/model methods не переопределяются; настройки сериализации
не меняются. In-process режим сохраняет прежний callable RPC путь.

В process-режиме TPOT всегда измеряется между host-visible outputs, даже
при синхронном scheduler: фоновый core продолжает работу без frontend step,
поэтому синхронизация после первого токена исказила бы измерение.
Завершением batch queue и scheduler управляет сам EngineCoreProc; frontend
не обращается к его внутренним очередям. `queued_batches` и `cleanup_steps`
при этом равны `null`, а `[request-end].cleanup_owner=engine_core_process`.
Финальная NPU-синхронизация остаётся вне TPOT. В логе выводится
`[execution] engine_core=process`, а в результате сохраняется
`effective.engine_core_mode`. Выбор core относится к реализации engine и
не меняет comparison hash нагрузки.

Флаг разрешён только для Omni и несовместим с target/MTP boundary tracing:
внешний core может начать запрос до RPC, активирующего hooks. Для проверки
process-режима уберите `--trace-mtp-boundary`, сохранив temperature=0,
число повторов и остальные параметры. `[repeat-check]` работает как прежде.
Оба режима остаются offline LLMEngine без HTTP-сервера.

Для точного воспроизведения известной серверной конфигурации доступны
`--max-model-len 133120` и `--cudagraph-capture-sizes 8`.
Без них сохраняются прежние настройки: ёмкость равна
`context_length + max(max_new_tokens, 17) + 8`, capture sizes —
`[1, depth + 1]` без дубликатов. Это настройки для выравнивания конфигураций,
а не подтверждённое исправление nondeterminism.

`--max-model-len` поддерживается обоими engines, должен вмещать prompt,
generation reserve и не превышать HF position limit. В direct он также
увеличивает выделяемый KV-кеш. `--cudagraph-capture-sizes` разрешён только
для Omni в ACL-режиме; размеры должны быть положительными, покрывать окно
проверки и не превышать token budget. Явные значения проверяются по
фактическим настройкам workers, чтобы vLLM не мог незаметно отбросить размер.
Изменение ёмкости или списка capture sizes меняет comparison hash;
явное указание прежних значений его не меняет.

Строка `[execution-config]` и `effective.execution_settings` содержат
фактические max length, token budget, capture sizes, hybrid marker и
npugraph_ex options. В `details.workers_before/after[].runtime_manifest`
добавлены выбранные environment values, пути и SHA256 ключевых Python
исходников. Сначала используются пути уже загруженных модулей, затем пути
из установленного пакета. Это не хеши весов или CANN binaries.

Отдельный аудит окружения можно сохранить без загрузки модели, импорта
torch/vLLM и обращения к NPU, запустив в соответствующем Python environment:

```bash
PYTHONPATH="$PWD/src" python -m vllm_bench.pangu_infer.audit --output runtime-manifest.json
```

Он фиксирует установленные файлы и переменные окружения на момент вызова;
поля `loaded=false` явно отличают их от модулей, загруженных worker.
Он не восстанавливает effective config уже завершённого HTTP-запуска.
Для одинакового package version хеши исходников могут различаться.

Для async и process-режима каждый запрос выводит `[request-end]`: число кешированных prompt-токенов,
начальную длину оставшейся очереди `queued_batches`, число дополнительных
`cleanup_steps` и время завершения `cleanup_seconds`, включая финальную
синхронизацию. В `records[]` эти поля называются `cached_prompt_tokens`,
`post_request_queued_batches`, `post_request_cleanup_steps` и
`post_request_cleanup_seconds`. Cleanup не входит в TPOT и число decode cycles.
Для синхронного режима поля cleanup равны нулю, прежняя граница замера сохранена.

При `temperature=0` для каждого prompt сравниваются повторы с первым измеренным
результатом этого prompt. Строка `[repeat-check]` и поле `repeat_consistency`
содержат `checked_pairs`, `matches` и `first_differences`. При расхождении
сохраняются индексы записей, позиция первого отличия, token IDs (либо длины)
и числа кешированных prompt-токенов. Разные `id` сравниваются отдельно.
При отсутствии повторов `matches=null`; для sampling проверка не выполняется.
Обычное сравнение speedup отклоняет greedy-запуски с `matches=false`;
`--correctness` позволяет исследовать их токены без speedup.

Флаг входит в идентификатор сравнения: запуски с разным scheduling и разными
границами замера не проходят обычную проверку сопоставимости производительности.

Для повторения HTTP-конфигурации с prefix caching используйте в `--engine omni`:

```text
--enable-prefix-caching --block-size 128 --mamba-block-size 128
```

Эти параметры совместимы с `--async-scheduling` и `--prefill-chunk-size 2048`.
Без них prefix caching остаётся выключенным, размеры блоков выбирает Omni.
`--mamba-block-size` требует включённого prefix caching; все три параметра
разрешены только для Omni. Direct пока использует собственный протокол кеша
без APC. Target trace/replay также требует выключенного prefix caching.

При включённом prefix caching Omni управляет состояниями MoME по блокам;
при выключенном используется одно текущее состояние на запрос. Кеширование
должно сохранять результат модели, но режимы используют разные metadata и
пути работы с состоянием. Совпадение выходов нужно проверить на NPU.
На старте строка `[cache]` показывает фактические `enable_prefix_caching`,
`block_size`, `mamba_block_size` и `mamba_cache_mode`. Они сохраняются в
`effective` и в снимках workers. Запуск отклоняется, если Omni изменил
запрошенное значение `enable_prefix_caching`.

Прогрев и повторы используют тот же engine; prefix cache между ними
не сбрасывается. Повторный prompt может вычисляться лишь частично.
`records[].cached_prompt_tokens` показывает число закешированных prompt-токенов,
которое сообщает `RequestOutput`; `null` означает отсутствие значения в API.
Эти настройки входят в идентификатор сравнения, чтобы отделять запуски
с разными режимами кеша и размерами блоков. Для сопоставления с HTTP укажите
одинаковые параметры в обоих запусках и новый `--output`.

Формат KV-кеша задаётся отдельно от точности весов и вычислений:

```text
--kv-cache-dtype hif8_ds_mla
```

Параметр передаётся обоим engines через штатный `EngineArgs.kv_cache_dtype`.
По умолчанию используется `auto`; `dtype` модели остаётся `bfloat16`.
Допустимые форматы проверяет установленный Omni/vLLM. В текущем Omni
`hif8_ds_mla` поддержан для DSA; SWA-кеш в этом режиме остаётся неквантизованным.
Формы, dtype и размещение кеша выбирают исходные cache specs и backend Omni,
дополнительного преобразования в direct нет. Исполнение HIF8 на NPU требует
проверки отдельно для каждого engine.

Строка `[cache]` и `effective.cache_dtype` показывают фактическую строку
конфигурации. В Omni она также проверяется на соответствие запрошенному
формату. Значение входит в идентификатор сравнения; обычное сравнение
производительности дополнительно отклоняет различие `effective.cache_dtype`.
`--correctness` позволяет сравнить выходные IDs разных форматов без speedup.
Отличие BF16/HIF8 может менять logits и greedy-выходы, поэтому для
сопоставления с HTTP нужно явно повторить его `--kv-cache-dtype`.
MXFP8-квантизация весов сама по себе не включает HIF8 KV-кеш.

Если в checkpoint указан другой класс, добавьте `--ignore-architecture`.
В обоих движках через штатный `EngineArgs.hf_overrides` переопределяется только
`architectures: ["PanguUltraMoEForCausalLM"]`; остальные поля конфигурации берутся
из checkpoint. `config.json` не изменяется. В `launch_plan.json` сохраняются
исходный `config` и отдельный `hf_overrides`, поэтому проверка изменения
checkpoint перед запуском продолжает работать. Для сравнения движков и
trace/replay передавайте флаг в обе команды; для matrix добавьте его в `options`.
Веса и остальные параметры по-прежнему должны быть совместимы с Ultra.

Для запуска без MTP замените `--depth 3` / `--depth 7` на `--no-mtp`
(или `--depth 0`). Флаг работает для обоих engines, greedy и sampling, всех
поддерживаемых режимов compile/ACL. Predictor не создаётся, его веса не
загружаются, speculative decoding отключён. Подходит checkpoint как с MTP,
так и без predictor layers. По умолчанию MTP остаётся включённым с depth 3.

`--prompt-ids /data/prompt.json` принимает JSON-массив ровно из 8192 token IDs
при стандартном context-length. Без файла с IDs или текстами оба engine получают одинаковый
детерминированный synthetic prompt, зависящий от seed и vocab_size. Для
воспроизводимого анализа настоящих запросов используйте явный файл.

### Текстовые запросы

Вместо `--prompt-ids` передайте `--prompts /data/prompts.json`:

```json
[
  {"id": 0, "input": "Расскажи о тензорном параллелизме."},
  {"id": 1, "input": "Напиши краткое объяснение MoE."}
]
```

`id` должен быть уникальным целым числом или строкой, `input` — строкой.
Токенизатор загружается локально из `--model`; для отдельного каталога
добавьте `--tokenizer /data/tokenizer`. Используется обычный `encode` с
`add_special_tokens=True`, без chat template. Токенизация производится при
планировании; для обоих engines сохраняются одинаковые IDs всех запросов
в `launch_plan.json`, вместе с количеством обрезанных и добавленных токенов.
Поэтому dry-run с текстами требует установленный `transformers` и файлы
токенизатора.

Длина нормализуется **в токенах** до `--context-length`: длинный prompt
обрезается справа (сохраняются первые токены), короткий дополняется слева
`pad_token_id`, а при его отсутствии — `eos_token_id`. Добавленные позиции
обрабатываются как часть контекста; отдельная padding mask не применяется.
Модель загружается один раз; все запросы выполняются последовательно,
с очисткой состояния и повторным использованием execution lanes direct.
Прогрев выполняется на первом запросе.

К прежней команде запуска добавьте:

```bash
--prompts /data/prompts.json --repeats 1
```

В `<output>/generations.csv` записываются колонки `id,text,tpot`.
`text` содержит только сгенерированное продолжение, декодированное с
`skip_special_tokens=True`. `tpot` — мс на decode-токен:
`decode_seconds * 1000 / decode_tokens`, без первого токена после prefill.
При отсутствии decode-токенов поле пустое. Токенизация и декодирование
не включаются в замеры. CSV сохраняется в UTF-8 с корректным экранированием
запятых, кавычек и переводов строк.

Каждый запрос выполняется `--repeats` раз (по умолчанию 3): CSV содержит
отдельную строку для каждого повтора с тем же `id`. Для одной строки на
запрос используйте `--repeats 1`. В `result.json` записи дополнительно
содержат `id` и `repeat`. Сравнение движков учитывает весь набор запросов
и их порядок. Текстовые batches поддерживаются в matrix через `options`;
trace/replay остаётся режимом одного запроса через `--prompt-ids`.

## Отдельный TP для MTP

Для direct задайте target TP и размер одной MTP-копии независимо:

```bash
--engine direct --tp 8 --mtp-tp 2 --depth 7
```

`--mtp-tp` принимает 1/2/4/8, должен делить target TP и не превышать его.
Без флага MTP использует target TP и общие с target embedding/LM head,
как прежде. При отличающемся TP MTP исполняется на первых `mtp-tp` rank
на **prefill и decode**. Например, при TP8/MTP2 веса MTP находятся только
на rank 0–1; rank 2–7 ждут результат proposal, затем участвуют в target
verification. Дополнительных MTP-копий на остальных rank нет.

С меньшим TP MTP получает собственные шардированные embedding и LM head,
загруженные из глобальных `model.embed_tokens.*` и `lm_head.*` checkpoint.
Его TP/EP/dispatch-группы и parallel config отделены от target. Target hidden
states уже доступны на участвующих rank; отдельного копирования через CPU нет.
После proposal rank 0 передаёт IDs всем target-rank через существующую target
TP device group. При sampling также передаются draft-распределения,
необходимые для verification. Эти передачи входят в end-to-end decode.

Работают depth 3/7, eager/compile/acl/compile-acl, текстовые запросы,
`--mtp-pad-to-8`, memory diagnostics и model timing. На неучаствующих rank
MTP timing rows содержат нули до TP MAX reduction: итоговое время берётся
с исполняющих rank, количество вызовов в сводке не умножается на число NPU.
В `result.json` сохраняются `effective.mtp_tp`, `effective.mtp_ranks` и TP
каждой execution lane на rank 0.

При разных TP каждый direct ACL-граф захватывает собственные attention
workspaces и task handles. Общий cache Omni по числу токенов и типу операции
не учитывает разное число локальных heads у target и MTP; переиспользовать его
workspace между этими графами небезопасно. Изоляция действует для `acl` и
`compile-acl`, не меняет graph pool и может увеличить расход памяти workspace.
Проверки CPU покрывают владение ресурсами; mixed-TP capture/replay требует
проверки на NPU.

Для проверки задержек static-kernel compilation при разных TP добавьте
`--mtp-no-static-kernels` к прежнему запуску и укажите новый `--output`:

```text
--tp 8 --mtp-tp 2 --depth 7 --mode compile-acl --mtp-no-static-kernels --memory-diagnostics
```

Флаг отключает `static_kernel_compile` и `super_kernel_optimize` только для
MTP, включая backend и ACL wrapper. MTP продолжает использовать `torch.compile`
и внешний ACL capture/replay; target сохраняет исходные настройки. Все остальные
опции backend, включая `frozen_parameter` и `clone_input`, сохраняются.
Флаг разрешён только для direct с положительной depth в `compile`/`compile-acl`;
без него поведение прежнее. На старте выводится подтверждение отключения;
в `result.json` настройки MTP записываются в `effective.mtp_compilation`.
Флаг входит в идентификатор сравнения: обычное сравнение производительности
не смешивает запуски с разными настройками компиляции MTP.

Причина проверки: static compiler backend создаёт Gloo-группу по
`LOCAL_WORLD_SIZE`, независимо от scoped TP модели. При MTP TP2 в мире TP8
в этот путь входят только rank 0–1, хотя backend ожидает все восемь rank.
Это может приводить к таймаутам MTP и нарушению порядка создания групп при
следующей компиляции target. Изменение TP или `LOCAL_WORLD_SIZE` не заменяет
отдельную поддержку группы компиляции в backend. В этом эксперименте
исходники Omni/backend и переменные окружения не изменяются.

Отличающийся MTP TP поддерживается только direct. В
`model_parallel_config.layer_parallel_config` разрешены переопределения вида
`"self_attn.o_proj": {"tp_size_or_ranks": 1}`: одноранговые группы сохраняют
реплицированный слой у обеих моделей. Пустой конфиг также разрешён.
Группы большего размера и x/y transforms пока отклоняются: Omni выбирает
эти общие группы по коротким именам слоя, одинаковым для target и MTP.
Изменения файлов Omni
не требуются; scoped переключение групп описано в `UPSTREAM.md`.
Для matrix используйте `"engines": ["direct"]`, `"tp": [8]` и
`"options": ["--mtp-tp", "2"]`.

Сначала проверьте урезанный checkpoint с `--temperature 0`: eager и
compile-acl отдельно для MTP1/2/4. Затем запускайте полную модель.
Шард весов MTP на каждой участвующей NPU увеличивается при уменьшении TP;
собственные embedding/head тоже требуют памяти. Наличие TP8 у target
не гарантирует, что MTP1/2 поместится рядом с ним. CPU tests проверяют
протокол, но NPU loading, численная корректность и графы этого пути требуют
проверки на сервере. Разный MTP TP меняет workload fingerprint; стандартный
benchmark compare его отклоняет. `compare --correctness` допускает такую
проверку greedy token IDs без вывода о speedup/Python overhead.

## Уменьшение памяти direct

Для удаления общего padding между страницами разных типов кеша добавьте:

```bash
--compact-kv-cache
```

У direct каждый слой владеет отдельным raw-буфером. По умолчанию размеры
страниц сохраняют общее hybrid-выравнивание Omni; с этим флагом cache spec
каждого слоя получает собственный размер страницы. Исходный spec не изменяется.
MLA/DSA/MoME backends получают фактические strides через свои штатные reshape
и metadata builders. Backend должен объявлять поддержку block-stride indexing.
Количество блоков, токенов, dtype, MoME history и резервный block 0 сохраняются.
Квантование кеша и ограничение истории не применяются.

Флаг действует для target и MTP, с одинаковым или разным TP; поддерживается
только direct. Строка `[cache] rank=... allocated_mib=... padding_saved_mib=...`
показывает размер кеша и фактическую экономию на каждом rank. Выигрыш зависит
от исходных cache specs; если padding отсутствовал, экономия равна нулю.
NPU-корректность нового layout необходимо проверить на сервере.

Другие способы уменьшения памяти: сократить `--context-length` (меняет
нагрузку) или `--max-new-tokens` (уменьшает резерв продолжения). При контексте
8192 сокращение генерации с 256 до 64 даёт небольшой эффект на KV-память.
`gpu_memory_utilization` в direct не ограничивает собственные cache allocations.
Если OOM возникает до `weights_loaded`, compact cache ещё не может помочь;
загрузка только используемых MTP heads уменьшает уже память весов.

После загрузки/packing и выделения кеша direct один раз синхронизирует NPU
и освобождает неиспользуемый резерв torch allocator через `empty_cache()`.
Это происходит до прогрева и замеров, чтобы ленивой инициализации HCCL
была доступна память, оставшаяся от временных packing-буферов.

### Prefill частями

Для контекста 8192 добавьте в direct или Omni:

```bash
--prefill-chunk-size 2048
```

По умолчанию флаг равен 0: prefill остаётся одним вызовом. При включении
`max_num_batched_tokens` равен указанному бюджету, `enable_chunked_prefill=true`.
В direct каждый начальный target/MTP проход выполняется последовательными
частями с сохранением KV/MoME-состояния и абсолютных позиций. Hidden states
соединяются для следующего MTP head; исходные shifted IDs режутся в том же
порядке. Для каждого eager chunk вычисляется последняя строка logits,
используется только результат последнего chunk. Sampling между частями нет.
Весь bootstrap остаётся внутри prefill timing; decode продолжает использовать
выбранный eager/compile/ACL режим и прежнее verification window.

Для PoC размер части должен превышать `depth+1`; если он меньше контекста,
`context-length` должен делиться на него без остатка. Это исключает короткий
хвост, который metadata builders могли бы классифицировать как decode.
Trace/replay с chunked prefill пока отклоняется. При сравнении движков
передавайте одинаковый флаг обоим: он входит в workload fingerprint.
В Omni число decode cycles отсчитывается от первого выданного токена,
исключая все prefill steps. Новый путь требует NPU проверки.

### Память HCCL

HCCL выделяет коммуникационную память отдельно от torch allocator.
Она составляет удвоенный размер `HCCL_BUFFSIZE` или настройки communicator:
см. [HcclGetHcclBuffer](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/commopdev/docs/en/api_ref/comm_opdev/control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md).
Например, запрос 2517630976 bytes равен 2401 MiB и соответствует двум
буферам по 1200 MiB с дополнительным 1 MiB. Для такого отказа можно проверить
`HCCL_BUFFSIZE=200`; `HCCL_MOE_DISPATCH_EP_BUFFSIZE` управляет только отдельной
MoE dispatch-группой. Chunking уменьшает активации, а размер HCCL-буфера
определяется настройкой HCCL. Для сравнения TPOT значения env должны совпадать
у сравниваемых запусков.

### Проверка Gloo отдельно от модели

Если появляются `gloo/transport/tcp ... Connection refused`, сначала проверьте
первый traceback rank перед ними: ошибка соединения может быть следствием
завершения одного из peer-процессов. После прерванного запуска проверьте
остаточные процессы своего запуска через `ps` и `npu-smi info`.
Launcher использует `torchrun --standalone`, без фиксированного rendezvous-порта.

Для CPU-проверки Gloo-разбиений world TP8 и одной MTP-копии TP4:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  --module vllm_bench.pangu_infer.check_gloo --mtp-tp 4 --timeout 60
```

Проверка не загружает модель и не выполняет HCCL-операций. Все rank создают
группы в одинаковом порядке; проверяются target-группы, одноранговые layer
groups и MTP-группы, выполняются CPU reductions и финальный barrier.
Лог содержит rank, PID, hostname, выбранный Gloo interface и `stage`.
Успех отмечается `[gloo-check] PASS`. Timeout ограничивает каждую distributed
операцию, а не суммарное время всех групп. Эта проверка отделяет базовый Gloo
transport от модели и HCCL, но не воспроизводит весь startup Omni.

Если все workers находятся на одном сервере, повторение проверки с
`GLOO_SOCKET_IFNAME=lo` позволяет отдельно проверить loopback. Переменная
управляет только Gloo interface; этот тест не меняет настройки HCCL.

## Acceptance после prefill

После prefill число `num_accepted_tokens` соответствует одному выданному
sampler-токену, а не размеру speculative-окна. Direct передаёт **1** начальному
MTP proposal и первому target verification; replay-with-MTP также начинает с 1.
После обычного verification используется `accepted_drafts + 1`. Это повторяет
порядок stock runner: sampling → обновление hybrid state counts → drafter.
MoME builder отдельно обрабатывает prefill; подмена этого счётчика на `depth+1`
меняет выбор истории в первом рекуррентном MTP-шаге, где computed tokens уже
равны длине prompt и prefill-маскирование не применяется. Соответствующий
[код vLLM 0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_model_runner.py#L4131).

Target replay с фиксированным acceptance из Omni не проверяет инициализацию
этого счётчика в самостоятельной генерации direct. Для неё есть отдельная
CPU-регрессия по depth 0/3/7; совпадение реальных NPU-выходов проверяется отдельно.

## Padding рекуррентных MTP-вызовов

`--mtp-pad-to-8` — отдельный контроль для `--engine direct` с включённым MTP.
Флаг дополняет однопозиционные рекуррентные MTP-вызовы до 8 строк. Для обычного
checkpoint с тремя predictor layers и depth 7 это шесть вызовов после первого
packed-прохода. Первый проход, prefill, target verification и target-only decode
не дополняются. В multi-head ветке depth 3 флаг не меняет packed-входы.

```bash
python -m vllm_bench.pangu_infer.run \
  --engine direct --model /models/pangu-ultra-12-mtp \
  --operator-config /configs/pangu-tp2.json \
  --tp 2 --depth 7 --mode compile-acl --temperature 0 \
  --mtp-pad-to-8 --output /results/direct-tp2-mtp7-pad8 --execute
```

IDs, positions и previous hidden states в хвосте заполнены нулями, KV slot mapping
содержит `-1` для padding. Полезная query остаётся одна: `query_start_loc=[0,1]`,
`max_query_len=1`, sequence length увеличивается на одну позицию. Physical
`num_actual_tokens` CommonAttentionMetadata равен 8, как в FULL graph ветке
Omni; MoME builder выводит полезное число позиций из query boundaries.
Из результата MTP возвращается только первая строка hidden states.

Флаг работает в eager, compile, acl и compile-acl; без флага поведение прежнее.
Omni с FULL graph сохраняет размер первого окна для рекуррентных шагов; его
eager ветка может использовать размер 1. Флаг принудительно задаёт 8 во всех
direct modes и не является обещанием совпадения всех остальных metadata Omni.
Параметры численной согласованности не переключаются.

В ACL режимах packed и padded MTP-вызовы получают разные execution lanes и
отдельные графы. Локальный подкласс Omni ACLGraphWrapper сохраняет собственные
task handles/workspaces каждого графа, поскольку оба имеют physical size 8
и одинаковые layer names. Без этого registry второго capture перезаписал бы
handles первого. Capture/replay и update/event protocol остаются исходными;
глобальных runtime-подмен нет. На NPU этот путь требует проверки.

В `details.execution_lanes` поля `tokens` и `query_tokens` показывают physical
размер и число полезных позиций. `--model-timing` печатает physical input count:
при depth 7 вместо последовательности `8,1,1,1,1,1,1` получится семь `8`.
Это контроль различий в вычислительной работе, а не отдельное доказательство
Python overhead.

Флаг можно добавить к существующему direct replay с `--replay-with-mtp`,
оставив прежнюю трассу Omni и выбрав новый output. Summary показывает
`mtp_pad_to_8`. Повторная запись Omni не нужна. Сам target replay без выполнения
MTP флаг не затрагивает. Для matrix используйте `engines: ["direct"]`, если
передаёте `--mtp-pad-to-8` в общем списке options.

## Диагностика первого MTP-окна между повторами Omni

Если greedy-повторы расходятся только с MTP, добавьте к прежней команде
`--trace-mtp-boundary`, сохраните `--depth 7 --temperature 0 --repeats 3`
и укажите новый `--output`. Настройки prefix cache, HIF8, размеров блоков,
chunked prefill, async scheduling и compile/ACL остаются прежними.
Диагностика разрешена только для Omni с положительной depth и temperature=0;
она несовместима с `--trace-target`, replay и profiling.

После прогрева на каждом worker устанавливаются штатные hooks у target
`logits_processor` и draft `model.logits_processor`. Они вызываются вне
captured model forward, поэтому фиксируют и ACL replay. После добавления
измеряемого запроса, до первого `engine.step()`, hooks активируются для
внутреннего ID, возвращённого `LLMEngine.add_request()`. Внешний ID может
отличаться: vLLM добавляет случайный суффикс. Hooks сохраняют:

- `initial_target`: последний prefill-вызов, который завершает prompt;
  logits промежуточных prefill-чанков пропускаются;
- `mtp_0` … `mtp_6` при depth 7: первый proposal;
- `first_verification`: первый следующий вызов основной модели.

Снимки содержат вход LM head, полные logits, argmax и accepted/computed
counts. Для target также сохраняются фактические input IDs и positions;
512 prefill-токенов не путаются с единственной строкой logits.
Вход LM head draft уже прошёл shared-head normalization.

Сохраняются выбранные состояния первого MoME-слоя target и первого MoME-слоя
draft. Принадлежность к draft определяется по штатному списку
`runner.drafter.attn_layer_names`, а не по префиксу имени слоя.
Логические блоки выбираются около начала последнего prefill-чанка
и конца prompt. Для текущего случая 7680 + 512 токенов при block size 128
это блоки 59, 63, 64. Физические IDs проверяются по GPU block table и
выделенным блокам запроса; null/PAD и недоступные блоки не читаются.
Постоянные metadata buffers builder читаются только для соответствующего
decode-вызова при full graphs и допустимой batch capacity. На prefill
и в eager они помечаются как неактуальные и не читаются.

В `mtp-boundary/rank-N/record-XXXXX.pt` записываются CPU tensors и обычные
контейнеры, пригодные для `torch.load(weights_only=True)`. Это снимки только
первого окна, а не всей генерации. JSON рядом и строка `[mtp-boundary]`
сравнивают каждый повтор с первым запросом того же prompt:

- `first_compute_difference`: первое отличие входа LM head, logits или argmax;
- `first_input_difference`: первое отличие target IDs/positions или GPU accepted counts;
- `first_state_difference`: первое отличие снимков MoME/metadata;
- `target_state_changes_during_mtp`: изменения выбранных target-состояний
  между prefill и drafting при неизменном физическом размещении блока.

Номера физических блоков между повторами не сравниваются. Выделенный блок
ещё не обязательно содержит актуальное состояние: старые байты будущего
блока 64 или draft-блока до его первого вызова не доказывают ошибку.
Поэтому различия вычисленных выходов и сырых состояний разделены.
`reference_record=null` обозначает первый запрос данного prompt.
Сводка всех ranks сохраняется в `mtp-boundary/report.json` и
`details.mtp_boundary_reports`. Неполное окно вызывает явную ошибку;
hooks снимаются при завершении и при исключении.
В ошибке неполного окна выводятся ожидаемый внутренний ID, число вызовов
hooks и IDs первого наблюдавшегося batch для target и MTP.

Metadata первой target-проверки из уже сохранённых снимков можно вывести
без новой генерации (нужен установленный torch):

```bash
python scripts/inspect_mtp_boundary.py <output> --rank 0 --records 0 1
```

Для сравнения содержимого выбранного начального блока MoME и предыдущего
блока добавьте `--compare-states`. Строка `[state-comparison]` сравнивает
последние MTP-снимки **до** первой target-проверки: после проверки состояние
уже перезаписано. Для каждого компонента (0=q, 1=kv, 2=o) выводятся
`row_max_abs_diff` по всем строкам и `selected_as_initial` для каждого повтора.
Число строк истории вычисляется как число строк состояния минус depth;
`accepted_history_rows` соответствует Python reference-формуле
`accepted - 1 + arange(kernel_width - 1)`, а `committed_tail_rows` показывает
хвост состояния. При accepted=1 и двух строках истории это строки 0 и 1.
Это интерпретация по reference-коду: чтение этих строк фактическим fused
NPU-оператором ещё не проверено. Метаданные берутся из снимка первой проверки;
неактуальные persistent buffers явно отклоняются.

CPU copies синхронизируют вычисления и могут скрыть гонку. Результат
помечается `diagnostic` и не используется для speedup. Обязательно проверяйте
`[repeat-check]`: сохранение расхождения со снимками и его исчезновение
имеют разную диагностическую ценность. Методы модели/runner, параметры
численной точности и содержимое кеша инструмент не изменяет.

### Корректировка первого target decode после prefill

При context 8192, MoME block size 128 и depth 7 диагностика показала:
prefill и первый MTP proposal совпадают, но первая target-проверка расходится.
Metadata выбирает initial block 64 и accepted=1. Перед проверкой блок 63
совпадает полностью, а строки 0 и 1 блока 64 имеют огромные различия
(до 1.36e35). Это указывает на выбор неактуального состояния после prefill.
Причинность нужно подтвердить повтором с корректировкой.

Это гипотеза о builder, а не доказанное отличие от рабочего HTTP-сервера.
Корректировка реализована в `vllm-bench`; checkout и установленный пакет
OmniInfer менять не нужно. Для проверки добавьте к прежней команде:

```bash
--fix-mome-prefill-state
```

Флаг поддерживается с `--engine omni` и `--engine direct`, depth 3/7 и
`use_mome=true`. Без него используется прежний путь. Численная политика входит
в `comparison_sha256`: при сравнении скорости обоих движков флаг должен совпадать.
`--omni-multiprocessing` можно использовать вместе с адаптером; ограничение
boundary trace на in-process запуск сохраняется.

Локальный `MomePrefillMetadataBuilder` наследует исходный builder. Для non-PD
**target** при `computed == prompt` accepted нормализуется до `depth + 1`:
при context 8192, block 128, depth 7 получаются initial block 63 и accepted=8,
то есть хвост prefill-состояния. Draft offsets, последующий target decode и PD
сохраняют прежнюю семантику. Счётчики sampler/runner и содержимое кеша не меняются.

В `direct` класс выбирается при создании MoME metadata lane. В `omni` явно
выбирается локальный подкласс NPUWorker: после штатной инициализации KV-кеша,
до прогрева/захвата графов, он создаёт локальные MoME builders и обновляет
ссылки штатного MTP proposer через его initializer. Остальные attention
builders остаются штатными. Глобальные методы и backend registry не подменяются.
Рабочий процесс сообщает число установленных builders; launcher проверяет его
и сохраняет `effective.mome_prefill_state_fix`. В Omni-логе появляется
`[execution] ... mome_prefill_state_fix=true`.

Для NPU-проверки нужен новый процесс и новый output-каталог с прежними
параметрами и этим флагом. В диагностическом запуске с `--trace-mtp-boundary`
ожидаются `repeat-check.matches=true`, initial block 63 и accepted=8 в metadata
первой target-проверки. Исполнение на NPU ещё требует проверки; совпадение
с HTTP-сервером не гарантируется одной корректировкой. После проверки чисел
TPOT измеряется отдельно без trace.

`patches/omniinfer/mome-target-prefill-state.patch` оставлен как контрольный
вариант для CPU-тестов во временной копии. Для запуска инструмента применять
его к OmniInfer не требуется.

## Численные CPU-тесты MoME без модели

Для проверки перехода prefill → первая target verification добавлены
`scripts/check_mome_cpu.py` и `tests/test_mome_boundary_torch_cpu.py`.
Нужны Python 3.10+, CPU PyTorch, git и исходный checkout OmniInfer;
checkpoint, vLLM, torch_npu и NPU не требуются. По умолчанию используется
соседняя папка `../omniinfer`; другой путь задаётся через `--omni-source`.

Можно использовать отдельное окружение:

```bash
python3 -m venv /tmp/pangu-mome-cpu
/tmp/pangu-mome-cpu/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/pangu-mome-cpu/bin/python scripts/check_mome_cpu.py \
  --omni-source ../omniinfer --threads 1 --output results/mome-cpu.json
```

Если подходящий torch уже установлен, достаточно последней команды с
соответствующим Python. Runner отключает
[автозагрузку device extensions](https://docs.pytorch.org/tutorials/unstable/python_extension_autoload.html)
перед импортом torch и явно создаёт тензоры на CPU. Исходники Omni загружаются
без записи `__pycache__`; кандидат патча применяется только к временной копии.
Исходный checkout не изменяется.

Проверяются реальные `build`, `build_for_drafting` и block-index arithmetic
из `omni/attention/backends/mome.py`, а также `causal_conv1d_ref` из
`omni/layers/ops/causal_conv1d.py`. Заглушки заменяют только импорты внешних
framework/hardware dependencies. Сам torch не заменяется; исправленный путь
исполняет настоящий локальный subclass поверх исходных методов builder.
Конструктор builder не проверяется: тест создаёт маленькие CPU buffers
и batch-one metadata. Split-helper поддерживает упорядоченные CPU fixtures;
SP/TP/HCCL протоколы этим тестом не проверяются.

Десять тестов охватывают:

- context 8192, depth 3/7, APC on/off, float32/bfloat16;
- выбор блока 64 / accepted=1 в текущем исходнике и блока 63 / accepted=depth+1
  в локальном адаптере;
- два разных содержимых повторно выделенной будущей страницы при одинаковом
  актуальном prefill-tail;
- сравнение с исходной reference convolution на непрерывном prompt + query;
- правильный target-only control и chunked prefill 2048/2048/2048/1536/512;
- соседние с границей блока длины 8184/8191/8192/8193/8320;
- сохранение draft/последующего target/PD metadata после кандидата правки;
- ветку обновления постоянных buffers на CPU, без захвата или replay графа;
- совпадение metadata локального адаптера с контрольной временной правкой
  исходника для target/draft, APC, PD и persistent buffers.

Выбор истории из APC-страницы выполняет явный тестовый адаптер по публичному
контракту CANN. Сама convolution — исходная PyTorch reference. Разный мусор
страницы моделируется известными sentinel values; реальный allocator vLLM
и фактический fused NPU-оператор не исполняются. Поэтому тест проверяет
предполагаемый механизм, а не воспроизводит полный сервер или модель.

JSON содержит версии, хеши исходного, локального и временно исправленного
builder и reference,
результаты отдельных контрпримеров, время и peak RSS процесса. Успешные
reproduction assertions означают, что воспроизведена зависимость текущего
исходника от неактуального состояния. Они не означают, что текущий builder
корректен. Candidate-fix assertions требуют точного совпадения с reference.
Тесты закрепляют поведение исходной ревизии 4cd855b2; при другой или уже
исправленной версии нужно смотреть хеши, ошибки применения патча и ожидания.

Если torch отсутствует, обычный unittest discovery пропускает эти десять
тестов. Явный runner возвращает exit code 2 и не создаёт успешный отчёт.
Код 0 возможен только при выполненных тестах без failures/errors/skips.
В текущем локальном окружении torch отсутствует, загрузка пакета недоступна
из-за DNS. Пользователь выполнил прежние девять численных тестов временной
правки: все прошли. Обновлённая suite проверяет уже локальный runtime-класс
и требует повторного численного запуска. Локально проверены инфраструктура
изоляции импортов, запрет записи bytecode, применение патча во временной
копии и корректный отказ runner при отсутствии torch. Для конфигурации
по умолчанию достаточно одного CPU-потока и нескольких гигабайт RAM;
фактический RSS после численного запуска попадёт в JSON.

## Диагностика расхождений target: запись Omni и replay в direct

Этот режим сравнивает основную модель на одинаковых входных окнах. Он нужен,
когда самостоятельные генерации расходятся: после первого отличающегося токена
обычное сравнение logits уже смешивает разные входы и состояния.

Используйте `eager`, `temperature=0` и `repeats=1`. Начните с 32 выходных токенов.
Checkpoint, operator/additional config, prompt, seed, TP, depth, max-new-tokens
и остальные параметры workload должны совпадать в обеих командах. В частности,
при replay остаётся depth 7: замена на `--no-mtp` изменила бы target state spec.

Пример из корня checkout, с настроенным абсолютным `PYTHONPATH`:

```bash
python -m vllm_bench.pangu_infer.run \
  --engine omni --model /models/pangu-ultra-12-mtp \
  --operator-config /configs/pangu-tp2.json \
  --tp 2 --depth 7 --mode eager --temperature 0 \
  --context-length 8192 --max-new-tokens 32 --repeats 1 \
  --trace-target --trace-layers all \
  --output /results/omni-target-trace --execute

python -m vllm_bench.pangu_infer.run \
  --engine direct --model /models/pangu-ultra-12-mtp \
  --operator-config /configs/pangu-tp2.json \
  --tp 2 --depth 7 --mode eager --temperature 0 \
  --context-length 8192 --max-new-tokens 32 --repeats 1 \
  --replay-target /results/omni-target-trace --trace-layers all \
  --output /results/direct-target-replay --execute
```

Если исходная команда использует `--prompt-ids`, `--additional-config` или другой
seed, передайте их в обе команды. Без `--execute` работает проверка плана.
Режимы trace/replay запускаются отдельно, через `run`, а не через `matrix`.

Omni прогревается обычным способом. Затем worker RPC устанавливает обычные
PyTorch forward hooks на target, его logits processor и выбранные decoder layers
на всех TP ranks. Записывается один измеряемый запрос, включая его prefill;
после него hooks удаляются. Файлы OmniInfer и методы модели не изменяются.

Direct загружает те же модели с той же speculative-конфигурацией, обнуляет
собственные кеши один раз и воспроизводит только target-вызовы. Входные окна,
позиции и значения MoME acceptance берутся из записи Omni. Собственные draft
предложения и sampling не выполняются. Кеши между target-вызовами сохраняются;
состояние из Omni в direct не копируется. Поэтому последующее расхождение
состояния остаётся видимым. Warmup генерации direct в replay не выполняются.

Если target replay совпал, следующий контроль — добавить к той же direct-команде
`--replay-with-mtp` и выбрать новый output. Повторная запись Omni не нужна.
В этом режиме перед каждым decode target-вызовом выполняется обычный
`DirectEngine.propose()`, включая первый proposal после prefill. Он получает
hidden states предыдущего target-вызова и anchor из следующего окна Omni.
Число committed-позиций и индекс sampling определяются по продвижению курсора
в записи; после prefill используется последняя позиция prompt. Собственные
предложения сохраняются для сравнения, но target продолжает получать исходные
окна и target acceptance из Omni. Это проверяет влияние исполнения MTP на
target-состояние и общие буферы, сохраняя одинаковые target-входы.

В `rank-N/draft-proposals.json` сохраняются предложения direct и соответствующая
часть окна Omni. Summary содержит `replay_with_mtp` и
`first_draft_proposal_difference` с номером target-вызова, rank, индексом draft
и двумя token IDs. Если исходное окно укорочено, сравнивается только его draft
префикс; `compared_tokens` показывает число сопоставленных предложений.
Несовпадение draft-предложений само по себе не означает ошибку: их проверяет
основная модель. `--replay-with-mtp` требует положительный depth.

Артефакты:

- `target-trace/manifest.json`: workload, effective config, версии и оборудование.
- `target-trace/rank-N/index.json`: число target-вызовов.
- `target-trace/rank-N/call-XXXXX.pt`: CPU tensors — входы, позиции, полные logits
  и выбранные логические поля attention metadata, отдельно для prefill/decode.
- `trace_comparison.json` в output direct: автоматическое сравнение всех ranks
  и вызовов, первая разница входов, metadata, logits и argmax. В stdout/runtime.log
  выводится краткий summary с top-5, top-1 margin и logits конкурирующего токена
  для первого расхождения argmax. `first_metadata_detail` показывает первые
  четыре отличающихся поля metadata с путями и значениями reference/candidate;
  полные ограниченные списки находятся в `calls[].metadata_differences`.

`--trace-layers all` дополнительно сохраняет tensor-выходы всех decoder layers
и активные состояния трёх MoME convolutions до/после target forward. Для prefill
выходы слоёв ограничены последней локальной строкой, для decode сохраняются
все строки. В полном prompt KV cache не копируется. Можно выбрать, например,
`--trace-layers 0,5,11`; без опции записываются только входы, metadata и logits.
Replay может сравнивать подмножество уже записанных слоёв. Для расширения набора
нужна новая запись Omni.

Сравнение metadata исключает физические номера блоков и адреса: аллокаторы
движков различаются. В отчёт входят только поля, явно перечисленные в
`target_trace.METADATA_FIELDS`; равенство этого набора не означает равенство всех
внутренних полей backend. MoME states сравниваются по активному блоку каждого
слоя независимо от его физического номера. Содержимое состояния перед prefill
может различаться после прогрева; при `num_computed_tokens=0` оно не обязательно
участвует в вычислении. Его отличие само по себе не доказывает ошибку.

Отчёт содержит exact equality, max absolute error, RMSE и `allclose` с явно
указанными `rtol=1e-3`, `atol=1e-3`. Эти пороги диагностические и не устанавливают
допустимую погрешность модели. Малое отличие logits само по себе не доказывает
ошибку. При смене argmax сначала смотрите top-1 margin, metadata и состояние
первого расходящегося слоя. Replay изолирует target-путь; он не проверяет
правильность самостоятельной генерации MTP-предложений.

Файлы `.pt` загружаются с `weights_only=True`. Их можно сравнить повторно на
машине с CPU PyTorch, без NPU и OmniInfer:

```bash
python -m vllm_bench.pangu_infer.trace_compare \
  --reference /results/omni-target-trace \
  --candidate /results/direct-target-replay \
  --output /results/target-comparison.json
```

CLI возвращает 1 при отличии входов, логических metadata или argmax. При
отличии только численных значений с одинаковым argmax — 0; точные метрики
сохраняются в отчёте. Source-запуск помечается `diagnostic`; replay создаёт
`trace_comparison.json`, а не benchmark `result.json`. Копирование тензоров на
CPU и запись файлов синхронизируют вычисления, поэтому время этих запусков
не используется для сравнения производительности. Полные logits и состояния
выбранных слоёв могут занимать значительное место — размер растёт с числом
target-вызовов, TP ranks и выбранных слоёв.

## Режимы исполнения

| `--mode` | Прямой runtime | Offline baseline |
| --- | --- | --- |
| `eager` | Без model compilation и ACL capture | Compilation=NONE, graphs=NONE |
| `compile` | Прямой `torch.compile(fullgraph=True, dynamic=False)` с NPU backend | Отклонён: штатный Omni decorator обходит compile вне FULL graph mode |
| `acl` | ACLGraphWrapper вокруг прямого model step | Compilation=NONE, FULL graphs; проверяется факт capture |
| `compile-acl` | Compile, затем ACLGraphWrapper | Штатная compilation + FULL graphs |

Prefill идёт eager. Начальное заполнение MTP входит в prefill, не в decode.
Инициализация и выполнение идут под `torch.no_grad()`: веса и runtime buffers
сохраняют version counters, необходимые NPU post-processing и компиляции.
`torch.inference_mode()` для этого пути не используется.
Модель, LM head и выбор нужных logits включены в прямой model step; sampling,
metadata preparation и фиксация принятого префикса находятся снаружи него.
У direct остаётся один host read accepted count на speculative cycle и
необходимая TP-координация; это намеренно простой Python engine.

В compile-режимах напрямую используется `NpuGraphExAdaptor` OmniInfer;
компиляции считаются собственным backend callback. Его настройки берутся
из того же additional config, что и в baseline. По умолчанию явно задаётся
`npugraph_ex_config.enable=true`, чтобы backend не оказался FX-eager bypass;
явное `false` в compile-режиме отклоняется.
Backend может иметь собственные механизмы graph execution: для интерпретации
compile-only отдельно смотрите trace и настройки установленного npugraph_ex.
Указание compile-only означает отсутствие внешнего ACLGraphWrapper.

`enforce_eager` не включается ни в одном режиме, поскольку он дополнительно
меняет операторные оптимизации OmniInfer. Настройки streams/fusion берутся
из operator config. Эффективные настройки сохраняются, и compare отклоняет
пару с различиями операторов, parallel config, block sizes, hardware или версий.

ACL capture выполняется после compile/прогрева. Inputs, positions, lengths,
slot mappings и MoME buffers имеют постоянные адреса. Оригинальный
ACLGraphWrapper оставлен для обновления attention graph tasks с растущими
длинами. Простой replay без этого обновления был бы некорректен.
Совместимые слои с одинаковыми backend и cache spec используют общую metadata;
для MTP она готовится только для активного predictor.

Перед измерениями выполняются полные warmup requests, затем caches и seed
сбрасываются. Если measured requests вызывают новую compilation/capture,
результат не принимается. Для direct также проверяется факт replay в warmup.

## Проверка корректности и сравнение

Сначала запустите direct с `--depth 0 --temperature 0 --mode eager`. Это
target-only reference. Затем direct с depth 3 и 7, а также обычный OmniInfer
на тех же входах и при том же TP:

```bash
python -m vllm_bench.pangu_infer.compare \
  --reference /results/target-only \
  --candidate /results/direct-tp2-mtp3-eager --correctness

python -m vllm_bench.pangu_infer.compare \
  --reference /results/omni-tp2-mtp3-eager \
  --candidate /results/direct-tp2-mtp3-eager
```

Для correctness допускаются разные depth/mode. Для speedup должны совпадать
checkpoint, исходные конфиги, prompt, TP, depth, mode и sampling. Greedy
расхождение возвращает ненулевой exit code, показывает первую позицию и
не публикует speedup. Проверка target-only нужна и для baseline: одинаковая
ошибка в двух speculative engines не должна считаться доказательством.

Sampling использует p/q rejection, residual `max(p-q, 0)` и target bonus при
полном принятии. Температура, top-k/top-p применяются к обоим распределениям.
Random draws выполняет rank 0 с broadcast решений. Greedy использует argmax
на одинаковых all-gather logits каждого rank, без дополнительных broadcast
и без создания vocabulary-sized one-hot probabilities. Одинаковый seed не
предполагает одинаковые sampled sequences между движками: порядок random draws
различается. Для sampling отчёт сравнивает throughput, но явно не утверждает
совпадение последовательностей или одинаковую длину принятого префикса.

## Матрица

Создайте JSON, пути внутри которого отсчитываются относительно этого файла:

```json
{
  "models": ["/models/pangu-ultra", "/models/pangu-ultra-12-mtp"],
  "operator_configs": {
    "1": "/configs/pangu-tp1.json",
    "2": "/configs/pangu-tp2.json",
    "4": "/configs/pangu-tp4.json",
    "8": "/configs/pangu-tp8.json"
  },
  "output": "/results/pangu-direct-matrix",
  "options": ["--temperature", "0", "--context-length", "8192"]
}
```

```bash
python -m vllm_bench.pangu_infer.matrix /configs/matrix.json
python -m vllm_bench.pangu_infer.matrix /configs/matrix.json --execute
```

Defaults: TP1/2/4/8 × depth3/7 × четыре direct modes и три baseline modes,
для каждого checkpoint. Для меньшего smoke test можно задать `tp`, `depths`,
`modes`, `engines` массивами. `depths: [0, 3, 7]` добавляет target-only runs;
их correctness comparison запускается явно. Процессы идут последовательно,
ошибки/OOM записываются, оставшиеся комбинации продолжаются. Успешные direct/omni
пары автоматически сравниваются. `matrix_results.json` обновляется после
каждого запуска; неподходящие пары не получают оценки overhead.

Для всей матрицы без MTP добавьте `--no-mtp` к команде `pangu_infer.matrix`:
это заменяет JSON `depths` на `[0]`, сохраняя выбранные TP, engines и modes.

## Метрики и профили

В каталоге запуска:

- `launch_plan.json`: конфиги, token IDs, sampling, выбранные predictor layers,
  идентификатор сравниваемой нагрузки, hash index/manifest и size/mtime shards;
- `runtime.log`, а при ошибке — `failure.json`;
- `result.json`: generated IDs каждого повтора, prefill/decode wall time,
  TPOT, tokens/s, acceptance и фактические настройки;
- `summary.csv`: компактные сводные показатели.

Большие tensor payloads при запуске повторно не хешируются. Для побайтовой
проверки сокращённого checkpoint используйте существующий verifier с hashes;
записанные size/mtime выявляют обычные изменения файлов между запусками.

Основной decode TPOT — wall time завершённого decode / число выданных токенов
после первого токена. Для direct берётся самый медленный TP-rank. Синхронизации
стоят на границах целого участка, без barrier между этапами. У baseline
по умолчанию выключено async scheduling; отдельный engine-core процесс
выключен в обоих режимах, TP workers остаются штатными. При синхронном
scheduling синхронизация workers после первого output исключена из decode,
завершающая синхронизация через RPC включена в его wall time. С флагом
`--async-scheduling` замер идёт между получением первого и последнего output
без промежуточной синхронизации; финальное ожидание NPU исключено из TPOT.
Контекст начинается с 8192 и растёт, это не fixed-context replay benchmark.
Итоговые времена и статистика `--model-timing` сводятся через существующую
TP CPU-группу (Gloo), вне измеряемого decode. Для них не создаётся HCCL
communicator на default process group. Broadcast при случайном sampling
использует существующую TP device group.
Последний speculative cycle может вычислить больше токенов, чем выдаётся
пользователю; замер включает работу до завершения соответствующего output.
В async-режиме оставшаяся работа после последнего output завершается вне замера.

Acceptance direct берётся из verification до терминального обрезания;
baseline восстанавливает её по приращениям outputs, поэтому последний
обрезанный цикл даёт нижнюю границу. Эти поля нельзя считать побитово
сопоставимым внутренним счётчиком.

Для диагностического запуска добавьте `--profile` и новый output. Direct
пишет CPU/NPU trace с диапазонами prefill, draft, target и sample/commit;
baseline включает штатный worker profiler. Такие результаты помечаются
`diagnostic`, и compare не использует их для speedup. Логирование внутри
измеряемого decode не выполняется.

### Диагностика памяти direct

Добавьте `--memory-diagnostics` и новый `--output`. Флаг выводит строки
`[memory]` на каждом TP rank после setup, загрузки весов и выделения кеша,
до и после первого вызова каждой execution lane, компиляции и ACL wrapper.
Они выводятся на экран и сохраняются в `runtime.log`. Поля `model`, `head`,
`tokens`, `query_tokens`, `start` и `draft_step` идентифицируют вызов.
Размеры `allocated_mib`, `reserved_mib`, `reserved_unallocated_mib` и
`free_mib` отделяют активные аллокации от резерва allocator и свободной памяти
устройства. При ошибке `model_failed.phase` указывает последний этап;
`acl_wrapper` включает внутренний static-kernel warmup и сам capture.
Исходное исключение сохраняется. Диагностика не добавляет вызовов модели
или синхронизаций; счётчики памяти читаются без дополнительной синхронизации.
Результат помечается `diagnostic` и не используется для speedup.

Строки `[execution]` разделяют первый вызов compiled callable и ожидание NPU:
`graph_warmup_forward_begin` → `graph_warmup_forward_end` →
`graph_warmup_sync_begin` → `graph_warmup_sync_end`. Они содержат монотонное
время, rank, модель, размеры входа, фактическое имя TP-группы и её ranks;
`*_end` также сообщает `elapsed_seconds`. Эти отметки не обращаются к NPU API.
Интервал forward включает создание backend и ленивую работу первого вызова:
`compile_end` сообщает только о возврате `backend.compile`, статические ядра
могут компилироваться позже. Интервал sync показывает ожидание уже поставленных
на устройство операций. Внешний ACL capture начинается после этих этапов.
`warmup_end`, `measured_begin` и `measured_end` отделяют полный прогрев от
измеряемой генерации на каждом rank.

Полный список `stage` приведён ниже. Каждый rank пишет собственные строки,
поэтому в общем логе этапы разных rank могут перемешиваться.

| Stage | Префикс | Описание |
|---|---|---|
| `setup_done` | `[memory]` | Конфигурация и распределённые группы созданы; загрузка модели ещё не началась. |
| `weights_loaded` | `[memory]` | Веса загружены и подготовлены. |
| `caches_allocated` | `[memory]` | Кеши и runtime-буферы выделены; выполнены синхронизация и освобождение неиспользуемого резерва allocator. |
| `model_begin` | `[memory]` | Начало первого вызова конкретного варианта исполнения: target/MTP, размер входа, head, prefill/decode. |
| `graph_warmup_begin` | `[memory]` | Начало подготовки модели перед внешним ACL capture. |
| `graph_warmup_forward_begin` | `[execution]` | Начало прямого вызова модели до внешнего ACL capture. В `compile-acl` вызов проходит через `torch.compile`. |
| `compile_begin` | `[memory]` | Вход в `backend.compile` после трассировки Dynamo. |
| `compile_end` | `[memory]` | Backend вернул callable; ленивая компиляция и первый запуск ещё могут продолжаться. |
| `graph_warmup_forward_end` | `[execution]` | Вызов модели вернул управление Python. `elapsed_seconds` включает компиляцию, если она происходила внутри вызова. |
| `graph_warmup_sync_begin` | `[execution]` | Начало ожидания завершения операций NPU после вызова. |
| `graph_warmup_sync_end` | `[execution]` | Синхронизация завершена. `elapsed_seconds` — время ожидания. |
| `graph_warmup_end` | `[memory]` | Подготовительный вызов и синхронизация завершены. |
| `acl_wrapper_begin` | `[memory]` | Начало первого вызова внешнего ACL wrapper: возможный внутренний прогрев и capture. |
| `acl_wrapper_end` | `[memory]` | Внешний wrapper завершил capture; последующая синхронизация тоже завершена. |
| `model_end` | `[memory]` | Первый вызов данного варианта исполнения успешно завершён. |
| `model_failed` | `[memory]` | Вызов завершился исключением; `phase` показывает последний отмеченный этап. Может появиться и после прогрева. |
| `warmup_end` | `[execution]` | Все прогревочные генерации на этом rank завершены. `elapsed_seconds` — полный прогрев. |
| `measured_begin` | `[execution]` | Начало блока измеряемых запросов. |
| `measured_end` | `[execution]` | Измеряемые запросы и включённая обработка model timing завершены; итоговое сохранение результатов ещё впереди. |

Для первого decode-вызова в `compile-acl` обычный порядок следующий:

```text
model_begin
  graph_warmup_begin
    graph_warmup_forward_begin
      compile_begin
      compile_end
    graph_warmup_forward_end
    graph_warmup_sync_begin
    graph_warmup_sync_end
  graph_warmup_end
  acl_wrapper_begin
  acl_wrapper_end
model_end
```

Последовательность повторяется для каждого нового варианта исполнения.
Обычные replay-вызовы не печатают её заново. Prefill выполняется в eager,
поэтому у него обычно только `model_begin` / `model_end`.

`gpu_memory_utilization` в direct передаётся в конфигурацию Omni, но не
ограничивает его собственное выделение кеша под `max_model_len`.

При ошибке packing direct выводит `[weight-pack]`: target/MTP scope и первые
четыре MXFP8 linear с нечётным количеством 32-элементных scale-блоков.
Сохраняются prefix, ключ layer config, фактический TP и формы weight/scales;
исходное исключение не заменяется. Эта диагностика не меняет упаковку весов
и не добавляет padding.

### Время вызовов target и MTP

Для direct добавьте `--model-timing` и новый `--output`. Флаг по умолчанию
выключен; для `--engine omni` он отклоняется при планировании. Для матрицы
укажите `"engines": ["direct"]` и `"options": ["--model-timing"]`.

Измеряется существующий model step: forward + LM head (`compute_logits`),
включая ACL replay/update при включённых графах. Копирование входов, построение
metadata и sampling находятся вне границы этого замера. На каждом TP-rank
выполняется `torch.npu.synchronize()` перед началом таймера и после model step;
время включает Python dispatch и завершение NPU-работы.

Учитываются только вызовы decode измеряемых запросов. Warmup, target prefill
и всё первоначальное заполнение MTP исключены, в том числе однотокенные MTP
продолжения начального prefill. При `--no-mtp` измеряется только target.

После завершения decode каждого запроса длительности соответствующих вызовов
агрегируются максимумом по TP-rank. До агрегации проверяется совпадение числа
записей; коммуникации для этой агрегации и печать не входят в decode wall time.
Только rank 0 выводит строки, которые launcher показывает на экране и сохраняет
в `runtime.log`:

```text
[model-time] repeat=1 cycle=0 model=target input_tokens=8 wall_ms_max_tp=12.345
[model-time] repeat=1 cycle=1 model=mtp step=0 head=0 input_tokens=8 wall_ms_max_tp=1.234
```

`repeat` начинается с 1, `cycle`, `step` и `head` — с 0. `input_tokens` —
фактическая длина входного вектора token IDs, включая все позиции verification,
а не длина KV-контекста и не число принятых предложений. Для target обычно
это 1 без MTP, 4 при depth 3 и 8 при depth 7. Для MTP отдельно показаны шаг
предложения и используемый predictor head; при depth 7 первый шаг decode
обычно получает 8 токенов, следующие — по одному.

В конце печатается `[model-time-summary]`: количество вызовов и median/min/max
в миллисекундах, сгруппированные по модели, input_tokens и head. Каждое значение
в этой статистике уже является максимумом по TP. Записи сохраняются в
`result.json` в `records[].model_timings`, сводка — в `model_timing_summary`.
Если decode отсутствует из-за начального EOS, лог содержит `no_decode_calls`.

Синхронизации меняют перекрытие операций, поэтому `--model-timing` помечает
результат как `diagnostic`. Его TPOT не является обычным benchmark; сравнение
speedup запрещено. `compare --correctness` допускает diagnostic результаты
для проверки совпадения greedy token IDs с обычным запуском. Без флага
дополнительных синхронизаций, агрегации и вывода замеров нет.

Разность wall time — разность двух путей исполнения, не готовая оценка чистого
Python overhead. Проверяйте одинаковые operator configs, kernel workload,
acceptance и trace: CPU launch gaps, preparation, sampling/commit и graph task
updates. Нельзя вычитать сумму длительностей NPU kernels из wall time, поскольку
операции и потоки перекрываются.

## Проверки на сервере

1. Сначала 12+MTP, TP1 или минимальный помещающийся TP, eager greedy, 32 output
   tokens; сравнить depth0/3/7 с target-only и stock OmniInfer.
2. Повторить при TP2/4/8; отдельно проверить prompt lengths на границах cache
   blocks, включая 8191/8192/8193. `--context-length` меняет synthetic input.
3. На тестовых prompts добиться первого/частичного rejection и полного
   принятия; проверить логику MoME по совпадению с target-only после нескольких
   циклов и сброса caches между запросами. `accepted_drafts` хранит каждый цикл.
4. Проверить compile, ACL и совместный режим против eager; в measured requests
   не должно появляться новых compilation/capture. Проверить trace replay.
5. Повторить sampling; выполнить optional CPU torch tests ниже. При расхождении
   greedy или ошибке API остановить интерпретацию performance до исправления.
6. Затем полная модель и матрица 8192/256, не менее трёх повторов, без profiler;
   отдельные profiler runs использовать для атрибуции overhead.
7. Проверить `--model-timing` при TP1/TP8, без MTP и с depth 3/7, в eager,
   compile и ACL-режимах. Сверить target `input_tokens=1/4/8`, исключение warmup
   и начального MTP, отсутствие новых compilation/capture. Через
   `compare --correctness` проверить greedy token IDs против запуска без флага.

```bash
PYTHONPATH="$PWD/src" python -m unittest discover -s tests
```

Базовые тесты не требуют ML-пакетов. При установленном CPU torch дополнительно
проверяются реальный tensor sampler, все позиции rejection, top-k/top-p и
статистическое распределение первого emitted token. CPU-проверки не подтверждают
packing весов, stateful MoME kernels, multi-rank collectives или ACL correctness.
