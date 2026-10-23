# Профилирование Pangu 505B на первых 12 слоях

Набор предназначен для Pangu 505B с 50 основными слоями, 3 MTP, 384 экспертами и top-8. Поддерживаются BF16 и MXFP8 checkpoint с архитектурой `OpenPanguV2ForCausalLM` или `PanguUltraMoEForCausalLM`; архитектура источника сохраняется. Приложенный reference config описывает исходный вариант V2/BF16, а quantization metadata берётся из настоящего checkpoint. Набор сохраняет настоящие слои 0–11 и позволяет запускать нагрузку и собирать NPU traces без изменения исходников OmniInfer. Целевой интерфейс сервера — vLLM 0.25.1 и просмотренный OmniInfer `4cd855b2`.

Подробное описание архитектуры, конкретного HF config и режимов бенчмарков: [Pangu 505B на русском](./PANGU_505B_RU.md).

Подготовка checkpoint, клиент бенчмарка, проверка файлов и анализ trace требуют только Python 3.10+. Запуск сервера требует уже установленного рабочего окружения OmniInfer/vLLM/torch_npu на NPU-машине. Скрипты ничего не устанавливают.

## Файлы

| Файл | Назначение |
| --- | --- |
| `prepare_checkpoint.py` | Потоковое копирование выбранных safetensors без загрузки модели |
| `verify_checkpoint.py` | Проверка структуры, конфигурации и необязательных SHA256 тензоров |
| `reference_config-hf.json` | Копия предоставленного исходного HF config |
| `config-12layers.json` | Предварительный сокращённый config для просмотра; экстрактор также создаёт `OUTPUT/config.json` |
| `serve.py` | План запуска или запуск постоянного сервера с NPU profiler |
| `benchmark.py` | Прогрев, замеры нагрузки и отдельные запросы под profiler |
| `suite.json` | Матрица 1K/8K/32K и concurrency 1/8, prefill и generation |
| `instrumentation/sitecustomize.py` | Необязательные CPU ranges с номерами слоёв |
| `analyze_trace.py` | Сводка CPU ranges из одного trace |
| `test_cpu.py` | CPU-проверки экстрактора и обработки потоковых ответов |
| `bundle_manifest.json` | SHA256 файлов переносимого набора |

Переносить нужно всю эту папку. Каталог OmniInfer рядом со скриптами не требуется, но его установленный Python-пакет и нужные runtime libraries должны быть доступны серверу.

Получить набор на сервере через GitHub:

```bash
git clone https://github.com/nekame/vllm-bench.git
cd vllm-bench
```

## Подготовка checkpoint

Источник должен быть локальным HF checkpoint в safetensors: один файл или shards с `model.safetensors.index.json`. Поддерживаются неквантованные BF16-веса и MXFP8 с `quantization_config.quant_method="mxfp8"` в исходном `config.json`. PyTorch `.bin`, другие quantization methods (INT8/INT4/HiF8), мультимодальные префиксы и CLA remapping пока не поддерживаются.

Сначала получить план без записи и без чтения tensor payload:

```bash
python prepare_checkpoint.py \
  --source /data/models/pangu505b-mxfp8 \
  --output /data/models/pangu505b-mxfp8-12layers \
  --dry-run
```

Затем сохранить checkpoint:

```bash
python prepare_checkpoint.py \
  --source /data/models/pangu505b-mxfp8 \
  --output /data/models/pangu505b-mxfp8-12layers \
  --max-shard-gib 4 \
  --buffer-mib 16 \
  --hash-tensors
```

По умолчанию читается `SOURCE/config.json`; для MXFP8 рекомендуется именно этот вариант. Не подменяйте выходной config файлом `config-12layers.json`: это пример для старого reference V2/BF16.

Если нужен `--config`, размеры и архитектура должны совпадать с источником. Отсутствующая в таком файле `quantization_config` наследуется из `SOURCE/config.json`. Если quantization config есть в обоих файлах, она должна совпадать целиком, включая `ignore` и communication flags. Противоречия отклоняются: структурный override не служит командой переквантовать модель. Источник с Ultra нельзя подменять приложенным V2 config.

Скрипт читает заголовки исходных shards, затем копирует только выбранные диапазоны байтов. RAM под tensor payload ограничена размером буфера; отдельно нужны память под заголовки, имена и index. Ненужные слои не читаются целиком. Содержимое выбранных тензоров не преобразуется. Для MXFP8 сохраняются исходные `F8_E4M3` веса, scales `U8`/`F8_E8M0`, BF16/FP32 тензоры, их имена и формы; веса не деквантуются и не квантуются повторно. Все тензоры scales сохраняемого слоя копируются вместе с его весами, а scales удалённых слоёв исключаются тем же правилом. Никакого импорта torch или создания модели нет.

Сохраняются все тензоры `model.layers.0.*` — `model.layers.11.*` и тензоры вне `model.layers.*`, включая embedding, LM head, финальные нормализации и mHC merge. Тензоры слоёв 12 и выше, в том числе MTP 50–52, исключаются. Для другого допустимого checkpoint prefix есть `--layer-prefix`; для этого checkpoint следует оставить стандартный `model.layers.`.

Новый `config.json` содержит:

```json
{
  "num_hidden_layers": 12,
  "num_nextn_predict_layers": 0,
  "dsa_layers": [0, 3, 6, 9],
  "swa_layers": [1, 2, 4, 5, 7, 8, 10, 11],
  "sliding_window_list": [512, 512, 512, 512, 512, 512, 512, 512],
  "block_post_layernorm_idx": [0, 4, 9]
}
```

Остальные размеры, эксперты, RoPE, MoME, mHC, sinks, precision и исходная `quantization_config` сохраняются. Поле `dtype="bfloat16"` совместимо с MXFP8: оно задаёт BF16-активации и неквантованные параметры, а формат quantized weights определяет `quantization_config`. Копируются обычные файлы кода HF config и tokenizer из корня checkpoint. Если custom code или tokenizer лежат во вложенных нестандартных каталогах, их нужно перенести отдельно; рекурсивного копирования произвольного дерева нет.

Выходной каталог должен отсутствовать. Источник не меняется. Запись идёт в соседний `.partial-*` каталог, который переименовывается после завершения. При ошибке partial сохраняется для диагностики; автоматического продолжения нет. Для повторного запуска выбирается новый output либо вручную удаляется только неудачный partial.

`--hash-tensors` вычисляет SHA256 исходных payload во время копирования. После переноса можно повторно прочитать выход и проверить hashes:

```bash
python verify_checkpoint.py /data/models/pangu505b-mxfp8-12layers --hashes
```

Без `--hashes` проверяются заголовки, index, число тензоров, размеры файлов и сокращённый config без полного чтения весов. `extraction_manifest.json` сохраняет разрешённый исходный config, hashes файла config и эффективного config, mapping слоёв, quantization method, признак наследования quantization metadata и статистику извлечения.

## Запуск постоянного сервера

Использовать те же NPU, TP/EP и operator settings, что и в предполагаемой полной модели. Уменьшение TP вместе с числом слоёв меняет размеры локальных матриц и коммуникации и мешает переносу результатов.

`--operator-config` обязателен: укажите JSON model/operator config из рабочего запуска полной модели. Скрипт не угадывает оптимизации A3/A5. Если подходящий JSON лежит в исходниках OmniInfer, сначала скопируйте его в отдельное место; скрипт исходники не редактирует. Размеры TP отдельных операторов внутри JSON должны соответствовать вашей схеме.

Пример для чистого TP8 с обычным V2 attention:

```bash
python serve.py \
  --model /data/models/pangu505b-mxfp8-12layers \
  --operator-config /data/profile-configs/pangu-target-operators.json \
  --output /data/profile-results/pangu12-server \
  --tp 8 \
  --patches low_latency \
  --attention-backend pangu \
  --max-model-len 33024 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096
```

Без `--execute` выводится только план. После проверки добавьте `--execute`. Сервер работает в foreground; stdout/stderr можно записывать через `tee`, сохранив проверку exit status через `set -o pipefail`.

`low_latency` в примере — набор patches для стандартного V2-пути. Если рабочая полная модель использует другой набор, передайте именно его. `--attention-backend deepseek` включает `OMNI_PANGU_V2_HIGH_THROUGHOUT=1`, но не меняет архитектуру на Ultra. Этот вариант требует соответствующего набора patches и operator config. Значение `--patches` по умолчанию берётся из уже заданного окружения, иначе используется `low_latency` для V2 и `high_throughout` для Ultra. `--attention-backend auto` выбирает `pangu` для V2 и `deepseek` для Ultra. Явно заданный `pangu` для Ultra отклоняется; архитектура не переписывается.

Для MXFP8 `serve.py` добавляет в `VLLM_PLUGINS` необходимые `omni-npu`, `omni_npu_patches` и `omni_custom_models`, сохраняя уже перечисленные plugins. Quantization method vLLM определяет из сохранённого `config.json`; принудительного переключения на BF16 weights нет. Аргумент сервера `--dtype bfloat16` оставлен для активаций, как требует реализация MXFP8 OmniInfer.

Сервер всегда запускается с DP1, без MTP, с отключённым prefix cache. Для expert parallelism дополнительно передайте `--enable-expert-parallel`, только если он нужен и в полном запуске. По умолчанию остаётся чистый tensor parallelism. Графы не отключаются автоматически; `--enforce-eager` предназначен для отдельного диагностического сравнения.

Скрипт удаляет из дочернего окружения переменные Hybrid/PD aligned decode, DP round-robin, force MoE load balance и автоматического token-trigger profiler. Конфликтующие benchmark patch directories отклоняются. Это предотвращает блокировку decode при DP1 и сохраняет естественный router. Дополнительные параметры vLLM задаются JSON-массивом CLI arguments через `--engine-cli-json`; изменение архитектуры, MTP, profiler и parallelism этим способом запрещено.

`server_manifest.json` содержит точную команду, HF config, operator config и параметры запуска. Он сохраняется до старта; существующий manifest не перезаписывается. Секреты окружения и API keys в него не выгружаются. Точные версии пакетов, NPU, драйвера, CANN и сетевую топологию следует сохранить рядом с логом запуска средствами вашего серверного окружения.

## Память и KV cache

Сокращение стека освобождает память весов и уменьшает KV на токен. При той же доле HBM сервер может вместить больше контекста, чем полная модель. Нельзя трактовать это как ускорение полной модели.

Для фиксированной нагрузки контролируйте `max-num-seqs`, token budget и длины запросов. Если известна подходящая вместимость исходного сервера, есть `--num-gpu-blocks-override` и `--kv-cache-memory-bytes`. Эти параметры не эквивалентны: одинаковый объём памяти при другом числе слоёв не означает одинаковую token capacity. Для hybrid cache groups соответствие нужно проверять по startup logs и фактическим cache specs.

Не ставьте `max-model-len=524288` только из-за поля HF config, если измеряете максимум 32K: лимит сервера здесь сознательно 33024. Для профиля 128K/512K отдельно увеличьте лимиты и добавьте случаи в suite, учитывая доступную память. Случаи с concurrency выше `max-num-seqs` отклоняются клиентом при переданном server manifest, чтобы скрытая очередь не выдавалась за полный active batch.

## Замеры производительности

Оставьте сервер загруженным и в другом терминале запустите:

```bash
python benchmark.py \
  --model-dir /data/models/pangu505b-mxfp8-12layers \
  --base-url http://127.0.0.1:8000 \
  --server-manifest /data/profile-results/pangu12-server/server_manifest.json \
  --output /data/profile-results/pangu12-benchmark
```

По умолчанию каждый случай прогревается двумя волнами, затем измеряется пять волн. Одна волна содержит `concurrency` одновременно отправляемых запросов и завершается после всех ответов. Это замер пачек с полным drain, а не open-loop нагрузка с заданным request rate. Модель между случаями не перезагружается.

Для первого короткого прогона:

```bash
python benchmark.py \
  --model-dir /data/models/pangu505b-mxfp8-12layers \
  --server-manifest /data/profile-results/pangu12-server/server_manifest.json \
  --select prefill-1k-c1 decode-1k-c1 \
  --warmup-waves 1 --measure-waves 2 \
  --output /data/profile-results/pangu12-smoke
```

`--dry-run` проверяет suite и локальные configs, выводит план и не подключается к серверу. Отдельный output нужен для каждого запуска; существующие результаты не перезаписываются.

Запросы идут на `/v1/completions` с готовыми token IDs, `return_token_ids=true`, `ignore_eos=true`, фиксированным числом выходных токенов и без добавления BOS. API key при необходимости читается из `OPENAI_API_KEY`. Base URL указывается без `/v1`.

Синтетические входы воспроизводимы по seed, но не представляют распределение естественного текста. Для приближения MoE routing к реальному workload используйте `--prompts-jsonl`:

```json
{"prompt_token_ids": [100, 101, 102]}
```

Это только пример формата. Каждый образец должен содержать не меньше токенов, чем самый длинный выбранный input; скрипт берёт префикс нужной длины. Токены должны быть получены tokenizer именно этой модели. Для измерения semantic quality сокращённый checkpoint не предназначен.

Результаты:

- `run_manifest.json`: workload, seed, model config, ответ `/v1/models` и server manifest, если передан.
- `requests.jsonl`: каждая warmup/measurement/profile request, времена прихода SSE chunks, число токенов, ошибки и hash входа.
- `summary.json`: результаты только измерительных волн по каждому случаю.
- `summary.csv`: компактная таблица для сравнения запусков; в profile mode метрики явно помечены как profile.

TTFT измеряется до первого chunk с output token IDs. TPOT — от первого до последнего токена, делённое на число последующих токенов. Если несколько токенов пришли в одном SSE chunk, их отдельные времена неизвестны; `all_chunks_single_token=false` явно отмечает это. `stream_chunk_gap_s` — интервалы chunks, а не обещание измерения времени каждого NPU decode-step.

`output_tokens_per_s_e2e` и `input_tokens_per_s_e2e` учитывают prefill, HTTP, очередь и drain. Prefill с одним выходным токеном — измерение с преобладанием prefill, но оно также включает LM head и выдачу токена. Generation с 128 токенами также содержит prefill; это не изолированная скорость kernels decode.

Клиент сверяет фактические input/output counts, usage и целостность stream. Ошибка или преждевременное завершение приводит к ненулевому exit status; сырые данные сохраняются. Профильные результаты не смешиваются с обычными замерами.

## Сбор NPU trace без повторной загрузки

Тот же сервер уже запущен с `--profiler-config`, поэтому загрузка checkpoint повторно не нужна. Клиент сначала прогревает выбранный случай, затем вызывает `/start_profile`, отправляет одну волну и гарантированно пытается вызвать `/stop_profile` после неё.

Prefill с одним выходным токеном:

```bash
python benchmark.py \
  --model-dir /data/models/pangu505b-mxfp8-12layers \
  --server-manifest /data/profile-results/pangu12-server/server_manifest.json \
  --mode profile --profile-kind prefill --select prefill-8k-c8 \
  --output /data/profile-results/pangu12-trace-prefill
```

Короткий trace с prefill и последующим decode:

```bash
python benchmark.py \
  --model-dir /data/models/pangu505b-mxfp8-12layers \
  --server-manifest /data/profile-results/pangu12-server/server_manifest.json \
  --mode profile --profile-kind generation --profile-tokens 8 \
  --select decode-8k-c8 \
  --output /data/profile-results/pangu12-trace-generation
```

NPU traces находятся в `pangu12-server/traces`, клиентские результаты — в указанном output. Profiler собирает CPU и NPU на workers; frontend profiling выключен. Не отправляйте постороннюю нагрузку во время такого окна.

`generation` не называется чистым decode: сбор начинается перед запросами и включает их prefill, а с chunked prefill — несколько prefill steps. Decode нужно выделять на timeline. Кроме того, конкурентные HTTP-запросы не гарантируют неизменный active batch на каждом шаге: это проверяется по trace. Gates для формирования batch здесь не используются.

Стек Python, shapes и memory profiling по умолчанию выключены, чтобы уменьшить накладные расходы. При необходимости сервер можно запустить с `--record-shapes` или `--profile-memory` для отдельного диагностического сеанса. Это меняет стоимость сбора. PyTorch NPU wrapper и его runtime остаются теми же, что в установленном OmniInfer.

## Метки слоёв и анализ

Опциональный `serve.py --annotate-layers` поддерживает V2 и Ultra и добавляет через отдельный `sitecustomize.py` ranges `pangu12/layer_000/forward`, ..., `pangu12/layer_011/forward` и `mhc_head`. Он не меняет файлы OmniInfer. В stderr должно появиться `[pangu12] CPU layer annotations installed`.

Это дополнительная Python-инструментация: её overhead входит в измерения, а графы/компиляция могут менять видимость ranges. Для baseline performance оставьте её выключенной; для разбора слоёв при необходимости проведите отдельный annotated запуск. `--enforce-eager` может помочь диагностике, но eager-профиль нельзя выдавать за профиль целевого graph mode.

```bash
python analyze_trace.py \
  /data/profile-results/pangu12-server/traces/WORKER/ASCEND_PROFILER_OUTPUT/trace_view.json \
  --output /data/profile-results/layer-cpu-ranges.json
```

Путь к trace нужно взять из фактического вывода profiler; поддерживается Chrome JSON и `.json.gz`. Анализируется один файл/rank. Скрипт выдаёт **CPU launch durations**, а не время асинхронных NPU kernels. Для NPU critical path, HCCL, памяти и перекрытия streams используйте device timeline в MindStudio/Perfetto и соответствующие profiler reports. Складывать kernel durations всех streams/ranks нельзя.

## Перенос результата на полную модель

12 слоёв сохраняют исходный префикс и его размеры. В полном стеке 1 dense+DSA, 2 dense+SWA, 17 MoE+DSA и 30 MoE+SWA. Для приблизительной оценки фиксированного шага можно использовать времена представительных блоков с такими весами, отдельно учитывая extra norms, embedding/LM head, границы стека, коммуникации и перекрытие.

Ограничения: нет последних слоёв 48/49, отсутствует MTP, финальная нормализация получает другое представление, меньше суммарный working set и давление на HBM/KV. MoE routing корректен для сохранённого префикса, но не описывает распределение экспертов поздних слоёв. Поэтому линейное умножение полного времени 12 слоёв на 50/12 недостоверно.

Для проверки масштабирования экстрактор поддерживает `--layers 6` и `--layers 9`; соответствующие checkpoint и конфиги создаются отдельно. Сравнивать их следует при одинаковых workload, TP/EP, kernels и режимах графов. Финальный эффект оптимизации нужно подтвердить одним контрольным запуском полной модели.

## CPU-проверки и источники

```bash
python test_cpu.py
```

Тесты используют маленькие синтетические BF16/MXFP8 safetensors checkpoint и SSE-ответы в памяти. Для MXFP8 проверяются обе архитектуры, побайтовая сохранность FP8-весов и scales, metadata, SHA256, наследование quantization config, отказ при конфликтующих/неподдерживаемых форматах и планы запуска без импорта ML runtime. Они проверяют сохранение tensor bytes, глобальных весов и исходных файлов, обрезку конфигурации, dry-run, точность подсчёта токенов и отказ при обрыве stream. Torch, модель, NPU и сетевые сокеты не используются.

Сверенные интерфейсы: [vLLM 0.25.1 CompletionRequest](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/entrypoints/openai/completion/protocol.py), [vLLM ProfilerConfig](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/config/profiler.py). В OmniInfer используются `omni/profiler/wrapper.py`, `omni/worker/npu_worker.py`, `omni/v1/models/pangu/pangu_v2_moe.py`. Поведение на реальном NPU-сервере здесь не проверялось.
