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
- Архитектура checkpoint — строго `PanguUltraMoEForCausalLM`. Поддерживаются
  MXFP8 и BF16 safetensors; конфигурация квантизации и scales не преобразуются.
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

Для запуска без MTP замените `--depth 3` / `--depth 7` на `--no-mtp`
(или `--depth 0`). Флаг работает для обоих engines, greedy и sampling, всех
поддерживаемых режимов compile/ACL. Predictor не создаётся, его веса не
загружаются, speculative decoding отключён. Подходит checkpoint как с MTP,
так и без predictor layers. По умолчанию MTP остаётся включённым с depth 3.

`--prompt-ids /data/prompt.json` принимает JSON-массив ровно из 8192 token IDs
при стандартном context-length. Без файла оба engine получают одинаковый
детерминированный synthetic prompt, зависящий от seed и vocab_size. Для
воспроизводимого анализа настоящих запросов используйте явный файл.

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

Артефакты:

- `target-trace/manifest.json`: workload, effective config, версии и оборудование.
- `target-trace/rank-N/index.json`: число target-вызовов.
- `target-trace/rank-N/call-XXXXX.pt`: CPU tensors — входы, позиции, полные logits
  и выбранные логические поля attention metadata, отдельно для prefill/decode.
- `trace_comparison.json` в output direct: автоматическое сравнение всех ranks
  и вызовов, первая разница входов, metadata, logits и argmax. В stdout/runtime.log
  выводится краткий summary с top-5, top-1 margin и logits конкурирующего токена
  для первого расхождения argmax.

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
выключены async scheduling и отдельный engine-core процесс; TP workers
остаются штатными. Синхронизация workers после первого output исключена из
decode, завершающая синхронизация через RPC включена в его wall time.
Контекст начинается с 8192 и растёт, это не fixed-context replay benchmark.
Последний speculative cycle может вычислить больше токенов, чем выдаётся
пользователю; время включает эту работу в обоих движках.

Acceptance direct берётся из verification до терминального обрезания;
baseline восстанавливает её по приращениям outputs, поэтому последний
обрезанный цикл даёт нижнюю границу. Эти поля нельзя считать побитово
сопоставимым внутренним счётчиком.

Для диагностического запуска добавьте `--profile` и новый output. Direct
пишет CPU/NPU trace с диапазонами prefill, draft, target и sample/commit;
baseline включает штатный worker profiler. Такие результаты помечаются
`diagnostic`, и compare не использует их для speedup. Логирование внутри
измеряемого decode не выполняется.

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
