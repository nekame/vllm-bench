# Отдельное профилирование MTP Pangu 505B

Все команды ниже выполняются из корня `vllm-bench` после настройки:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Что реализовано и что проверено

Инструмент запускает отдельный `torchrun` job на восьми NPU, загружая MTP и необходимые общие веса. HTTP, EngineCore, scheduler и основная 50-слойная модель не запускаются. Используются библиотеки vLLM и OmniInfer: это не независимый перенос модели на чистый PyTorch.

Адаптер предназначен для **vLLM 0.25.1**, **OmniInfer `4cd855b2`**, **V1 NPUModelRunner**. Поддержаны ветки V2/Ultra и BF16/MXFP8 в экстракторе. Launcher включает `VLLM_USE_V2_MODEL_RUNNER=0`; recorder нужно подключать к полному запуску с тем же V1 runner. Другие версии private API и V2 runner автоматически не подменяются.

Для bootstrap используется HCCL world с отдельными CPU gloo groups, `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=0`: это обходит CUDA-specific требования split-PG validator vLLM к внешнему launcher. Такой вариант инициализации групп нужно учитывать при сравнении с окружением, где split-PG включён.

Буфер `cu_num_draft_tokens` не считается обязательным атрибутом V1 runner. При выключенном `fix_multi_mtp_kvcache` synthetic и snapshot работают без него. Если этот путь включён, standalone создаёт zero-initialized CPU/NPU buffer через `_make_buffer`; replay при необходимости создаёт его и восстанавливает записанные counts. Recorder полного запуска не подставляет нули вместо отсутствующего обязательного состояния.

**На машине разработки выполнены только CPU-проверки. NPU bootstrap, реальная загрузка MXFP8, распределённые collectives, capture/replay и численная эквивалентность ещё требуют server smoke tests.** Наличие реализации этих путей не означает, что они уже прошли аппаратную проверку. При ошибке адаптер прекращает выполнение; dummy attention, случайные веса вместо checkpoint и незаметный переход на eager не используются.

## Параллелизм и нагрузка

| `--topology` | TP | DP | EP | Batch 1 на DP-rank |
| --- | --- | --- | --- | --- |
| `tp8` | 8 | 1 | выключен | Одна последовательность на весь job |
| `dp8-ep8` | 1 | 8 | включён, эксперты распределены на 8 NPU | Восемь независимых последовательностей |

Все восемь процессов должны участвовать в согласованных MTP-вызовах. Multi-node и преобразование реальной записи между topology не реализованы. Для каждого режима нужен соответствующий operator config из рабочего запуска.

Синтетические defaults: контекст 8192, batch 1 на DP-rank, depths `3 7`, два прогрева, пять измерений, greedy sampling. Все длины и числа повторов доступны в CLI. Заполнение контекста выполняется настоящими MTP-блоками на синтетических token IDs/hidden states. Это формирует согласованное состояние cache, но не воспроизводит распределение активаций target model и нагрузку на экспертов production-запроса.

Основная метрика включает MTP forward, logits и sampling, а также CPU-работу proposer. Verify target, rejection sampling по target logits и HTTP в неё не входят. `--depths 1` даёт цикл с одним predictor forward и его logits/sampling.

## Важная развилка depth=3 и depth=7

В конфиге три predictor-слоя: 50, 51, 52. Сам predictor выбирает слой по `spec_step_idx % 3`, **но полный OmniInfer proposer не всегда передаёт этот индекс**.

В [patch_eagle.propose](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/vllm_patches/patches/common/patch_eagle.py#L843) multi-MTP включён только при `n_predict > 1` и `num_speculative_tokens <= n_predict`:

- depth=3: слои **50 → 51 → 52**, с packed multi-MTP путём;
- depth=7: обычная ветка, без передачи `spec_step_idx`, то есть **слой 50 семь раз**, с обычным обновлением позиций/metadata.

Это поведение просмотренного commit. Оно сохраняется в standalone-профайлере и отражается в `expected_layers`/`layer_sequence`; depth=7 не заменяется искусственным обходом 50/51/52. При сравнении с более новым OmniInfer эту развилку необходимо сверить заново.

## Подготовка checkpoint

Источником должен быть **полный исходный checkpoint** с MTP, а не наши первые 12 слоёв: из последних MTP уже удалён.

```bash
python -m vllm_bench.mtp_profile.prepare_mtp_checkpoint \
  --source /data/models/pangu505b-mxfp8 \
  --output /data/models/pangu505b-mxfp8-mtp \
  --hash-tensors --dry-run
```

Для копирования повторите команду без `--dry-run`. Экстрактор не импортирует torch, не выделяет tensor buffers модели и копирует payloads побайтно, включая MXFP8 scales. По умолчанию shard до 4 GiB, копирующий buffer 16 MiB. Уже существующий output не перезаписывается.

```bash
python -m vllm_bench.mtp_profile.verify_mtp_checkpoint /data/models/pangu505b-mxfp8-mtp --hash-tensors
```

Сохраняются исходные имена/индексы 50–52, predictor-specific нормы/проекции и global embedding/lm_head. Исходный `num_hidden_layers=50` сохраняется как начало диапазона MTP. **Этот каталог нельзя запускать как обычную target model через `vllm serve`**: standalone loader явно конструирует только MTP.

В полном запуске [set_shared_weight](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/v1/models/pangu/pangu_v2_moe_mtp.py#L413) привязывает общие веса target. В standalone используются извлечённые target embedding/lm_head, даже если в checkpoint присутствуют predictor-local дубликаты. Все три predictor head используют один объект общей выходной головы; собственные head norms остаются раздельными. Переименование входных ключей делает локальный adapter загрузчика, исходные файлы OmniInfer не меняются.

## Синтетический запуск

Сначала проверьте план — по умолчанию модели и NPU не инициализируются, output не создаётся:

```bash
python -m vllm_bench.mtp_profile.profile_mtp \
  --model /data/models/pangu505b-mxfp8-mtp \
  --operator-config /data/config/operators-dp8-ep8.json \
  --topology dp8-ep8 \
  --context-length 8192 --batch-per-dp-rank 1 --depths 3 7 \
  --execution eager \
  --output /data/profile-results/mtp-dp8-eager
```

Для запуска на сервере добавьте `--execute`. Launcher сам вызывает восемь процессов `torch.distributed.run`; оборачивать `profile_mtp.py` ещё одним `torchrun` не нужно. Для TP8 используйте `--topology tp8`, соответствующий operator config и новый output.

Каждый depth запускается отдельным последовательным torchrun job с собственной загрузкой небольшого MTP-checkpoint и собственными speculative buffers. Это важно: `num_speculative_tokens` влияет на cache и формы буферов, поэтому depth=3 не измеряется с буферами конфигурации depth=7. В replay сохраняется configured depth исходного сервера, даже если записанный вызов запросил меньше токенов.

В режиме `compiled` компиляция сохраняется, а `cudagraph_mode` становится `NONE`. В `graph` используется `FULL` и реальные `ACLGraphWrapper` для predictor-слоёв. Передать целевые настройки можно через `--compilation-config /data/config/compilation.json`; содержимое — JSON object vLLM CompilationConfig. При необходимости передайте также исходный `--additional-config` и `--patches`.

```bash
python -m vllm_bench.mtp_profile.profile_mtp \
  --model /data/models/pangu505b-mxfp8-mtp \
  --operator-config /data/config/operators-dp8-ep8.json \
  --topology dp8-ep8 --depths 3 7 \
  --execution graph --compilation-config /data/config/compilation.json \
  --output /data/profile-results/mtp-dp8-graph --execute
```

Eager выключает компиляцию и графы, а также дополнительные оптимизации через штатный eager config OmniInfer. Сравнение режимов описано в [COMPILATION_EAGER_RU.md](../reference/COMPILATION_EAGER_RU.md).

## Реальные входы: запись полного запуска

Recorder подключается только по явному запросу через отдельный `sitecustomize.py`. Файлы установленного OmniInfer не редактируются. Поддержан hook загрузки **V1 NPUModelRunner**; в логе должно появиться `[pangu-mtp] recorder attached on rank …`.

Запускайте wrapper в том же контейнере/окружении, где запускается модель, чтобы пути `PYTHONPATH` были доступны worker-процессам:

```bash
python -m vllm_bench.mtp_profile.record_mtp \
  --output /data/mtp-capture --skip 2 --count 1 \
  --max-gib-per-snapshot 8 --execute \
  -- bash /data/scripts/launch-full-model.sh
```

Без `--execute` команда только печатает environment и исходную команду. Wrapper не меняет speculative settings полного запуска: depth и parallelism задаются в вашем `launch-full-model.sh`. Для отдельных записей depth=3 и depth=7 нужны соответствующие реальные запуски/вызовы.

У полного запуска должен быть явно задан `OMNI_CUSTOM_MODEL_CONFIG_PATH`. Не сочетайте этот wrapper с нашим `--annotate-layers`: обе инструментации используют `sitecustomize.py`. Для DP8 подавайте сбалансированную нагрузку на все ranks. Первая версия replay принимает записи с одинаковыми input-token counts и batch size на восьми ranks; пустой/незавершённый rank делает запись непригодной.

После двух пропущенных proposer-вызовов на каждом rank записывается один цикл. Граница — **внешний `propose`**, до graph replay, поэтому запись не зависит от повторного вызова Python-кода отдельных слоёв.

Сохраняются входные tokens/positions/hidden states, sampling metadata и RNG, draft KV/MoME caches, используемые runner buffers, block tables, configured depth, настройки cache/компиляции/operator config и контрольные hidden states/logits/tokens. Target cache и target weights не снимаются.

Cache snapshot в этой версии содержит **полные выделенные draft-cache tensors**, а не только занятые блоки. Ограничение 8 GiB применяется к одному snapshot на rank; при превышении recorder явно останавливает запуск. Восемь ranks означают до 64 GiB для такого набора снимков, плюс контрольные выходы и накладные расходы. Компактизация blocks пока не реализована. Если MTP и target делят одну raw cache allocation, запись отклоняется.

Recorder добавляет synchronization и CPU copies: времена этого запуска не являются baseline. Stateful penalties, token constraints и объекты metadata, которые нельзя явно сериализовать, отклоняются. Для первого capture используйте greedy sampling без дополнительных logits processors. JSON tree и tensor-only `torch.save` загружаются с `weights_only=True`, без десериализации произвольных pickle-объектов.

Готовая запись:

```text
/data/mtp-capture/cycle-0000/
  rank-000/capture.json, before/, reference/
  …
  rank-007/capture.json, before/, reference/
```

`capture.json` появляется только после успешного завершения записи rank. Незавершённые каталоги не переиспользуйте как готовый capture.

## Replay без target model

```bash
python -m vllm_bench.mtp_profile.profile_mtp \
  --model /data/models/pangu505b-mxfp8-mtp \
  --operator-config /data/config/operators-dp8-ep8.json \
  --topology dp8-ep8 --input replay \
  --capture /data/mtp-capture/cycle-0000 \
  --execution graph \
  --output /data/profile-results/mtp-replay --execute
```

В replay depth и batch берутся из записи. Если передаётся `--depths`, он должен содержать ровно записанный depth. Нельзя превратить один capture depth=3 в capture depth=7.

Topology, HF config, operator config и версии пакетов должны совпадать. Runtime limits, cache settings, sampling options и часть draft environment восстанавливаются из capture. Compilation options и additional config по умолчанию также берутся из записи; можно явно передать отдельную compilation config для сравнения режима исполнения. Eager/compiled/graph применяют свои правила отключения графов поверх этих настроек. Произвольные compiler callbacks в capture не поддержаны.

Перед таймингами выполняется отдельная численная проверка hidden states/logits и совпадения предложенных tokens. Допуски `--atol` и `--rtol` по умолчанию равны 0.05; для вашей точности/backend их нужно оценить по `comparison-DEPTH.json`. Это не гарантия побитового совпадения MXFP8. При провале проверки timing-прогон прекращается.

## Границы измерений и результаты

Перед каждым повтором восстанавливаются одинаковые cache, metadata и входы; device input buffers переиспользуются. Восстановление, disk I/O и межранговый barrier находятся вне измеряемого интервала. Wall time измеряется до завершения NPU-работы всего draft-цикла, без промежуточных synchronization между forward/logits/sampling.

Graph capture и прогрев выполняются до измерений. Проверяются стабильность device pointers/layout входов и отсутствие новых graph entries после прогрева. При нарушении вместо потенциально неверного профиля будет ошибка; автоматического eager fallback нет. Такой отказ на сервере требует разбора соответствующего private-API/graph пути.

Trace собирается отдельным повтором, чтобы profiler overhead не попадал в baseline latency. Внутри него CPU ranges `mtp/forward`, `mtp/logits`, `mtp/sampling` помогают ориентироваться; они не являются NPU-временами и не размечают слои внутри replay.

`--diagnostic-stages` дополнительно выполняет отдельный запуск с synchronization на границах частей. Его результаты хранятся отдельно от baseline; сумма частей не обязана равняться времени обычного цикла из-за потери перекрытия.

Результаты включают:

- `launch_plan.json` и `command.json` — исходный план и список torchrun commands.
- `depth-D/rank-NNN/effective_config.json` — фактические версии и настройки.
- `depth-D/rank-NNN/results.json` — latency повторов, число graph entries и память.
- `depth-D/rank-NNN/comparison-DEPTH.json` — проверка реального replay.
- `depth-D/rank-NNN/traces/` — данные torch_npu profiler.
- `summary.json`/`summary.csv` — median critical-path latency и суммарные proposed tokens/s. Critical path определяется максимумом времени по ranks для каждого повтора.

Для offline export используйте существующий инструмент:

```bash
python -m vllm_bench.tools.export_profile /data/profile-results/mtp-dp8-graph --execute
```

`depth-D/failure-rank-N.json` и traceback содержат диагностику ошибки; такой запуск не считать валидным результатом. Acceptance rate и выигрыш end-to-end speculative decoding этот инструмент не измеряет.

## Проверка на сервере

CPU-проверки запускаются командой `python3 -m unittest discover -s tests`. Они проверяют файлы, CLI, сериализацию/восстановление через mocks и изоляцию ML imports, но не исполнение модели.

Порядок server smoke tests: проверить extracted checkpoint; выполнить synthetic eager depth=1, затем 3/7 на TP8 и DP8/EP8; проверить compiled и graph с целевыми configs; получить небольшую реальную запись; подтвердить numerical replay в исходной topology; проверить непустые NPU traces и отсутствие повторного capture в timing window. Зафиксировать версии и фактическое потребление памяти. До прохождения этих шагов runtime остаётся аппаратно непроверенным.

Основные реализации: [standalone runtime](../../src/vllm_bench/mtp_profile/mtp_runtime.py), [recorder](../../src/vllm_bench/mtp_profile/mtp_capture.py), [V2 predictor](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/v1/models/pangu/pangu_v2_moe_mtp.py), [Ultra predictor](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/v1/models/pangu/pangu_ultra_moe_mtp.py), [Omni proposer](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/vllm_patches/patches/common/patch_eagle.py), [native graph wrapping](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/worker/npu_model_runner.py#L694).
