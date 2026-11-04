# Отладка greedy MTP verification и MoME

Этот сценарий подключает диагностику к полной или сокращённой модели Pangu с MTP. Он записывает draft tokens, target argmax, bonus token, результат acceptance и проверку относительно CPU-эталона greedy prefix acceptance. Дополнительные режимы сохраняют target inputs, MoME, attention и исходный Ki indexer. Для Ultra на TP2/TP8 есть собственный сервер без профайлера и автоматическое воспроизведение одной командой.

Для замеров времени standalone MTP используйте [профилирование MTP](../mtp_profile/README.md). Диагностические копии на CPU и синхронизации меняют тайминги и могут скрывать гонки; результаты этого запуска не являются baseline производительности.

## Требования

- Python 3.10+ и рабочее окружение OmniInfer/vLLM/torch_npu с V1 NPUModelRunner.
- Для wrapper — модель с настроенными MTP и topology в вашем серверном launch script. Wrapper эти настройки не меняет. Новый launcher ниже сам формирует отладочный запуск Ultra.
- Только greedy sampling (`all_greedy`); после включения записи другой sampling вызывает явную ошибку.
- Для `--rejection-windows` необходимы eager (`--enforce-eager`), отключённый prefix caching (`--no-enable-prefix-caching`) и один активный запрос. Не посылайте постороннюю нагрузку.
- Новая директория результатов. По умолчанию записывается rank 0; другой rank выбирается переменной `PANGU_SPEC_DEBUG_RANK`.

Отдельных конфигов этого сценария пока нет. Используются настройки вашего полного запуска и параметры CLI. Исходники OmniInfer не изменяются. Не сочетайте wrapper с MTP recorder или `--annotate-layers`: каждый использует свой `sitecustomize.py`, конфликт отклоняется до старта.

## Воспроизведение на двух NPU одной командой

`mtp_debug.reproduce` сам запускает сервер, дожидается API, прогревает, отправляет одинаковые запросы, сравнивает Ki и завершает собственный сервер с workers. Нужен Linux, рабочее окружение A5/OmniInfer и checkpoint Ultra, который помещается на двух доступных NPU. Сокращённый checkpoint можно подготовить средствами `layer_profile.prepare_checkpoint --keep-mtp`; уменьшение TP не требует перекопирования весов, но operator config должен быть согласован с TP2. `layer_profile` остаётся отдельным инструментом профилирования.

Из корня checkout полная команда:

```bash
bash scripts/reproduce_ki.sh \
  --model /data/models/pangu505b-mxfp8-12layers-mtp \
  --operator-config /data/profile-configs/pangu-target-operators-tp2.json \
  --output /data/profile-results/tp2-ki-repro \
  --execute
```

Без `--execute` команда только печатает план: каталог output не создаётся, модель/ML runtime не импортируются. Эквивалентный Python entrypoint после настройки `PYTHONPATH`: `python3 -m vllm_bench.mtp_debug.reproduce`. Пути модели и operator config замените на свои; output должен быть новым. В operator JSON требуются `operator_optimization_config.use_quant_lightning_indexer_v2=true` и `use_noncontiguous_kv=true`. Полный рабочий JSON с TP2-настройками операторов нужно предоставить явно, launcher не переписывает его.

Defaults: TP2/DP1/PP1, сокращённая `PanguUltraMoEForCausalLM`, MTP depth=1, eager, prefix cache выключен, `hif8_ds_mla`, block size=128, один активный запрос. Сервер получает `profiler=null`; legacy `VLLM_TORCH_PROFILER_DIR` и автоматический token profiler выключаются. Profiler endpoints и layer annotations не используются. Штатное определение объёма памяти/KV при инициализации vLLM выполняется как обычно. **Ki-копии и синхронизации всё равно добавляют overhead: это диагностический запуск.** Для него MTP-веса обязательны, число predictor-слоёв в checkpoint не изменяется.

Один synthetic prompt длиной 8192 token IDs генерируется по seed=123 и сохраняется в `prompt.json` с SHA256. Два прогрева и два тестовых запроса получают один и тот же prompt, `temperature=0`, `top_p=1`, `top_k=0`, `seed=123`, выход ровно 128 токенов (`ignore_eos`, одинаковые min/max tokens). Между запросами меняется только request ID. Прогрев не захватывается; затем launcher сам выставляет ARM до первого тестового prefill и LABEL `request-01`, `request-02`. Вручную переключать ARM/LABEL не нужно.

Изменяемые параметры: `--input-tokens`, `--output-tokens`, `--seed`, `--requests` (>=2), `--warmup-requests`, `--num-speculative-tokens`, `--tp 2|8`, `--port`, настройки памяти и capture. Например, `--requests 16 --tensor-budget-mib 2048` увеличивает число запросов и per-worker budget. Token IDs с тем же seed/config повторяются между запусками; полный массив хранится в файле. `max-model-len` по умолчанию равен сумме входа и выхода, `max-num-batched-tokens` — длине входа, чтобы prompt поместился в один prefill. При явном переопределении лимиты проверяются. Число разрешённых sampler cycles автоматически увеличивается для выбранного количества запросов.

Полные returned token IDs и hashes находятся в `requests.jsonl`. Каждый тестовый запрос после первого сравнивается с `request-01`; Ki-отчёты и переносимые реальные входы сохраняются в `comparisons/`. Основные файлы:

- `run_manifest.json`, `prompt.json` — параметры, фактическая команда и вход;
- `server.log`, `server/server_manifest.json` — startup/runtime logs и серверный manifest;
- `capture/` — Ki sidecars, verification, tensors и лимиты обоих rank;
- `result.json`, `summary.txt` — общий результат и три короткие строки для передачи.

Пришлите только `result`, `first_difference`, `evidence` из `summary.txt`. Результат `OUTPUT_TOKENS_DIFFER` означает, что ответы различаются при отсутствии найденного отличия в захваченном prefill Ki. Частичный capture имеет `INCOMPLETE_CAPTURE`; отсутствие rank не считается успешным сравнением. Различия — диагностический результат, exit code=0; ошибки запуска/HTTP/анализа/остановки или неполный capture дают код 1. Полные сведения остаются в JSON/logs, не нужно переписывать длинные массивы.

Сервер работает отдельной группой процессов. При нормальном завершении, ошибке, Ctrl+C или SIGTERM launcher удаляет ARM и завершает только свою группу: SIGTERM, ожидание `--shutdown-timeout 30`, затем SIGKILL при необходимости. Занятый порт вызывает отказ; существующий сервер не используется и не останавливается. Startup timeout задаётся `--startup-timeout 1800`, timeout запроса — `--request-timeout 1800`. Запросы автоматически не повторяются. Ход запуска/запросов виден в stderr, подробности старта — в `server.log`.

### Отдельный отладочный сервер

Для ручной подачи запросов используйте `mtp_debug.serve`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m vllm_bench.mtp_debug.serve \
  --model /data/models/pangu505b-mxfp8-12layers-mtp \
  --operator-config /data/profile-configs/pangu-target-operators-tp2.json \
  --output /data/profile-results/tp2-debug-server \
  --execute
```

Этот entrypoint только запускает постоянный сервер без profiler. По умолчанию capture лежит в `OUTPUT/capture`; здесь ARM/LABEL нужно переключать вручную по инструкции ниже. Общая сборка engine command/env используется обоими server CLI, но `layer_profile.serve` сохраняет прежний profiler, defaults и manifest. В `--engine-cli-json` нового debug-сервера разрешены только дополнительные параметры: управляемые topology, profiler, cache, адрес API и лимиты меняются именованными флагами, а не дублирующими overrides.

CPU-тесты проверяют legacy CLI, TP2/TP8, фиксированные payloads, ARM/LABEL, SSE/token IDs, неполный capture, ошибки и очистку процессов, включая принудительное завершение дочернего процесса. Loopback HTTP integration test может пропускаться в sandbox без сокетов. Запуск настоящего vLLM на двух NPU здесь не выполнен. Отсутствие различия на сокращённой TP2-модели не исключает проблему полной TP8-модели.

## План и запуск

Из корня репозитория настройте путь:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Сначала проверьте план без создания output и запуска модели:

```bash
python -m vllm_bench.mtp_debug.debug_spec \
  --output /data/profile-results/mtp-debug \
  --limit 1000 \
  -- bash /data/scripts/launch-full-model.sh
```

Добавьте `--execute` перед `--`, чтобы запустить модель. В логе ожидается `[spec-debug] sampler hook installed`. Wrapper передаёт окружение дочерним workers, поэтому должен работать в том же контейнере, что и модель.

После старта модели выполните прогрев. Затем в другом терминале разрешите запись:

```bash
printf '%s\n' 'request-1' > /data/profile-results/mtp-debug/LABEL
touch /data/profile-results/mtp-debug/ARM
```

`LABEL` необязателен: его текст попадёт в события и поможет связать их с нагрузкой. Запись начинается только после появления `ARM`, ограничивается `--limit` вызовами sampler на выбранном rank и прекращается при удалении `ARM`. Лимит действует на время жизни процесса и не сбрасывается повторным созданием `ARM`.

Теперь отправьте короткий greedy-запрос своим клиентом. Сначала проверьте обычную sampler-диагностику, затем включайте дополнительные снимки в новом запуске.

## Rejection windows и fingerprints

```bash
python -m vllm_bench.mtp_debug.debug_spec \
  --output /data/profile-results/mtp-rejections \
  --limit 1000 \
  --rejection-windows --mome-layers 0 \
  --execute -- bash /data/scripts/launch-full-model-eager.sh
```

После прогрева создайте `ARM` в новом output и подавайте по одному запросу. В логе должен появиться `[spec-debug] target hook installed`.

`--mome-layers` принимает индексы слоёв через запятую либо `all`. По умолчанию выбран слой 0. Флаг `--state-fingerprints` вместе с `--rejection-windows` добавляет хеширование активных MoME blocks до и после выбранных свёрток. Это дополнительные CPU copies и синхронизации; используйте их отдельным диагностическим запуском.

## Результаты

- `rank-R-pid-P.jsonl` — события verification, draft/target/bonus token IDs, accepted counts, top-2 logits и `matches_reference`.
- `rejections/rank-R-pid-P/cycle-NNNNNN.json` — окна rejection с предыдущим, текущим и следующим verification и target/MoME metadata.
- `hooks-pid-P.json` — перечень установленных target/MoME hooks и их ограничения.
- `ARM` и `LABEL` — пользовательское управление записью и метка нагрузки.

Окно может оставаться `awaiting_next_verification`, если следующего verification ещё нет, либо завершиться как `request_changed_before_next_verification` при смене запроса. Такие окна нельзя считать полным сопоставлением соседних шагов. После завершения сравнивайте `expected_next_runner_count` и фактические counts через сводку:

```bash
python -m vllm_bench.mtp_debug.summarize_rejections \
  /data/profile-results/mtp-rejections --limit 20
```

`matches_reference=false` указывает на расхождение записанного greedy verification с CPU-эталоном. Поле `next_runner_count_matches_output=false` показывает расхождение counts следующего target шага с выходом sampler. Это диагностические признаки, требующие разбора конкретного окна, а не готовое доказательство причины ошибки.

## Сравнение двух запросов

Offline-инструмент `compare_rejections` читает JSONL и/или JSON окон rejection без ML-зависимостей. Сначала выведите записанные запросы:

```bash
python -m vllm_bench.mtp_debug.compare_rejections \
  /data/profile-results/mtp-rejections --list-requests
```

Сравните две метки из одного запуска:

```bash
python -m vllm_bench.mtp_debug.compare_rejections \
  /data/profile-results/mtp-rejections \
  --left-label request-1 --right-label request-2 \
  --output /data/profile-results/comparison.json
```

Для двух отдельных файлов/каталогов передайте второй путь позиционным аргументом. Если метки не задавались или повторяются, выбирайте через `--left-request` и `--right-request` (request ID из списка). Каждая сторона должна однозначно выбирать один запрос; rank должен совпадать. Можно сравнивать два отдельных JSON окна. Полезнее передавать весь каталог: исходный JSONL сохраняет проходы между окнами.

Инструмент объединяет повторные копии циклов из перекрывающихся окон и восстанавливает **наблюдаемый принятый префикс** из sampler outputs. Отвергнутые draft IDs в него не включаются. Проходы сопоставляются по абсолютной позиции первого target input, а не по cycle/номеру строки. Разная длина принятого выхода может менять границы циклов без изменения последовательности токенов. После первого различия в префиксе дальнейшие проходы не сравниваются как одинаковый контекст.

В краткой сводке:

- `matched_forwards` — число сопоставленных проходов на одинаковом записанном префиксе;
- `first_observed_token_difference` — первая известная позиция с разными принятыми token IDs;
- `first_logits_difference` и `first_metadata_difference` — первые различия до расхождения префикса;
- `first_fingerprint_observation` — первое различие/недоступность записанных fingerprints;
- `fingerprint_pairs_compared` — сколько пар fingerprints действительно проверено; ноль означает, что равенство состояний не проверялось.

Полный JSON содержит исходные пути событий, номера циклов обеих сторон, различия metadata, top-2 logits и fingerprints. `--output` требует нового файла, существующие записи не перезаписываются. Физические slot/block/cache IDs исключаются из смыслового сравнения metadata. Для единственного MoME-блока сравнивается содержимое независимо от его физического номера. Несколько блоков автоматически не сопоставляются. Разные хеши полного блока обозначаются `whole_block_differs`, а не ошибкой: speculative и неиспользуемые слоты входят в хеш. Образец из первых 16 значений также не гарантирует совпадение используемой истории.

Logit deltas вычисляются только для одинаковых token IDs, присутствующих в обеих top-2. Строки, обусловленные различающимися draft-префиксами, пропускаются. Допуски `--atol` и `--rtol` по умолчанию равны `1e-5`; они фильтруют численные отличия logits, но не различия token IDs или точных SHA256.

`top1_changed` сохранён для совместимости и означает изменение первого ID из `topk(2)`, а не обязательно фактического выбора sampler. При равных максимальных logits порядок top-2 может меняться при неизменном `argmax`. Поля `left_actual_choice`, `right_actual_choice`, `actual_choice_changed` берутся из `verify.target_argmax` или `verify.bonus_ids`; `top2_has_tie` отмечает нулевой margin. При отсутствии фактического выбора в записи значение `actual_choice_changed` равно `null`.

Ограничения: prompt и история до начала записи отсутствуют, их одинаковость остаётся внешним условием теста. При разных начальных позициях записи, пробеле в известном префиксе или отсутствии общего target-прохода инструмент не подтверждает равенство контекста и отражает это в `skipped`/`unmatched_forward_positions`. При `matched_forwards=0` сравнение неинформативно. Обрезанные target inputs/positions отклоняются; обрезанная metadata отмечается отдельно. Новые вызовы модели или повторное снятие данных для анализа существующей записи не нужны.

## Поиск события, выдавшего конкретный токен

Число из `first_observed_token_difference.path` — абсолютная позиция выданного токена, а не token ID и не начало target-прохода. Для поиска соответствующего события используйте:

```bash
python -m vllm_bench.mtp_debug.locate_output \
  /data/profile-results/mtp-rejections \
  --position 8210 --output /data/profile-results/token-8210.json
```

По умолчанию проверяются все записанные запросы. Для одного запроса добавьте `--label request-1` или `--request REQUEST_ID`. Принимается также отдельный JSONL или JSON окна. Повторные копии события удаляются; упоминания позиции в input/metadata не считаются выдачей токена. Если выдающий проход отсутствует в записи, запрос попадает в `not_found_in_recorded_outputs`.

Краткая сводка показывает запрос, фактический `token_id`, `cycle`, `source`, `source_row`, accepted count и `output_matches_selection`. Полный JSON дополнительно сохраняет relevant top-2 logits, target inputs/positions, MoME metadata/fingerprints и описание предыдущего записанного verification. Источники:

- `accepted_draft` — принятый draft-токен; дополнительно проверяется совпадение с target argmax;
- `target_correction` — target argmax первой отклонённой позиции;
- `bonus` — дополнительный токен после принятия всего draft.

Для depth=1 проход с первым input на позиции P может выдать токены P+1 и P+2. Инструмент вычисляет нужный offset по фактической длине выхода и accepted count, а не ищет число по тексту файла. `raw_top2_has_tie` и `processed_top2_has_tie` относятся к relevant строке; реальный выбор берётся из verifier inputs. Processed bonus logits текущая инструментация не снимает, поэтому для bonus доступна только raw top-2. Полный MoME-блок по-прежнему включает speculative/неиспользуемые слоты: разные SHA256 не являются самостоятельным доказательством ошибки.

Краткую выборку полей для разбора решений target можно вывести сохранённой командой (нужен `jq`). Из корня репозитория:

```bash
bash scripts/extract_token_choices.sh /path/to/token-8210.json
```

## Автоматический поиск первого расхождения

Чтобы не двигаться по позициям вручную, запустите сохранённый скрипт:

```bash
bash scripts/find_first_difference.sh /path/to/capture --output first-difference.json
```

Скрипт сам добавляет абсолютный путь к `src` в `PYTHONPATH` и запускает `vllm_bench.mtp_debug.find_divergence`. По умолчанию выбираются `request-01` и `request-02`; selector может быть как `LABEL`, так и request ID. Другую пару задайте через `--left NAME --right NAME`. Каждый selector должен однозначно выбирать один запрос того же rank.

Инструмент ищет самое раннее отличие **принятых токенов** на общей записанной позиции, затем находит оба выдавших его события через output spans. Границы циклов могут различаться: например, bonus одного запроса сравнивается с correction другого. Краткая сводка сразу показывает позицию, token IDs, циклы, источники, conditioning inputs/positions и relevant top-2 logits. Полный JSON дополнительно содержит target/MoME snapshots обоих producer-проходов и предыдущие verification.

`recorded_prefix_complete_before_difference=true` означает, что записанная часть префикса от общего начала до найденного различия полностью известна и совпадает. Prompt до начала записи по-прежнему отсутствует. При пробелах, разных начальных позициях либо различии уже первого записанного токена статус обозначает неполноту префикса; найденное отличие нельзя объявлять первым во всей генерации. Если producer отсутствует, но токен виден на входе более позднего прохода, это явно отмечается вместо угадывания источника. Номер позиции заранее знать не требуется; существующие файлы отчётов не перезаписываются.

## Точные тензоры MoME

Хеш полного блока `[3, 8192]` не показывает, какая строка различается, а прежний `sample` охватывает лишь первые 16 значений строки 0. Для следующего уровня диагностики включите `--mome-tensors` в новом запуске:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m vllm_bench.mtp_debug.debug_spec \
  --output /data/profile-results/mome-tensors \
  --rejection-windows --mome-tensors --mome-layers 0 \
  --limit 1000 --execute -- bash /data/scripts/launch-full-model-eager.sh
```

Серверный скрипт должен включать `--enforce-eager --no-enable-prefix-caching`. Сохраните выбранные depth и topology; для минимального воспроизведения рекомендуется depth=1. После прогрева создайте `ARM`, меняйте `LABEL` и подавайте одинаковые greedy-запросы по одному. По умолчанию снимаются все три свёртки слоя 0 на выбранном rank (по умолчанию 0). `--state-fingerprints` можно дополнительно оставить для прежнего формата, но новые descriptors уже содержат точный SHA256 полного тензора и каждой строки.

Новый режим сохраняет только single-request decode. Prefill явно помечается `skipped` без больших дампов входа; его итоговая история видна в state перед первым decode. Для каждой выбранной свёртки записываются `x_before`, выбранный блок `states_before`, результат `y_after`, блок `states_after` и, при inplace, `x_after`. Если inplace-оператор возвращает `None`, в `y_after` сохраняется изменённый `x`, с `y_source=inplace_x` и `return_kind=none`; исходный `None` передаётся вызывающему коду без подмены. При обычном возврате тензора `y_source=returned_tensor`. Для `None` без inplace результат отмечается недоступным, а остальные снимки сохраняются; выход не угадывается по входу. Снимок перед вызовом превращается в независимые байты до возможной модификации исходного тензора. Аргументы и возвращаемый результат оригинальной функции не изменяются.

Файлы `.bin` лежат в `tensors/rank-R-pid-P/forward-F/call-C/`. Их descriptors в `mome_calls[].tensor_capture` содержат относительный путь, исходный dtype, shape, little-endian byte order, byte count, SHA256, исходные strides и послотные hashes/samples. Числа не конвертируются в FP32 перед сохранением. Формат не использует pickle. Сохраняйте весь каталог capture при переносе. Manifest hooks дополнительно содержит версии окружения, backend/precision-флаги и хеши весов выбранных свёрток.

Ограничения по умолчанию: **16 МиБ на тензор, 512 МиБ суммарных payloads на выбранный worker-процесс**. Настройки — `--tensor-max-mib` и `--tensor-budget-mib`. При достижении любого ограничения точная запись останавливается на остаток процесса, с причиной в `tensor-status-rank-R-pid-P.json` и затронутом событии; sampling/metadata продолжают работать до `--limit`. Неполный вызов имеет `status=stopped`, его нельзя трактовать как полный снимок. Тензоры не обрезаются и не подменяются хешами. Ошибка записи/сериализации завершается явной ошибкой. Вне armed target forward снимки не выполняются.

После эксперимента запустите анализатор:

```bash
bash scripts/compare_mome.sh /data/profile-results/mome-tensors \
  --left request-01 --right request-02 --output mome-comparison.json
```

Он автоматически находит первую различающуюся позицию принятых токенов. Для конкретной позиции добавьте `--position 8201`. Если вход — отдельный JSON окна, корень payloads обычно находится среди родительских каталогов; при необходимости задайте `--capture-root /path/to/capture`.

Анализ работает стандартной библиотекой без Torch/NPU, поддерживает BF16, FP16, FP32 и FP64. Проверяются shape, byte order, размер файла и SHA256. Старые записи без точных тензоров, пропущенные/повреждённые файлы и остановленные снимки обозначаются как `unavailable`/`partial`, а не как совпадение.

Отчёт содержит для каждой свёртки:

- `states_before.rows` и `states_after.rows`: точное равенство каждой строки, число побитово отличающихся элементов, максимальную абсолютную разницу конечных значений и до 16 первых отличий с координатами;
- `x_before.rows` и `y_after.rows`: сравнение активаций по общим абсолютным позициям, участвующим в рассматриваемом решении; padding и будущие позиции не используются для этого сравнения;
- полное сравнение захваченных тензоров, parameters/weights и convolution metadata;
- `observations`, разделяющие различие сохранённого состояния, текущих активаций и результата при равных снятых аргументах.

`different_elements` считает точные битовые различия, включая signed zero и NaN payloads. Численная метрика `max_abs_finite` рассматривает только конечные пары; NaN/Inf учитываются отдельно. Для строк координаты относятся к строке (например, номер канала); её номер находится в `row`. Разная длина query не мешает сравнению общей позиции активаций, но полные тензоры с разными shape не объявляются одинаковыми.

При width=3 и accepted count 2 строки `[1, 2]` помечаются как **кандидатная история** по описанному wrapper-контракту. Контракт установленного внешнего kernel здесь не подтверждён: сохраняются все строки, а разные SHA не объявляются доказательством неверной используемой истории. Разные начала producer-проходов также означают разные границы истории. Наблюдение `output_differs_with_equal_captured_arguments_requires_kernel_investigation` появляется только на одинаковом записанном префиксе и начале прохода, при равных полных state/x, известных совпадающих параметрах/весах и metadata. Это направление расследования, а не доказательство ошибки kernel.

CPU-проверки покрывают dtype/целостность, лимиты, inplace-снимки, пропуск prefill, совместимость старых записей, сравнение строк и поиск события. На NPU новый режим ещё требует smoke test: подтвердить hooks и отсутствие `.bin` до `ARM`, получить два запроса, проверить `tensor_capture.status=complete`, выполнить offline-анализ и отдельно проверить достижение малого лимита. Синхронизации/CPU copies/disk I/O меняют тайминги и могут скрывать гонку; эти запуски не подходят для TPOT baseline.

## Автоматический срез HiF8 DSA attention

Этот режим предназначен для поиска расхождения между выходами `qa_conv`/`compresskv_conv` и входом `o_conv`. Поддерживается текущая реализация OmniInfer на A5: `hif8_ds_mla`, noncontiguous KV, latent 512, rotary 64, sparse count 2048, DP=1/TP=2 или 8/PP=1, без CP/KVSP. При неподдерживаемой конфигурации запуск отказывается включать этот режим; неподдерживаемый формат фактического вызова отмечается в capture. Исходники OmniInfer не меняются: hooks устанавливаются только в workers запущенного процесса.

Нужен **новый** capture, старые дампы MoME не содержат аргументы attention. Запускайте из корня checkout:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m vllm_bench.mtp_debug.debug_spec \
  --output /data/profile-results/attention-tensors \
  --rejection-windows --attention-tensors --mome-layers 0 \
  --attention-steps-per-request 32 \
  --limit 1000 --execute -- bash /data/scripts/launch-full-model-eager.sh
```

Серверный скрипт должен включать `--enforce-eager --no-enable-prefix-caching`. Сохраните depth=1 для первого теста и отправляйте одинаковый вход по одному запросу, с `temperature=0`, `seed=123`. После прогрева, перед первым тестовым запросом:

```bash
printf '%s\n' request-01 > /data/profile-results/attention-tensors/LABEL
touch /data/profile-results/attention-tensors/ARM
```

После полного завершения первого запроса, перед вторым:

```bash
printf '%s\n' request-02 > /data/profile-results/attention-tensors/LABEL
```

`--attention-tensors` включает `--mome-tensors` и по умолчанию записывает **все rank текущего TP**. `--attention-ranks 0,3` позволяет снять часть rank, но полное TP-сравнение будет отмечено как неполное. Без нового режима прежний default rank=0 сохраняется. Старые анализаторы получили `--rank`, по умолчанию 0, чтобы выбирать нужную копию события в многопроцессном capture.

В `target_forward.attention_trace` сохраняются:

- Фактический вызов quantized lightning indexer: quantized query, веса, query scales, действующие key/scales по логическим позициям, lengths, параметры и упорядоченный результат. Поддерживается также позиционная сигнатура `cann_ops_transformer.quant_lightning_indexer`.
- Фактический вызов HiF8 sparse attention: поглощённый query, sparse indices, выбранные packed KV строки, dummy value, sinks при наличии, lengths, параметры и выход kernel. Packed строка разбирается на 512 байт latent, 128 байт BF16 rotary и 16 байт FP32 scales. HiF8 сравнивается **по байтам**, без собственного численного декодера.
- Вход и выход `_post_attn_absorb`, идентичность W_UV, полный вход/выход `o_conv`, точные MoME состояния и параметры выбранного слоя. Новые collectives/barriers не добавляются; аргументы и возвращаемые значения оригинальных вызовов сохраняются.

KV сохраняется через действующий `block_table` и реальный размер страницы аргумента оператора. Физические номера страниц остаются для аудита; сравнение идёт по логическим позициям. Для копирования используется basic slice каждой нужной страницы на CPU, затем выбор строк по байтовым смещениям: NPU advanced indexing (`tensor[pages, slots]` / `aclnnIndex`) не вызывается. Каждая нужная физическая страница копируется один раз за снимок; весь cache на CPU не переносится. Indexer получает всю действующую историю, attention — объединение реально выбранных строк. Неиспользуемая аллокация KV не дампится. Для attention payloads сохраняется SHA256 полного файла, без тысяч отдельных строковых samples/hashes; точные строки сравниваются offline по `.bin`.

Ограничение `--attention-steps-per-request` относится к первым decode-проходам каждого request ID после включения записи; prefill не расходует квоту. После квоты точные attention/MoME snapshots пропускаются с явной причиной. Лимиты payloads общие для MoME и attention: по умолчанию 16 МиБ на один тензор и 512 МиБ **на worker**, то есть до 1 ГиБ для TP2 и 4 ГиБ для TP8. При необходимости увеличьте `--tensor-max-mib`/`--tensor-budget-mib`; достигнутый лимит останавливает точные дампы этого worker до перезапуска. Проверяйте `tensor-status-rank-*-pid-*.json` и `hooks-pid-*.json`.

После получения двух запросов:

```bash
bash scripts/compare_attention.sh /data/profile-results/attention-tensors \
  --left request-01 --right request-02 --output attention-comparison.json
```

Команда сама находит первое отличие принятых токенов и сравнивает захваченные операнды на одинаковом записанном префиксе до него. Из диагностики исключаются строки с отвергнутым speculative conditioning и позиции после расхождения. Разные границы forward и пробелы истории не угадываются, а отмечаются как недостаточное покрытие. Проверяются все выбранные слои и rank текущего TP; локальные выходы проекции соединяются в порядке TP rank и сверяются с входом `o_conv` каждого rank. Внутри каждого запроса дополнительно проверяется передача выхода attention во вход проекции.

Создаются три файла, существующие файлы не перезаписываются:

- `attention-comparison.json` — все сравнения, метрики, сведения о первом отличающемся токене, состояния, TP reconstruction и ограничения покрытия.
- `attention-comparison.summary.json` — компактная сводка.
- `attention-comparison.summary.txt` — короткий текст для передачи. **Для начала пришлите первые три строки: `result`, `first_boundary`, `evidence`.**

`first_boundary` содержит position, layer, rank, этап и поле. Численные метрики включают `different_elements`, `max_abs_finite`, относительную L2 и максимальное расстояние ULP; для HiF8/opaque данных — различающиеся байты. Порядок indices и совпадение их множества проверяются отдельно. NaN/Inf, metadata/weights и нарушения протокола отражаются автоматически. Различия полного MoME state остаются отдельными наблюдениями: используемая kernel история пока не доказана.

Чтобы вывести сравнение четырёх входов indexer (`query`, `query_dequant_scale`, `weights`, `key_scales`) из уже готового полного отчёта:

```bash
bash scripts/inspect_indexer.sh attention-comparison.json
```

По умолчанию rank/position/layer берутся из `first_boundary`. Для конкретного среза добавьте `--rank 0 --position 8193 --layer 0`. Скрипт выводит context и четыре строки с `exact_equal`, статусом и доступными метриками; alias `query_scale` поддерживается. Отсутствующие или неоднозначные поля явно отмечаются и не считаются совпадением. Нужен полный JSON, бинарные дампы для этой команды не требуются.

Если key/scales сравниваются по байтам, скрипт сам переводит `first_different_offsets` в логические позиции по длине строки и `compared_positions`. Для короткого вывода только истории используйте `bash scripts/inspect_indexer.sh attention-comparison.json --history-summary`: три строки содержат context с вариантом оператора, `key_rows` и `key_scales`. `sampled_different_positions` относится к сохранённым первым смещениям; полный список позиций известен только при `all_different_positions_known=true`. Число отличающихся байтов не является численной величиной ошибки scales. Режим indexer нужно определять по оператору и его параметрам: `hif8_ds_mla` описывает attention KV и сам по себе не доказывает dtype scales indexer.

Если `first_boundary` указывает на `indexer_kernel_input.metadata`, выполните `bash scripts/inspect_indexer.sh attention-comparison.json --difference-summary`. По умолчанию берётся контекст first_boundary; явные `--rank/--position/--layer` также поддерживаются. Три строки показывают первые три отличающихся элемента metadata, равенство операндов/выходов выбранного контекста и первое найденное отличие вне этого поля во всём записанном сравнении. `null` или `unavailable` не означают равенство. Отсутствие другого отличия относится только к сравнимому capture, особенно при квоте. Metadata сохраняется в исходном отчёте как реальный аргумент оператора с непроверенной семантикой; этот дополнительный вывод не доказывает, что metadata не влияет на результат или что все аргументы kernel равны.

Основные значения `result`:

| Значение | Что найдено |
| --- | --- |
| `INDEXER_INPUT_OR_PARAMETERS_DIFFER` / `INDEXER_HISTORY_DIFFERS` | Первое отличие в quantized indexer operands или его истории |
| `DSA_SELECTED_INDICES_DIFFER` | Изменился результат indexer либо indices фактического attention |
| `ATTENTION_SELECTED_KV_DIFFERS` / `ATTENTION_KERNEL_INPUT_DIFFERS` | Различаются выбранные KV либо остальные входы attention |
| `ATTENTION_OUTPUT_DIFFERS_WITH_EQUAL_SELECTED_INPUTS` | Выход attention различается при равных снятых действующих операндах и параметрах; направление для проверки kernel |
| `POST_ATTENTION_PROJECTION_DIFFERS` | Первое отличие на выходе проекции |
| `TP_RECONSTRUCTION_MISMATCH` | Вход o_conv не совпал с конкатенацией снятых TP chunks |
| `INCOMPLETE_CAPTURE` | Не хватает rank, операндов, истории или проверенного сопоставления; это не вывод о причине |
| `PROTOCOL_INVARIANT_FAILURE` / `NONFINITE_VALUES_OBSERVED` | Нарушен проверяемый протокол либо обнаружены NaN/Inf |
| `NO_DIFFERENCE_IN_CAPTURED_ATTENTION` | В сравнимой записанной части attention отличий нет; полная генерация этим не проверена |

Другие этапы, например Q/KV MoME input/output или параметры исполнения, обозначаются собственным `*_DIFFERS`. Даже при неполном покрытии наблюдаемая `first_boundary` остаётся в отчёте вместе с причиной ограничения. Это первая наблюдаемая граница, **не доказанная первопричина**; Q/KV и indexer могут выполняться параллельно. Равенство prompt до начала записи предполагается извне.

CPU-тесты используют синтетические файлы всех восьми rank и mocks фактических wrappers, проверяют позиционную сигнатуру indexer, `None`/tuple returns, paging, квоты/лимиты, пропуск отвергнутых строк, пропуски истории, диагностику каждого участка и TP reconstruction. На NPU требуется smoke test: подтвердить отсутствие дампов до ARM, полные indexer/attention/projection stages после него, покрытие восьми rank, новые отчёты и явную остановку при малом budget. Синхронные копирования могут скрыть гонку и меняют TPOT; измерять производительность с этим режимом нельзя.

## Равенство исходного ki до квантования

Если standalone-тест проходит, следующий срез — настоящий неквантованный вход `ki` оператора `indexer_quant_cache`. Для него нужен **новый запуск сервера** с дополнительным `--indexer-ki`. Старый capture содержит quantized cache, но не этот вход.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m vllm_bench.mtp_debug.debug_spec \
  --output /data/profile-results/indexer-ki \
  --rejection-windows --indexer-ki --mome-layers 0 \
  --execute -- bash /data/scripts/launch-full-model-eager.sh
```

`--indexer-ki` включает прежние attention/MoME tensors и все rank по умолчанию. Сохраняйте DP=1/TP=2 или 8/PP=1, `--enforce-eager --no-enable-prefix-caching`, выбранный depth и greedy sampling. Требуется именно ветка QLIv2 MXFP8; другой indexer явно отвергается при установке hooks. После прогрева задайте `LABEL=request-01`, создайте `ARM` **до подачи первого тестового запроса**, дождитесь его полного завершения, смените LABEL на `request-02` и подайте тот же вход. Это важно для захвата всего prefill с позиции 0.

Новый hook снимает аргументы **фактического** `indexer_quant_cache` внутри выбранного target `_update_cache`, включая prefill и decode:

- `ki_before` — точный `ki[T,128]` после projection/norm/RoPE, перед квантованием;
- `ki_after` — вход после вызова для проверки отсутствия модификации;
- `key_after`, `scales_after` — результат непосредственно после вызова, собранный по реальным активным write slots;
- полные input IDs/positions текущего forward, отображение строк Ki в позиции, активные и padding slots, параметры оператора, shape/strides/storage offsets кэшей.

Все NPU-копирования используют прежний безопасный basic-page-slice → CPU → byte offsets путь, без advanced indexing. Возвращаемое значение оригинального оператора, включая `None`, не изменяется. Для decode ссылки на уже снятые cache operands indexer добавляются к новой записи без повторного дампа истории.

Записи лежат в отдельных `ki-rank-R-pid-P.jsonl`: они записываются после target forward **независимо от sampler**, поэтому prefill не потеряется из-за отсутствия speculative verification. В прежних sampler-логах сохраняется ограничение snapshots до 64 значений; для Ki sidecars IDs/positions снимаются полностью. До ARM Ki-снимки не выполняются. `--indexer-ki-steps-per-request 32` ограничивает число target forwards на request ID **вместе с prefill-чанками**. При большом числе чанков увеличьте квоту, чтобы захватить первый decode. Общие лимиты `.bin` прежние: 16 МиБ на тензор, 512 МиБ на worker, на Ki/MoME/attention совместно; настройки `--tensor-max-mib` и `--tensor-budget-mib`.

После двух запросов:

```bash
bash scripts/compare_indexer_ki.sh /data/profile-results/indexer-ki \
  --output ki-comparison.json
```

По умолчанию выбираются `request-01`/`request-02`, число rank из topology capture и все захваченные слои. Старые sidecars без topology используют default TP8. Другие LABEL/request ID задаются через `--left`/`--right`; `--expected-ranks` задаёт явное ожидаемое число rank и не может противоречить записанному TP. Отсутствующий worker не уменьшает ожидаемый TP.

Анализатор сопоставляет **prefill** по абсолютным позициям, проверяет полноту префикса от нуля и точное совпадение input IDs, затем сравнивает Ki и записанные key/scales. Разные границы prefill-чанков допустимы и отражаются через export provenance. Внутри каждого запроса prompt-кэш после записи сопоставляется с его чтением перед первым decode indexer, с учётом захваченных записей того же decode. Изменение Ki оператором проверяется и для prefill, и для decode. Между запросами decode Ki не объявляется равным/разным по совпадению отдельных токенов: различающиеся speculative conditioning и ненаблюдаемая история требуют отдельной проверки.

Результат сохраняется в `ki-comparison.json`, `.summary.json` и `.summary.txt`. Пришлите **только три строки из summary.txt**: `result`, `first_difference`, `evidence`. Величина ошибки первой различающейся строки Ki, её позиция, число отличающихся строк и байтов вычисляются автоматически. Неполнота, повреждённые payloads, другой prompt, пропущенный prefill/первый decode и отсутствие rank отмечаются явно.

| Result | Наблюдение |
| --- | --- |
| `KI_INPUT_DIFFERS` | Неквантованный Ki уже различается перед оператором; дальше проверяем формирование `wk → norm → RoPE` |
| `KI_MUTATED_BY_CACHE_CALL` | Вход Ki изменился между снимками до/после вызова |
| `CACHE_OUTPUT_DIFFERS_WITH_EQUAL_RECORDED_KI` | Записанные key/scales различаются при равном записанном Ki; физическая адресация/layout могут различаться, это не доказательство недетерминизма kernel |
| `CACHE_CHANGED_AFTER_WRITE` | Prompt-кэш при первом decode-чтении отличается от последней захваченной записи; возможны последующая запись, lifetime или адресация |
| `NO_DIFFERENCE_IN_CAPTURED_PREFILL_KI` | Различия исходного Ki, записи prompt-кэша и его сохранности не воспроизвелись в сравнимом срезе |
| `INCOMPLETE_CAPTURE` | Не хватает данных для полного вывода; обнаруженное отличие и причины ограничений остаются в отчёте |

Анализатор также экспортирует пару реальных входов из выбранных prefill-вызовов в `ki-comparison-replays/left/input.json` и `right/input.json`, с собственными `.bin`. При найденном различии выбирается prefill-вызов, содержащий его позицию; при отсутствии различия — первый полный prefill. Для случая мутации decode входа export остаётся prefill-парой. Метаданные содержат исходные позиции, slots, параметры и layout. Экспорт не требует переноса исходного capture при дальнейшем replay:

```bash
bash scripts/test_indexer_quant_cache.sh \
  --input ki-comparison-replays/left/input.json \
  --output /data/profile-results/real-ki-left --execute
```

В standalone сохраняется точный Ki выбранного вызова; slots/layout варьируются по standalone-сценариям, это не полное воспроизведение allocator/streams модели. Отчёты и каталоги replay не перезаписываются. CPU-тесты проверяют независимый prefill sidecar, полный snapshot сверх 64 токенов, сопоставление чанков, различение трёх источников расхождения, scale reshape `[1,4] → [1,2,2]`, реальную сигнатуру и inplace-снимки через mocks, ошибки/лимиты и переносимость replay. На NPU новый hook ещё требует smoke test; синхронные снимки могут скрыть гонку.

## Standalone-тест indexer_quant_cache без модели

Для отдельной проверки записи MXFP8 K-кэша QLI v2 не нужны checkpoint, сервер vLLM или TP8. Нужен один A5 NPU и серверное Python-окружение с `torch`, `torch_npu`, CANN и установленным `cann_ops_transformer`, содержащим `indexer_quant_cache`.

```bash
bash scripts/test_indexer_quant_cache.sh \
  --output /data/profile-results/indexer-quant-cache --execute
```

Без `--execute` команда только показывает план и не импортирует ML-пакеты. Тест работает в отдельном процессе на `npu:0`; другой NPU задаётся через `--device npu:N` с учётом видимости устройств в окружении. Прогон не подходит для измерения TPOT.

По умолчанию фиксированный CPU-generated вход `ki` имеет shape `[8192,128]`, dtype BF16, seed=123. Случайные строки нормализованы по RMS; несколько первых строк заменены нулями и значениями около степеней двойки. Это синтетические данные, а не восстановленные активации модели. `--pattern random` отключает специальные строки, `--tokens`, `--dtype`, `--seed` меняют вход. Один и тот же вход используется во всех испытаниях.

Оператор вызывается с той же сигнатурой, что в ветке QLI v2 модели: `indexer_quant_cache(key, scales, ki, slots_int32, quant_mode="mxfp8", round_scale=True)`. Key хранится как E4M3, четыре scale-байта на строку — в `uint8`. Default `--block-size 128`; подставьте фактический размер страницы indexer на сервере, если он другой. Shared layout повторяет расположение страницы `[attention: B*656][indexer key: B*128][scales: B*4]`, с соответствующими strides и offsets. Для дополнительного padding можно задать `--page-bytes`.

Семь сценариев, по 20 повторов каждого:

| Case | Что меняется |
| --- | --- |
| `shared` | Повторные вызовы с новыми буферами общего strided storage |
| `contiguous` | Key/scales находятся в отдельных contiguous-буферах |
| `reuse` | Один и тот же кэш перезаписывается без очистки между повторами |
| `poison` | До вызова память заполняется поочерёдно байтами 0, 165, 255 |
| `permuted` | Логические страницы размещены в другом порядке физических страниц |
| `shifted` | Первый slot сдвинут к концу страницы, вход пересекает границы страниц |
| `chunked` | Тот же `ki` записывается чанками; default `--chunk-size 256` |

Список можно сократить через `--cases shared,reuse,poison`, число повторов изменить через `--repeats 100`. Первый выбранный case даёт общий reference. Сравниваются точные байты key/scales после восстановления **логического** порядка строк, а также повторы внутри каждого case. Отдельно проверяется неизменность исходного `ki`, guard-страниц, padding, attention-области и всех slots, которые оператор не должен записывать. GPU advanced indexing не используется: снимки сырых буферов копируются на CPU. После всей записи выполняется синхронизация; `--sync-each-call` дополнительно синхронизирует каждый чанк.

Результат — `result.json` и короткий `summary.txt`. Для разбора достаточно прислать три строки `result`, `first_failure`, `failure_counts`. При первом отличии сохраняются reference/actual bytes логических key/scales в `first-failure/`. Общий результат различает:

- `PASS_IN_TESTED_CASES` — все проверенные представления совпали, вход и guards не изменены.
- `REPEATED_CASE_DIFFERS` — результат отличается от первого повтора того же case; для fresh/poison сценариев учитывайте изменение аллокации/начального заполнения.
- `SCENARIO_DEPENDENT_OUTPUT` — повторы case стабильны, но отличаются от общего reference, например при разбиении на чанки. Это наблюдение, не доказательство ошибки оператора.
- `OUT_OF_SLOT_WRITE` или `INPUT_MUTATED` — изменились байты вне разрешённых slots либо исходный вход.
- `ERROR` — вызов или подготовка не выполнены; ошибка и traceback записаны в JSON, CPU/fake fallback отсутствует.

Фиксированный вход сохраняется как exact `.bin` с descriptor в `input.json`. Его можно переиграть без повторной генерации:

```bash
bash scripts/test_indexer_quant_cache.sh \
  --input /data/profile-results/indexer-quant-cache/input.json \
  --output /data/profile-results/indexer-quant-cache-replay --execute
```

`--input` также принимает descriptor реального **неквантованного** `ki` в формате `TensorWriter`, shape `[T,128]` или `[T,1,128]`. Пути `.bin` должны разрешаться относительно JSON, SHA256/размеры проверяются. Для real-input replay используйте export нового `--indexer-ki` capture и `compare_indexer_ki.sh`, описанный выше. Обычный attention capture без этого флага не содержит исходный Ki: quantized `key_rows` и `key_scales` не подходят для его точного восстановления.

CPU-проверки покрывают layout, адресацию, guard-проверки, repeat/scenario классификацию, загрузку raw-входа, планирование и сигнатуру вызова через контролируемые mocks. Численного независимого эталона MXFP8 здесь нет. Успешный standalone-тест исключает наблюдаемое расхождение только в проверенных условиях: он не доказывает равенство исходных `ki` модели и не воспроизводит её конкурирующие streams. Сам NPU-прогон здесь не выполнялся.

## Проверки и ограничения

```bash
python -m unittest discover -s tests
```

CPU-тесты проверяют greedy acceptance, arming/limit, сопоставление rejection windows, границы запросов и target/MoME hooks через mocks. Проверки структуры дополнительно проверяют импорт hooks из нового пакета и отказы при конфликте инструментаций. Реальный NPU-запуск диагностики после реорганизации здесь не выполнялся.

Серверный smoke test: запустить полный V1 server, прогреть до `ARM`, подтвердить отсутствие записей до включения, получить события после включения и проверить лимит. Отдельно проверить eager rejection windows на одном запросе, сводку и fingerprints выбранного слоя. Сохранить версии окружения, команду запуска и логи рядом с результатами.
