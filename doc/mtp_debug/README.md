# Отладка greedy MTP verification и MoME

Этот сценарий подключает диагностику к существующему **полному** запуску Pangu с MTP. Он записывает draft tokens, target argmax, bonus token, результат acceptance и проверку относительно CPU-эталона greedy prefix acceptance. Дополнительный режим сохраняет target inputs и MoME metadata вокруг rejection.

Для замеров времени standalone MTP используйте [профилирование MTP](../mtp_profile/README.md). Диагностические копии на CPU и синхронизации меняют тайминги и могут скрывать гонки; результаты этого запуска не являются baseline производительности.

## Требования

- Python 3.10+ и рабочее окружение OmniInfer/vLLM/torch_npu с V1 NPUModelRunner.
- Полная модель с настроенными MTP и topology в вашем серверном launch script. Wrapper эти настройки не меняет.
- Только greedy sampling (`all_greedy`); после включения записи другой sampling вызывает явную ошибку.
- Для `--rejection-windows` необходимы eager (`--enforce-eager`), отключённый prefix caching (`--no-enable-prefix-caching`) и один активный запрос. Не посылайте постороннюю нагрузку.
- Новая директория результатов. По умолчанию записывается rank 0; другой rank выбирается переменной `PANGU_SPEC_DEBUG_RANK`.

Отдельных конфигов этого сценария пока нет. Используются настройки вашего полного запуска и параметры CLI. Исходники OmniInfer не изменяются. Не сочетайте wrapper с MTP recorder или `--annotate-layers`: каждый использует свой `sitecustomize.py`, конфликт отклоняется до старта.

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

Этот режим предназначен для поиска расхождения между выходами `qa_conv`/`compresskv_conv` и входом `o_conv`. Поддерживается текущая реализация OmniInfer на A5: `hif8_ds_mla`, noncontiguous KV, latent 512, rotary 64, sparse count 2048, DP=1/TP=8, без CP/KVSP. При неподдерживаемой конфигурации запуск отказывается включать этот режим; неподдерживаемый формат фактического вызова отмечается в capture. Исходники OmniInfer не меняются: hooks устанавливаются только в workers запущенного процесса.

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

`--attention-tensors` включает `--mome-tensors` и по умолчанию записывает **все восемь rank**. `--attention-ranks 0,3` позволяет снять часть rank, но полное TP-сравнение будет отмечено как неполное. Без нового режима прежний default rank=0 сохраняется. Старые анализаторы получили `--rank`, по умолчанию 0, чтобы выбирать нужную копию события в многопроцессном capture.

В `target_forward.attention_trace` сохраняются:

- Фактический вызов quantized lightning indexer: quantized query, веса, query scales, действующие key/scales по логическим позициям, lengths, параметры и упорядоченный результат. Поддерживается также позиционная сигнатура `cann_ops_transformer.quant_lightning_indexer`.
- Фактический вызов HiF8 sparse attention: поглощённый query, sparse indices, выбранные packed KV строки, dummy value, sinks при наличии, lengths, параметры и выход kernel. Packed строка разбирается на 512 байт latent, 128 байт BF16 rotary и 16 байт FP32 scales. HiF8 сравнивается **по байтам**, без собственного численного декодера.
- Вход и выход `_post_attn_absorb`, идентичность W_UV, полный вход/выход `o_conv`, точные MoME состояния и параметры выбранного слоя. Новые collectives/barriers не добавляются; аргументы и возвращаемые значения оригинальных вызовов сохраняются.

KV сохраняется через действующий `block_table` и реальный размер страницы аргумента оператора. Физические номера страниц остаются для аудита; сравнение идёт по логическим позициям. Для копирования используется basic slice каждой нужной страницы на CPU, затем выбор строк по байтовым смещениям: NPU advanced indexing (`tensor[pages, slots]` / `aclnnIndex`) не вызывается. Каждая нужная физическая страница копируется один раз за снимок; весь cache на CPU не переносится. Indexer получает всю действующую историю, attention — объединение реально выбранных строк. Неиспользуемая аллокация KV не дампится. Для attention payloads сохраняется SHA256 полного файла, без тысяч отдельных строковых samples/hashes; точные строки сравниваются offline по `.bin`.

Ограничение `--attention-steps-per-request` относится к первым decode-проходам каждого request ID после включения записи; prefill не расходует квоту. После квоты точные attention/MoME snapshots пропускаются с явной причиной. Лимиты payloads общие для MoME и attention: по умолчанию 16 МиБ на один тензор и 512 МиБ **на worker**, то есть до 4 ГиБ для TP8. При необходимости увеличьте `--tensor-max-mib`/`--tensor-budget-mib`; достигнутый лимит останавливает точные дампы этого worker до перезапуска. Проверяйте `tensor-status-rank-*-pid-*.json` и `hooks-pid-*.json`.

После получения двух запросов:

```bash
bash scripts/compare_attention.sh /data/profile-results/attention-tensors \
  --left request-01 --right request-02 --output attention-comparison.json
```

Команда сама находит первое отличие принятых токенов и сравнивает захваченные операнды на одинаковом записанном префиксе до него. Из диагностики исключаются строки с отвергнутым speculative conditioning и позиции после расхождения. Разные границы forward и пробелы истории не угадываются, а отмечаются как недостаточное покрытие. Проверяются все выбранные слои и восемь rank; локальные выходы проекции соединяются в порядке TP rank и сверяются с входом `o_conv` каждого rank. Внутри каждого запроса дополнительно проверяется передача выхода attention во вход проекции.

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

## Проверки и ограничения

```bash
python -m unittest discover -s tests
```

CPU-тесты проверяют greedy acceptance, arming/limit, сопоставление rejection windows, границы запросов и target/MoME hooks через mocks. Проверки структуры дополнительно проверяют импорт hooks из нового пакета и отказы при конфликте инструментаций. Реальный NPU-запуск диагностики после реорганизации здесь не выполнялся.

Серверный smoke test: запустить полный V1 server, прогреть до `ARM`, подтвердить отсутствие записей до включения, получить события после включения и проверить лимит. Отдельно проверить eager rejection windows на одном запросе, сводку и fingerprints выбранного слоя. Сохранить версии окружения, команду запуска и логи рядом с результатами.
