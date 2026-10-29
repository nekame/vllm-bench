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
bash doc/mtp_debug/extract_token_choices.sh /path/to/token-8210.json
```

## Проверки и ограничения

```bash
python -m unittest discover -s tests
```

CPU-тесты проверяют greedy acceptance, arming/limit, сопоставление rejection windows, границы запросов и target/MoME hooks через mocks. Проверки структуры дополнительно проверяют импорт hooks из нового пакета и отказы при конфликте инструментаций. Реальный NPU-запуск диагностики после реорганизации здесь не выполнялся.

Серверный smoke test: запустить полный V1 server, прогреть до `ARM`, подтвердить отсутствие записей до включения, получить события после включения и проверить лимит. Отдельно проверить eager rejection windows на одном запросе, сводку и fingerprints выбранного слоя. Сохранить версии окружения, команду запуска и логи рядом с результатами.
