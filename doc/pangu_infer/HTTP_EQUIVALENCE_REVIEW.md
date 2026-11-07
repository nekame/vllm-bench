# Повторное ревью расхождения с HTTP

Проверка от 2026-11-07, после `8d24514`, совместно с двумя агентами Astra.
OmniInfer проверялся только чтением, никаких изменений его исходников нет.

Пользователь подтвердил: после `--fix-mome-prefill-state` повторы стали
одинаковыми, но отличие от HTTP осталось примерно со второго–третьего токена.
Это подтверждает устранение наблюдаемой нестабильности, а не полное совпадение
с HTTP или правильность всех fused NPU вычислений. Новых NPU-запусков в этом
ревью не было.

## Найденное семантическое отличие

Локальный адаптер и исходный HTTP-путь теперь могут выполнять первую target
verification с разной историей MoME. При значениях из прежней трассы:

| Поле | Исходный builder | Локальный адаптер |
| --- | --- | --- |
| computed / prompt | 8192 / 8192 | 8192 / 8192 |
| исходный runner accepted | 1 | 1 |
| effective accepted | 1 | 8 |
| initial logical block, APC и block=128 | 64 | 63 |

В `omni/attention/backends/mome.py`, `build`, нормализация accepted использует
`computed < prompt`; равенство пропускается. В
`_compute_prefix_caching_block_indices` равенство уже попадает в speculative
ветку: `ceil((computed - accepted + depth + 1) / block_size) - 1`.
При depth=7 это даёт приведённые индексы. Локальный `mome_prefill.py` исправляет
только target на этой границе, не меняя исходный runner accepted.

Первый выходной токен выбирается после prefill. Следующий выбирается уже
первой target verification. Поэтому различие здесь совместимо с наблюдаемым
ранним расхождением; оно не обязано менять самый первый токен. Совпадение
нескольких argmax также не доказывает равенство logits.

Стабильный HTTP может повторять неправильное чтение: содержимое выбранной
страницы может воспроизводиться. Это возможное объяснение, а не доказанный
диагноз удалённого сервера. Его фактические source hashes и аргументы fused
оператора в момент первой verification пока неизвестны.

## Откуда берётся accepted=1

Дополнительно просмотрены первичные исходники **vLLM v0.25.1**, а не выданная
за серверную версия из Dockerfile:

- [InputBatch.add_request](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_input_batch.py#L428)
  задаёт начальный accepted=1.
- [GPUModelRunner._prepare_inputs](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_model_runner.py#L1902)
  в async non-align ветке снова заполняет accepted единицами; GPU-коррекция
  предназначена для запросов с drafts на предыдущем target-шаге.
- [Обновление запросов](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_model_runner.py#L1221)
  различает первую verification и последующие: перед первой предыдущий
  target-шаг был prefill, его prev_num_draft_len равен нулю.
- [GPU update kernel](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/spec_decode/utils.py#L518)
  не исправляет accepted у запроса без previous drafts.

В локальном Omni `NPUModelRunner._refresh_mome_num_prompt_tokens` публикует
реальную длину prompt после подготовки входов. HTTP-фронтенд сам по себе
не исключает проверку `computed == prompt`. Это цепочка для просмотренной
версии vLLM; равенство серверных исходников ещё не установлено. Audit теперь
включает fingerprints InputBatch, spec-decode utils и LLMEngine.

## Проверенные альтернативы

- `omni_worker.py` создаёт builders из исходных `kv_cache_spec` и config,
  сохраняет bound prompt buffer, обновляет ссылки proposer штатным
  `initialize_attn_backend` до прогрева. Потерянной draft-инициализации
  или дополнительных численных изменений здесь не найдено.
- Локальный `build_for_drafting` сохраняет исходную synthesis-политику.
  Target-коррекция не применяется к draft-вызовам.
- В исходном Eagle имеется `num_accepted_tokens.fill_(depth + 1)` внутри
  `set_inputs_first_pass`, но он ограничен `fix_multi_mtp_kvcache` /
  multi-head режимом. Он не является общим исправлением первого target
  decode при рекуррентном использовании одного MTP head.
- Создание LLMEngine в corrected path повторяет значимую последовательность
  штатной factory с явным выбором worker. Новый дефект этого пути не найден.
- HTTP пользователя не передаёт `min_tokens`; отличие репозиторного HTTP
  benchmark-клиента по этому параметру к его запуску не относится.
- При temperature=0 draft использует argmax до top-k. Изменение seed/top-k
  не объясняет уже отличающиеся вычисленные hidden states.

Есть отдельная гипотеза о первой **single-token draft** после финального
draft-prefill: она тоже может достичь `computed == prompt` с accepted=1.
Локальный адаптер намеренно сохраняет её исходную семантику. Это общий путь
обоих запусков; сам по себе он не объясняет разный greedy target-результат
при правильной verification. Без проверки фактических prefill writes нельзя
переносить target-коррекцию на draft. Дополнительная правка не внесена.

## Граница вывода и следующая диагностика

Новой доказанной ошибки обвязки относительно HTTP не найдено. Применять ещё
настройки или менять draft-код на основании одного несовпадения не следует.
Первое установленное отличие сейчас — исправленная и исходная политика
выбора target MoME history.

Когда доступна диагностика HTTP, полезен один сопоставимый снимок первой
verification: loaded source hashes, builder class, computed/prompt, raw и
effective accepted, фактический `cache_indices`, initial-state index,
две выбранные строки history **до** target forward, LM-head input и argmax.
Это локализует чтение состояния, не требует серии переборов параметров.

Текущий boundary trace снимает значения в hook LM head после forward и
использует runner block table, а не фактический builder cache-index buffer.
Его снимки нельзя выдавать за прямую запись аргументов и history до NPU
оператора. Изменение trace сейчас не вносилось.
