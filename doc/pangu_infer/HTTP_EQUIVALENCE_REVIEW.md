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

## Результаты проверки истории запросов после `0cde5b4`

Пользователь сообщил следующие результаты при одинаковых входах:

| Вариант | Результат |
| --- | --- |
| `--warmup 0`, 16 повторов | Первый ответ A, остальные B |
| Прогрев, затем `cold-first` | Первый ответ A, остальные B |
| `cold-each` | Каждый ответ A |
| Прогрев ответами по 1024 токена, `reuse` | Каждый ответ B |

Оба результата отличаются от HTTP. A и B также расходятся примерно со
второго–третьего токена. Следовательно, одной длиной warmup совпадение с HTTP
не объясняется. Сброс prefix entries после уже выполненного warmup возвращает
A, хотя raw NPU states не обнуляются. Это существенно сужает подозрение до
пути APC reuse/пересчёта и его последствий; прежняя недетерминированность
сырых страниц сама по себе этот воспроизводимый рисунок не объясняет.

Совпадение token_ids[0] доказывает только равенство выбранного prefill argmax.
Равенство финальных prefill hidden states/logits ещё не установлено. При
квантованном кеше и выключенной batch-invariant арифметике разный объём
пересчёта также может влиять на численные результаты. Различие A/B само
по себе не является доказательством ошибки восстановления MoME state.

### Дополнительная зацепка в draft chunked prefill

В просмотренном Omni `EagleProposerPatch.prepare_next_token_ids_padded`
строит backup token через
`request.get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)`.
При discard текущего sample выбирается этот backup.
`set_inputs_first_pass` сдвигает prompt IDs и вставляет next_token_ids
в последнюю позицию каждого request slice.

Если num_tokens_no_spec равен полной длине prompt (как при
[добавлении request в vLLM v0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_input_batch.py#L324)), промежуточный chunk, заканчивающийся
на 2048, получает в последний shifted slot prompt[8191], тогда как для
непрерывного MTP-prefill нужен prompt[2048]. Это конкретная условная
несогласованность индексации, а не случайный sampled token. Ревью с Astra
не нашло в просмотренном sample/propose пути общего пропуска drafter для
таких промежуточных чанков.

Путь общий для исходного HTTP и нашего Omni. Он может влиять на draft
state и acceptance, но ещё не доказан как причина target-output расхождения:
правильная greedy verification должна отклонять неверные draft proposals.
Идентичность версии/пути на сервере и фактические сохранённые snapshots
не проверены. Дополнительная правка на основании этой зацепки не внесена.

Также пока не проверена совместимость draft MoME с Mamba APC lookup:
Omni MomeManager наследует его без отдельной адаптации, а просмотренный
vLLM v0.25.1 MambaManager принимает, но не использует drop_eagle_block.
Hybrid coordinator при этом увеличивает lookup length на блок для Eagle
групп перед предполагаемым drop. Для одинакового prompt это ещё не доказывает
неправильный checkpoint; нужны реальные группы и возвращённые длины tables.
`mamba_cache_mode=align` не является обходным решением: Omni явно запрещает
его для MoME из-за отсутствия state-copy contract.

Следующий локализующий снимок — последний target prefill LM-head input/logits
и первая verification для A и B. Существующий boundary trace уже сохраняет
эти две точки; для проверки самого чтения history всё ещё нужны реальные
аргументы/cache rows до model forward, как описано выше. Новая серия
параметрических вариантов по этим результатам не обоснована.
