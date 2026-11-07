# Omni2: серверный путь без HTTP listener

`--engine omni2` использует функции HTTP-launcher установленного vLLM,
вместо нашего синхронного цикла `LLMEngine.step()`:

1. Штатные `cli_env_setup`, `make_arg_parser`, `validate_parsed_serve_args`.
2. `build_async_engine_client(args)` с серверным UsageContext.
3. Штатные `get_supported_tasks` и `init_app_state`.
4. `openai_serving_completion.create_completion(CompletionRequest(...))`.
5. Чтение полученного completion-stream непосредственно в процессе.

Сохраняются AsyncLLM/AsyncMPClient, tokenizer, OnlineRenderer, преобразование
HTTP sampling defaults, native request IDs и MTP proposal policy. Запускается
`uvloop`, как в штатном сервере. Сокет HTTP, uvicorn и FastAPI middleware
не запускаются. Engine IPC и коммуникации NPU остаются штатными.
Это путь **`/v1/completions`**, не chat endpoint с chat template.

Реализация вызывает установленный серверный код, не копирует и не патчит его.
API/signatures сверены по первичным исходникам **vLLM v0.25.1**:
[launcher](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/entrypoints/openai/api_server.py),
[completion serving](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/entrypoints/openai/completion/serving.py).
Работа установленной на сервере версии и NPU требует проверки. Одинаковые
настройки вашего shell-launcher всё ещё нужно передать; одного имени движка
недостаточно для гарантии совпадения выходов.

## Запуск

В прежней команде замените `--engine omni` на `--engine omni2` и укажите новый
output-каталог. Уберите `--fix-mome-prefill-state`, `--trace-mtp-boundary`
и `--trace-target`: omni2 использует штатный worker/builder. Исходный `omni`
и его необязательная MoME-корректировка сохраняются.

Сохраните прежние значения model/operator/additional config, TP, depth,
контекста/выхода, mode, dtype кеша, block sizes, chunk size и sampling.
Поддерживаются полные/урезанные Ultra checkpoint, TP1/2/4/8, depth0/3/7,
eager, acl, compile-acl и текстовые JSON-входы существующего инструмента.
Compile-only отклоняется, как в старом omni: native decorator его обходит.

Для воспроизведения последовательности HTTP «первый запрос, затем повторы»:

```bash
--engine omni2 --warmup 0 --repeats 16
```

По умолчанию omni2 не добавляет собственных warmup-запросов; штатная
инициализация/компиляция/прогрев движка остаются. Для direct/omni default
по-прежнему два дополнительных запроса. Явные `--warmup`,
`--omni-warmup-tokens`, `--omni-cache-history` также поддерживаются.
Без нашего прогрева результат отмечается diagnostic; изменения compile
counters сохраняются. С прогревом компиляция во время замера отклоняется.

`--omni-multiprocessing` не требуется: используется native AsyncMPClient.
Если флаг указан, env VLLM_ENABLE_V1_MULTIPROCESSING явно становится 1.
Локальная MoME-корректировка запрещена; native speculative config получает
только method=mtp и num_speculative_tokens. Draft/rejection defaults
определяются самим установленным сервером, а не нашим runtime.

Используйте **то же окружение, что для HTTP**. В отличие от старого launcher,
omni2 не принуждает MRv1, AOT=0, draft top-k=-1 и не удаляет унаследованные
переменные benchmark gates. Operator config задаётся нашим обязательным
аргументом; штатные Omni plugins добавляются к allow-list. Если директория
patches/backend явно установлена в окружении, она сохраняется; иначе
используются прежние defaults Ultra high_throughout.

Tokenizer теперь действительно инициализируется. Для урезанного checkpoint
без tokenizer-файлов можно передать `--tokenizer /path/to/full/model`,
в том числе вместе с `--prompt-ids` или синтетическим prompt.

## Параметры, которых нет в run.py

`--omni2-server-args server.json` добавляет native аргументы из JSON-массива:

```json
["--generation-config", "vllm"]
```

Пример задаёт generation_config=vllm, **только если так настроен HTTP**.
Default без этого аргумента остаётся штатным auto. Передаются только
дополнительные аргументы, без `vllm serve` и model positional argument.
Содержимое массива фиксируется в launch plan и comparison hash.
Используются полные имена `--option`, не короткие aliases.

Модель, worker, tokenizer, TP/depth, compilation и остальные контролируемые
параметры нельзя незаметно заменить через этот файл: используйте соответствующие
аргументы run.py. Установленный parser проверяет остальные native параметры.
Launcher-only опции, требующие отдельного frontend/плагина, отклоняются,
когда их путь не поддержан. Multi-API-server, headless и grpc не поддерживаются.

## Что сохраняется и как считается время

- `native_server_argv.json`: фактические аргументы штатного parser.
- `native_server_config.json`: effective config snapshot, worker class,
  speculative depth, generation_config и native sampling defaults.
- `result.json`: token IDs, native текст, cache hits, finish reason,
  requested/actual output lengths, effective worker config и source hashes.
- `generations.csv`: прежние `id,text,tpot`, с текстом native completion.

Native CompletionRequest получает уже подготовленные token IDs, stream=true,
return_token_ids=true, add_special_tokens=false, ignore_eos согласно run.py.
Min_tokens и detokenize вручную не меняются. Если renderer вернул другие
prompt IDs, stream содержит ошибку, usage не совпал с token IDs или нет
finish/DONE, результат не сохраняется как успешный.

TTFT измеряется до первого token-bearing SSE frame, decode — от него до
последнего token-bearing frame. Время trailing usage/DONE сохранено отдельно
в `stream_tail_seconds`. Сеть исключена; подготовка ответа и сериализация
штатного completion handler остаются. Перед/после каждого запроса нет
NPU synchronization RPC или ручного drain-loop.

Async output может объединять токены в первом frame. Сохраняются
`first_output_tokens` и `stream_token_counts`; decode denominator исключает
все токены первого frame, чьё раздельное время не наблюдалось.
Не следует приравнивать один SSE frame к одному шагу модели.

`acceptance_rate=null`, `cycles=null`: SSE не предоставляет достоверного
числа proposals/accepted drafts. После запросов вызывается native
`do_log_stats()`, чтобы вывести накопленные штатные метрики. Сравнивайте
acceptance по этой статистике, а не по числу SSE chunks.

Для аудита зарегистрирован прежний worker extension с именованным read-only
RPC. Он не имеет constructor или overrides; worker/builder остаются native.
RPC вызывается до/после серии измерений, не внутри запросов. Проверяются
effective dtype, token budgets, TP, cache mode, compilation/capture и отсутствие
локальной MoME-корректировки. Это дополнительная диагностика относительно
обычного HTTP frontend, не часть его генерации.

Native OpenAI defaults отмечены отдельной comparison identity. Для сравнения
token IDs со старым omni/direct используйте `compare --correctness`; speedup
при разных defaults автоматически не объявляется. Для разбивки холодных и
тёплых запросов сохраняется `prefix_cache_groups`.

CPU tests проверяют routing, native helper lifecycle, parser/env propagation,
delta token IDs, prompt/usage validation, errors/закрытие stream и engine,
coalesced first frame и исключение trailing usage из decode. Они не
загружают модель и не подтверждают равенство NPU-выходов с HTTP.
