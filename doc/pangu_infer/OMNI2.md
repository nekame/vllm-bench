# Omni2: серверный путь без HTTP listener

`--engine omni2` использует функции HTTP-launcher установленного vLLM,
вместо нашего синхронного цикла `LLMEngine.step()`:

1. Штатные `cli_env_setup`, `make_arg_parser`, `validate_parsed_serve_args`.
2. `build_async_engine_client(args)` с серверным UsageContext.
3. Штатные `get_supported_tasks`, `build_app` и `init_app_state`.
4. `app.router.lifespan_context(app)` вокруг прогрева и всех запросов.
5. `openai_serving_completion.create_completion(CompletionRequest(...))`
   и чтение completion-stream непосредственно в процессе.

Сохраняются AsyncLLM/AsyncMPClient, tokenizer, OnlineRenderer, преобразование
штатное преобразование sampling-параметров, native request IDs и MTP proposal policy. Запускается
`uvloop`, как в штатном сервере. Native lifespan сам запускает периодическую
статистику и `freeze_gc_heap`; при выходе освобождает app state до shutdown
движка. FastAPI app создаётся штатно, но HTTP listener/uvicorn не запускаются.
Запросы вызывают completion handler напрямую, без ASGI request wrappers
и сетевого планирования задач. Engine IPC и коммуникации NPU остаются штатными.
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

Для подтверждённой конфигурации HTTP с async и decode-only графами:

```bash
--engine omni2 --mode compile-acl --async-scheduling \
  --cudagraph-mode FULL_DECODE_ONLY --cudagraph-capture-sizes 8
```

Без явных флагов сохранены defaults до `5ca1807`: async выключен,
graph mode `FULL`, capture sizes `[1, depth+1]` (для depth7 `[1, 8]`).
Для сравнения с данным HTTP задавайте async и graph policy явно, как выше.
Удаление этих флагов возвращает прежнюю политику запуска; параметры не
сохраняются между запусками. Явный `--no-async-scheduling` эквивалентен default.
В `[execution]` выводится фактическое разрешённое значение из VllmConfig.

В `compile-acl` compilation JSON также явно задаёт
`mode: 3` (`VLLM_COMPILE`), как до коммита `5ca1807`. Удаление этого поля
в `5ca1807` отменено: выбор graph policy не должен неявно менять compilation mode.
Фактический mode строго проверяется и записывается в результат;
при аудите дополнительно проверяется реальная компиляция/capture.
`acl` и `eager` явно задают compilation mode 0. Точное имя graph mode
проверяется после создания движка: наличие любого full graph недостаточно.
Для прежних `omni`/`direct` defaults не изменены: async off,
graph `FULL`, capture sizes `[1, depth+1]`. Новый `--cudagraph-mode`
разрешён только stock engines в ACL-режиме и участвует в comparison identity.

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
По умолчанию worker audit выключен: результат diagnostic, а изменение
worker compile counters неизвестно (`null`). При включённом аудите без
нашего прогрева результат также diagnostic, counters сохраняются.
При аудите с прогревом компиляция во время замера отклоняется.

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
["--generation-config", "auto"]
```

По умолчанию omni2 задаёт generation_config=vllm, чтобы sampling defaults
из файла модели не заменяли настройки инструмента. Пример явно разрешает
model defaults для неуправляемых параметров. Temperature/top_p/top_k и
верхний предел длины всё равно задаются нашими generation overrides. Передаются только
дополнительные аргументы, без `vllm serve` и model positional argument.
Содержимое массива фиксируется в launch plan и comparison hash.
Используются полные имена `--option`, не короткие aliases.

Модель, worker, tokenizer, TP/depth, compilation и остальные контролируемые
параметры нельзя незаметно заменить через этот файл: используйте соответствующие
аргументы run.py. Установленный parser проверяет остальные native параметры.
Launcher-only опции, требующие отдельного frontend/плагина, отклоняются,
когда их путь не поддержан. Multi-API-server, headless и grpc не поддерживаются.

## Приоритет sampling-параметров

Temperature, top_p и top_k из CLI/плана передаются native серверу через
`--override-generation-config` и явно в каждом CompletionRequest. Значение
`top_k=0` теперь также передаётся: отсутствие ключа раньше позволяло
подставить top_k из generation_config при temperature>0.
Seed и ignore_eos передаются в запросе. Max_tokens задаётся отдельно для
измерения и warmup; server-wide max_new_tokens равен максимуму этих лимитов,
чтобы не обрезать прогрев. Файлы модели не редактируются.

Сообщение vLLM «Default sampling parameters overridden by generation_config»
описывает серверные defaults, а не доказанную замену явно заданной temperature.
В [native CompletionRequest](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/entrypoints/openai/completion/protocol.py#L244)
значение из файла используется, когда поле запроса равно None. Явный ноль
сохраняется. Поэтому прежняя строка temperature=1 сама по себе не объясняет
разные greedy ответы.

Один раз до warmup выполняется native `get_max_tokens` и
`CompletionRequest.to_sampling_params`. В `[sampling]`,
`sampling_parameters.json` и `details.sampling` записаны:

- `requested`: наши параметры запроса;
- `server_defaults`: defaults, полученные от serving class;
- `resolved`: итог native преобразования в SamplingParams.

Это preview **до InputProcessor, EOS/tokenizer обработки и GPU execution**,
не снимок worker SamplingMetadata. В каждом запросе сохраняется штатный
handler; его методы не подменяются. Preview не выполняет RPC или NPU copies.
Если запрошенная temperature=0 не сохранилась при преобразовании, запуск
останавливается. При greedy native SamplingParams нормализует top_p=1,
top_k=0 и min_p=0; seed=-1 и очень малая положительная temperature также
могут нормализоваться самим vLLM и видны в preview.

## Что сохраняется и как считается время

- `native_server_argv.json`: фактические аргументы штатного parser.
- `sampling_parameters.json`: requested/default/resolved sampling preview.
- `native_server_config.json`: effective config snapshot, worker class,
  speculative depth, generation_config и native sampling defaults.
- `result.json`: token IDs, native текст, cache hits, finish reason,
  requested/actual output lengths, frontend config и source hashes;
  worker config добавляется только с `--omni2-audit`.
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
числа proposals/accepted drafts. Native lifespan периодически вызывает
`do_log_stats()`, как в HTTP. Принудительного вызова после серии нет;
короткий запуск может завершиться до очередного интервала логирования.
Сравнивайте acceptance по штатной статистике, а не по числу SSE chunks.

По умолчанию `--omni2-audit` выключен: worker extension не регистрируется,
никаких дополнительных audit RPC перед/после запросов нет. В логе это
`native_lifespan=true worker_audit=false`. Native IPC движка сохраняется.

При необходимости `--omni2-audit` включает прежний read-only worker extension
и два audit RPC на границах серии. Он не имеет constructor или overrides;
внутри запросов дополнительной синхронизации нет. Это диагностическое
вмешательство относительно штатного HTTP; для проверки повторяемости
начинайте с обычного запуска **без** этого флага.

Без аудита `effective.settings_source=frontend_vllm_config`:
operator/parallel/hardware worker и `compiled_during_measurement` неизвестны
и записаны как `null`; `workers_before/after=null`, `details.worker_audit=false`.
Исходный operator JSON не выдаётся за эффективный worker config. С аудитором
сохраняются реальные worker settings и выполняются проверки capture/counters.
Обычные frontend проверки модели, TP/depth, dtype, бюджетов, cache и
заданного compilation/graph mode выполняются в обоих режимах.

Native OpenAI defaults/lifecycle отмечены отдельной comparison identity.
Для сравнения token IDs со старым omni/direct используйте `compare --correctness`.
При неизвестных worker fields сравнение токенов разрешено с явным
`unverified_worker_settings`; известные расхождения и разные checkpoint/prompt
по-прежнему отклоняются. Сравнение speedup требует `--omni2-audit` и прогрева,
неаудированный результат для него не используется. Для разбивки холодных и
тёплых запросов сохраняется `prefix_cache_groups`.

CPU tests проверяют routing, native helper lifecycle, parser/env propagation,
delta token IDs, prompt/usage validation, errors/закрытие stream и engine,
coalesced first frame и исключение trailing usage из decode. Отдельно
проверены native app/lifespan вокруг генерации и сохранения результата,
удаление app state перед shutdown и отсутствие RPC в обычном режиме. Они не
загружают модель и не подтверждают равенство NPU-выходов с HTTP.

Первая версия `e092779` была запущена пользователем на NPU: три ответа
различались между собой и с HTTP. Это не подтверждение численной
эквивалентности. Последующее добавление lifespan и отключение аудита
устраняют конкретные отличия bootstrap, но пока не доказаны как
исправление этого расхождения. Параметры компиляции, cudagraph и sampling
при этом не менялись.

В запуске `2733c88` пользователь также получил разные ответы. Затем он
обнаружил реальные отличия argv: `--no-async-scheduling` и graph `FULL`
вместо HTTP `FULL_DECODE_ONLY`, sizes `[8]`. В `5ca1807` добавлены явные
graph-флаги, но одновременно изменены defaults и удалён явный mode=3.
Пользователь сообщил вывод из одних `!` и с новыми флагами, и после их
удаления. Причина этого вывода пока не установлена. Изменение defaults
отменено, явные новые флаги сохранены; mode=3 также восстановлен.

При проверке без ML-пакетов исходники launcher до `5ca1807` извлечены во
временную папку и сравнены с текущим launcher на одном CPU fixture:
в 18 вариантах depth0/3/7, eager/acl/compile-acl и async off/on native argv совпадает, кроме
двух намеренно добавленных sampling-параметров (`generation-config` и
`override-generation-config`). Это подтверждает восстановление прежних
аргументов запуска, но не доказывает эквивалентность вычислений на NPU.

По vLLM v0.25.1 при стандартном O2 mode=None разрешается в VLLM_COMPILE=3
до применения зависимых optimization defaults; NPUPlatform не переопределяет
apply_config_platform_defaults. По этому пути явное и автоматическое значение
3 совпадают. Считать удаление поля установленной причиной `!` нельзя.
FULL и FULL_DECODE_ONLY проходят has_full_cudagraphs/has_mode(FULL), поэтому
ветка обёрток target/MTP включена в обоих случаях. Однако decode-only меняет
dispatch смешанных batch; proposer берёт runtime mode и padding descriptor
из target runner. Это реальное отличие выполнения, но оно также используется
в рабочем HTTP, поэтому само по себе не объясняет регрессию omni2.

Источники native defaults: [VllmConfig v0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/config/vllm.py#L1088),
[CUDAGraphMode v0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/config/compilation.py#L48).
