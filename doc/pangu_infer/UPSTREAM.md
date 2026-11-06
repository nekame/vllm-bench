# Происхождение локального кода

Источник — read-only checkout `omniinfer`, commit
`4cd855b29b5eb0a78bca645fad39e3181c35d099`.

| Локальный код в `src/vllm_bench/pangu_infer/model/` | Источник | Изменения |
| --- | --- | --- |
| `pangu_ultra_moe.py` | `omni/v1/models/pangu/pangu_ultra_moe.py` | Удалены `support_torch_compile` и post-forward plugin decorator; attention-классы импортируются из локального адаптера; post-load вызывается явно после packing |
| `pangu_ultra_moe_mtp.py` | `omni/v1/models/pangu/pangu_ultra_moe_mtp.py` | Удалён compile decorator; при одинаковом TP embedding/head общие с target; при отдельном MTP TP создаются собственные embedding/head и загружаются из глобальных checkpoint-весов; post-load вызывается явно |
| `attention.py` | `MomeAttentionMixin._build_mome_conv` из `npu_mla.py`; `NPUMoMEPatch.MomeAttention` из `patch_mome_hybrid.py` | Явный MoME-класс и обычные подклассы attention вместо зависимости от динамического `vllm.model_executor.layers.npumome`; arithmetic, convolution operators и state spec сохранены |
| `acl_graph.py` | `omni/compilation/acl_graph.py`, `ACLGraphWrapper._update_graph_tasks` | При разных TP или `--mtp-pad-to-8`: отдельные task handles и workspaces каждого графа, изоляция registry при capture; протокол dynamic kwargs/update/events сохранён |

`compilation.py` адаптирует non-static ветку `NpuGraphExAdaptor.compile` из
`omni/compilation/npugraph_ex.py` для `--mtp-no-static-kernels`. Локальный backend
принимает отдельные MTP options; tuple output, FakeTensor tracing context и
прочие параметры компиляции сохранены. Для ACL wrapper создаётся поверхностная
копия VllmConfig с отдельным additional_config и выключенными static/super kernels;
static-forward registry остаётся общим. Глобальный конфиг Omni не изменяется.
Без флага обе модели используют исходный backend. Поддержка subgroup в static
compiler здесь не реализуется; эксперимент требует проверки на сервере.

Оригинальные license/copyright headers сохранены. MoME adapter содержит
атрибуцию исходному коду; его patch-источник использует Apache-2.0.

MTP adapter дополнительно принимает список используемых физических predictor
layers из плана. При рекуррентном использовании первого head остальные не
создаются и не загружаются. HF-конфигурация и политика выбора head сохраняются.

Остальные файлы нового пакета реализуют launcher, управление буферами и
последовательный цикл batch 1. Новых monkey patches, import hooks,
`sitecustomize` и динамической генерации исходников в runtime нет.

`boundary_trace.py` использует штатные `register_forward_hook` на target и draft
logits processors, установленные через `collective_rpc` после прогрева.
Исходные model/runner methods не заменяются. Доступ к буферам, группам кеша
и builder fields привязан к текущим NPUModelRunner/Eagle/MoME API.
Hooks работают вне captured model forward; CPU snapshots синхронизируют
исполнение и требуют отдельной NPU-проверки. Это диагностика, не benchmark.

`mtp_parallel.py` использует штатный `patch_tensor_parallel_group` из
`omni/vllm_patches/patches/common/patch_parallel_state.py`. Его протокол
scoped state заимствован для явного переключения `_EP` и `_MOE_DISPATCH_EP`:
операторы Omni обращаются к этим process-global группам и не принимают их
аргументами. Контекст охватывает создание, загрузку, packing, компиляцию,
capture/replay и исполнение MTP; target-группы восстанавливаются в `finally`.
Функции, классы и import bindings не заменяются. Этот путь рассчитан на
последовательный single-thread runtime PoC, без конкурентных model forwards;
зависимость от внутренних имён групп привязана к указанной версии Omni/vLLM.

Для базового запуска PoC изменения в OmniInfer не требуются. Зависимости на его startup-регистрации
остаются: NPU platform, linear/quantization/custom operators, cache specs,
static sinks и необходимые vLLM adaptations загружаются обычными плагинами
`omni-npu,omni_npu_patches,omni_custom_models` с `OMNI_VLLM_PATCHES_DIR=high_throughout`.
Плагин `omni-npu` обязателен и в allow-list `VLLM_PLUGINS`: vLLM применяет
этот фильтр к platform entry points, а не только к general plugins.
Это штатные patches установленного OmniInfer, а не новые подмены PoC.

Отдельная корректировка по результатам repeat-диагностики сохранена в
`patches/omniinfer/mome-target-prefill-state.patch`. Она меняет исходный
`NPUMomeAttentionMetadataBuilder.build`: non-PD target на границе
`computed == prompt` читает prefill-tail с accepted=`num_spec + 1`.
Draft-вызовы явно помечаются через `build_for_drafting`, их offsets сохранены;
последующий target decode и PD не меняются. Патч включает regression tests
для глубин 3/7, последующего decode, draft, PD и смешанного batch.
Checkout OmniInfer в этой рабочей папке не изменяется: патч применяется
пользователем на сервере, если он решит проверить исправление исходного
builder. Локально проверены применение к временной копии, обратимость и
синтаксис; torch/NPU validation требует серверного запуска.

Только определения нейросети в `model/` требуют ML-пакетов уже при импорте,
как исходные классы OmniInfer. Они импортируются внутри `load_models` NPU worker;
все CLI, `--help`, планирование и служебные модули сохраняют CPU-only import.

В прямом цикле нет Engine, scheduler, NPUModelRunner или EagleProposer.
`VllmConfig`, forward context, distributed groups, metadata builders и
`ACLGraphWrapper` оставлены как интерфейсы, необходимые существующим операторам.
ACL wrapper обслуживает обновление graph tasks с динамическими длинами attention.
При разных TP или padding-контроле используется локальный подкласс. В исходном
Omni task registry ключ состоит из physical input size и layer name, поэтому
отдельные direct MTP-графы 8/query=8 и 8/query=1 перезаписали бы handles друг друга.
Workspace cache имеет ключ physical input size и workspace function, без TP и
форм входов. Target TP8 и MTP TP2 с одинаковым числом токенов могут ошибочно
использовать один workspace при разном числе локальных attention heads.
Подкласс на время capture предоставляет пустые словари tasks/workspaces данного
размера, сохраняет собственные GraphTaskEntry/workspaces и обновляет именно их.
Предыдущие словари восстанавливаются в finally, включая ошибку capture.
Сам capture, replay и graph pool остаются в исходном wrapper; глобальные функции
не подменяются. При одинаковых TP без padding используется исходный wrapper.
Изменение требует NPU проверки; CPU tests проверяют ownership handles/events/
workspaces и восстановление registry на stand-ins.

При обновлении OmniInfer сначала сравните перечисленные исходники с локальными
копиями. Протокол MoME acceptance, политика multi-MTP и graph-task API привязаны
к данной ревизии. Несовпадение установленной версии нужно проверять на сервере,
а не исправлять универсальными runtime-подменами.

Проверенные upstream-интерфейсы vLLM 0.25.1:
[LLMEngine](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/engine/llm_engine.py),
[plugin loader](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/plugins/__init__.py),
[forward context](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/forward_context.py),
[LogitsProcessor](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/logits_processor.py),
[MambaBase](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/mamba/abstract.py).

`--omni-multiprocessing` использует публичный
`LLMEngine.from_engine_args(enable_multiprocessing=True)` и штатный SyncMPClient
с фоновым EngineCoreProc. HTTP AsyncLLM использует AsyncMPClient с тем же
процессным core; frontend API и транспорт получения outputs остаются разными.
См. [AsyncLLM](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/engine/async_llm.py)
и [EngineCoreClient](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/engine/core_client.py).
В process-режиме frontend не шагает по приватной batch queue; она обслуживается
штатным core loop. Этот путь позволяет проверить различие lifecycle без
изменения OmniInfer или model methods; его влияние на repeat divergence
ещё требует NPU-проверки.

MP RPC использует штатный `worker_extension_cls` с двумя новыми методами:
`pangu_bench_synchronize` и `pangu_bench_worker_state`. Extension без
конструктора и без overrides импортируется по qualified path; frontend
передаёт строковые method names. В in-process режиме остаются исходные
callables. Это устраняет несовместимость функций с безопасным msgpack
transport SyncMPClient без включения insecure serialization.
См. [WorkerWrapperBase](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/worker_base.py)
и [serial_utils](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/serial_utils.py).

`LLMEngine.add_request()` возвращает внутренний request ID после
`InputProcessor.assign_request_id()`: к внешнему ID может добавляться
случайный суффикс. Boundary trace активируется по возвращённому ID до
первого `step()`, без отключения штатной рандомизации IDs.
См. [InputProcessor](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/engine/input_processor.py).
