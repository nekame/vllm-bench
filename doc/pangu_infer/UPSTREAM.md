# Происхождение локального кода

Источник — read-only checkout `omniinfer`, commit
`4cd855b29b5eb0a78bca645fad39e3181c35d099`.

| Локальный код в `src/vllm_bench/pangu_infer/model/` | Источник | Изменения |
| --- | --- | --- |
| `pangu_ultra_moe.py` | `omni/v1/models/pangu/pangu_ultra_moe.py` | Удалены `support_torch_compile` и post-forward plugin decorator; attention-классы импортируются из локального адаптера; post-load вызывается явно после packing |
| `pangu_ultra_moe_mtp.py` | `omni/v1/models/pangu/pangu_ultra_moe_mtp.py` | Удалён compile decorator; при одинаковом TP embedding/head общие с target; при отдельном MTP TP создаются собственные embedding/head и загружаются из глобальных checkpoint-весов; post-load вызывается явно |
| `attention.py` | `MomeAttentionMixin._build_mome_conv` из `npu_mla.py`; `NPUMoMEPatch.MomeAttention` из `patch_mome_hybrid.py` | Явный MoME-класс и обычные подклассы attention вместо зависимости от динамического `vllm.model_executor.layers.npumome`; arithmetic, convolution operators и state spec сохранены |
| `acl_graph.py` | `omni/compilation/acl_graph.py`, `ACLGraphWrapper._update_graph_tasks` | Только при `--mtp-pad-to-8`: подкласс wrapper сохраняет task handles и workspace references каждой записи графа отдельно; протокол dynamic kwargs/update/events сохранён |

Оригинальные license/copyright headers сохранены. MoME adapter содержит
атрибуцию исходному коду; его patch-источник использует Apache-2.0.

MTP adapter дополнительно принимает список используемых физических predictor
layers из плана. При рекуррентном использовании первого head остальные не
создаются и не загружаются. HF-конфигурация и политика выбора head сохраняются.

Остальные файлы нового пакета реализуют launcher, управление буферами и
последовательный цикл batch 1. Новых monkey patches, import hooks,
`sitecustomize` и динамической генерации исходников в runtime нет.

`mtp_parallel.py` использует штатный `patch_tensor_parallel_group` из
`omni/vllm_patches/patches/common/patch_parallel_state.py`. Его протокол
scoped state заимствован для явного переключения `_EP` и `_MOE_DISPATCH_EP`:
операторы Omni обращаются к этим process-global группам и не принимают их
аргументами. Контекст охватывает создание, загрузку, packing, компиляцию,
capture/replay и исполнение MTP; target-группы восстанавливаются в `finally`.
Функции, классы и import bindings не заменяются. Этот путь рассчитан на
последовательный single-thread runtime PoC, без конкурентных model forwards;
зависимость от внутренних имён групп привязана к указанной версии Omni/vLLM.

Изменения в OmniInfer не требуются. Зависимости на его startup-регистрации
остаются: NPU platform, linear/quantization/custom operators, cache specs,
static sinks и необходимые vLLM adaptations загружаются обычными плагинами
`omni-npu,omni_npu_patches,omni_custom_models` с `OMNI_VLLM_PATCHES_DIR=high_throughout`.
Плагин `omni-npu` обязателен и в allow-list `VLLM_PLUGINS`: vLLM применяет
этот фильтр к platform entry points, а не только к general plugins.
Это штатные patches установленного OmniInfer, а не новые подмены PoC.

Только определения нейросети в `model/` требуют ML-пакетов уже при импорте,
как исходные классы OmniInfer. Они импортируются внутри `load_models` NPU worker;
все CLI, `--help`, планирование и служебные модули сохраняют CPU-only import.

В прямом цикле нет Engine, scheduler, NPUModelRunner или EagleProposer.
`VllmConfig`, forward context, distributed groups, metadata builders и
`ACLGraphWrapper` оставлены как интерфейсы, необходимые существующим операторам.
ACL wrapper обслуживает обновление graph tasks с динамическими длинами attention.
Для необязательного padding-контроля используется локальный подкласс: в исходном
Omni registry ключ состоит из physical input size и layer name, поэтому два
отдельно захваченных direct MTP-графа 8/query=8 и 8/query=1 перезаписали бы
handles друг друга. Подкласс после capture сохраняет ссылки на собственные
GraphTaskEntry/workspaces и обновляет именно их. Сам capture, replay и graph pool
остаются в исходном wrapper; глобальные функции и registry не подменяются.
Без флага используется исходный wrapper. Изменение требует NPU проверки;
CPU tests проверяют ownership handles/events/workspaces на stand-ins.

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
