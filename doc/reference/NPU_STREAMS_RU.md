# Hardware streams при профилировании Pangu 505B

Все команды ниже выполняются из корня `vllm-bench` после настройки:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Документ описывает дорожки `Stream …` на NPU timeline, а не CPU threads. Основан на OmniInfer `4cd855b29b5eb0a78bca645fad39e3181c35d099`. На этой машине NPU-запуск не выполнялся; конкретное назначение дорожек пользовательской трассы без самой трассы не установлено.

## Что означают несколько streams

Stream задаёт последовательность device-задач. Несколько streams позволяют перекрывать независимые вычисления, обмены данными и вспомогательные операции. Наличие нескольких дорожек ещё не означает одновременную работу: фактическое перекрытие зависит от зависимостей, событий синхронизации и ресурсов NPU.

Единого параметра «использовать N hardware streams» в проверенном коде OmniInfer не найдено. Дополнительные streams создаются отдельными подсистемами. Например, [get_stream](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/v1/utils.py#L89) создаёт именованные `torch.npu.Stream` по мере необходимости.

На timeline могут одновременно присутствовать основной compute stream, streams attention и MoE, streams sampling, коммуникационные streams HCCL и служебные streams графов/runtime. Номер stream сам по себе не определяет его назначение.

## Переключатели OmniInfer

Параметры находятся в секции `operator_optimization_config` JSON, который передаётся нашему `serve.py` через `--operator-config`. Это не поля HF `config.json` и не параметры `benchmark.py`.

Определения и значения по умолчанию: [ModelOperatorOptConfig](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/model_config/config_loader/loader.py#L133).

| Параметр | Назначение | Значение по умолчанию в dataclass |
| --- | --- | --- |
| `enable_multi_stream` | Дополнительные streams в attention; в DSA управляет decode side stream | `false` |
| `split_q_up_in_multistream` | Разделение Q-проекции для перекрытия частей attention prolog; зависит от основного multistream-пути | `false` |
| `li_prolog_multi_stream` | Streams `wi_stream` и `ki_stream` для частей lightning indexer | `false` |
| `shared_expert_multi_stream` | Перекрытие shared expert с routed-expert веткой MoE | `true` |
| `enable_mhc_multistream` | Перекрытие mHC Sinkhorn с проекциями на боковом stream | `false` |
| `sampler_multi_stream` | Дополнительные streams sampling | `true` |
| `mome_sp_allgather_multi_stream` | Боковой stream для AllGather состояния MoME при использовании соответствующего MoME-SP пути | `true` |

Это defaults класса, а не гарантия эффективной конфигурации сервера: JSON, режим запуска, устройство и выбранная реализация могут менять применение настроек. `enable_multi_stream` не является глобальным выключателем всех streams.

Реализации:

- [Создание attention streams в DSA](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/v1/layers/attention/npu_dsa.py#L913).
- [Shared-expert multistream в MXFP8](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/layers/quantization/mxfp8.py#L984); способ перекрытия дополнительно выбирает `shared_expert_parallel_schedule` (`with_finalize` либо `with_routed_experts_cv`, когда этот путь поддерживается).
- [Sampling](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/sample/ops/topk_topp_sampler.py#L71).
- [MoME-SP AllGather](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/attention/backends/utils.py#L1487).

## Диагностическое отключение перекрытия

Для отдельного сравнения измените следующие поля в копии рабочего operator config, сохранив остальные поля и секции:

```json
{
  "enable_multi_stream": false,
  "split_q_up_in_multistream": false,
  "li_prolog_multi_stream": false,
  "shared_expert_multi_stream": false,
  "enable_mhc_multistream": false,
  "sampler_multi_stream": false,
  "mome_sp_allgather_multi_stream": false
}
```

Это фрагмент содержимого `operator_optimization_config`, а не полный operator config. После изменения сервер нужно перезапустить. Для установления причины лучше менять по одному механизму, оставляя ту же нагрузку и тот же режим компиляции.

Отключение этих оптимизаций не гарантирует один stream: остаются коммуникации, runtime, другие оптимизации вроде prefetch и механизмы графов. В частности, [ACLGraphWrapper после replay обновляет attention tasks на отдельном update stream](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/compilation/acl_graph.py#L387), используя межпоточные события для соблюдения порядка.

Отключение перекрытия меняет расписание и время выполнения. Такой запуск помогает разбирать зависимости, но его TPOT и длительности нельзя считать характеристиками исходного рабочего режима. Для baseline сохраните целевые настройки.

## Что не является числом hardware streams

`mhc_num_stream: 4` в [reference config](../../conf/common/reference_config-hf.json) задаёт число residual streams архитектуры mHC. Это размерность представления модели, а не количество device queues. Для уменьшения числа дорожек на trace менять его нельзя.

Также вызов `set_stream_limit` в [limit_core_num](https://gitee.com/huting233/omniinfer/blob/4cd855b29b5eb0a78bca645fad39e3181c35d099/omni/v1/utils.py#L112) ограничивает доступные stream вычислительные ресурсы Cube/Vector; это не лимит количества streams.

Влияние графов и eager описано отдельно: [компиляция, ACL Graph и eager](COMPILATION_EAGER_RU.md).
