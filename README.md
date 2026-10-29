# vllm-bench

Инструменты для профилирования и отладки vLLM/OmniInfer на Ascend NPU. Сейчас сценарии рассчитаны на Pangu 505B: сокращённый стек основных слоёв, standalone MTP и диагностика speculative decoding.

| Направление | Назначение | Руководство |
| --- | --- | --- |
| `layer_profile` | Подготовка checkpoint на 12 слоях (также 6/9), HTTP benchmark и NPU traces | [Профилирование слоёв](doc/layer_profile/README.md) |
| `mtp_profile` | MTP checkpoint, synthetic profiling, запись и replay реальных входов без target model | [Профилирование MTP](doc/mtp_profile/README.md) |
| `mtp_debug` | Проверка greedy acceptance, rejection windows и состояния MoME в полном запуске | [Отладка MTP](doc/mtp_debug/README.md) |

## Начало работы

Для подготовки checkpoint, CPU-проверок, планирования запуска и анализа файлов нужен Python 3.10+. Для выполнения модели нужно уже работающее серверное окружение OmniInfer/vLLM/torch_npu. Инструментарий ничего не устанавливает и не редактирует исходники установленного OmniInfer.

```bash
git clone https://github.com/nekame/vllm-bench.git
cd vllm-bench
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m vllm_bench.layer_profile.serve --help
python -m vllm_bench.mtp_profile.profile_mtp --help
python -m vllm_bench.mtp_debug.debug_spec --help
```

Команды запуска модели и wrappers по умолчанию показывают план; фактический запуск требует `--execute`. Экстракторы checkpoint используют `--dry-run` для планирования и записывают файлы без этого флага. Подробные команды и требования к model/operator config приведены в руководствах.

Пакет называется `vllm_bench`, репозиторий — `vllm-bench`. Установка через pip не требуется. После настройки абсолютного `PYTHONPATH` команды работают и из другого каталога; переданные пользователем относительные пути отсчитываются от текущего рабочего каталога.

## Организация

- `src/vllm_bench/` — исходники трёх направлений, общие функции в `common`, анализ и экспорт в `tools`.
- `conf/` — конфиги и матрицы нагрузки по направлениям; `conf/common/reference_config-hf.json` служит справочным примером.
- `doc/` — руководства сценариев и общие справочные материалы.
- `tests/` — CPU-проверки без NPU и ML-зависимостей.
- `scripts/` — shell-команды для запуска анализа и извлечения диагностических полей.
- `bundle_manifest.json` — SHA256 файлов переносимого набора и статус проверок.

При переносе сохраняйте структуру всего репозитория. Checkpoint, capture, traces и результаты запусков храните отдельно от исходников. Operator configs берите из рабочего запуска целевой модели: приложенный HF reference config их не заменяет.

Три инструментации на основе `sitecustomize.py` включаются явно и используются по одной на запуск. Для standalone MTP не должно быть активных hooks записи или диагностики.

## Проверки

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests
```

При реорганизации прошли 46 CPU-тестов, включая запуск из другого каталога, поиск конфигов, команды workers и подключение hooks на mock-модулях. Эти проверки не выполняют модель на NPU. Standalone MTP, capture/replay и диагностика требуют серверных smoke tests; порядок описан в руководствах. Ранее подтверждённый offline export описан в руководстве профилирования слоёв и не заменяет проверку новых запусков.

## Справочные материалы и расширение

- [Архитектура Pangu 505B и интерпретация измерений](doc/reference/PANGU_505B_RU.md).
- [NPU streams](doc/reference/NPU_STREAMS_RU.md).
- [Compilation, ACL Graph и eager](doc/reference/COMPILATION_EAGER_RU.md).
- [Добавление сценариев, переход со старых команд и обновление набора](doc/CONTRIBUTING.md).
