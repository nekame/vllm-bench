# Развитие vllm-bench

## Запуск и зависимости

Работаем из checkout без установки пакета:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests
```

Используйте пакетные импорты `vllm_bench.*` и команды `python -m`. Не добавляйте исходники отдельных сценариев в `sys.path`. Путь к `src` должен доходить до всех дочерних workers как абсолютный `PYTHONPATH`; общие helpers находятся в `common.runtime_env`. Для запуска MTP worker используется `torch.distributed.run --module vllm_bench.mtp_profile.mtp_runtime`.

Справка, планирование и работа с checkpoint должны использовать стандартную библиотеку Python 3.10+. ML-зависимости импортируйте внутри runtime-функций, которым они необходимы. Обычный импорт модуля не должен активировать инструментацию или запускать модель.

## Добавление варианта

1. Добавьте пакет `src/vllm_bench/<scenario>/` с пустым либо документирующим `__init__.py`. CLI реализуйте в отдельных модулях с `main()` и блоком `if __name__ == "__main__"`.
2. Создайте `doc/<scenario>/README.md`: назначение, требования, конфиги, входы, dry-run, запуск, результаты, ограничения, CPU-проверки и порядок аппаратной проверки.
3. При необходимости добавьте реальные конфиги в `conf/<scenario>/`. Общие справочные конфиги лежат в `conf/common/`. Не коммитьте серверные секреты и локальные пути в качестве универсальных defaults.
4. Для встроенных конфигов используйте `common.paths.CONF_ROOT`; пути, явно переданные пользователем, разрешайте относительно текущего каталога. Форматы результатов существующих сценариев не меняйте в ходе структурных переносов.
5. Добавьте необходимые CPU-тесты в `tests/test_*.py`, включая проверку планирования без ML runtime. Зарегистрируйте направление строкой в корневом README.
6. Вынесите функции в `common`, если ими действительно пользуются несколько направлений. Общие средства анализа файлов размещайте в `tools`. Один сценарий не должен импортировать CLI другого ради общей настройки.
7. Если нужен новый `sitecustomize.py`, держите его в `instrumentation` своего сценария и подключайте только явно. Добавьте режим в `common.runtime_env.INSTRUMENTATION`, проверку конфликтов и тест импорта hook в дочернем процессе. На один запуск допускается одна инструментация.

Универсального реестра сценариев нет: новый пакет и строка README достаточны. Привязку к Pangu отражайте в требованиях сценария; разделение по моделям понадобится при появлении второй реализации.

## Переход со старых команд

После настройки `PYTHONPATH` замените запуск файла запуском соответствующего модуля. CLI arguments остаются прежними.

| Старая команда | Новая команда |
| --- | --- |
| `python prepare_checkpoint.py` | `python -m vllm_bench.layer_profile.prepare_checkpoint` |
| `python verify_checkpoint.py` | `python -m vllm_bench.layer_profile.verify_checkpoint` |
| `python serve.py` | `python -m vllm_bench.layer_profile.serve` |
| `python benchmark.py` | `python -m vllm_bench.layer_profile.benchmark` |
| `python prepare_mtp_checkpoint.py` | `python -m vllm_bench.mtp_profile.prepare_mtp_checkpoint` |
| `python verify_mtp_checkpoint.py` | `python -m vllm_bench.mtp_profile.verify_mtp_checkpoint` |
| `python profile_mtp.py` | `python -m vllm_bench.mtp_profile.profile_mtp` |
| `python record_mtp.py` | `python -m vllm_bench.mtp_profile.record_mtp` |
| `python debug_spec.py` | `python -m vllm_bench.mtp_debug.debug_spec` |
| `python summarize_rejections.py` | `python -m vllm_bench.mtp_debug.summarize_rejections` |
| `python analyze_trace.py` | `python -m vllm_bench.tools.analyze_trace` |
| `python export_profile.py` | `python -m vllm_bench.tools.export_profile` |
| `python test_cpu.py` | `python -m unittest discover -s tests` |

Внешние server scripts, ссылающиеся на старые пути, необходимо обновить аналогично. Обёрток в корне нет. Конфиги перенесены в `conf/common/` и `conf/layer_profile/`; default suite находится автоматически.

## Обновление переносимого набора

После изменения файлов обновите SHA256. Из корня checkout:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
paths = [root / '.gitignore', root / 'README.md']
for directory in ('src', 'conf', 'doc', 'tests'):
    paths.extend(path for path in (root / directory).rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts
                 and path.suffix in {'.py', '.json', '.md', '.sh'})
manifest_path = root / 'bundle_manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['files'] = {path.relative_to(root).as_posix():
                     hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(paths)}
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
PY
```

Манифест не включает себя, `.git`, кэши, локальные файлы и результаты. Этот шаг пересчитывает только hashes: раздел `validation` обновляйте по фактическим проверкам текущей версии. Не переносите автоматически прежнее утверждение об успешных тестах на изменённый код. Для NPU отдельно сохраняйте статус, окружение и выполненные smoke tests; CPU-проверки не подтверждают исполнение модели.
