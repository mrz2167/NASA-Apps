# NASA-Apps
NASA Space Apps 2026 — Planet X and SPHEREx challenge.

Сайт, который показывает снимки SPHEREx любого участка неба как видео и отвечает не только «что движется», но и «из чего это сделано».

## Установка

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate — macOS/Linux
pip install -r requirements.txt
```

## Живой сайт

**https://nasa-skywatch-production.up.railway.app/** (Railway, проект `nasa-skywatch`)

Передеплой из папки с `cache/` (кэш в git не входит, поэтому `--no-gitignore`; лишнее отсекает `.railwayignore`):
```
railway up --no-gitignore --ci
```

## Docker (рекомендуется)

```
docker compose up -d --build       # http://localhost:8077/
docker compose logs -f             # логи
docker compose down                # остановить
```
- В образ запечён `cache/` пяти тестовых объектов (~200 МБ, образ ~1,7 ГБ): демо работает без архива NASA.
- `docker-compose.yml` подключает `./cache` томом — новые места неба сохраняются между перезапусками.
- Для хостинга (Railway/Render/Fly): тот же `Dockerfile`, порт берётся из `$PORT`. Перед сборкой положите `cache/` из архива на Drive.

## Запуск сайта целиком

```
pip install -r requirements.txt
uvicorn main:app --port 8077
```
Открыть **http://localhost:8077/** — сайт (`web/index.html`) и API с одного адреса, реальные данные SPHEREx.
`?api=demo` — симулированное небо без бэкенда; `?api=https://host` — другой сервер.

Проверено в браузере: все 5 кейсов открываются, клик по цели даёт верный вердикт сайта
(Barnard — Nearby star on the move, WISE 0855 и eps Ind B — Looks like a cold brown dwarf, RW Aur и R Dra — Variable star),
кейс 04 закрывается на реальных данных.

Правки во фронтенде: координаты целей — на время данных SPHEREx (J2000 Барнарда за краем кадра);
`API_BASE` определяется сам; вердикт и стрелка учитывают `motion.significant`, незначимое движение помечено «within noise».

## Бэкенд для сайта (LUMEN Skywatch): `main.py` по контракту [API.md](API.md)

```
pip install -r requirements.txt
uvicorn main:app --port 8077          # порт любой свободный
```
В `index.html`: `const CONFIG = { API_BASE: "http://localhost:8077", PIX_ARCSEC: 6.15 };`

| Эндпоинт | Что отдаёт |
|---|---|
| `GET /api/frames?ra&dec&fov&detector` | PNG по эпохам (asinh, север вверх), разности (янтарный/синий), `sources` с именами Gaia DR3, `target.positions` и `molecules` |
| `GET /api/object?ra&dec` | имя, `status` (known/candidate/star), каталог, движение ± ошибка и каталожное, переменность, молекулы, спектр, вердикт |
| `GET /api/health` | проверка |

- Сначала берётся полный набор снимков из `cache/`, если его нет — быстрый режим из архива (~2 мин на новое место), если архив лёг — что есть в кэше.
- Пиксели читаются прямо из открытого бакета AWS `https://nasa-irsa-spherex.s3.amazonaws.com/` (`cloud.py`): только нужные строки снимка (~1,8 МБ вместо 65 МБ), результат совпадает с вырезкой IRSA пиксель в пиксель. Сервис вырезок IRSA — запасной путь; `SKYWATCH_SOURCE=irsa` — только он. Список снимков по-прежнему из IRSA TAP. Бакет в us-east-1: держите сервер в US East, тогда новое место грузится за секунды.
- При первом запросе кадров в фоне грузится Gaia DR3 для всего поля (~20 с, кэш `cache/gaia/`), дальше клики по звёздам сверяются мгновенно.
- Проверено на 5 объектах NASA: `/api/frames` 10–18 с первый раз, `/api/object` ~2 с.
- Расширения к контракту (фронтенд может игнорировать): `meta`, `motion.arcsec_per_yr_err`, `motion.significant`, `variability.label`, `verdict`, `simbad_url`.

## Для сайта (Питонист 4): одна функция `get_sky`

```python
from sky import get_sky, get_target, SkyUnavailable

try:
    sky = get_sky(ra, dec, quick=True, on_progress=lambda i, n: bar.progress(i / n))
except SkyUnavailable as e:
    st.error(str(e))

sky = get_target("barnard")          # тестовые объекты: barnard, eps_ind_b, wise0855, rw_aur, r_dra
```

| Ключ | Что внутри |
|---|---|
| `frames` | `[(дата, детектор, 2D)]` — кадры для мигания |
| `by_det` | `{детектор: [(дата, кадр)]}` |
| `info` | подписи кадров: `date, detector, n_exposures, wavelength_um` |
| `diffs` | «что изменилось»: `[(дата, детектор, 2D)]` |
| `wcs`, `shape`, `coord` | общая сетка кадров и центр |
| `cutouts` | исходные вырезки — для спектра (Питонист 2) |
| `source` | `"archive"` или `"cache"` |
| `warnings` | строки для показа пользователю |

- `quick=True` (по умолчанию): ≤ 3 снимка на (эпоху, детектор), ~50 вырезок. Повторный запрос того же места — мгновенно из кэша (центр может отличаться до 3″).
- `offline=True`: только кэш, без сети — запасной вариант на демо.
- Архив не отвечает и кэша нет → `SkyUnavailable` с понятным текстом.

### Полный разбор объекта: `analyze(sky)`

```python
from sky import get_target, analyze
a = analyze(get_target("wise0855"))          # online=True — ещё SIMBAD (зеркало) и Gaia
a["card"]      # name, verdict, speed_text, direction_deg, variability, molecule_badges, simbad_url
a["motion"]    # speed_arcsec_yr ± speed_err_arcsec_yr, direction_deg, detectors, positions (для следа)
a["variability"], a["spectrum"] {"wavelength","flux","error"}, a["scores"]
```

Проверка на 5 объектах NASA (все вердикты верные):

| Объект | Измерили | Каталог | Вердикт |
|---|---|---|---|
| Барнард | 10,3 ± 3,7″/год, 352° | 10,4″/год, 356° | близкая звезда с большим движением |
| eps Ind B | 4,4 ± 0,6″/год, 125°; метан 0,82 | 4,7″/год, 122° | коричневый карлик |
| WISE 0855 | 7,6″/год, 286° | 8,2″/год, 275° | коричневый карлик |
| RW Aur | не движется; размах 78% | переменная | переменная звезда |
| R Dra | не движется; размах 53% | переменная | переменная звезда |

Как это сделано (`identify.py`):
- **Детектор** выбирается там, где объект заметнее всего относительно остальных источников кадра (холодный карлик — на 4–5 мкм). Если два детектора дают противоречащее движение, остаётся более надёжный.
- **Центр объекта** — центр масс в окне ±2 пикселя вокруг самого яркого пикселя; у пересвеченных звёзд — центр «дыры» пересвета (она симметрична).
- **Движение** значимо, если > 1″/год и > 2 ошибок; ошибка — по разбросу точек и расхождению детекторов.
- **Переменность**: размах > 30% и > 5σ шума; у пересвеченных звёзд — по крыльям. Шумные кадры (мало снимков) выкидываются (`epochs.good_frames`).
- `now_coord` берёт тестовые объекты из `targets.py`, остальные — из SIMBAD (зеркало Гарварда).
- Ограничения: у пересвеченных звёзд (Барнард, R Dra) спектра нет; у WISE 0855 на коротких волнах в апертуру попадает соседняя звезда, поэтому метан занижен.

### Картинки и анимации: `render.py` (всё в памяти, сразу в `st.image`)

```python
from render import blink_gif, frame_png, diff_png
from targets import predicted_track

gif = blink_gif(sky, detector=3, zoom_arcsec=90)                          # мигание, bytes
gif = blink_gif(sky, 3, zoom_arcsec=90, track=predicted_track("barnard"))  # + след по каталогу
st.image(gif)
png = frame_png(sky, index=0)      # один кадр;  diff_png(sky, index=0) — «что изменилось»
```

- Яркость: asinh-растяжка, одни пределы на все кадры детектора (мигание не «дышит» само).
- Пиксели увеличены без сглаживания; север вверх, восток влево; масштабная линейка 30″.
- Красные уголки — объект запроса; жёлтые полые кружки — след движения (не закрывают слабый объект).
- Пересвеченные ядра ярких звёзд (Барнард, R Dra) для показа заливаются белым (`fill_saturated`). Для науки — исходные кадры.
- Шумные кадры из малого числа снимков пропускаются (`drop_sparse=True`).
- `track` — список SkyCoord на кадр или функция `date -> SkyCoord`; Питонист 3 может передать измеренные положения.
- Готовые GIF для презентации: `out/anim/`.
- Для R Dra разность бесполезна (огромное пересвеченное ядро) — переменность ярких звёзд показывать графиком яркости по крыльям.

## Данные (Питонист 1) — готово

| Файл | Функция | Что делает |
|---|---|---|
| `data.py` | `find_images(ra, dec, size_deg=0.1)` | таблица снимков: `url, obs_id, release, time (MJD), detector, wl_min_um, wl_max_um`; дубли DR1/QR3 убраны |
| `data.py` | `load_cutout(url, time=None)` | словарь: `image, flags, variance, zodi, wcs, spectral_wcs, time, detector, header` |
| `data.py` | `load_cutouts(table)` | все вырезки таблицы, параллельно, через кэш |
| `data.py` | `bad_pixels(cutout)` | маска плохих пикселей по флагам |
| `epochs.py` | `build_frames(cutouts, coord, size_deg)` | `[(дата эпохи, детектор, 2D-кадр)]` на общей сетке |
| `epochs.py` | `make_target_wcs(coord, size_deg)` | та же сетка, что у кадров — для `centroid_sky` и т.п. |
| `epochs.py` | `frames_by_detector(frames)` | `{детектор: [(дата, кадр)]}` — для мигания |
| `epochs.py` | `build_frames(..., info=[])` | заполнит `info` подписями кадров: число экспозиций, длина волны в точке, даты |
| `epochs.py` | `diff_frames(frames)` | режим «что изменилось»: кадр минус первая эпоха (с подгонкой яркости) |
| `epochs.py` | `select_exposures(table, per_group=3)` | быстрый режим: ≤ N снимков на (эпоху, детектор) — для живого сайта |
| `targets.py` | `coord_at(name, when)` | координаты 5 тестовых объектов на дату (из SIMBAD/Gaia DR3, записаны в файл — работает без сети) |
| `targets.py` | `simbad_coord(name, when)` | любой объект по имени из SIMBAD на дату; `simbad()` — клиент на зеркале **simbad.harvard.edu** (основной CDS бывает недоступен) |

Пример:

```python
from astropy.coordinates import SkyCoord
from data import find_images, load_cutouts
from epochs import build_frames, make_target_wcs

coord = SkyCoord(133.75989, -7.24221, unit="deg")      # WISE 0855 на 2026 год
table = find_images(coord.ra.deg, coord.dec.deg, 0.1)
cutouts = load_cutouts(table)                           # первый раз долго, потом из cache/
frames = build_frames(cutouts, coord, 0.1)
wcs, shape = make_target_wcs(coord, 0.1)
```

**Готовый кэш** (чтобы не качать заново): [Google Drive](https://drive.google.com/drive/folders/1ODZE-CC1bClYx-5pgoSHZX4qoegocZFu?usp=sharing) → скачать `cache_*.zip` и распаковать в корень репозитория, чтобы получилась папка `cache/`.

Проверка целиком: `python check_data.py --target barnard` (объекты: `barnard, eps_ind_b, wise0855, rw_aur, r_dra`; `--per-group 3` — быстрый режим) → `out/wise0855/frames.png`, `blink_D*.gif`, `frames.npz`.

### Что важно знать остальным

- **Релиз:** запрос идёт по таблицам `spherex.plane` / `spherex.artifact`; сейчас по WISE 0855 приходит только DR1 (258 экспозиций). Если придут дубли из QR3, остаётся DR1.
- **Эпохи:** эпоха = разрыв > 30 дней, но не длиннее 45 дней. У WISE 0855 три эпохи (DR1); у Barnard, eps Ind B и RW Aur уже есть и QR3 за август–октябрь 2026. R Dra лежит в глубоком поле у полюса эклиптики: 2493 снимка подряд, поэтому там помесячные кадры и быстрый режим.
- **Разности:** звёзды оставляют красно-синие следы — пиксель крупный (6″), сдвиг меньше пикселя уже заметен. Для переменных звёзд это не мешает, для слабых движущихся объектов смотрите мигание, а не разность.
- **Вырезка 0,1°** = 58×59 пикселей по 6,16″. В кэше хранится облегчённая копия без таблицы EPSF (~55 КБ вместо 4 МБ).
- **Архив медленный:** поиск ~6–40 с, одна вырезка ~20–30 с (8 потоков параллельно). Первый прогон объекта — 10–15 минут. Поэтому всё в `cache/`; передавайте папку `cache/` друг другу, чтобы не качать заново.
- **Фон:** в `image` зодиакальный свет НЕ вычтен. `build_frames` вычитает `zodi` и остаточную медиану сам. Для фотометрии (Питонист 2) — вычитайте фон кольцом, как в плане.
- **Флаги:** `bad_pixels()` отбрасывает OVERFLOW, HOT, COLD, NONFUNC, MISSING_DATA, NONLINEAR, TRANSIENT, SUR_ERROR. Флаг OUTLIER специально не используется: движущийся объект может попасть в «выбросы».
- **Кадр одного детектора** — медиана всех экспозиций детектора за эпоху. Внутри детектора длина волны в точке объекта немного разная от экспозиции к экспозиции — для мигания это нормально; для спектра используйте отдельные вырезки и `spectral_wcs`, а не кадры.

### Для Питонистов 2 и 3: подводные камни на тестовых объектах

- **Переменность нельзя мерить по кадрам «как есть».** Внутри одного детектора длина волны в точке объекта от эпохи к эпохе разная (см. `wavelength_um` в `frames_info.json`): у R Dra в D1 — от 0,77 до 0,99 мкм. У красной звезды это само по себе меняет яркость. Честно: для каждой эпохи строить спектр (вырезки + `spectral_wcs`) и сравнивать поток на одной и той же длине волны.
- **R Dra пересвечен.** Звезда очень яркая в ИК: центр помечен флагами переполнения/нелинейности (`bad_pixels()` их выбрасывает — отсюда тёмные кольца на кадрах). Апертурная фотометрия по центру даст мусор; мерить по крыльям или по большой апертуре с маской плохих пикселей, и честно писать «яркость оценена по крыльям».
- **R Dra — глубокое поле:** 2493 экспозиции, скачано в быстром режиме (`--per-group 3`, 102 вырезки). Для спектра можно докачать больше.
