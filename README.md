# NASA-Apps
NASA Space Apps 2026 — Planet X and SPHEREx challenge.

Сайт, который показывает снимки SPHEREx любого участка неба как видео и отвечает не только «что движется», но и «из чего это сделано».

## Установка

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate — macOS/Linux
pip install -r requirements.txt
```

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

Проверка целиком: `python check_data.py` → `out/wise0855/frames.png`, `blink_D*.gif`, `frames.npz`.

### Что важно знать остальным

- **Релиз:** запрос идёт по таблицам `spherex.plane` / `spherex.artifact`; сейчас по WISE 0855 приходит только DR1 (258 экспозиций). Если придут дубли из QR3, остаётся DR1.
- **Эпохи:** у WISE 0855 их три — май 2025, ноябрь–декабрь 2025, апрель–май 2026. Эпоха = разрыв > 30 дней.
- **Вырезка 0,1°** = 58×59 пикселей по 6,16″. В кэше хранится облегчённая копия без таблицы EPSF (~55 КБ вместо 4 МБ).
- **Архив медленный:** поиск ~6–40 с, одна вырезка ~20–30 с (8 потоков параллельно). Первый прогон объекта — 10–15 минут. Поэтому всё в `cache/`; передавайте папку `cache/` друг другу, чтобы не качать заново.
- **Фон:** в `image` зодиакальный свет НЕ вычтен. `build_frames` вычитает `zodi` и остаточную медиану сам. Для фотометрии (Питонист 2) — вычитайте фон кольцом, как в плане.
- **Флаги:** `bad_pixels()` отбрасывает OVERFLOW, HOT, COLD, NONFUNC, MISSING_DATA, NONLINEAR, TRANSIENT, SUR_ERROR. Флаг OUTLIER специально не используется: движущийся объект может попасть в «выбросы».
- **Кадр одного детектора** — медиана всех экспозиций детектора за эпоху. Внутри детектора длина волны в точке объекта немного разная от экспозиции к экспозиции — для мигания это нормально; для спектра используйте отдельные вырезки и `spectral_wcs`, а не кадры.
