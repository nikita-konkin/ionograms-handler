"""Russian.

**The Latin domain abbreviations stay Latin.** MUF, LOF, foF2, hmF2, IRI, SNR,
MAE, SAO, tx and rx are not translated, and neither are the estimator names,
the model origins (`legacy`, `trained`, `imported`) or the framework and
capability values. They are what the plot axes, the database columns, the SAO
records, the `.h5` field names and the importer's own command line say, and a
console that renders МПЧ where the data says MUF makes an operator translate
in their head every time they compare the two. Only the prose around them is
Russian; units (МГц, км) are Russian, because those have settled forms nobody
reads as a different quantity.

Every key here must exist in `en.py` and vice versa -- see the parity test.
"""

from __future__ import annotations

MESSAGES: dict[str, str] = {

    # -- chrome ---------------------------------------------------------
    "app.title": "ионограммы",
    "chrome.language": "язык",

    "nav.console": "консоль",
    "nav.series": "ряды",
    "nav.soundings": "зондирования",
    "nav.sources": "источники",
    "nav.archives": "архивы",
    "nav.forecast": "прогноз",
    "nav.api": "api",

    # -- units ------------------------------------------------------------
    "unit.s": "с",
    "unit.m": "м",
    "unit.h": "ч",
    "unit.d": "д",
    "unit.km": "км",
    "unit.mhz": "МГц",

    # -- shared -----------------------------------------------------------
    "common.from": "с",
    "common.to": "по",
    "common.apply": "применить",
    "common.clear": "сбросить",
    "common.open": "открыть",
    "common.yes": "да",
    "common.previous": "назад",
    "common.next": "вперёд",
    "common.col.circuit": "tx → rx",
    "common.pager": "{first}–{last} из {total}",
    "common.pager.empty": "0 из {total}",

    # -- soundings --------------------------------------------------------
    "soundings.title": "зондирования",
    "soundings.hint": "найдено: {n}",
    "soundings.filter.tx": "tx",
    "soundings.filter.tx.any": "любой",
    "soundings.filter.format": "формат",
    "soundings.filter.format.any": "любой",
    "soundings.filter.picks": "снятые значения",
    "soundings.filter.picks.any": "любые",
    "soundings.filter.picks.some": "хотя бы одно",
    "soundings.filter.picks.none": "нет",
    "soundings.col.time": "время (UTC)",
    "soundings.col.format": "формат",
    "soundings.col.sweep": "полоса, МГц",
    "soundings.col.gate": "строб, км",
    "soundings.col.complete": "полнота",
    "soundings.col.picks": "снято",
    "soundings.empty": "По этим фильтрам ничего не найдено.",
    "soundings.empty.clear": "Сбросить их.",

    # -- one sounding -----------------------------------------------------
    "sounding.notfound": "не найдено",
    "sounding.all": "все зондирования",
    "sounding.earliest": "самое раннее зондирование",
    "sounding.latest": "самое позднее зондирование",
    "sounding.arrows":
        "&larr; и &rarr; перелистывают зондирования по времени.",
    "sounding.col.path": "трасса",
    "sounding.col.sweep": "полоса",
    "sounding.col.gate": "строб",
    "sounding.col.file": "файл",
    "sounding.percent_complete": "пройдено {pct}%",

    "sounding.ionogram": "ионограмма",
    "sounding.gate_label": "строб дальности:",
    "sounding.gate.auto": "авто",
    "sounding.gate.full": "полный",
    "sounding.plot_label": "график:",
    "sounding.plot.interactive": "интерактивный",
    "sounding.plot.rendered": "готовый",
    "sounding.scaling_label": "обработка:",
    "sounding.method_title": "чей след нарисован",
    "sounding.sao": "скачать SAO.XML",
    "sounding.sao_title": "все три записи, SAO.XML 5.0",
    "sounding.auto_note":
        "<b>авто</b> подгоняет окно под то место, где находится эхо. Продукт "
        "v2 в режиме поиска хранит &plusmn;3998 км, а след занимает несколько "
        "сотен, поэтому на полном размахе он выглядит волоском в пустом поле.",
    "sounding.rendered_note":
        "Рисуется по запросу, а не заранее &mdash; 288 изображений в сутки "
        "заранее это 105 тыс. файлов в год, которые в основном никто не "
        "открывает.",
    "sounding.no_scaling": "нет обработки",
    "sounding.scaling_failed":
        "Сохранённые извлечения ниже не затронуты &mdash; они записаны при "
        "загрузке. Попробуйте <a href=\"{url}\">готовый</a> график, ему "
        "запись SAO не нужна.",
    "sounding.show_points": "снятые точки",
    "sounding.show_raster": "растр",
    "sounding.show_marks": "MUF / LOF",
    "sounding.legend_hint":
        "&mdash; щелчок по строке легенды скрывает след, двойной щелчок "
        "оставляет только его.",
    "sounding.relative_pill": "относительная дальность",
    "sounding.relative_note":
        "Нулю дальности этой записи доверять нельзя: разности вдоль следа "
        "верны, начало отсчёта &mdash; нет. Не читайте ось как групповую "
        "дальность.",
    "sounding.traces_points":
        "{method} &mdash; {traces} {trace_unit}, {points} {point_unit}",
    "sounding.declined": "этот метод отказался &mdash; значение не снято",
    "sounding.letter_title": "квалифицирующая буква UAG-23A: {letter}",
    "sounding.model_note":
        "Модельные строки помечены моделью, которая их утверждает, и никогда "
        "не используются для правки измерения. IRI и измеренный MUF временами "
        "расходятся на несколько МГц, и то, что они записаны рядом, не решает, "
        "кто из них прав.",
    "sounding.scaled_in": "Обработано за {cost} с.",

    "sounding.extractions": "извлечения",
    "sounding.col.method": "метод",
    "sounding.col.muf": "MUF",
    "sounding.col.lof": "LOF",
    "sounding.col.range": "дальность, км",
    "sounding.col.snr": "SNR, дБ",
    "sounding.col.run": "серия",
    "sounding.col.hops": "скачки",
    "sounding.col.scatter": "разброс",
    "sounding.col.flags": "признаки",
    "sounding.no_pick": "не снято",
    "sounding.limited_title":
        "значение снято на верхней границе полосы: MUF является нижней оценкой",
    "sounding.loflim_title": "LOF ушёл ниже нижней границы полосы",
    "sounding.stored_note":
        "Записано при загрузке. Признаки качества идут вместе со значением, "
        "никогда отдельно &mdash; <b>limited</b> означает, что MUF является "
        "нижней оценкой, а не измерением. Панель выше обрабатывает продукт "
        "заново, поэтому она может расходиться с этими строками, если "
        "детекторы с тех пор изменились.",

    "sounding.axis.freq": "Частота (МГц)",
    "sounding.axis.range": "Действующая дальность (км)",
    "sounding.hop": "скачок",

    # -- forecast ---------------------------------------------------------
    "forecast.title": "прогноз",
    "forecast.hint":
        "зарегистрировано {models} {model_unit}, работает {n} на {circuits} "
        "{circuit_unit}",
    "forecast.live": "в работе",
    "forecast.col.circuit": "трасса",
    "forecast.col.param": "параметр",
    "forecast.col.model": "модель",
    "forecast.col.last_issue": "последний выпуск",
    "forecast.col.state": "состояние",
    "forecast.no_model": "нет модели",
    "forecast.stale": "устарело",
    "forecast.ok": "норма",
    "forecast.no_circuits":
        "Пока ни на одной трассе нет достаточно данных для прогноза.",
    "forecast.overtaken": "обойдена",
    "forecast.drift":
        "{circuit} / {param}: <b>{model}</b> даёт {mae} МГц на {lead}, тогда "
        "как <b>{baseline}</b> даёт {baseline_mae}, по {n} {pair_unit}. "
        "Показано, но не снято с работы.",
    "forecast.nothing_live":
        "Ничего не работает. Для свежего развёртывания это нормальное "
        "состояние, а не сбой &mdash; зарегистрируйте модель командой "
        "<code>python -m services.prediction.importer</code>, затем назначьте "
        "её ниже. До тех пор <code>/forecast</code> отдаёт пустоту, а не "
        "что-то непроверенное.",

    "forecast.models": "модели",
    "forecast.col.origin": "происхождение",
    "forecast.col.inputs": "входы",
    "forecast.golden.recorded": "эталон записан",
    "forecast.golden.absent": "эталона нет",
    "forecast.unbound": "без трассы",
    "forecast.of": "по",
    "forecast.state.active": "В РАБОТЕ",
    "forecast.state.comparison": "для сравнения",
    "forecast.activate": "назначить",
    "forecast.retire": "снять",
    "forecast.activate.modelled_title":
        "Обучена по модельной цели, поэтому её можно сравнивать сколько "
        "угодно, но нельзя назначить. Отказывает схема базы, а не эта "
        "страница.",
    "forecast.activate.unbound_title":
        "Не привязана ни к одной трассе, поэтому прогнозировать ей нечего. "
        "Загрузите её заново с --tx и --rx.",
    "forecast.nothing_registered":
        "Ничего не зарегистрировано. <code>python -m "
        "services.prediction.importer &lt;файл&gt; --param muf</code> "
        "регистрирует модель; маршрута по HTTP для этого намеренно нет, потому "
        "что регистрация модели означает запуск кода из файла на общем томе.",

    "forecast.leaderboard": "рейтинг",
    "forecast.leaderboard.sub": "{circuit} &middot; {param} &middot; MAE (МГц)",
    "forecast.circuit_label": "трасса:",
    "forecast.col.subject": "предмет",
    "forecast.col.pairs": "пар",
    "forecast.baselines_divider": "&mdash; опорные методы &mdash;",
    "forecast.baseline": "опорный",
    "forecast.censored_note":
        "Часы с цензурой &mdash; значение снято на верхней границе полосы или "
        "на её дне &mdash; оцениваются односторонне и считаются отдельно, "
        "чтобы оценка сверху не размывала число выше. Пересчёт: <code>python "
        "-m services.prediction.scoring --once</code>.",
    "forecast.nothing_scored":
        "Для этой трассы ещё ничего не оценено. Запустите <code>python -m "
        "services.prediction.scoring --once</code> или дождитесь следующего "
        "прохода <code>infer</code> &mdash; оценка идёт следом за ним. До тех "
        "пор таблица выше говорит, что модель <i>из себя представляет</i>, но "
        "не насколько она хороша.",
    "forecast.col.what": "что это",
    "forecast.baseline.persistence":
        "значение, снятое одну заблаговременность назад; на горизонте 24 ч это "
        "вчера в ту же минуту UTC, что даром улавливает суточный ход",
    "forecast.baseline.recurrence":
        "один оборот Солнца назад; стандартный оперативный опорный метод для КВ",
    "forecast.baseline.iri":
        "сохранённое опорное значение IRI в контрольной точке трассы &mdash; "
        "уже построено и проверено. Только MUF: IRI ничего не говорит о "
        "поглощении, которое задаёт LOF",
    "forecast.baseline.harmonic":
        "суточные гармоники плюс зенитный угол Солнца, подогнанные строго до "
        "оцениваемого окна, чтобы метод не стал оракулом в собственном рейтинге",

    "forecast.js.no_token":
        "Нет управляющего токена. Сначала вставьте его на странице консоли.",
    "forecast.js.activate_confirm":
        "Назначить {name} прогнозом {param} для трассы {circuit}?\n\n"
        "То, что работает на этой трассе сейчас, будет снято тем же действием.",
    "forecast.js.retire_confirm":
        "Снять {name}?\n\n"
        "Её строки и её прогнозы сохраняются, поэтому действие обратимо: "
        "достаточно назначить её снова.",
    # -- archives ---------------------------------------------------------
    "archives.title": "архивы",
    "archives.hint": "зарегистрировано {n} {unit}",
    "archives.mounted": "подключённый архив",
    "archives.col.host": "папка на хосте",
    "archives.col.seen_as": "видна здесь как",
    "archives.col.state": "состояние",
    "archives.not_reported": "не сообщено",
    "archives.primary": "основная",
    "archives.not_readable": "не читается",
    "archives.mounted_empty": "подключена, но пуста",
    "archives.ok": "норма",
    "archives.in_container":
        "Работа идёт в контейнере, поэтому &laquo;видна здесь как&raquo; "
        "&mdash; это точка монтирования, а папка на хосте &mdash; то, что "
        "задано в <code>deploy/.env</code>.",
    "archives.one_root":
        "Корень один. Чтобы индексировать папку на другом диске, добавьте ей "
        "собственную строку <code>volumes:</code> <i>и</i> перечислите её путь "
        "в контейнере в <code>ARCHIVE_ROOTS</code> &mdash; и то и другое, "
        "затем разверните заново. Со страницы путь добавить нельзя: файловая "
        "система контейнера фиксируется при его запуске.",
    "archives.fault.unreachable":
        "<b>Монтирование на месте, но хранилище за ним не отвечает.</b> "
        "<code>{root}</code> существует внутри контейнера, и любое чтение "
        "оттуда завершается ошибкой &mdash; {error}. "
        "<code>ARCHIVE_HOST_PATH</code> здесь <i>ни при чём</i>, и повторное "
        "развёртывание не поможет: чините на хосте, где смонтирован том, и "
        "сканирование возобновится само. До тех пор ничего нельзя ни "
        "зарегистрировать, ни проиндексировать.",
    "archives.fault.denied":
        "Монтирование существует, но этот процесс не имеет права его читать "
        "&mdash; {error}. api работает под uid 10001; этому uid нужен доступ "
        "на чтение к папке на хосте. Пока это не исправлено, ни одну папку "
        "зарегистрировать нельзя.",
    "archives.fault.missing":
        "По пути <code>{root}</code> ничего не читается. В Docker это значит, "
        "что привязанного тома нет или его источник пропал &mdash; задайте "
        "<code>ARCHIVE_HOST_PATH</code> в <code>deploy/.env</code> и "
        "разверните заново. Пока это не исправлено, ни одну папку "
        "зарегистрировать нельзя.",
    "archives.fault.empty":
        "Монтирование существует, но пусто. Привязанный том, источник которого "
        "переименовали на хосте, по-прежнему виден внутри контейнера &mdash; "
        "как пустой каталог, и тогда любое сканирование будет честно и вечно "
        "сообщать &laquo;0 на диске&raquo;.",
    "archives.intro":
        "Папки внутри <code>{root}</code>, которые этот сервер держит "
        "проиндексированными. Сканирование прогоняет конвейер по всему новому, "
        "поэтому каждое загруженное зондирование приходит с уже выведенными "
        "характеристиками &mdash; MUF, LOF, групповая дальность и SNR в базе, "
        "а также полная обработка за страницей каждого зондирования и его "
        "<code>sao.xml</code>. Отдельного шага для их вычисления нет.",
    "archives.rescan_on":
        "Включённые папки пересканируются автоматически каждые {minutes} мин. "
        "Сканирования идут по одному: они нагружают процессор и держат базу, "
        "поэтому два закончатся позже, чем одно за другим.",
    "archives.rescan_off":
        "Автоматическое пересканирование выключено "
        "(<code>ARCHIVE_SCAN_INTERVAL_S=0</code>), поэтому новые файлы "
        "появятся, только когда кто-нибудь нажмёт сканирование.",
    "archives.col.name": "имя",
    "archives.col.path": "путь",
    "archives.col.format": "формат",
    "archives.col.methods": "методы",
    "archives.col.soundings": "зондирований",
    "archives.col.last_scan": "последнее сканирование",
    "archives.col.result": "результат",
    "archives.disabled": "отключена",
    "archives.any": "любой",
    "archives.set": "задать",
    "archives.scan_now": "сканировать",
    "archives.disable": "отключить",
    "archives.enable": "включить",
    "archives.remove": "удалить",
    "archives.nothing_registered":
        "Пока ничего не зарегистрировано. Пока папки здесь нет, этот сервер "
        "индексирует только то, по чему кто-то вручную прогнал "
        "<code>services.api.ingest</code>.",

    "archives.add": "добавить папку",
    "archives.add_note":
        "Путь задаётся относительно <code>{root}</code>. Индексировать можно "
        "только папки внутри этого корня &mdash; в контейнере это "
        "единственный путь, смонтированный как <code>/archive</code>, поэтому "
        "папка вне его не просто нечитаема, она невидима, и сканирование "
        "отчиталось бы об успехе, ничего не загрузив.",
    "archives.cands_note":
        "Папки с суточными данными зондирования и дни внутри каждой из них. "
        "Регистрация одной покрывает все дни, которые в ней есть, и все дни, "
        "которые приёмник добавит позже &mdash; сканирование рекурсивно, "
        "поэтому новая дневная папка не требует здесь никаких действий.",
    "archives.col.folder": "папка",
    "archives.col.root": "корень",
    "archives.col.holds": "содержит",
    "archives.mount_itself": "(само монтирование)",
    "archives.day_folders": "{n} {unit}",
    "archives.inside":
        "внутри <code>{path}</code> &mdash; регистрируйте одно или другое, "
        "но не оба",
    "archives.stored_absolute": "хранится абсолютным",
    "archives.indexed": "проиндексировано: {n}",
    "archives.holds": "{n} {unit}",
    "archives.unreadable": "ничего, что этот сервер умеет читать",
    "archives.registered": "зарегистрирована",
    "archives.use": "выбрать",
    "archives.field.path": "путь",
    "archives.field.name": "имя",
    "archives.name_placeholder": "(по умолчанию совпадает с путём)",
    "archives.field.format": "формат",
    "archives.add_button": "добавить",
    "archives.methods": "методы",
    "archives.all": "все",
    "archives.methods_note":
        "Добавленный позже метод при следующем сканировании охватит уже "
        "имеющиеся зондирования папки &mdash; всё, что уже вычислено, "
        "сохраняется.",
    "archives.methods_note.cnn":
        "Метод, который здесь не может работать, отключён, а не просто "
        "отмечен предупреждением: <code>already_done</code> считает "
        "зондирование законченным, только когда по нему есть строка для "
        "<i>каждого</i> запрошенного метода, поэтому выбор метода, который "
        "никогда не даёт строки, заставлял бы пересканировать всю папку на "
        "каждом проходе, вечно.",
    "archives.looking": "смотрим, что смонтировано &hellip;",

    # -- series -----------------------------------------------------------
    "series.title": "ряды",
    "series.hint": "{n} {unit}",
    "series.h2": "Параметры во времени",
    "series.nothing_ingested": "пока ничего не загружено",
    "series.circuit": "трасса",
    "series.all_overlaid": "все (наложением)",
    "series.no_picks": "для этого метода ничего не снято",
    "series.day": "день",
    "series.all": "все",
    "series.reference": "опорная модель",
    "series.off": "выкл",
    "series.off_note":
        "выкл пропускает модель, а вместе с ней и сеть, которая ей может "
        "понадобиться.",
    "series.bare_date":
        "Одна дата без времени охватывает весь этот день с любого конца.",
    "series.family.forecast": "прогноз",
    "series.family.context": "верх полосы / hmF2",
    "series.drag_hint":
        "&mdash; протяните, чтобы приблизить, двойной щелчок сбрасывает, "
        "щелчок по точке открывает её ионограмму.",
    "series.hue_note":
        "<b>Цвет &mdash; это параметр, форма &mdash; это источник.</b> Маркеры "
        "измерены, линии &mdash; модель, поэтому синий маркер и синяя "
        "пунктирная линия рядом с ним &mdash; это MUF этой трассы и MUF по "
        "IRI, а разрыв между ними и есть то, что рисует панель невязок.",
    "series.hue_note.multi":
        "Когда наложено несколько трасс, цвет означает <b>трассу</b> &mdash; "
        "два MUF разных трасс одним цветом это ровно то сравнение, к которому "
        "нельзя подталкивать, &mdash; поэтому сначала рисуются только MUF и "
        "его модель. Каждая трасса моделируется в своей контрольной точке; "
        "легенда их группирует.",
    "series.hollow_note":
        "<b>Полые маркеры &mdash; это границы, а не измерения.</b> Полый "
        "маркер MUF стоял на верхнем краю полосы, значит настоящий MUF равен "
        "ему или выше; полый маркер LOF стоял на дне полосы, значит настоящий "
        "LOF равен ему или ниже. Они нарисованы, а не отброшены &mdash; тихо "
        "убрать любой из них значило бы притянуть кривую к середине полосы. В "
        "статистику невязок ниже они не входят; это отдельное решение, и о нём "
        "сказано там же.",
    "series.fof2_note":
        "<b>foF2 здесь не измерен.</b> Наклонный зонд никогда не видит "
        "вертикального падения. Это измеренный MUF, пересчитанный обратно по "
        "закону секанса при hmF2&nbsp;=&nbsp;{hmf2}&nbsp;км на одном скачке, "
        "что и делает его сопоставимым с моделью или с ближайшим вертикальным "
        "ионозондом. Собственный foF2 у IRI &mdash; это выход модели, "
        "пересчитанный тем же способом в обратную сторону, чтобы получить "
        "стоящий рядом MUF.",
    "series.lof_note":
        "<b>LOF, а не LUF.</b> Раздел 9 ITU-R P.533-13 определяет наименьшую "
        "<i>применимую</i> частоту через требуемое отношение сигнал/шум и "
        "медиану за месяц &mdash; свойство службы и месяца, а ни того ни "
        "другого у одного зондирования нет. Наклонный зонд снимает наименьшую "
        "<i>наблюдаемую</i> частоту, и это она и есть. Она следит за "
        "поглощением в области D, то есть идёт за Солнцем, а не за слоем F2: "
        "MUF, который движется, пока LOF под ним стоит на месте, стоит "
        "посмотреть ещё раз.",
    "series.summary": "сводка",
    "series.col.n": "n",
    "series.col.muf_median": "медиана MUF",
    "series.col.at_ceiling": "у потолка",
    "series.col.lof_median": "медиана LOF",
    "series.col.at_floor": "у дна",
    "series.col.fof2_median": "медиана foF2",
    "series.col.vs_iri": "против IRI: n",
    "series.col.bias": "смещение",
    "series.col.rms": "СКО",
    "series.col.r": "r",
    "series.path_km": "{km} км",
    "series.hops": ", скачков: {n}",
    "series.limited_title": "снято на верхнем краю полосы: нижняя оценка",
    "series.loflim_title": "LOF ушёл ниже дна полосы: верхняя оценка",
    "series.excluded_title": "нижние оценки, исключены",
    "series.reference_off": "модель выключена",
    "series.no_pair": "не с чем сравнивать",
    "series.bias_note":
        "<b>Смещение &mdash; это медиана разности измеренного и IRI по тем "
        "парам, которые являются измерениями.</b> Нижние оценки, посчитанные "
        "в столбце <i>у потолка</i>, из неё исключены: значение, прижатое к "
        "верхнему краю полосы, говорит, что ионосфера выдерживала <i>не "
        "менее</i> этого, и засчитать его как невязку значило бы выдать "
        "потолок полосы регистратора за ошибку модели. На трассе, ограниченной "
        "потолком, это большая часть светового дня &mdash; поэтому число "
        "исключённых точек печатается рядом с числом использованных.",
    "series.iri_note":
        "IRI &mdash; это климатология, управляемая сглаженным солнечным "
        "индексом, а для недавнего месяца такого индекса ещё не существует "
        "&mdash; см. <code>muf/reference/indices.py</code>. Расхождение в "
        "несколько МГц считайте нормой, а интересна <i>форма</i> невязки: "
        "постоянное смещение &mdash; это вопрос масштаба, суточное &mdash; "
        "нет.",

}

#: Three forms, not two: 1 зондирование, 2 зондирования, 5 зондирований.
#: `i18n._form` decides which, from the last digit and the last two digits.
PLURALS: dict[str, dict[str, str]] = {
    "sounding": {"one": "зондирование", "few": "зондирования",
                 "many": "зондирований"},
    "trace": {"one": "след", "few": "следа", "many": "следов"},
    "point": {"one": "точка", "few": "точки", "many": "точек"},
    "model": {"one": "модель", "few": "модели", "many": "моделей"},
    "circuit": {"one": "трасса", "few": "трассы", "many": "трасс"},
    "pair": {"one": "паре", "few": "парам", "many": "парам"},
    "folder": {"one": "папка", "few": "папки", "many": "папок"},
    "dayfolder": {"one": "дневная папка", "few": "дневные папки",
                  "many": "дневных папок"},
}
