"""The live routes: the agent posts batches, browsers read a snapshot or a stream."""

from __future__ import annotations

import base64
import random
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")


def batch(rows=None, columns=None):
    rows = rows or random.randint(20, 300)
    head = {"type": "sounding", "tx": "cyprus1",
            "t0": 1792161010 + 300 * random.randint(0, 99),
            "f_lo_mhz": 7.5, "f_step_mhz": 0.04096,
            "columns": columns or random.randint(10, 610),
            "range_lo_km": 2551.3, "range_step_km": 7.32, "rows": rows,
            "km_per_ms": 300.0}
    cells = np.random.randint(0, 256, rows, dtype=np.uint8)
    return [head, {"type": "column", "tx": "cyprus1", "t0": head["t0"], "index": 0,
                   "db": base64.b64encode(cells.tobytes()).decode("ascii")}]


def agent():
    return {"Authorization": "Bearer ctl"}


def push(client, events, station="YOS", **kw):
    return client.post("/stations/%s/live" % station, json={"events": events}, **kw)


def page(client, query="?station=YOS"):
    return client.get("/ui/live" + query).text


def snapshot(client):
    return client.get("/stations/YOS/live").json()


def test_agent_token_may_push_live_events(client):
    assert push(client, batch(), headers=agent()).status_code == 200, (
        "the agent token could not push live events")


def queue(client, station, count):
    for _ in range(count):
        client.post("/stations/%s/commands" % station,
                    json={"name": random.choice(["start", "stop"])},
                    headers=agent()).raise_for_status()


def test_live_reply_counts_the_commands_queued_for_its_station(client):
    count = random.randint(1, 5)
    queue(client, "YOS", count)
    assert push(client, batch(), headers=agent()).json()["queued"] == count, (
        "the live reply did not count the commands queued for the station")


def test_live_reply_does_not_count_another_stations_commands(client):
    queue(client, "NOR", random.randint(1, 5))
    assert push(client, batch(), headers=agent()).json()["queued"] == 0, (
        "the live reply counted commands queued for another station")


def test_live_reply_leaves_the_commands_it_counts_for_the_agent(client):
    count = random.randint(1, 5)
    queue(client, "YOS", count)
    push(client, batch(), headers=agent())
    assert len(client.get("/stations/YOS/commands",
                          headers=agent()).json()["commands"]) == count, (
        "the live reply took commands it was only meant to count")


def test_live_reply_does_not_count_commands_already_taken(client):
    queue(client, "YOS", random.randint(1, 5))
    client.get("/stations/YOS/commands", headers=agent())
    assert push(client, batch(), headers=agent()).json()["queued"] == 0, (
        "the live reply counted commands the agent had already taken")


def test_empty_live_push_still_counts_the_queue(client):
    count = random.randint(1, 5)
    queue(client, "YOS", count)
    assert push(client, [], headers=agent()).json()["queued"] == count, (
        "a live push with no events did not count the queue")


def test_live_push_without_a_token_is_refused(client):
    assert push(client, batch()).status_code == 401, (
        "an anonymous caller pushed live events")


def test_malformed_batch_is_refused_as_a_bad_request(client):
    events = batch()
    events[1]["index"] = events[0]["columns"] + random.randint(0, 9)
    assert push(client, events, headers=agent()).status_code == 400, (
        "a column past its sounding was accepted")


def test_batch_without_an_event_list_is_refused(client):
    assert push(client, "column", headers=agent()).status_code == 400, (
        "a batch without a list of events was accepted")


def test_snapshot_shows_a_pushed_column(client):
    events = batch()
    push(client, events, headers=agent())
    assert snapshot(client)["current"]["cyprus1"]["data"]["0"] == events[1]["db"], (
        "the snapshot did not show the column just pushed")


def test_event_stream_opens_with_a_snapshot_of_the_station(client):
    push(client, batch(), headers=agent())
    client.app.state.live.close()
    assert client.get("/stations/YOS/live/events").text.startswith(
        'event: snapshot\ndata: {"station":"YOS"'), (
        "the event stream did not open with the station's snapshot")


def test_event_stream_is_served_as_server_sent_events(client):
    client.app.state.live.close()
    assert client.get("/stations/YOS/live/events").headers["content-type"].startswith(
        "text/event-stream"), "the event stream is not text/event-stream"


def test_live_page_names_its_station(client):
    assert 'data-station="YOS"' in page(client), (
        "the live page does not name the station it watches")


def test_live_page_offers_start_to_whoever_may_control_a_station(client):
    assert '<button data-needs="station" data-command="start">' in page(client), (
        "the live page offers no Start gated on the station capability")


def test_live_page_offers_stop_to_whoever_may_control_a_station(client):
    assert '<button data-needs="station" data-command="stop">' in page(client), (
        "the live page offers no Stop gated on the station capability")


def test_live_page_disables_its_controls_for_whoever_may_not(client):
    assert "gate('station')" in page(client), (
        "the live page never gates its controls on the station capability")


def test_live_page_queues_its_commands_through_the_command_route(client):
    assert ("control('/stations/' + encodeURIComponent(STATION) + '/commands', "
            "'POST'") in page(client), (
        "the live page does not queue its commands through "
        "POST /stations/{id}/commands")


def test_live_page_labels_start_in_russian(client):
    assert "Пуск</button>" in page(client, "?station=YOS&lang=ru"), (
        "the live page's Start is not labelled in Russian")


def test_live_page_titles_its_strips_without_a_fixed_span(client):
    assert '"Signal/noise"' in page(client, "?station=YOS&lang=en"), (
        "the live page's SNR strip still claims a fixed span")


def test_live_page_zooms_its_strips_in_to_six_hours(client):
    assert "const MIN_SPAN_S = 6 * 3600;" in page(client), (
        "the live page cannot zoom its strips in to six hours")


def test_live_page_says_how_long_a_strip_spans(client):
    assert "T('live.js.span_h'" in page(client), (
        "the live page does not say how long a strip spans")


def test_live_page_defaults_to_a_station_that_is_live(client):
    push(client, batch(), "NOR", headers=agent())
    assert 'data-station="NOR"' in page(client, ""), (
        "the live page did not pick the station that is live")


def test_what_the_agent_sends_the_server_accepts(client, make_lfs, tmp_path):
    from services.agent import live as feed
    from services.agent.config import StationConfig

    (tmp_path / "2026.02.04").mkdir()
    windows = random.randint(2, 6)
    noise = np.random.normal(0, 1, (2, windows * 16384))
    make_lfs(noise[0] + 1j * noise[1], name="2026.02.04/cyprus1_20260204_000010.lfs")

    def post(events):
        push(client, events, headers=agent()).raise_for_status()

    feed.Feed(StationConfig(station="YOS", live_dir=tmp_path),
              lambda: ["STATUS cyprus1 capturing 0.0100 0"],
              post, lambda: 1770163210 + 30).step()
    assert len(snapshot(client)["current"]["cyprus1"]["data"]) == windows, (
        "the server did not take every column the agent sent")


def test_live_page_says_a_stop_waits_for_the_sounding_in_progress(client):
    assert "T('live.js.stop_queued'" in page(client), (
        "the live page does not say a queued stop waits for the sounding in progress")


def test_live_page_shows_the_chirp_time_after_the_band(client):
    text = page(client, "?station=YOS&lang=en")
    assert text.find("Chirp time, s</th>") > text.find("Band, MHz</th>") > 0, (
        "the schedule does not show the chirp time after the band")


def test_live_page_names_the_chirp_time_in_russian(client):
    assert "Время ЛЧМ, с</th>" in page(client, "?station=YOS&lang=ru"), (
        "the schedule does not name the chirp time in Russian")


def test_live_page_steps_its_control_ionogram_through_the_archive(client):
    assert "/live/archive?tx=" in page(client), (
        "the live page cannot step back through the archive")


def test_live_page_reads_moscow_time_three_hours_ahead_of_utc(client):
    assert "const MSK_S = 3 * 3600;" in page(client), (
        "the live page does not read the strips in Moscow time")


def test_live_page_fills_a_strip_until_the_next_sounding(client):
    assert "const HOLD_S = 1800;" in page(client), (
        "the live page does not hold a strip column until the next sounding")


def test_live_page_hides_transmitters_the_schedule_switched_off(client):
    assert "addEventListener('ionozond-schedule'" in page(client), (
        "the live page does not follow the schedule's transmitters")


def test_live_page_asks_how_far_each_transmitter_is(client):
    assert "/live/path?tx=" in page(client), (
        "the live page cannot place its ionograms' height axis")


def test_live_page_measures_height_over_a_spherical_earth(client):
    assert "const EARTH_KM = 6371;" in page(client), (
        "the live page has no spherical Earth to measure height over")


def test_live_page_fills_its_plots_with_the_plot_colour(client):
    assert "g.fillStyle = css('--plot');" in page(client), (
        "the live page does not fill its plots with the theme's plot colour")


def test_live_page_keeps_height_labels_apart(client):
    assert "const TICK_GAP_PX = 12;" in page(client), (
        "the live page lets crowded height labels overlap")


def test_live_page_labels_its_colour_bars_as_densely_as_its_axes(client):
    assert "ticks(lo, hi, Math.max(4, Math.floor(ph / 34)))" in page(client), (
        "the live page labels its colour bars more sparsely than its axes")


def test_live_page_can_label_the_strips_every_hour(client):
    assert "const HOUR_STEPS = [1, 2, 3, 6, 12, 24];" in page(client), (
        "the live page cannot label its strips more often than every 3 hours")


def test_live_page_carries_a_schedule_panel(client):
    assert '<section class="panel" id="sched" hidden>' in page(client), (
        "the live page has no schedule panel")


def test_live_page_reads_the_schedule_in_force(client):
    assert "'/stations/' + encodeURIComponent(STATION) + '/ionozond'" in page(client), (
        "the live page does not read GET /stations/{id}/ionozond")


def test_live_page_checks_each_edit_without_queueing_it(client):
    assert ("control('/stations/' + encodeURIComponent(STATION) + '/ionozond/check', "
            "'POST'") in page(client), (
        "the live page does not check an edit through /ionozond/check")


def test_live_page_queues_a_schedule_change_as_set_ionozond(client):
    assert "name: 'set_ionozond'" in page(client), (
        "the live page does not queue a schedule change as set_ionozond")


def test_live_page_offers_apply_to_whoever_may_control_a_station(client):
    assert '<button data-needs="station" id="sched-apply"' in page(client), (
        "the live page offers no Apply gated on the station capability")


def test_live_page_asks_once_more_before_applying_over_warnings(client):
    assert "T('live.js.sched_confirm'" in page(client), (
        "the live page does not ask again before applying over warnings")


def test_live_page_says_a_written_change_waits_for_the_sounding(client):
    assert "T('live.js.sched_pending'" in page(client), (
        "the live page does not say a written change awaits the sounding in progress")


def test_live_page_says_when_the_sounder_file_has_drifted(client):
    assert "T('live.js.sched_drift'" in page(client), (
        "the live page does not say when chirp_config.py has drifted")


def test_live_page_starts_each_band_half_the_full_sample_rate_below_cf(client):
    assert "const sr = (shown.sounder_config.settings || {}).sample_rate_khz;" in page(
        client), "the live page starts a band half the decimated sample rate below cf"


def test_live_page_names_the_schedule_in_russian(client):
    assert "Расписание</h2>" in page(client, "?station=YOS&lang=ru"), (
        "the live page's schedule panel is not named in Russian")


def test_live_page_offers_the_h5_archive_switch(client):
    assert '<input type="checkbox" data-general="h5_archive">' in page(client), (
        "the live page offers no .h5 archive switch")


def test_live_page_offers_the_local_h5_retention(client):
    assert '<input data-general="nas_keep_h5_days">' in page(client), (
        "the live page offers no local .h5 retention")


def test_live_page_shows_the_nas_destination(client):
    assert "T('live.js.sched_nas', {dest: cfg.nas_dest})" in page(client), (
        "the live page does not show the NAS destination")


def test_live_page_offers_regenerate_to_whoever_may_control_a_station(client):
    assert '<button data-needs="station" id="sched-regenerate"' in page(client), (
        "the live page offers no Regenerate gated on the station capability")


def test_live_page_queues_regenerate_as_regenerate_ionozond(client):
    assert "name: 'regenerate_ionozond'" in page(client), (
        "the live page does not queue Regenerate as regenerate_ionozond")


def test_live_page_shows_what_regenerate_would_change(client):
    assert "cfg.sounder, cfg.regenerated" in page(client), (
        "the live page does not show what Regenerate would change")


def test_live_page_asks_once_more_before_regenerating(client):
    assert "T('live.js.regen_confirm'" in page(client), (
        "the live page does not ask again before regenerating")


def test_the_drift_notice_names_regenerate_not_the_console(client):
    from services.api.locale import en
    assert "dialog" not in en.MESSAGES["live.js.sched_drift"], (
        "the drift notice still sends the operator to the console's dialog")


def test_live_page_lists_the_site_settings(client):
    assert '<table id="site">' in page(client), (
        "the live page does not list the site settings")


def test_live_page_queues_a_site_change_as_set_ionozond_site(client):
    assert "name: 'set_ionozond_site'" in page(client), (
        "the live page does not queue a site change as set_ionozond_site")


def test_live_page_offers_site_apply_to_whoever_may_control_a_station(client):
    assert '<button data-needs="station" id="site-apply"' in page(client), (
        "the live page offers no site Apply gated on the station capability")


@pytest.mark.parametrize("unit", ["ionozond-nas.service", "ionozond-prune.service"])
def test_live_page_offers_run_now_for_each_job(client, unit):
    assert 'data-run="%s"' % unit in page(client), (
        "the live page offers no Run now for %s" % unit)


def test_live_page_queues_run_now_as_a_start_of_the_unit(client):
    assert "{name: 'start', params: {unit: b.dataset.run}}" in page(client), (
        "the live page does not queue Run now as a start of the job unit")


def test_live_page_reads_the_last_runs_from_the_health_report(client):
    assert "'job:' + b.dataset.run" in page(client), (
        "the live page does not read the jobs' last runs from the health report")


def test_live_page_names_the_site_settings_in_russian(client):
    assert "Настройки станции</h3>" in page(client, "?station=YOS&lang=ru"), (
        "the live page's site settings are not named in Russian")


def complete(client, t0, tx="cyprus1", columns=3, rows=8):
    head = {"type": "sounding", "tx": tx, "t0": t0, "f_lo_mhz": 7.5,
            "f_step_mhz": 0.04096, "columns": columns, "range_lo_km": 2551.3,
            "range_step_km": 7.32, "rows": rows, "km_per_ms": 300.0}
    cells = lambda: base64.b64encode(  # noqa: E731
        np.random.randint(0, 256, rows, dtype=np.uint8).tobytes()).decode("ascii")
    push(client, [head] + [{"type": "column", "tx": tx, "t0": t0, "index": i,
                            "db": cells()} for i in range(columns)],
         headers=agent()).raise_for_status()
    return head


def archive(client, head, value):
    snr = base64.b64encode(bytes([value]) * head["columns"]).decode("ascii")
    pdp = base64.b64encode(bytes([value]) * head["rows"]).decode("ascii")
    event = dict({k: v for k, v in head.items() if k != "type"}, type="summary",
                 snr=snr, pdp=pdp)
    push(client, [event], headers=agent()).raise_for_status()
    return snr


def summaries(client, since, until, tx="cyprus1"):
    return client.get("/stations/YOS/live/summaries", params={
        "tx": tx, "since": since, "until": until}).json()


def test_completed_sounding_is_read_back_through_the_range_route(client):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    complete(client, t0)
    assert [s["t0"] for s in summaries(client, t0 - 1, t0 + 1)["summaries"]] == [t0], (
        "a completed sounding was not read back through the range route")


def test_range_route_leaves_out_a_summary_outside_the_range(client):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    complete(client, t0)
    assert summaries(client, t0 + 1, t0 + 600)["summaries"] == [], (
        "the range route returned a summary outside the range")


def test_range_route_names_the_earliest_summary(client):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    complete(client, t0)
    complete(client, t0 + 300)
    assert summaries(client, t0 + 299, t0 + 301)["earliest"] == t0, (
        "the range route did not name the earliest summary the station has")


def test_archived_summary_does_not_replace_a_live_one(client):
    head = complete(client, 1792161010 + 300 * random.randint(0, 99))
    archive(client, head, random.randint(1, 255))
    got = summaries(client, head["t0"], head["t0"])["summaries"]
    assert got[0]["source"] == "live", (
        "a summary from the archive replaced the live one")


def test_live_summary_replaces_an_archived_one(client):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    archive(client, {"type": "sounding", "tx": "cyprus1", "t0": t0, "f_lo_mhz": 7.5,
                     "f_step_mhz": 0.04096, "columns": 3, "range_lo_km": 2551.3,
                     "range_step_km": 7.32, "rows": 8, "km_per_ms": 300.0}, 9)
    complete(client, t0)
    assert summaries(client, t0, t0)["summaries"][0]["source"] == "live", (
        "a live summary did not replace the one from the archive")


def test_archived_summary_is_read_back_with_its_cells(client):
    head = {"type": "sounding", "tx": "cyprus1",
            "t0": 1792161010 + 300 * random.randint(0, 99), "f_lo_mhz": 4.0,
            "f_step_mhz": 0.05, "columns": random.randint(5, 50),
            "range_lo_km": 2000.0, "range_step_km": 5.0,
            "rows": random.randint(5, 50), "km_per_ms": 150.0}
    snr = archive(client, head, random.randint(1, 255))
    assert summaries(client, head["t0"], head["t0"])["summaries"][0]["snr"] == snr, (
        "a summary from the archive did not come back with its cells")


def test_days_route_counts_each_transmitter_per_day(client):
    t0 = 1792195200 + 300 * random.randint(0, 200)
    for k in range(3):
        complete(client, t0 + 300 * k, tx="NIC1")
    days = client.get("/stations/YOS/live/days").json()["days"]
    assert days == {"NIC1": {time.strftime("%Y-%m-%d", time.gmtime(t0)): 3}}, (
        "the days route did not count the summaries of each transmitter per day")


def test_live_page_warns_when_echoes_leave_the_window(client):
    assert "T('live.js.echo_outside'" in page(client), (
        "the live page does not warn when echoes sit outside the window")


def test_live_page_listens_for_where_the_echo_is(client):
    assert "kind === 'echo'" in page(client), (
        "the live page does not listen for where the echo is")


def test_live_page_reads_history_through_the_range_route(client):
    assert "'/live/summaries?tx='" in page(client), (
        "the live page does not read its history through the range route")


def test_live_page_scrolls_history_longer_than_three_days(client):
    assert "const SHOWN_DAYS = 3;" in page(client), (
        "the live page does not show three days of history at a time")


def test_kept_summary_is_committed_for_other_connections(client):
    from services.api import db
    t0 = 1792161010 + 300 * random.randint(0, 99)
    complete(client, t0)
    assert len(db.summaries(db.connect(), "YOS", "cyprus1", t0, t0, 10)) == 1, (
        "a kept summary was not committed where another connection could read it")


def stored(client, make_chirp_h5, t0, tx="cyprus1", rx="YOS"):
    path = make_chirp_h5(np.random.exponential(1.0, (random.randint(20, 60), 600)),
                         t0=float(t0), txname=tx, station_name=rx,
                         name="lfm_ionogram-%s-%s-ch0-000-%.2f.h5" % (tx, rx, t0))
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0))
    conn = client.app.state.db
    conn.execute("INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at)"
                 " VALUES (?, ?, ?, ?, ?, ?)",
                 (path.name, str(path), when, tx, rx, when))
    conn.commit()
    return path


def archived(client, tx="cyprus1", **bound):
    return client.get("/stations/YOS/live/archive", params=dict(bound, tx=tx))


def test_the_archive_offers_the_newest_sounding_before_a_time(client, make_chirp_h5):
    t0 = 1792161010 + 600 * random.randint(0, 99)
    for k in range(3):
        stored(client, make_chirp_h5, t0 + 300 * k)
    assert archived(client, before=t0 + 600).json()["t0"] == t0 + 300, (
        "the archive did not offer the sounding just before the time")


def test_the_archive_offers_the_oldest_sounding_after_a_time(client, make_chirp_h5):
    t0 = 1792161010 + 600 * random.randint(0, 99)
    for k in range(3):
        stored(client, make_chirp_h5, t0 + 300 * k)
    assert archived(client, after=t0).json()["t0"] == t0 + 300, (
        "the archive did not offer the sounding just after the time")


def test_an_archived_sounding_carries_every_column(client, make_chirp_h5):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    stored(client, make_chirp_h5, t0)
    view = archived(client, before=t0 + 1).json()
    assert len(view["data"]) == view["columns"], (
        "an archived sounding did not carry a column per frequency step")


def test_the_archive_says_when_nothing_is_earlier(client, make_chirp_h5):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    stored(client, make_chirp_h5, t0)
    assert archived(client, before=t0).status_code == 404, (
        "the archive did not say it holds nothing earlier")


def test_the_archive_keeps_to_its_station(client, make_chirp_h5):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    stored(client, make_chirp_h5, t0, rx="DOB")
    assert archived(client, before=t0 + 1).status_code == 404, (
        "the archive offered another station's sounding")


def test_the_archive_does_not_take_a_receiver_that_differs_only_in_case(
        client, make_chirp_h5):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    stored(client, make_chirp_h5, t0, rx="yos")
    assert archived(client, before=t0 + 1).status_code == 404, (
        "the archive offered a sounding from a receiver named apart only by case")


def test_a_renamed_station_steps_through_its_old_receivers_soundings(
        client, make_chirp_h5):
    from services.api import db
    t0 = 1792161010 + 300 * random.randint(0, 99)
    complete(client, t0 - 3600)
    stored(client, make_chirp_h5, t0)
    db.rename_station(client.app.state.db, "YOS", "yos-2")
    assert client.get("/stations/yos-2/live/archive",
                      params={"tx": "cyprus1", "before": t0 + 1}).status_code == 200, (
        "a renamed station cannot step through soundings filed under its old name")


def test_the_archive_needs_a_bound(client):
    assert archived(client).status_code == 400, (
        "the archive answered without a time to step from")


def ground(tx, rx):
    from types import SimpleNamespace

    from muf import calibrate
    return calibrate.ground_range_km(SimpleNamespace(
        is_oblique=True, tx_latitude=tx[0], tx_longitude=tx[1],
        rx_latitude=rx[0], rx_longitude=rx[1]))


def test_a_path_is_measured_from_the_schedule_the_station_reported(client):
    from tests.test_api_sounder import health
    health(client)
    got = client.get("/stations/yoshkar-ola/live/path", params={"tx": "cyprus1"}).json()
    assert got["ground_km"] == pytest.approx(ground((35, 34), (56.38, 47.53))), (
        "the path was not measured from the reported schedule")


def test_a_path_is_measured_from_the_registry_without_a_report(client):
    from muf.stations import default_registry
    sites = default_registry()
    tx, rx = sites.station("NIC1"), sites.station("yoshkar-ola")
    got = client.get("/stations/yoshkar-ola/live/path", params={"tx": "NIC1"}).json()
    assert got["ground_km"] == pytest.approx(ground(
        (tx.latitude, tx.longitude), (rx.latitude, rx.longitude))), (
        "the path was not measured from the station registry")


def test_an_unknown_path_is_not_measured(client):
    got = client.get("/stations/yoshkar-ola/live/path",
                     params={"tx": "nowhere%d" % random.randint(0, 999)})
    assert got.status_code == 404, "a path nobody can place was measured"


def picked(client, t0, muf, lof=None, tx="cyprus1", rx="YOS", method="contour",
           limited=0, loflim=0):
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0))
    conn = client.app.state.db
    cur = conn.execute("INSERT INTO sounding (file, path, datetime, tx, rx,"
                       " ingested_at) VALUES (?, ?, ?, ?, ?, ?)",
                       ("%s-%s-%d.h5" % (tx, rx, t0), "x.h5", when, tx, rx, when))
    conn.execute("INSERT INTO extraction (sounding_id, method, muf, lof, limited,"
                 " loflim) VALUES (?, ?, ?, ?, ?, ?)",
                 (cur.lastrowid, method, muf, lof, limited, loflim))
    conn.commit()


def muf(client, tx="cyprus1", **params):
    return client.get("/stations/YOS/live/muf", params=dict(params, tx=tx))


def test_the_live_muf_is_the_newest_soundings(client):
    t0, value = 1792161010 + 300 * random.randint(0, 99), random.uniform(5, 30)
    picked(client, t0 - 300, value + 1)
    picked(client, t0, value)
    assert muf(client).json()["muf"] == pytest.approx(value), (
        "the live MUF was not the newest sounding's")


def test_the_live_muf_says_which_sounding_it_is(client):
    t0 = 1792161010 + 300 * random.randint(0, 99)
    picked(client, t0, 20.0)
    assert muf(client).json()["t0"] == t0, "the live MUF did not say its sounding"


def test_the_live_muf_says_when_it_is_a_lower_bound(client):
    picked(client, 1792161010, 30.0, limited=1)
    assert muf(client).json()["limited"] is True, (
        "a MUF at the top of the sweep was not marked a lower bound")


def test_a_newest_sounding_without_a_pick_is_not_covered_by_an_older_one(client):
    t0 = 1792161010 + 300 * random.randint(1, 99)
    picked(client, t0 - 300, 20.0)
    picked(client, t0, None)
    assert muf(client).json()["muf"] is None, (
        "an older pick stood in for the newest sounding, which had none")


def test_the_live_muf_keeps_to_its_station(client):
    picked(client, 1792161010, 20.0, rx="DOB")
    assert muf(client).status_code == 404, "the live MUF came from another station"


def test_the_live_muf_reads_the_method_asked_for(client):
    picked(client, 1792161010, 20.0, method="algo")
    assert muf(client, method="algo").json()["muf"] == 20.0, (
        "the live MUF did not read the method asked for")


def test_the_live_muf_refuses_a_method_that_does_not_exist(client):
    assert muf(client, method="guess").status_code == 400, (
        "the live MUF accepted a method that does not exist")


def test_live_page_asks_for_each_transmitters_muf(client):
    assert "/live/muf?tx=" in page(client), (
        "the live page does not show the pipeline's MUF")


def test_live_page_shows_planned_against_recorded(client):
    assert 'id="tally"' in page(client), (
        "the live page does not show the planned and recorded captures")


def test_live_page_names_planned_against_recorded_in_russian(client):
    assert "Запланировано и записано" in page(client, "?station=YOS&lang=ru"), (
        "the planned and recorded table was not titled in Russian")
