#!/usr/bin/env python3
"""Behavior-snapshot harness for revv's network-facing commands.

    python3 tests/snapshot_cli.py > snapshot.txt

Captures the exact stdout (and, separately, stderr) of `revv bench`,
`revv compare`, `revv status` and `revv toggle` against a FAKE in-process
HTTP server, with revv's own `time` binding replaced by a deterministic
fake clock. The result is byte-identical across runs on the same machine,
which is the whole point: a refactor of revv.py can be proven behavior-
preserving by diffing this file's output before and after.

Stdlib only, Python 3.9+. Does NOT modify revv.py or any other file, and
does NOT need a GPU or a real llama-server -- everything the client talks
to is the fake server defined below.
"""

import os
import shutil
import sys

# --- REVV_HOME and terminal-detection env vars must be set BEFORE revv is
# imported: REVV_HOME because module-level path constants (RUN_FILE, etc.)
# are computed at import time, and NO_COLOR/TERM/COLUMNS because _COLOR is
# also a module-level constant computed at import time (we override it again
# below, belt-and-braces, but the env vars are what a real invocation would
# set anyway).
HOME = "/tmp/revv-clisnap-home"
shutil.rmtree(HOME, ignore_errors=True)
os.makedirs(HOME, exist_ok=True)
os.environ["REVV_HOME"] = HOME
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["COLUMNS"] = "80"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
import revv  # noqa: E402

revv._COLOR = False
revv._git_sha = lambda: None

import argparse            # noqa: E402
import contextlib          # noqa: E402
import io                  # noqa: E402
import json                # noqa: E402
import re                  # noqa: E402
import socket               # noqa: E402
import threading            # noqa: E402
import time as real_time    # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

# ---------------------------------------------------------------------------
# The deterministic clock. revv reaches time through its own module-global
# `time`, so replacing revv.time (not the `time` module itself) is enough:
# every wall-clock/ttft/uptime figure revv prints becomes a fixed function of
# how many times revv's code called time.time(), which is exactly the
# property a refactor must preserve.
# ---------------------------------------------------------------------------


class _FakeTime:
    def __init__(self):
        self._t = 1_700_000_000.0

    def time(self):
        self._t += 0.25
        return self._t

    def sleep(self, s):
        self._t += float(s)

    def monotonic(self):
        return self._t


# The exact value _FakeTime().time() returns on its FIRST call, from a fresh
# instance. cmd_status's uptime line is the only place in the scenarios below
# that calls time.time() more than zero times before printing, and it always
# calls it exactly once, so this constant lets us pin the uptime line to a
# fixed, human-legible value (see status_normal).
FIRST_CALL_VALUE = 1_700_000_000.0 + 0.25

HAS_NVIDIA_SMI = shutil.which("nvidia-smi") is not None

# ---------------------------------------------------------------------------
# Fake server: /health, /_revv/status, /_revv/mode, /v1/chat/completions.
# Lives entirely in this file and uses the REAL time module -- only revv's
# own binding is swapped, never this file's.
# ---------------------------------------------------------------------------

REVV_MODE_DESC = "tuned: MTP n=2 speculation, q8_0 KV, thinking off"
STOCK_MODE_DESC = ("llama-server defaults for this model: thinking on, "
                    "f16 KV, no speculation")
MODE_DESC = {"revv": REVV_MODE_DESC, "stock": STOCK_MODE_DESC}

# Mirrors _status_obj() in revv.py: exactly the keys the client code reads.
DEFAULT_STATUS = {
    "mode": "revv",
    "mode_description": REVV_MODE_DESC,
    "model": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
    "tier": "12gb",
    "context": 16384,
    "kv": "q8_0",
    "line": "dense",
    "levers": ["speculation", "quantized kv", "thinking off"],
    "tuning_is_noop": False,
    "backend_alive": True,
    "modes_identical": False,
    "stock_description": "llama-server defaults, all layers on the GPU",
    "build": "IQ3_XXS",
    "speculation": True,
    "backend_port": 41414,
    "backend_pid": 4242,
    "supervisor_pid": 4241,
    "requests": 7,
    "last_decode_tps": 37.42,
}


def regime(rate=37.86, n_tok=400, reasoning_chars=0, draft_n=0, draft_acc=0,
           content_chunks=5):
    """One canned chat-completion response.

    `rate` is predicted_per_second; None means "the server did not report
    decode timings at all" (used for the compare_no_server_timings
    scenario). `content_chunks` only matters for streaming (SSE) responses:
    it is the number of delta.content chunks emitted before the final
    timings chunk, and it is what controls the CLIENT-observed decode rate
    in _timed_generation (chunks - 1, divided by a fixed 0.25s under the
    fake clock -- see the module docstring in cmd_compare's scenarios
    below for why that denominator is always 0.25).
    """
    return {"rate": rate, "n_tok": n_tok, "reasoning_chars": reasoning_chars,
            "draft_n": draft_n, "draft_acc": draft_acc,
            "content_chunks": content_chunks}


DEFAULT_REGIME = regime()
# Arm A=0, arm B=0, arm C>0: thinking_check's PASS branch.
PASS_PROBES = [regime(reasoning_chars=0), regime(reasoning_chars=0),
               regime(reasoning_chars=60)]


def _nonstream_body(rgm):
    reasoning = "R" * rgm.get("reasoning_chars", 0)
    body = {
        "choices": [{
            "index": 0,
            "message": {"content": "def foo():\n    pass\n",
                        "reasoning_content": reasoning},
            "finish_reason": "stop",
        }],
        "usage": {"completion_tokens": rgm.get("n_tok") or 0},
    }
    if rgm.get("rate") is not None:
        body["timings"] = {
            "predicted_per_second": rgm["rate"],
            "predicted_n": rgm.get("n_tok") or 0,
            "draft_n": rgm.get("draft_n", 0),
            "draft_n_accepted": rgm.get("draft_acc", 0),
        }
    return body


def _sse_body(rgm):
    n_chunks = rgm.get("content_chunks", 3)
    lines = []
    for i in range(n_chunks):
        obj = {"choices": [{"index": 0, "delta": {"content": "tok%d " % i},
                            "finish_reason": None}]}
        # Compact separators so the raw bytes contain the literal substring
        # '"content":"', which is exactly what revv's _timed_generation
        # scans for (no space after the colon -- see revv.py's time-to-
        # first-token detection).
        lines.append(b"data: " + json.dumps(obj, separators=(",", ":"))
                     .encode("utf-8"))
    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    if rgm.get("rate") is not None:
        final["timings"] = {
            "predicted_per_second": rgm["rate"],
            "predicted_n": rgm.get("n_tok") or 0,
            "draft_n": rgm.get("draft_n", 0),
            "draft_n_accepted": rgm.get("draft_acc", 0),
        }
    lines.append(b"data: " + json.dumps(final, separators=(",", ":"))
                 .encode("utf-8"))
    lines.append(b"data: [DONE]")
    return b"\n".join(lines) + b"\n"


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 + explicit Content-Length: one request per connection,
        # no chunked-transfer edge cases -- same choice revv.py's own proxy
        # handler makes, for the same reason.
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *a):
            pass  # never print: stdout/stderr are globally redirected while
                  # this thread may be mid-request, and a stray print here
                  # would land in the captured snapshot nondeterministically.

        def _send_json(self, code, obj):
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                self._send_json(200, {})
            elif path == "/_revv/status":
                self._send_json(200, state["status"])
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            if path == "/_revv/mode":
                want = None
                try:
                    want = json.loads(body.decode("utf-8")).get("mode")
                except ValueError:
                    pass
                obj = dict(state["status"])
                if want:
                    obj["mode"] = want
                    obj["mode_description"] = MODE_DESC.get(
                        want, obj.get("mode_description"))
                obj["switch_seconds"] = 12.5
                self._send_json(200, obj)
            elif path == "/v1/chat/completions":
                try:
                    payload = json.loads(body.decode("utf-8")) if body else {}
                except ValueError:
                    payload = {}
                regimes = state["chat_regimes"]
                rgm = regimes[state["call_count"] % len(regimes)]
                state["call_count"] += 1
                if payload.get("stream"):
                    data = _sse_body(rgm)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._send_json(200, _nonstream_body(rgm))
            else:
                self._send_json(404, {"error": "not found"})

    return Handler


def start_server(status_obj, chat_regimes):
    state = {"status": status_obj, "chat_regimes": chat_regimes,
             "call_count": 0}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


def stop_server(httpd):
    httpd.shutdown()
    httpd.server_close()


def make_dead_port():
    """A port nothing is listening on: bind, then immediately close."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Capture + scrub + report
# ---------------------------------------------------------------------------

def scrub(text, port):
    if port is not None:
        text = re.sub(r'\b%d\b' % port, "<port>", text)
    text = text.replace(HOME, "<home>")
    text = text.replace(REPO_ROOT, "<repo>")
    return text


def print_report(name, stdout_text, stderr_text, rc):
    print("===== SCENARIO: %s =====" % name)
    sys.stdout.write(stdout_text)
    if stdout_text and not stdout_text.endswith("\n"):
        sys.stdout.write("\n")
    print("--- stderr ---")
    if stderr_text:
        sys.stdout.write(stderr_text)
        if not stderr_text.endswith("\n"):
            sys.stdout.write("\n")
    print("exit=%d" % rc)
    print()


def capture(name, port, fn):
    """Run fn() with revv.time faked and stdout/stderr captured, then print."""
    out = io.StringIO()
    err = io.StringIO()
    rc = 0
    revv.time = _FakeTime()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                result = fn()
                rc = result if isinstance(result, int) else 0
            except SystemExit as e:
                code = e.code
                if code is None:
                    rc = 0
                elif isinstance(code, int):
                    rc = code
                else:
                    rc = 1
    finally:
        revv.time = real_time
    print_report(name, scrub(out.getvalue(), port),
                 scrub(err.getvalue(), port), rc)


# ---------------------------------------------------------------------------
# Per-command scenario runners
# ---------------------------------------------------------------------------

def scenario_bench(name, status_overrides, measured, probes):
    status_obj = dict(DEFAULT_STATUS)
    status_obj.update(status_overrides)
    # 1 warmup + BENCH_REQUESTS(4) measured calls, then the 3 thinking_check
    # probe arms -- 8 chat completions total, in exactly that order.
    chat_regimes = [measured] * (1 + revv.BENCH_REQUESTS) + list(probes)
    httpd, port = start_server(status_obj, chat_regimes)
    base = "http://127.0.0.1:%d" % port
    try:
        capture(name, port, lambda: revv.cmd_bench(
            argparse.Namespace(url=base, timeout=30.0)))
    finally:
        stop_server(httpd)


def scenario_status(name, status_overrides, dead=False, with_run_file=False,
                    run_started_at=None):
    if not dead and HAS_NVIDIA_SMI:
        # cmd_status's detect_gpus() block is unconditional once it is
        # reached, and its numbers come from the real driver -- not
        # reproducible across machines. Skip rather than snapshot noise.
        print("===== SCENARIO: %s =====" % name)
        print("SKIPPED (nvidia-smi present)")
        print("--- stderr ---")
        print("exit=n/a")
        print()
        return
    if dead:
        port = make_dead_port()
        base = "http://127.0.0.1:%d" % port
        capture(name, port, lambda: revv.cmd_status(
            argparse.Namespace(url=base)))
        return
    status_obj = dict(DEFAULT_STATUS)
    status_obj.update(status_overrides)
    httpd, port = start_server(status_obj, [DEFAULT_REGIME])
    base = "http://127.0.0.1:%d" % port
    try:
        if with_run_file:
            revv.write_run_file(
                pid=1234, port=port, host="127.0.0.1", model="x.gguf",
                tier="12gb", backend_pid=4242, started_at=run_started_at,
                log=os.path.join(revv.REVV_HOME, "logs", "revv.log"))
        capture(name, port, lambda: revv.cmd_status(
            argparse.Namespace(url=base)))
    finally:
        if with_run_file:
            revv.clear_run_file()
        stop_server(httpd)


def scenario_toggle(name, status_overrides, mode_arg=None, dead=False):
    if dead:
        port = make_dead_port()
        base = "http://127.0.0.1:%d" % port
        capture(name, port, lambda: revv.cmd_toggle(
            argparse.Namespace(url=base, mode=mode_arg)))
        return
    status_obj = dict(DEFAULT_STATUS)
    status_obj.update(status_overrides)
    httpd, port = start_server(status_obj, [DEFAULT_REGIME])
    base = "http://127.0.0.1:%d" % port
    try:
        capture(name, port, lambda: revv.cmd_toggle(
            argparse.Namespace(url=base, mode=mode_arg)))
    finally:
        stop_server(httpd)


def scenario_compare(name, status_overrides, regimes):
    status_obj = dict(DEFAULT_STATUS)
    status_obj.update(status_overrides)
    httpd, port = start_server(status_obj, regimes)
    base = "http://127.0.0.1:%d" % port
    try:
        capture(name, port, lambda: revv.cmd_compare(
            argparse.Namespace(url=base, max_tokens=2048, timeout=60.0)))
    finally:
        stop_server(httpd)


# ---------------------------------------------------------------------------
# The 21 scenarios
# ---------------------------------------------------------------------------

def main():
    # bench: target/reference = BENCH_REF_PATCHED / 1.025 ~= 36.94 t/s for
    # the registered IQ3_XXS build (manifest is always absent here, so
    # patched is always False). BENCH_REF_NOSPEC * 1.10 = 24.75 t/s is the
    # no-speculation-regime cutoff. See cmd_bench's "reading" section.

    # 1. on target: ratio ~1.015, in [0.95, 1.05].
    scenario_bench("bench_on_target", {},
                   regime(rate=37.5, n_tok=400, draft_n=120, draft_acc=95),
                   PASS_PROBES)

    # 2. above reference: ratio ~1.62 > 1.05.
    scenario_bench("bench_above_reference", {},
                   regime(rate=60.0, n_tok=400, draft_n=150, draft_acc=140),
                   PASS_PROBES)

    # 3. no-speculation regime: mean 21.0 < 24.75, draft_n=0.
    scenario_bench("bench_nospec_regime", {},
                   regime(rate=21.0, n_tok=400, draft_n=0, draft_acc=0),
                   PASS_PROBES)

    # 4. slightly low: ratio ~0.893, in [0.85, 0.95).
    scenario_bench("bench_slightly_low", {},
                   regime(rate=33.0, n_tok=400, draft_n=100, draft_acc=70),
                   PASS_PROBES)

    # 5. well below: ratio ~0.812 < 0.85, AND mean (30.0) >= 24.75 so the
    #    no-speculation-regime branch (checked earlier in cmd_bench) does
    #    NOT fire first. NOTE: the spec's illustrative "~15.0" would in
    #    fact land in the nospec branch (mean < 24.75 fires unconditionally
    #    on mean alone, regardless of draft_n) -- 30.0 is the value that
    #    actually reaches "well below". See the report for detail.
    scenario_bench("bench_well_below", {},
                   regime(rate=30.0, n_tok=400, draft_n=80, draft_acc=50),
                   PASS_PROBES)

    # 6. thinking leak: non-stream responses carry reasoning_content, and
    #    probe arm A > 0 -> thinking_check's FAIL branch.
    leak_measured = regime(rate=37.5, n_tok=400, reasoning_chars=40,
                           draft_n=120, draft_acc=95)
    leak_probes = [regime(reasoning_chars=30), regime(reasoning_chars=25),
                  regime(reasoning_chars=5)]
    scenario_bench("bench_thinking_leak", {}, leak_measured, leak_probes)

    # 7. thinking inconclusive: all three probe arms return 0.
    inconclusive_probes = [regime(reasoning_chars=0), regime(reasoning_chars=0),
                           regime(reasoning_chars=0)]
    scenario_bench("bench_thinking_inconclusive", {},
                   regime(rate=37.5, n_tok=400, draft_n=120, draft_acc=95),
                   inconclusive_probes)

    # 8. unregistered build: status build=None -> dense stand-in reference.
    scenario_bench("bench_unregistered_build", {"build": None},
                   regime(rate=37.0, n_tok=400, draft_n=120, draft_acc=95),
                   PASS_PROBES)

    # 9. status, normal: run file present -> deterministic uptime line.
    #    cmd_status calls time.time() exactly once (for the uptime line)
    #    before ever reaching detect_gpus(), so the first fake-clock value
    #    is always FIRST_CALL_VALUE; started_at is back-solved from it so
    #    the printed uptime is a fixed "1h 2m".
    scenario_status("status_normal", {}, with_run_file=True,
                    run_started_at=FIRST_CALL_VALUE - 3725.0)

    # 10. status, backend down.
    scenario_status("status_backend_down", {"backend_alive": False})

    # 11. status, tuning is a no-op for this model.
    scenario_status("status_noop", {"tuning_is_noop": True})

    # 12. status, no server (closed port) -> "down:" text, exit 1.
    scenario_status("status_no_server", {}, dead=True)

    # 13. toggle, default direction (mode=None): revv -> stock.
    scenario_toggle("toggle_default", {"mode": "revv"}, mode_arg=None)

    # 14. toggle, explicit mode equal to current -> "Already in ... mode."
    scenario_toggle("toggle_explicit_same", {"mode": "revv"}, mode_arg="revv")

    # 15. toggle, tuning is a no-op, mode=None -> the "nothing to switch" note.
    scenario_toggle("toggle_noop", {"tuning_is_noop": True}, mode_arg=None)

    # 16. toggle, no server (closed port) -> die(), exit 1.
    scenario_toggle("toggle_no_server", {}, dead=True)

    # 17. compare, normal: distinct token counts/rates so the ratio, tokens-
    #     spent and acceptance lines all print, with no transport note (the
    #     client-observed rate is tuned to track the server-reported rate).
    stock_warmup = regime(rate=20.0, n_tok=150, content_chunks=6)
    stock_measured = regime(rate=20.0, n_tok=800, content_chunks=6)
    revv_warmup = regime(rate=45.0, n_tok=150, content_chunks=12,
                         draft_n=100, draft_acc=90)
    revv_measured = regime(rate=45.0, n_tok=300, content_chunks=12,
                           draft_n=100, draft_acc=90)
    scenario_compare("compare_normal", {},
                     [stock_warmup, stock_measured, revv_warmup, revv_measured])

    # 18. compare, tuning is a no-op -> "nothing to compare".
    scenario_compare("compare_noop", {"tuning_is_noop": True}, [DEFAULT_REGIME])

    # 19. compare, modes identical -> "refusing to run this comparison".
    scenario_compare("compare_modes_identical", {"modes_identical": True},
                     [DEFAULT_REGIME])

    # 20. compare, transport gap: very few SSE content chunks relative to
    #     the reported decode rate, so the client-observed rate is far below
    #     the server-reported one and the transport-diagnostic note fires.
    gap_stock_warmup = regime(rate=50.0, n_tok=150, content_chunks=2)
    gap_stock_measured = regime(rate=50.0, n_tok=800, content_chunks=2)
    gap_revv_warmup = regime(rate=55.0, n_tok=150, content_chunks=12)
    gap_revv_measured = regime(rate=55.0, n_tok=300, content_chunks=3)
    scenario_compare("compare_transport_gap", {},
                     [gap_stock_warmup, gap_stock_measured,
                      gap_revv_warmup, gap_revv_measured])

    # 21. compare, no server timings at all -> the "did not report decode
    #     timings" note (and no transport note, since overhead_pct is None
    #     whenever server_tps is None).
    no_timings = regime(rate=None, n_tok=None, content_chunks=5)
    scenario_compare("compare_no_server_timings", {}, [no_timings])


if __name__ == "__main__":
    main()
