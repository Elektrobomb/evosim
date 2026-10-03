"""Run the sim in a background thread and serve the viewer at http://localhost:8000."""
import argparse
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from sim import config as C
from sim.genome import Genome
from sim.presets import SEEDS
from sim.world import World

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Runner:
    def __init__(self, seed, pop, genome_file=None, seeds=tuple(SEEDS)):
        self.genome_file = genome_file
        self.seeds = seeds
        self.lock = threading.Lock()
        self.speed = 1.0            # sim s per real s; 0 paused, -1 max
        self.rate = 0.0             # measured
        self.epoch = 0
        self.pop = pop
        self.reset(seed)
        threading.Thread(target=self._loop, daemon=True).start()

    def reset(self, seed=None):
        w = World(seed)
        if self.genome_file:
            with open(self.genome_file) as f:
                genomes = [Genome.from_dict(json.load(f))]
        else:
            genomes = [SEEDS[s]() for s in self.seeds]
        w.populate(genomes, self.pop)
        w.add_food(w.rng.uniform([0, 0], [w.W, w.H], (120, 2)), np.full(120, C.FOOD_ENERGY))
        with self.lock:
            self.world = w
            self.epoch += 1

    def _loop(self):
        last = time.perf_counter()
        debt = 0.0
        window_t, window_sim = last, 0.0
        while True:
            now = time.perf_counter()
            dt_real, last = now - last, now
            if self.speed == 0:
                debt = 0.0
                time.sleep(0.02)
            elif self.speed < 0:
                end = now + 0.03
                while time.perf_counter() < end:
                    with self.lock:
                        self.world.tick()
                    window_sim += C.BRAIN_DT
                time.sleep(0.002)          # let the HTTP thread grab the lock
            else:
                debt = min(debt + dt_real * self.speed, 0.5)
                while debt >= C.BRAIN_DT:
                    with self.lock:
                        self.world.tick()
                    debt -= C.BRAIN_DT
                    window_sim += C.BRAIN_DT
                    if time.perf_counter() - now > 0.1:   # can't keep up; drop the backlog
                        debt = 0.0
                        break
                time.sleep(0.004)
            if now - window_t > 0.5:
                self.rate = window_sim / (now - window_t)
                window_t, window_sim = now, 0.0


class Handler(BaseHTTPRequestHandler):
    runner = None
    # Keep-alive: a new connection per poll is slow on Windows. Needs Content-Length.
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        r = self.runner
        if u.path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            sel = int(q["sel"][0]) if q.get("sel") else None
            epoch = int(q.get("epoch", ["-1"])[0])
            since = int(q.get("since", ["0"])[0]) if epoch == r.epoch else 0
            with r.lock:
                snap = r.world.snapshot(selected=sel, since=since)
            snap.update(epoch=r.epoch, speed=r.speed, rate=round(r.rate, 2))
            return self._send(200, json.dumps(snap, separators=(",", ":")))
        if u.path == "/api/genome":
            cid = int(q["id"][0])
            with r.lock:
                c = next((c for c in r.world.creatures if c.id == cid), None)
                if c is None:
                    return self._send(404, '{"error":"no such creature"}')
                body = json.dumps(dict(id=c.id, gen=c.gen, genome=c.genome.to_dict()), indent=1)
            return self._send(200, body)
        self._send(404, '{"error":"not found"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(n) or b"{}")
        r = self.runner
        cmd = msg.get("cmd")
        if cmd == "speed":
            r.speed = float(msg["value"])
        elif cmd == "reset":
            r.reset(msg.get("seed"))
        elif cmd == "food_rate":
            with r.lock:
                r.world.food_rate = float(msg["value"])
        elif cmd == "pop_target":
            with r.lock:
                v = int(msg["value"])
                r.world.pop_target = v if v > 0 else None
        elif cmd == "feed":
            with r.lock:
                w = r.world
                pts = np.array([msg["x"], msg["y"]]) + w.rng.normal(0, 2.5, (12, 2))
                w.add_food(np.clip(pts, 1, [w.W - 1, w.H - 1]), np.full(12, C.FOOD_ENERGY))
        elif cmd == "clone":
            with r.lock:
                w = r.world
                c = next((c for c in w.creatures if c.id == int(msg["id"])), None)
                if c is not None:
                    w.spawn(c.genome.mutate(w.rng), energy=20.0, gen=c.gen + 1, parent=c.id)
        elif cmd == "kill":
            with r.lock:
                for c in r.world.creatures:
                    if c.id == int(msg["id"]):
                        c.energy = -1.0
        else:
            return self._send(400, '{"error":"unknown cmd"}')
        self._send(200, '{"ok":true}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pop", type=int, default=24, help="starting population")
    ap.add_argument("--genome", help="seed the world from a saved genome JSON instead of the presets")
    ap.add_argument("--seeds", default=",".join(SEEDS),
                    help=f"comma-separated preset seeds, used in turn (from: {', '.join(SEEDS)})")
    a = ap.parse_args()
    seeds = tuple(s.strip() for s in a.seeds.split(",") if s.strip())
    if not seeds or any(s not in SEEDS for s in seeds):
        ap.error(f"--seeds must name some of: {', '.join(SEEDS)}")
    Handler.runner = Runner(a.seed, a.pop, a.genome, seeds)
    # Also listen on ::1: browsers try it first for localhost, and on Windows a refused
    # ::1 stalls 0.3-2 s per request.
    servers = [ThreadingHTTPServer(("127.0.0.1", a.port), Handler)]
    if socket.has_ipv6:
        try:
            servers.append(IPv6Server(("::1", a.port), Handler))
        except OSError:
            pass
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Evolving creatures running at http://localhost:{a.port}  (Ctrl+C to stop)")
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass


class IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


if __name__ == "__main__":
    main()
