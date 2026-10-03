"""Hand-built starting creatures."""
import math

from . import cppn as P
from .config import CORE, BONE, MUSCLE, CART, EYE, CPPN_COORD_SCALE, CPPN_EXISTS_SLOPE
from .genome import BodyGenome, BrainGenome, Genome, IN_OSC_S, IN_OSC_C, reserve_node_ids


def box_cells(boxes):
    """Cell dict for (x0, x1, a0, a1, type) boxes: x0 <= x <= x1, a0 <= |y| <= a1."""
    cells = {(x, s * a): (t, 0.0) for x0, x1, a0, a1, t in boxes
             for x in range(x0, x1 + 1) for a in range(a0, a1 + 1) for s in ((1, -1) if a else (1,))}
    cells[(0, 0)] = (CORE, 0.0)
    return dict(sorted(cells.items()))


def box_cppn(boxes, k=4.0, margin=2.0):
    """A CPPN painting non-overlapping `boxes`. Each box edge is a shared tanh step on x or
    |y| (position = bias / weight); a box node ANDs its steps, and outputs read 4·(box + 1),
    so the right type wins by 8 and `exists` clears its threshold by >= `margin`."""
    net = P.Cppn()
    steps = {}

    def step(src, edge, sign):
        key = (src, edge, sign)
        if key not in steps:
            h = steps[key] = net.add_node("tanh", bias=k * sign * edge)
            net.connect(src, h, -k * sign / CPPN_COORD_SCALE)
        return steps[key]

    logits = {BONE: P.OUT_BONE, MUSCLE: P.OUT_MUSCLE, CART: P.OUT_CART, EYE: P.OUT_EYE}
    # A long body can't clear the full slope at its tip and still miss it beside the core,
    # so a link from d cancels part of the slope.
    far = max(math.hypot(x, a) for x0, x1, a0, a1, _ in boxes for x in (x0, x1) for a in (a0, a1))
    slope = CPPN_EXISTS_SLOPE * CPPN_COORD_SCALE
    eff = min(slope, (8.0 - 2 * margin) / max(far - 1, 1))
    if eff < slope:
        net.connect(P.IN_D, P.OUT_EXISTS, (slope - eff) / CPPN_COORD_SCALE)
    net.nodes[P.OUT_EXISTS].bias = eff * far + margin
    for x0, x1, a0, a1, t in boxes:
        hs = [step(P.IN_X, x1 + 0.5, 1), step(P.IN_X, x0 - 0.5, -1), step(P.IN_AY, a1 + 0.5, 1)]
        if a0 > 0:
            hs.append(step(P.IN_AY, a0 - 0.5, -1))
        b = net.add_node("tanh", bias=-k * (len(hs) - 1))
        for h in hs:
            net.connect(h, b, k)
        net.connect(b, P.OUT_EXISTS, 4.0)
        net.nodes[P.OUT_EXISTS].bias += 4.0 - 8.0 / len(boxes)
        if t in logits:
            net.connect(b, logits[t], 4.0)
            net.nodes[logits[t]].bias += 4.0
    return net


def wave_brain(eyes, motors, amp=4.0, steer=4.0, freq=3.0, wave=0.5, diff_gain=8.0,
               diff_off=3.0):
    """Oscillator-driven travelling wave (`wave` wavelengths front to back, sides in
    antiphase) with one sensor per eye and one motor per (x, ±1).

    Steering is differential: per side, hidden tanh(diff_gain·(own − other) − diff_off)
    excites that side's motors, and the motor bias cancels its resting value. Wiring eyes
    straight to motors made a food patch co-contract every muscle (ARCHITECTURE §6a)."""
    br = BrainGenome()
    br.osc_freq = freq
    sens = {}
    for i, (x, y) in enumerate(eyes):
        sens[1 if y > 0 else -1] = br.add_node("sensor", id=10 + i, pos=(x, y))
    xs = sorted({x for x, _ in motors}, reverse=True)
    span = max(xs[0] - xs[-1], 1)
    diff = bool(steer) and len(sens) == 2
    rest = math.tanh(-diff_off)                                 # difference node at rest
    gain = steer / (math.tanh(diff_gain - diff_off) - rest) if diff else steer
    nid = 20
    ms = []
    for x, side in motors:
        phi = -2 * math.pi * wave * (xs[0] - x) / span * (len(xs) - 1) / len(xs)
        m = br.add_node("motor", id=nid, pos=(x, side), tau=0.05,
                        bias=-gain * rest if diff and side in sens else 0.0)
        nid += 1
        ms.append((m, side))
        ph = phi + (0 if side > 0 else math.pi)
        br.connect(IN_OSC_S, m, amp * math.cos(ph))
        br.connect(IN_OSC_C, m, amp * math.sin(ph))
    if steer:
        src = sens
        if diff:
            src = {}
            for side in (1, -1):
                src[side] = br.add_node("hidden", id=nid, bias=-diff_off, tau=0.05)
                nid += 1
                br.connect(sens[side], src[side], diff_gain)
                br.connect(sens[-side], src[side], -diff_gain)
        for m, side in ms:
            if side in src:
                br.connect(src[side], m, gain)
    reserve_node_ids(nid)
    return br


def eel_boxes(segments=3):
    """Core nose, eyes flanking a bone, then a cartilage spine flanked by muscle."""
    tail = 2 * segments + 1
    return [(0, 0, 0, 0, CORE), (-1, -1, 0, 0, BONE), (-1, -1, 1, 1, EYE),
            (-tail, -2, 0, 0, CART), (-tail, -2, 1, 1, MUSCLE)]


def swimmer(segments=3, amp=4.0, steer=4.0, freq=3.0, wave=0.5):
    """The seed eel, 22 cells, one motor per side every two cells (x = -3, -5, -7).

        y=+1   M  M  M  M  M  M  E           (+y = left when facing +x)
        y= 0   C  C  C  C  C  C  B  core
        y=-1   M  M  M  M  M  M  E
              x=-7 ...           -1  0
    """
    motors = [(-(3 + 2 * s), side) for s in range(segments) for side in (1, -1)]
    return _boxed(eel_boxes(segments), [(-1, 1), (-1, -1)], lineage="eel",
                  motors=motors, amp=amp, steer=steer, freq=freq, wave=wave)


def _boxed(boxes, eyes, lineage, sym_prob=0.8, motors=None, **brain):
    """box_cppn body plus wave_brain; motors default to every (x, ±1) muscle."""
    body = BodyGenome(box_cppn(boxes), sym_prob=sym_prob)
    assert body.cells == box_cells(boxes), "box CPPN doesn't paint its layout"
    if motors is None:
        motors = sorted({(x, 1 if y > 0 else -1) for (x, y), (t, _) in body.cells.items()
                         if t == MUSCLE and abs(y) == 1})
    return Genome(body, wave_brain(eyes, motors, **brain), mut_rate=1.0, lineage=lineage)


TADPOLE = [(0, 0, 0, 0, CORE), (-1, -1, 0, 0, BONE), (-1, -1, 1, 1, EYE),
           (-3, -2, 0, 0, CART), (-3, -2, 1, 1, MUSCLE), (-4, -4, 0, 0, CART)]


def tadpole():
    """The small seed, 11 cells: just small enough for the eel to eat. Muscles beat in
    unison at 2 Hz; the cartilage tail tip doubles its speed. Keep steer below amp or it
    circles near food.

        y=+1         M  M  E
        y= 0   C  C  C  B  core
        y=-1         M  M  E
              x=-4       -1  0
    """
    return _boxed(TADPOLE, [(-1, 1), (-1, -1)], lineage="tadpole",
                  amp=5.0, steer=4.0, freq=2.0, wave=0.0)


SEEDS = {"eel": swimmer, "tadpole": tadpole}
