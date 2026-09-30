# Cell 40 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v18.0 M90 (Wang/Ludvig harness ported onto ours; serving unchanged))
# ===== v18.0 = M90: WANG/LUDVIG HARNESS (board 11.49/11.64, commit 6420079) PORTED ONTO OURS; SERVING UNCHANGED =====
"""Their per-turn message (5 lines), their system prompt texts and order, their python sandbox (host + sandbox +
segmentation: id/area/bbox objects, frame_diff, last_transition.diff), and no re-fed conclusions (M24 ledger off).
Sources are their files at commit 6420079, embedded verbatim except: the sandbox imports its segmentation from a
private module name, and the intermediate-frames text states M76's actual layout (frames after the current image)."""
import importlib.util as _il90
import os as _os90
import sys as _sys90
from typing import Any

_W90_SEG_SRC = '"""Connected-component segmentation of a single frame layer.\n\nThis module is intentionally self-contained -- standard library only, no project\nimports, no ``from __future__`` import -- so its source can be spliced verbatim into\nthe Python-tool sandbox bootstrap, where project packages are not importable.\n"""\n\nimport hashlib\n\n_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))\n# clockwise Moore-neighbour offsets, starting at NW\n_CW = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))\n_CW_INDEX = {off: i for i, off in enumerate(_CW)}\n\n\ndef _trace_outer_contour(cells, start):\n    """Moore-neighbour trace of a 4-connected component\'s outer perimeter, clockwise."""\n    if len(cells) == 1:\n        return [start]\n\n    contour = [start]\n    b = start\n    prev = (start[0], start[1] - 1)  # W neighbour: outside the component since start is reading-order-min\n    second = None\n    for _ in range(8 * len(cells) + 16):\n        idx = _CW_INDEX[(prev[0] - b[0], prev[1] - b[1])]\n        nxt = None\n        for k in range(1, 9):\n            off = _CW[(idx + k) % 8]\n            cand = (b[0] + off[0], b[1] + off[1])\n            if cand in cells:\n                nxt = cand\n                back = _CW[(idx + k - 1) % 8]\n                new_prev = (b[0] + back[0], b[1] + back[1])\n                break\n        if nxt is None:\n            break\n        if second is None:\n            second = nxt\n        elif b == start and nxt == second:  # Jacob\'s stopping criterion\n            break\n        contour.append(nxt)\n        prev, b = new_prev, nxt\n\n    if len(contour) > 1 and contour[-1] == contour[0]:\n        contour.pop()\n    return contour\n\n\ndef _corner_points(contour):\n    """Reduce a traced contour loop to only the points where its direction changes."""\n    if len(contour) <= 2:\n        return list(contour)\n    m = len(contour)\n    corners = []\n    for i in range(m):\n        prev, cur, nxt = contour[i - 1], contour[i], contour[(i + 1) % m]\n        d_in = (cur[0] - prev[0], cur[1] - prev[1])\n        d_out = (nxt[0] - cur[0], nxt[1] - cur[1])\n        if d_in != d_out:\n            corners.append(cur)\n    return corners\n\n\ndef _object_id(cells, color):\n    """Translation-invariant identity of an object: its color plus its cell shape,\n    normalized so the top-left of its bounding box is the origin. Same shape + color\n    => same id regardless of position, so objects can be matched across frames.\n    Identical-looking objects share an id."""\n    min_r = min(r for r, _ in cells)\n    min_c = min(c for _, c in cells)\n    norm = sorted((r - min_r, c - min_c) for r, c in cells)\n    payload = repr((color, norm)).encode()\n    return hashlib.sha1(payload).hexdigest()[:8]\n\n\nNODE_FIELDS = (\n    "id", "color", "area", "bbox", "boundary", "children",\n)\n\n# Names that read like node fields but are not, mapped to what to use instead. Keeps a\n# wrong guess from failing as a confusing TypeError several lines later.\n_NODE_FIELD_HINTS = {\n    "hash": "id",\n    "pixels": "area (an int cell count, not a list of coordinates)",\n    "px": "area (an int cell count)",\n    "n_pixels": "area",\n    "size": "area",\n    "cells": "area for the count, or boundary/bbox for the shape",\n    "coords": "boundary for the outline, or bbox for position",\n    "shape": "bbox, as [min_row, min_col, max_row, max_col]",\n    "centroid": "bbox -- midpoint is ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)",\n    "h": "bbox -- height is bbox[2] - bbox[0] + 1",\n    "w": "bbox -- width is bbox[3] - bbox[1] + 1",\n    "x": "bbox[1] / bbox[3] (columns)",\n    "y": "bbox[0] / bbox[2] (rows)",\n}\n\n\nclass Node(dict):\n    """One segmentation node. Behaves as a plain dict; a missing key raises with a\n    pointer to the right field name rather than a bare ``KeyError``."""\n\n    def __missing__(self, key):\n        hint = _NODE_FIELD_HINTS.get(key)\n        if hint is not None:\n            raise KeyError(\n                f"node has no field {key!r}; use {hint}. "\n                f"Node fields: {\', \'.join(NODE_FIELDS)}."\n            )\n        raise KeyError(\n            f"node has no field {key!r}. Node fields: {\', \'.join(NODE_FIELDS)}."\n        )\n\n\ndef segment_layer(layer, color_chars):\n    """Segment one frame layer into connected-component nodes.\n\n    Pass a single layer (if the frame has multiple) and ``color_chars``, the ARC\n    color-symbol mapping (indexed by integer color value -> single-char label). The\n    layer is partitioned into 4-connected components of equal integer value via flood\n    fill, and each component becomes a node. Nodes are listed in reading order of\n    their top-most-left-most cell.\n\n    Each node is a dict with:\n      - ``id``: the object\'s identity -- a short string derived from its color plus its\n        cell shape normalized to a top-left origin, so the same-looking object gets the\n        same id regardless of position or frame (lets objects be matched across frames;\n        identical-looking objects share an id).\n      - ``color``: the component\'s ARC color character (looked up in ``color_chars``).\n      - ``area``: number of cells in the component (an int, not a coordinate list).\n        Counts only the component\'s own cells -- enclosed children are separate\n        components and are not included, though ``bbox`` does span them.\n      - ``bbox``: ``[r0, c0, r1, c1]`` -- the component\'s inclusive bounding box.\n      - ``boundary``: the component\'s outer perimeter as an ordered, clockwise list of\n        ``[row, col]`` corner points -- a Moore-neighbour trace reduced to only the\n        vertices where the contour changes direction (enclosed holes are not traced).\n      - ``children``: ids of components directly enclosed by this node. A is a child of\n        B only if B is the innermost component that fully surrounds A (every path from A\n        to the grid edge crosses B), which yields a clean nesting tree. When several\n        enclosed objects look identical their ids coincide; disambiguate spatially by\n        filtering on ``bbox``.\n\n    Returns a dict with:\n      - ``nodes``: list of the node dicts above, in reading order.\n      - ``adjacency_list``: sorted, de-duplicated list of ``[id_a, id_b]`` pairs for\n        components that share a 4-connected edge (includes parent/child pairs, since\n        they physically touch).\n    """\n    height = len(layer)\n    width = len(layer[0]) if height else 0\n\n    # connected components, 4-connectivity. Reading-order scan => component ids are\n    # already ordered by top-most-left-most cell (each layer is 64x64, so it is unique).\n    comp_id = [[-1] * width for _ in range(height)]\n    components = []  # each: {"value": int, "cells": set[(r, c)], "start": (r, c)}\n    for sr in range(height):\n        for sc in range(width):\n            if comp_id[sr][sc] != -1:\n                continue\n            value = layer[sr][sc]\n            cid = len(components)\n            cells = set()\n            stack = [(sr, sc)]\n            comp_id[sr][sc] = cid\n            while stack:\n                r, c = stack.pop()\n                cells.add((r, c))\n                for dr, dc in _ORTH:\n                    nr, nc = r + dr, c + dc\n                    if 0 <= nr < height and 0 <= nc < width and comp_id[nr][nc] == -1 and layer[nr][nc] == value:\n                        comp_id[nr][nc] = cid\n                        stack.append((nr, nc))\n            components.append({"value": int(value), "cells": cells, "start": (sr, sc)})\n\n    n = len(components)\n\n    # adjacency between components: any two components with 4-adjacent cells\n    adj_pairs = set()\n    for r in range(height):\n        for c in range(width):\n            cid = comp_id[r][c]\n            if r + 1 < height and comp_id[r + 1][c] != cid:\n                other = comp_id[r + 1][c]\n                adj_pairs.add((min(cid, other), max(cid, other)))\n            if c + 1 < width and comp_id[r][c + 1] != cid:\n                other = comp_id[r][c + 1]\n                adj_pairs.add((min(cid, other), max(cid, other)))\n\n    # containment: for each component b, flood-fill its complement inward from the grid\n    # border; any component whose cells are never reached is enclosed by b.\n    enclosers = [set() for _ in range(n)]\n    for b in range(n):\n        reached = [[False] * width for _ in range(height)]\n        stack = []\n        for r in range(height):\n            for c in (0, width - 1):\n                if comp_id[r][c] != b and not reached[r][c]:\n                    reached[r][c] = True\n                    stack.append((r, c))\n        for c in range(width):\n            for r in (0, height - 1):\n                if comp_id[r][c] != b and not reached[r][c]:\n                    reached[r][c] = True\n                    stack.append((r, c))\n        while stack:\n            r, c = stack.pop()\n            for dr, dc in _ORTH:\n                nr, nc = r + dr, c + dc\n                if 0 <= nr < height and 0 <= nc < width and not reached[nr][nc] and comp_id[nr][nc] != b:\n                    reached[nr][nc] = True\n                    stack.append((nr, nc))\n        for a in range(n):\n            if a == b:\n                continue\n            ar, ac = components[a]["start"]\n            if not reached[ar][ac]:\n                enclosers[a].add(b)\n\n    # parent = innermost encloser. enclosers are transitive, so along a nesting chain the\n    # innermost component is the one that is itself most deeply enclosed.\n    children = [[] for _ in range(n)]\n    for a in range(n):\n        if enclosers[a]:\n            parent = max(enclosers[a], key=lambda e: (len(enclosers[e]), -e))\n            children[parent].append(a)\n    for child_list in children:\n        child_list.sort()\n\n    object_ids = [\n        _object_id(components[cid]["cells"], color_chars[max(0, min(15, components[cid]["value"]))])\n        for cid in range(n)\n    ]\n\n    nodes = []\n    for cid in range(n):\n        comp = components[cid]\n        color = color_chars[max(0, min(15, comp["value"]))]\n        boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))\n        cells = comp["cells"]\n        rows_ = [r for r, _ in cells]\n        cols_ = [c for _, c in cells]\n        r0, c0, r1, c1 = min(rows_), min(cols_), max(rows_), max(cols_)\n        nodes.append(\n            Node(\n                {\n                    "id": object_ids[cid],\n                    "color": color,\n                    "area": len(cells),\n                    "bbox": [r0, c0, r1, c1],\n                    "boundary": [[r, c] for r, c in boundary],\n                    "children": [object_ids[child] for child in children[cid]],\n                }\n            )\n        )\n\n    adjacency_list = sorted(\n        {tuple(sorted((object_ids[a], object_ids[b]))) for a, b in adj_pairs}\n    )\n    adjacency_list = [list(pair) for pair in adjacency_list]\n\n    return {"nodes": nodes, "adjacency_list": adjacency_list}\n'
_W90_SANDBOX_SRC = '"""Lightweight isolated runner for analyzer Python tool calls."""\nfrom __future__ import annotations\n\nimport inspect\nimport json\nimport os\nimport queue\nimport signal\nimport subprocess\nimport sys\nimport tempfile\nimport threading\nimport textwrap\nimport time\nfrom typing import Any, Callable\n\nimport m90_wang_segmentation as _segmentation\nfrom inference.utils.grid_utils import ARC_COLOR_CHARS\n\n\n_SANDBOX_BOOTSTRAP = textwrap.dedent(\n    r"""\n    import builtins\n    import contextlib\n    import io\n    import json\n    import os\n    import sys\n    import traceback\n\n    try:\n        import resource\n    except ImportError:  # pragma: no cover\n        resource = None\n\n    COLOR_CHARS = ""\n\n    __SEGMENTATION_SOURCE__\n\n    HOST_STDOUT = sys.stdout\n\n    SAFE_MODULES = {\n        "bisect",\n        "collections",\n        "copy",\n        "fractions",\n        "functools",\n        "heapq",\n        "itertools",\n        "json",\n        "math",\n        "operator",\n        "random",\n        "re",\n        "statistics",\n        "string",\n    }\n    SAFE_BUILTINS = {\n        "abs",\n        "all",\n        "any",\n        "ascii",\n        "bin",\n        "bool",\n        "bytearray",\n        "bytes",\n        "callable",\n        "chr",\n        "complex",\n        "dict",\n        "dir",\n        "divmod",\n        "enumerate",\n        "Exception",\n        "filter",\n        "float",\n        "format",\n        "frozenset",\n        "getattr",\n        "hasattr",\n        "hash",\n        "hex",\n        "int",\n        "isinstance",\n        "issubclass",\n        "iter",\n        "len",\n        "list",\n        "map",\n        "max",\n        "min",\n        "next",\n        "oct",\n        "ord",\n        "pow",\n        "print",\n        "range",\n        "repr",\n        "reversed",\n        "round",\n        "set",\n        "slice",\n        "sorted",\n        "str",\n        "sum",\n        "tuple",\n        "TypeError",\n        "type",\n        "ValueError",\n        "RuntimeError",\n        "zip",\n    }\n\n\n    def _send(payload):\n        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\\n")\n        HOST_STDOUT.flush()\n\n\n    def _recv():\n        line = sys.stdin.readline()\n        if not line:\n            raise EOFError("sandbox input closed")\n        return json.loads(line)\n\n\n    class FrameView:\n        def __init__(self, *, ascii, step, level, shape, grid):\n            self.ascii = ascii\n            self.step = step\n            self.level = level\n            self.shape = tuple(shape)\n            self._grid = grid\n            self._segmentation = None\n\n        @property\n        def segmentation(self):\n            if self._segmentation is None:\n                self._segmentation = segment_layer(self._grid, COLOR_CHARS)\n            return self._segmentation\n\n        def __str__(self):\n            rows, cols = self.shape\n            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"\n\n        __repr__ = __str__\n\n\n    DIFF_CELL_DETAIL_LIMIT = 12\n\n\n    DIFF_GROUP_FIELDS = ("from_color", "to_color", "count", "bbox", "cells")\n\n    # Names that read like diff-group fields but are not. ``from``/``to`` were renamed\n    # to ``from_color``/``to_color`` because they hold a single ARC color char, not a\n    # node -- the old names invited ``group[\'from\'][\'color\']``.\n    DIFF_GROUP_FIELD_HINTS = {\n        "from": "from_color (a single color char, e.g. \'B\')",\n        "to": "to_color (a single color char, e.g. \'B\')",\n        "color": "from_color / to_color",\n        "colors": "from_color / to_color",\n        "n": "count",\n        "size": "count",\n        "pixels": "count",\n        "px": "count",\n    }\n\n\n    class DiffGroup(dict):\n        def __missing__(self, key):\n            hint = DIFF_GROUP_FIELD_HINTS.get(key)\n            fields = ", ".join(DIFF_GROUP_FIELDS)\n            if hint is not None:\n                raise KeyError(\n                    f"diff group has no field {key!r}; use {hint}. "\n                    f"Diff-group fields: {fields}."\n                )\n            raise KeyError(\n                f"diff group has no field {key!r}. Diff-group fields: {fields}."\n            )\n\n        def __repr__(self):\n            if self["count"] <= DIFF_CELL_DETAIL_LIMIT:\n                return dict.__repr__(self)\n            folded = dict(self)\n            folded["cells"] = (\n                f"<{self[\'count\']} cells; inspect group[\'cells\'] for coordinates>"\n            )\n            return repr(folded)\n\n        __str__ = __repr__\n\n\n    def frame_diff(before_frame, after_frame):\n        # Group every cell change by color transition. Large coordinate lists are\n        # folded only in the representation; the object retains every cell.\n        if not isinstance(before_frame, FrameView) or not isinstance(after_frame, FrameView):\n            raise TypeError("frame_diff(before, after) expects two frame views.")\n        if before_frame.shape != after_frame.shape:\n            raise ValueError(\n                "frame_diff requires equal frame shapes; "\n                f"got {before_frame.shape} and {after_frame.shape}."\n            )\n\n        grouped = {}\n        cells_changed = 0\n        rows, cols = before_frame.shape\n        for r in range(rows):\n            for c in range(cols):\n                before_value = before_frame._grid[r][c]\n                after_value = after_frame._grid[r][c]\n                if before_value == after_value:\n                    continue\n                before_color = COLOR_CHARS[max(0, min(15, int(before_value)))]\n                after_color = COLOR_CHARS[max(0, min(15, int(after_value)))]\n                grouped.setdefault((before_color, after_color), []).append([r, c])\n                cells_changed += 1\n\n        groups = []\n        for (before_color, after_color), cells in grouped.items():\n            rs = [cell[0] for cell in cells]\n            cs = [cell[1] for cell in cells]\n            groups.append(\n                DiffGroup(\n                    {\n                        "from_color": before_color,\n                        "to_color": after_color,\n                        "count": len(cells),\n                        # Flat [r0, c0, r1, c1], same layout as a segmentation node\'s bbox.\n                        "bbox": [min(rs), min(cs), max(rs), max(cs)],\n                        "cells": cells,\n                    }\n                )\n            )\n        groups.sort(\n            key=lambda group: (-group["count"], group["from_color"], group["to_color"])\n        )\n        return {"cells_changed": cells_changed, "groups": groups}\n\n\n    diff = frame_diff\n\n\n    class HistoryEntryView:\n        def __init__(self, *, action, frame, anim=None):\n            self.action = action\n            self.frame = frame\n            self._anim = anim if isinstance(anim, list) else None\n\n        def __str__(self):\n            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"\n\n        __repr__ = __str__\n\n\n    class TransitionView:\n        def __init__(self, *, action, before_frame, after_frame, result):\n            self.action = action\n            self.before_frame = before_frame\n            self.after_frame = after_frame\n            self.frame = after_frame\n            self.result = dict(result) if isinstance(result, dict) else {}\n            self._diff = None\n            self._anim = None\n            self._frames = None\n\n        @property\n        def frames(self):\n            if self._frames is None:\n                self._frames = [_m111_frame(enc, self.after_frame) for enc in (self._anim or [])]\n            return self._frames\n\n        intermediate_frames = frames\n        animation_frames = frames\n\n        @property\n        def diff(self):\n            if self._diff is None and self.before_frame is not None and self.after_frame is not None:\n                self._diff = frame_diff(self.before_frame, self.after_frame)\n            return self._diff\n\n        def __str__(self):\n            return (\n                "ActionTransitionView("\n                f"action={self.action!r}, "\n                f"before_frame={self.before_frame}, "\n                f"after_frame={self.after_frame})"\n            )\n\n        __repr__ = __str__\n\n\n    def _m111_frame(encoded, ref):\n        grid = [[int(ch, 16) for ch in row] for row in str(encoded).split("/")]\n        ascii = "\\n".join("".join(COLOR_CHARS[v] if v < len(COLOR_CHARS) else "?" for v in row) for row in grid)\n        return FrameView(ascii=ascii, step=getattr(ref, "step", 0), level=getattr(ref, "level", 0),\n                         shape=[len(grid), len(grid[0]) if grid else 0], grid=grid)\n\n\n    def _frame_from_payload(payload):\n        if not isinstance(payload, dict):\n            return None\n        return FrameView(\n            ascii=str(payload.get("ascii", "")),\n            step=int(payload.get("step", 0)),\n            level=int(payload.get("level", 0)),\n            shape=payload.get("shape", [0, 0]),\n            grid=payload.get("grid", []),\n        )\n\n\n    def _history_from_payload(payload):\n        items = []\n        for entry in payload or []:\n            if not isinstance(entry, dict):\n                continue\n            items.append(\n                HistoryEntryView(\n                    action=str(entry.get("action", "")),\n                    frame=_frame_from_payload(entry.get("frame")),\n                    anim=entry.get("anim"),\n                )\n            )\n        return items\n\n\n    def _transitions_from_history(history, last_action_result):\n        transitions = []\n        for index, entry in enumerate(history):\n            action = str(getattr(entry, "action", "") or "").strip()\n            if not action:\n                continue\n            before_frame = history[index - 1].frame if index > 0 else None\n            transition = TransitionView(\n                action=action,\n                before_frame=before_frame,\n                after_frame=entry.frame,\n                result={},\n            )\n            transition._anim = getattr(entry, "_anim", None)\n            transitions.append(transition)\n        if transitions and isinstance(last_action_result, dict):\n            transitions[-1].result = dict(last_action_result)\n        return transitions\n\n\n    def _json_safe(value):\n        if value is None or isinstance(value, (str, int, float, bool)):\n            return value\n        if isinstance(value, dict):\n            return {str(key): _json_safe(item) for key, item in value.items()}\n        if isinstance(value, (list, tuple, set)):\n            return [_json_safe(item) for item in value]\n        return str(value)\n\n\n    def _sanitize_exception(exc):\n        extracted = traceback.extract_tb(exc.__traceback__)\n        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]\n        lines = ["Traceback (most recent call last):"]\n        for frame in user_frames or extracted[-1:]:\n            lines.append(f\'  File "<python_tool>", line {frame.lineno}, in {frame.name}\')\n        lines.append(f"{exc.__class__.__name__}: {exc}")\n        return "\\n".join(lines)\n\n\n    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n        root = str(name or "").split(".", 1)[0]\n        if root not in SAFE_MODULES:\n            raise ImportError(f"Module \'{name}\' is not allowed in the sandbox.")\n        return builtins.__import__(name, globals, locals, fromlist, level)\n\n\n    def _set_limits(timeout_seconds):\n        if resource is None:\n            return\n        cpu_limit = max(1, int(timeout_seconds)) + 1\n        for limit, value in (\n            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),\n            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),\n            (getattr(resource, "RLIMIT_NOFILE", None), 32),\n        ):\n            if limit is None:\n                continue\n            try:\n                resource.setrlimit(limit, (value, value))\n            except (OSError, ValueError):\n                pass\n\n\n    def _normalize_actions(actions):\n        if isinstance(actions, str):\n            items = [actions]\n        elif isinstance(actions, dict):\n            items = [actions]\n        elif isinstance(actions, (list, tuple)):\n            items = list(actions)\n        else:\n            raise TypeError(\n                "action(actions) expects a string, an action object, or a list of action strings/objects."\n            )\n        if not items:\n            raise ValueError("action(actions) requires at least one action.")\n\n        normalized = []\n        for index, item in enumerate(items, start=1):\n            if isinstance(item, str):\n                action_name = item.strip()\n                if not action_name:\n                    raise ValueError(f"Action {index} is empty.")\n                normalized.append({"action": action_name})\n                continue\n            if isinstance(item, dict):\n                action_name = str(item.get("action", "")).strip()\n                if not action_name:\n                    raise ValueError(f"Action {index} is missing an `action` field.")\n                entry = {"action": action_name}\n                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):\n                    raise ValueError(\n                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."\n                    )\n                if "row" in item:\n                    entry["row"] = item.get("row")\n                if "col" in item:\n                    entry["col"] = item.get("col")\n                normalized.append(entry)\n                continue\n            raise TypeError(f"Action {index} must be a string or a dict.")\n        return normalized\n\n\n    def main():\n        initial = _recv()\n        global COLOR_CHARS\n        COLOR_CHARS = str(initial.get("color_chars") or "")\n        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))\n        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()\n        if sandbox_cwd:\n            os.chdir(sandbox_cwd)\n        _set_limits(timeout_seconds)\n\n        action_results = []\n        stdout = io.StringIO()\n        runtime_globals = {\n            "__builtins__": {\n                name: getattr(builtins, name)\n                for name in SAFE_BUILTINS\n            },\n            "result": None,\n            "frame_diff": frame_diff,\n            "diff": diff,\n        }\n        runtime_globals["__builtins__"]["__import__"] = _safe_import\n\n        def _refresh_state(state_payload):\n            current_frame = _frame_from_payload(state_payload.get("current_frame"))\n            history = _history_from_payload(state_payload.get("history"))\n            last_action_result = state_payload.get("last_action_result")\n            action_result = (\n                dict(last_action_result) if isinstance(last_action_result, dict) else {}\n            )\n            transitions = _transitions_from_history(history, action_result)\n            last_transition = transitions[-1] if transitions else None\n\n            runtime_globals["current_frame"] = current_frame\n            runtime_globals["latest_frame"] = current_frame\n            runtime_globals["history"] = history\n            runtime_globals["transitions"] = transitions\n            runtime_globals["last_transition"] = last_transition\n            runtime_globals["previous_frame"] = (\n                last_transition.before_frame if last_transition is not None else None\n            )\n            runtime_globals["last_action_frame"] = (\n                last_transition.after_frame if last_transition is not None else None\n            )\n            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None\n            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]\n            runtime_globals["last_action_result"] = action_result\n\n        def action(actions):\n            normalized_actions = _normalize_actions(actions)\n            _send({"type": "action", "actions": normalized_actions})\n            reply = _recv()\n            if reply.get("type") == "action_error":\n                raise RuntimeError(str(reply.get("error", "action failed")))\n            if reply.get("type") != "action_result":\n                raise RuntimeError("Invalid action response from sandbox host.")\n            action_result = reply.get("action_result") or {}\n            action_results.append(action_result)\n            _refresh_state(reply.get("state") or {})\n            return action_result\n\n        runtime_globals["action"] = action\n\n        def update_memory(world_model=None, goal_model=None, action_model=None,\n                          recent_findings=None, open_questions=None, plan=None,\n                          cross_level_notes=None):\n            fields = {\n                name: value\n                for name, value in {\n                    "world_model": world_model,\n                    "goal_model": goal_model,\n                    "action_model": action_model,\n                    "recent_findings": recent_findings,\n                    "open_questions": open_questions,\n                    "plan": plan,\n                    "cross_level_notes": cross_level_notes,\n                }.items()\n                if value is not None\n            }\n            if not fields:\n                raise ValueError(\n                    "update_memory() needs at least one field, e.g. "\n                    "update_memory(world_model=...)."\n                )\n            non_string = [name for name, value in fields.items() if not isinstance(value, str)]\n            if non_string:\n                raise TypeError(\n                    f"update_memory() fields must be strings: {\', \'.join(sorted(non_string))}"\n                )\n            _send({"type": "memory", "fields": fields})\n            reply = _recv()\n            if reply.get("type") == "memory_error":\n                raise RuntimeError(str(reply.get("error", "update_memory failed")))\n            if reply.get("type") != "memory_result":\n                raise RuntimeError("Invalid update_memory response from sandbox host.")\n            return {"updated": list(reply.get("updated") or [])}\n\n        runtime_globals["update_memory"] = update_memory\n        _refresh_state(initial.get("state") or {})\n\n        try:\n            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")\n            with contextlib.redirect_stdout(stdout):\n                exec(compiled, runtime_globals, runtime_globals)\n            _send(\n                {\n                    "type": "final",\n                    "stdout": stdout.getvalue(),\n                    "result": _json_safe(runtime_globals.get("result")),\n                    "action_results": _json_safe(action_results),\n                }\n            )\n        except Exception as exc:\n            _send(\n                {\n                    "type": "error",\n                    "error": _sanitize_exception(exc),\n                    "stdout": stdout.getvalue(),\n                    "action_results": _json_safe(action_results),\n                }\n            )\n\n\n    if __name__ == "__main__":\n        main()\n    """\n).replace("__SEGMENTATION_SOURCE__\\n", inspect.getsource(_segmentation))\n\n\ndef _sanitize_host_error_text(text: str) -> str:\n    if not str(text or "").strip():\n        return "Sandbox process exited unexpectedly."\n    return "Sandbox process exited unexpectedly."\n\n\ndef _sandbox_env() -> dict[str, str]:\n    return {\n        "PYTHONUNBUFFERED": "1",\n        "PYTHONIOENCODING": "utf-8",\n        "PYTHONDONTWRITEBYTECODE": "1",\n        "HOME": "/tmp",\n        "TMPDIR": "/tmp",\n        "PATH": os.environ.get("PATH", ""),\n    }\n\n\ndef _send_json_line(handle: Any, payload: dict[str, Any]) -> None:\n    handle.write(json.dumps(payload, ensure_ascii=False) + "\\n")\n    handle.flush()\n\n\ndef _kill_process_group(process: subprocess.Popen[str]) -> None:\n    try:\n        os.killpg(process.pid, signal.SIGKILL)\n    except OSError:\n        try:\n            process.kill()\n        except OSError:\n            pass\n\n\ndef _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:\n    try:\n        process.wait(timeout=timeout)\n        return\n    except subprocess.TimeoutExpired:\n        _kill_process_group(process)\n    except OSError:\n        return\n\n    try:\n        process.wait(timeout=timeout)\n    except (subprocess.TimeoutExpired, OSError):\n        pass\n\n\ndef run_sandboxed_python(\n    *,\n    code: str,\n    timeout_seconds: int,\n    initial_state: dict[str, Any],\n    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],\n    memory_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,\n) -> dict[str, Any]:\n    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:\n        host_action_results: list[dict[str, Any]] = []\n        try:\n            process = subprocess.Popen(\n                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],\n                stdin=subprocess.PIPE,\n                stdout=subprocess.PIPE,\n                stderr=subprocess.PIPE,\n                text=True,\n                encoding="utf-8",\n                cwd=sandbox_dir,\n                env=_sandbox_env(),\n                start_new_session=True,\n            )\n        except OSError:\n            return {\n                "error": "Sandbox process could not start.",\n                "stdout": "",\n                "action_results": [],\n            }\n        assert process.stdin is not None\n        assert process.stdout is not None\n        assert process.stderr is not None\n\n        stdout_queue: queue.Queue[str | None] = queue.Queue()\n\n        def _stdout_reader() -> None:\n            for raw_line in process.stdout:\n                stdout_queue.put(raw_line)\n            stdout_queue.put(None)\n\n        threading.Thread(target=_stdout_reader, daemon=True).start()\n\n        _send_json_line(\n            process.stdin,\n            {\n                "code": code,\n                "timeout_seconds": timeout_seconds,\n                "sandbox_cwd": sandbox_dir,\n                "state": initial_state,\n                "color_chars": ARC_COLOR_CHARS,\n            },\n        )\n\n        deadline = time.monotonic() + max(1, int(timeout_seconds))\n        while True:\n            remaining = deadline - time.monotonic()\n            if remaining <= 0:\n                _kill_process_group(process)\n                _wait_for_process_exit(process)\n                return {\n                    "error": f"Tool timed out after {timeout_seconds}s",\n                    "stdout": "",\n                    "action_results": list(host_action_results),\n                }\n\n            try:\n                line = stdout_queue.get(timeout=remaining)\n            except queue.Empty:\n                continue\n            if line is None:\n                stderr = process.stderr.read()\n                _wait_for_process_exit(process)\n                return {\n                    "error": _sanitize_host_error_text(stderr),\n                    "stdout": "",\n                    "action_results": list(host_action_results),\n                }\n\n            try:\n                message = json.loads(line)\n            except json.JSONDecodeError:\n                stderr = process.stderr.read()\n                _kill_process_group(process)\n                _wait_for_process_exit(process)\n                return {\n                    "error": "Sandbox process returned an invalid response.",\n                    "stdout": "",\n                    "action_results": list(host_action_results),\n                }\n\n            msg_type = str(message.get("type", "")).strip()\n            if msg_type == "action":\n                try:\n                    action_result_payload = action_handler(list(message.get("actions") or []))\n                except Exception:  # noqa: BLE001\n                    _send_json_line(\n                        process.stdin,\n                        {\n                            "type": "action_error",\n                            "error": "action failed in sandbox host.",\n                        },\n                    )\n                    continue\n                raw_action_result = action_result_payload.get("action_result") or {}\n                if isinstance(raw_action_result, dict):\n                    host_action_results.append(dict(raw_action_result))\n                _send_json_line(\n                    process.stdin,\n                    {\n                        "type": "action_result",\n                        "action_result": raw_action_result,\n                        "state": action_result_payload.get("state") or {},\n                    },\n                )\n                continue\n\n            if msg_type == "memory":\n                if memory_handler is None:\n                    _send_json_line(\n                        process.stdin,\n                        {\n                            "type": "memory_error",\n                            "error": "update_memory is not available in this session.",\n                        },\n                    )\n                    continue\n                try:\n                    memory_payload = memory_handler(dict(message.get("fields") or {}))\n                except Exception:  # noqa: BLE001\n                    memory_payload = {"error": "update_memory failed in sandbox host."}\n                if memory_payload.get("error"):\n                    _send_json_line(\n                        process.stdin,\n                        {\n                            "type": "memory_error",\n                            "error": str(memory_payload["error"]),\n                        },\n                    )\n                else:\n                    _send_json_line(\n                        process.stdin,\n                        {\n                            "type": "memory_result",\n                            "updated": list(memory_payload.get("updated") or []),\n                        },\n                    )\n                continue\n\n            if msg_type in {"final", "error"}:\n                _wait_for_process_exit(process)\n                return {\n                    "stdout": str(message.get("stdout", "") or ""),\n                    "result": message.get("result"),\n                    "error": str(message.get("error", "") or ""),\n                    "action_results": list(message.get("action_results") or host_action_results),\n                }\n\n            _wait_for_process_exit(process)\n            return {\n                "error": "Sandbox process returned an unknown message type.",\n                "stdout": "",\n                "action_results": list(host_action_results),\n            }\n'
_W90_TEXTS = {'STRUCTURED_RUNTIME_STATE_ADDENDUM': "\n\nRuntime variables inside every `python` tool call:\n- `current_frame` is a lightweight frame view for the latest environment state.\n- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n- `current_frame.ascii` is a single newline-delimited string containing the latest board rendered with the letter-coded ARC color symbols.\n- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n- Each node in `segmentation['nodes']` is one 4-connected same-color object. The nodes are ordered top-most-left-most, and each one has exactly these fields: `id`, `color`, `area`, `bbox`, `boundary`, `children`.\n- `id` is an 8-character signature of the object's color and shape that ignores its position -- equal ids mean the same color and shape wherever it sits, so use it to track an object across frames or to spot multiple identical objects in one frame. Identical-looking objects therefore SHARE one id; an id names a shape, not a particular node.\n- `color` is the ARC color character. `area` is the cell count as an int. `bbox` is `[min_row, min_col, max_row, max_col]`, so height is `bbox[2] - bbox[0] + 1` and width is `bbox[3] - bbox[1] + 1`. `boundary` is the clockwise outer-perimeter corner points as `[row, col]`. `children` are the ids of objects fully enclosed by this one.\n- `segmentation['adjacency_list']` is a list of `[id_a, id_b]` id pairs whose objects share an edge. Because identical objects share an id, resolve a pair back to specific nodes by filtering `segmentation['nodes']` on that id together with `bbox`.\n- `current_frame.step` is the current environment step count.\n- `current_frame.level` is the current level number.\n- `current_frame.shape` is a `(rows, cols)` tuple.\n- The raw numeric grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view of the board -- objects, colors, shapes, containment, adjacency, and cross-frame object ids. Use `current_frame.ascii` only to read a small, specific region; do not scan the whole board with it.\n- `history` is a chronological list of action/frame snapshots.\n- `history` is a Python list of objects, not a dict.\n- Each history entry exposes only `.action` and `.frame`; entries are not subscriptable like `entry['action']`.\n- Each `history[i].frame` is the frame after `history[i].action`; each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n- Important history semantics: when `history` is non-empty, `history[-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `history[-2].frame` when available.\n- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n- `last_action_frame` is the post-action frame for `last_action`; it matches `current_frame` after a real action.\n- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), `.result`, and `.frames`.\n- `last_transition` is `transitions[-1]` or `None`. Its `.result` mirrors `last_action_result`; older transitions may have an empty `.result`. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `history[-1].frame`.\n- `last_action_result` is the persisted result dict from the most recent `action(...)` call. It remains available across later Python inspection calls that do not call `action(...)`, and is `{}` before any action result exists. Read transition metadata from fields/keys such as `last_action_result['board_changed']`, `last_action_result['done']`, `last_action_result['level_completed']`, `last_action_result['game_over']`, `last_action_result['run_complete']`, `last_action_result['reward']`, and `last_action_result['valid_actions']`.\n- `valid_actions` is the current list of valid action names.\n- Call `action(actions)` to execute one or more real environment actions from Python.\n- Pass `action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_result` are refreshed.\n", 'MULTIMODAL_CONTEXT_ADDENDUM': '\n\nMultimodal context:\n- The latest user turn includes an attached image of the current ARC grid; earlier turns are text only.\n- The image and `current_frame.ascii` are two representations of the same current frame.\n- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n', 'PYTHON_ADDENDUM': "\n\nPython tool guidance:\n- Use `current_frame.segmentation` as your primary view of the board -- objects, colors, containment, adjacency, and cross-frame object ids.\n- Use `current_frame.ascii` only to read a small, specific region of the board when `segmentation` is not enough; never use it to scan or summarize the whole board.\n- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\n- The only tool is `python`; call it with one ephemeral `code` string.\n- Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n- For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`. `history[-1].frame` is the current frame, so comparing it to `current_frame` only compares the board to itself.\n- Maintain a compact working world model: what entities or regions exist, what actions seem to do, what the goal likely is, what remains uncertain, and what plan best fits the evidence so far.\n- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n- Optimize for the shortest reliable sequence that advances the current goal as described by your world model. If confidence is low, program a discriminating probe and revise the world model from the result.\n- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n- Inspect current and history frames from Python instead of describing frames freehand.\n- Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs, coordinates, counts, or tiny local crops.\n- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. It's fine to write a lot of python code, just make the output short and interpretable\n- A strong default loop is: summarize the board, infer the desired environment change, write a small scorer or search over candidate sequences, execute the best probe or plan with `action(...)`, then inspect again until you understand exactly what changed.\n- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n- Call `action(...)` inside Python rather than returning action text in the chat.\n- `action(...)` accepts an ordered list of one or more actions. Once your code has selected a reliable sequence, it is often useful to batch it.\n- You can also call `action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n", 'GAME_PRIORS_ADDENDUM': '\n\nPriors and probing:\n- Approach an unfamiliar scene the way an experienced player does: start from priors about what the objects and the actions are likely to do, then probe to confirm or discard them. Priors are the hypotheses you test first, not conclusions.\n- Action priors worth considering early: the arrow actions usually move or steer something, whether a piece, a cursor, or a selection; `SPACE` often rotates, switches, or cycles the active object, or commits an attempt; `MOUSE` often selects or toggles whatever sits under it, or triggers the thing it lands on. Any of these can be wrong in a given game.\n- Objects often advertise their function through shape and color: repeated small tokens tend to be collectibles or slots, an enclosure tends to be a container or a wall, a single distinct shape tends to be either the piece you control or the target, and shared colors tend to mark things that belong together.\n- Work from mechanics outward to the goal. Once most objects and action effects are pinned down, the goal usually shows itself in what the board rewards or changes, and you can probe the goal the same way: form a candidate, test it cheaply, revise.\n- The goal is sometimes legible before the mechanics are. When it is, use it to decide which mechanics are worth probing at all, and stop probing once the ones the current goal needs are settled.\n', 'TURN_PROTOCOL_ADDENDUM': '\n\nPer-turn protocol:\n- Each user turn carries only what changed: the outcome of the previous action sequence, the current step and level, and the valid actions right now. Everything else you need is already in this system prompt and in the conversation so far.\n- Before executing new actions, restate your working world model as revised by the newest evidence. Carry forward what still holds, and drop or correct anything that recent evidence contradicts.\n- Keep that revision to short assistant text before the tool call. Helpful optional prefixes are `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and `Cross-level notes:`.\n- Then use `python` to inspect the evidence, refine the world model from the newest history, score candidate actions or short sequences against the goal as you currently understand it, and call `action(...)` with the best action or ordered batch.\n', 'INTERMEDIATE_FRAMES_ADDENDUM': '\n\nIntermediate frames:\n- When an action changes the board over several steps, those in-between frames are attached after the current grid image, oldest first, under their own header.\n- Use them to see how the board reached its current state: what moved, in which direction, and what it passed through or collided with.\n- Only the current grid image is the live board; the earlier frames are already past.\n'}
_W90_DIR = "/tmp/m90_wang"
_STAT90 = {"installed": False, "sandbox_swapped": False, "system_prompt_chars": 0, "user_prompts": 0, "ledger_off": False}


def _m90_load(name, source):
    path = _os90.path.join(_W90_DIR, name + ".py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    spec = _il90.spec_from_file_location(name, path)
    mod = _il90.module_from_spec(spec)
    _sys90.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install90():
    import inference.agent.tool_agent as T
    import inference.agent.python_tool_sandbox as S
    if getattr(T.ToolAgent._build_user_prompt, "_m90", False):
        raise RuntimeError("M90 installed twice")
    _os90.makedirs(_W90_DIR, exist_ok=True)
    _m90_load("m90_wang_segmentation", _W90_SEG_SRC)
    W = _m90_load("m90_wang_sandbox", _W90_SANDBOX_SRC)
    S.run_sandboxed_python = W.run_sandboxed_python
    S._SANDBOX_BOOTSTRAP = W._SANDBOX_BOOTSTRAP
    T.run_sandboxed_python = W.run_sandboxed_python
    _STAT90["sandbox_swapped"] = True
    for k in ("STRUCTURED_RUNTIME_STATE_ADDENDUM", "MULTIMODAL_CONTEXT_ADDENDUM", "PYTHON_ADDENDUM"):
        setattr(T, k, _W90_TEXTS[k])
    frames_on = "m76_summary" in globals()

    def _build_system_prompt(*, tool_output_tokens: int) -> str:
        prompt = "You are a coding agent solving a grid-based puzzle game."
        prompt += T.GAME_OVERVIEW_ADDENDUM
        prompt += _W90_TEXTS["STRUCTURED_RUNTIME_STATE_ADDENDUM"]
        if T.current_grid_image_enabled():
            prompt += _W90_TEXTS["MULTIMODAL_CONTEXT_ADDENDUM"]
        if frames_on:
            prompt += _W90_TEXTS["INTERMEDIATE_FRAMES_ADDENDUM"]
        prompt += T.VISUAL_GAME_ADDENDUM
        prompt += _W90_TEXTS["GAME_PRIORS_ADDENDUM"]
        prompt += _W90_TEXTS["PYTHON_ADDENDUM"]
        prompt += T.COMPACT_TOOL_SESSION_ADDENDUM.format(tool_output_tokens=tool_output_tokens)
        prompt += _W90_TEXTS["TURN_PROTOCOL_ADDENDUM"]
        _STAT90["system_prompt_chars"] = len(prompt)
        return prompt
    T._build_system_prompt = _build_system_prompt

    ns = dict(_format_valid_action_line=T._format_valid_action_line, Any=Any, Frame=T.Frame, HistoryEntry=T.HistoryEntry)
    exec('def _build_user_prompt(\n    self,\n    action_num: int,\n    *,\n    valid_actions: list[str] | None,\n    current_frame: Frame | None = None,\n    history_entries: list[HistoryEntry] | None = None,\n    previous_step_summary: dict[str, Any] | None = None,\n) -> str:\n    history_entries = history_entries or []\n    current_step = max(current_frame.step if current_frame is not None else 0, max(0, action_num)) + 1\n    current_level = current_frame.level if current_frame is not None else 1\n    summary_level = None\n    if previous_step_summary is not None:\n        try:\n            summary_level = int(previous_step_summary.get("level"))\n        except (TypeError, ValueError):\n            summary_level = None\n    if summary_level is not None:\n        current_level = max(current_level, summary_level)\n    observed_max_level = max(\n        [current_level, *[entry.frame.level for entry in history_entries if entry.frame is not None]],\n        default=current_level,\n    )\n    lines: list[str] = []\n    if previous_step_summary:\n        count = previous_step_summary.get("executed_count")\n        try:\n            normalized_count = int(count) if count is not None else None\n        except (TypeError, ValueError):\n            normalized_count = None\n        action_label = "action" if normalized_count == 1 else "actions"\n        lines.append(f"The code executed {normalized_count or 0} {action_label} in the previous sequence.")\n        executed_actions = previous_step_summary.get("executed_actions")\n        rendered_actions: list[str] = []\n        if isinstance(executed_actions, list):\n            rendered_actions = [str(name).strip() for name in executed_actions if str(name).strip()]\n        if rendered_actions:\n            action_prefix = "Executed actions (first 10):" if len(rendered_actions) > 10 else "Executed actions:"\n            lines.append(f"{action_prefix} {\', \'.join(rendered_actions[:10])}.")\n        else:\n            lines.append("Executed actions: none.")\n        if previous_step_summary.get("run_complete"):\n            lines.append("You have completed the run!")\n        elif previous_step_summary.get("level_transition"):\n            lines.append("You have progressed to a new level!")\n        else:\n            lines.append("You are still on the same level.")\n        if previous_step_summary.get("game_over"):\n            lines.append("The game is over.")\n    elif (current_frame is not None and current_frame.step > 0) or action_num > 0:\n        lines.append("No previous action sequence was captured.")\n    else:\n        lines.append("No previous sequence has been executed yet.")\n    state_line = f"Current state: step {current_step}, level {current_level}"\n    if observed_max_level > current_level:\n        state_line += f" out of observed max level {observed_max_level} so far"\n    state_line += "."\n    lines.extend(\n        [\n            state_line,\n            f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",\n        ]\n    )\n    # if action_num == 0:\n    #     lines.append(\n    #         "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame."\n    #     )\n    # else:\n    #     lines.append(\n    #         "Focus on what changed most recently, update the target environment change if needed, and separate gameplay-object changes from HUD-only changes."\n    #     )\n    return "\\n".join(lines)', ns)
    their = ns["_build_user_prompt"]

    def _build_user_prompt(self, action_num, *a, **k):
        _STAT90["user_prompts"] += 1
        return their(self, action_num, *a, **k)
    _build_user_prompt._m90 = True
    T.ToolAgent._build_user_prompt = _build_user_prompt
    g = globals()
    if callable(g.get("_line_for")):
        g["_line_for"] = lambda msg: None      # M24: no ledger lines -> nothing injected into the system message
        if "_RESERVE" in g:
            g["_RESERVE"] = 0
        _STAT90["ledger_off"] = True
    _STAT90["installed"] = True
    return True


def m90_summary():
    return dict(_STAT90)

install90()
print('V1800_M90 installed', __import__('json').dumps(m90_summary()), flush=True)

# ===== v18.1a = M111: ANIMATION FRAMES IN THE SANDBOX (transition.frames) =====
"""WHY (v18.1 transcripts + engine replay of its 9,000 actions, 2026-09-29):
  * 22 of 25 public games have actions whose in-between frames differ from BOTH the board before and the settled board
    after (information that exists only mid-animation): sk48 255 actions, tu93 273, su15 226, g50t 213, lf52 208,
    bp35 119, sb26 108, sp80 23 (26 frames per shot), tn36 42. The model gets at most four of them, as 256 px images
    (8x8 cells per visual token) on the next turn, and NONE in Python: `transitions` hold only before/after.
  * Four of v18.1's six biggest time sinks are games whose decisive feedback is mid-animation: sp80 L2 (123 min,
    stuck), tn36 L2 (116, stuck), s5i5 L3 (106, stuck), cd82 L3 (107, stuck); sb26 L5 85 min. Thinking sentences about
    animations / mid-frames: tn36 256, sp80 215, sb26 194. The model asks for exactly this: "Maybe there are extra
    history entries for animation frames? Let me check" (tn36), "I only have 2 mid-frames ... can't know without the
    animation" (sp80). WALL_ANALYSIS (Sep 22): "evidence destroyed by the final frame" = 18.8% of available score.
  * Tried before only as images (M76) or as text in the turn message (M103, v19.3: lost). Python access was proposed
    on Sep 24 and never built. This is data on request: nothing is added to any turn message.
HOW:
  1. host: after each engine action, the frames the engine returned (GameState.all_frames) are kept for this game's
     agent: in-between frames only (the settled board is the history entry itself), consecutive duplicates and a tail
     equal to the settled board dropped, at most M111_MAX_FRAMES (even subsample, first and last kept), for the last
     M111_KEEP_ACTIONS animated actions. Encoded compactly (rows of hex digits joined by '/').
  2. payload: the sandbox state's history entries carry "anim" for those actions (copies; M86's cached payload dicts
     are never mutated). The game is identified by the ToolAgent running the python call (thread-local).
  3. sandbox: TransitionView.frames (aliases intermediate_frames / animation_frames): FrameView objects with the same
     API as current_frame (.ascii, .segmentation, frame_diff), oldest first, [] when the action returned one board.
  4. action results: `animation_frames` (in-between boards kept, one count per executed action) is added to the
     dict action() returns, only when an action in that call animated -- the model prints these results constantly,
     so it sees the count at the moment it matters.
  5. prompt: one clause in the `transitions` line and one line in the intermediate-frames section.
READ: m111_summary()
"""
import threading as _th111

M111_KEEP_ACTIONS = 8
M111_MAX_FRAMES = 32
_TL111 = _th111.local()
_STAT111 = {"actions": 0, "animated": 0, "frames_kept": 0, "subsampled": 0, "payload_attach": 0, "results_tagged": 0,
            "payload_len_mismatch": 0, "errors": 0, "last_error": ""}


def m111_summary():
    return dict(_STAT111)


def _m111_err(exc):
    _STAT111["errors"] += 1
    _STAT111["last_error"] = f"{type(exc).__name__}: {exc}"[:200]


def _m111_grid(frame):
    """taaf Frame (.data ndarray) / .grid / dict / nested list -> list of lists of ints, or None."""
    data = getattr(frame, "data", None)
    if data is None:
        data = getattr(frame, "grid", None)
    if data is None and isinstance(frame, dict):
        data = frame.get("data", frame.get("grid"))
    if data is None:
        data = frame
    rows = data.tolist() if hasattr(data, "tolist") else data
    try:
        out = [[int(v) for v in row] for row in rows]
    except (TypeError, ValueError):
        return None
    return out if out and all(out) else None


def m111_select(grids, max_frames=M111_MAX_FRAMES):
    """All frames of one action (last = settled board) -> the in-between boards worth keeping, oldest first."""
    if len(grids) <= 1:
        return []
    final = grids[-1]
    mids = []
    for g in grids[:-1]:
        if mids and g == mids[-1]:
            continue
        mids.append(g)
    while mids and mids[-1] == final:
        mids.pop()
    if len(mids) > max_frames:
        _STAT111["subsampled"] += 1
        n = len(mids)
        idx = sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})
        mids = [mids[i] for i in idx]
    return mids


def _m111_enc(grid):
    return "/".join("".join("%x" % (v & 15) for v in row) for row in grid)


def install111():
    import inference.framework.solver as S
    import inference.agent.tool_agent as T
    sess = S._HarnessGameSession
    if getattr(sess._execute_action, "_m111", False):
        raise RuntimeError("M111 installed twice")
    orig_exec = sess._execute_action

    def _execute_action(self, action, **kw):
        n0 = len(self.history_entries)
        out = orig_exec(self, action, **kw)
        try:
            _STAT111["actions"] += 1
            if len(self.history_entries) == n0 + 1:
                frames = list(getattr(self.game.current_state, "all_frames", ()) or ())
                keep = m111_select([g for g in (_m111_grid(f) for f in frames) if g])
                target = getattr(self, "analyzer", None) or self
                target.__dict__.setdefault("_m111_batch", []).append(len(keep))
                if keep:
                    _STAT111["animated"] += 1
                    _STAT111["frames_kept"] += len(keep)
                    store = target.__dict__.setdefault("_m111_anim", {})
                    store[n0] = [_m111_enc(g) for g in keep]
                    for k in sorted(store)[:-M111_KEEP_ACTIONS]:
                        del store[k]
        except Exception as exc:
            _m111_err(exc)
        return out

    _execute_action._m111 = True
    sess._execute_action = _execute_action

    orig_run = T.ToolAgent._run_python_tool

    def _run_python_tool(self, *a, **k):
        prev = getattr(_TL111, "agent", None)
        _TL111.agent = self
        try:
            return orig_run(self, *a, **k)
        finally:
            _TL111.agent = prev

    _run_python_tool._m111 = True
    T.ToolAgent._run_python_tool = _run_python_tool

    orig_hp = T._ascii_history_view_payload

    def _ascii_history_view_payload(history_entries):
        out = orig_hp(history_entries)
        try:
            ag = getattr(_TL111, "agent", None)
            store = getattr(ag, "_m111_anim", None) if ag is not None else None
            if store:
                if len(out) != len(history_entries):
                    _STAT111["payload_len_mismatch"] += 1
                    return out
                out = list(out)
                for idx, enc in store.items():
                    if 0 <= idx < len(out) and isinstance(out[idx], dict):
                        out[idx] = dict(out[idx], anim=enc)
                _STAT111["payload_attach"] += 1
        except Exception as exc:
            _m111_err(exc)
        return out

    T._ascii_history_view_payload = _ascii_history_view_payload

    orig_compact = T.ToolAgent._compact_action_result

    def _compact_action_result(self, *a, **k):
        out = orig_compact(self, *a, **k)
        try:
            counts = self.__dict__.pop("_m111_batch", None)
            if isinstance(out, dict) and counts and any(counts):
                out = dict(out, animation_frames=counts)
                _STAT111["results_tagged"] += 1
        except Exception as exc:
            _m111_err(exc)
        return out

    T.ToolAgent._compact_action_result = _compact_action_result
    return True


def _m111_selftest():
    g0 = [[0, 0], [0, 0]]; g1 = [[1, 0], [0, 0]]; g2 = [[1, 1], [0, 0]]
    assert m111_select([g0]) == [] and m111_select([g1, g1, g2, g2]) == [g1]
    assert len(m111_select([[[i % 16]] for i in range(100)] + [[[0]]])) == M111_MAX_FRAMES
    assert _m111_enc([[10, 1], [0, 15]]) == "a1/0f"
    _STAT111["subsampled"] = 0
    return True


install111()
print('V181A_M111 installed', _m111_selftest(), __import__('json').dumps(m111_summary()), flush=True)

# ===== v19.0 = M99: OFFICIAL ACTION SEMANTICS -- ACTION7 IS UNDO, AND THE MODEL MAY RESET THE LEVEL =====
"""WHY (2026-09-27, CPU audit of v18.1/6/7/7s/8/12 transcripts + the 25 public game sources + arcengine 0.9.3):
  * docs.arcprize.org/actions: "ACTION7 -- Simple undo action" (human key Z); "RESET -- Initializes or restarts the game or
    level state". All 6 public games that offer ACTION7 (ar25 bp35 lf52 sb26 sk48 su15) implement it as pop-and-restore of
    the previous state; checked on the engine (undo_check.py): the move's cells revert, only the edge step-counter differs,
    and on a fresh level ACTION7 changes nothing.
  * v14.14 made ACTION7 executable but labelled it "ACTION7" ("no semantic alias"). The model therefore probes it once on a
    fresh level, where there is nothing to undo, sees no change and writes it off -- or guesses "flap/jump", "commit",
    "shoot". Transcripts: "maybe the game has an undo? ACTION7?", "actions are real; I can't undo".
  * RESET is filtered out of valid_actions (_engine_action_names) AND refused by step_env ("RESET is not valid right now":
    it is never in available_actions); only the harness's own GAME_OVER auto-reset can issue it. The system prompt never
    mentions it. Across six v18 runs 21.5% of analysis steps come after the model has itself declared the current level
    unwinnable / irreversible / stuck; 57 such levels were never cleared (993 steps).
  * Engine check (reset_l2.py): a RESET on level 2 restores level 2's opening board and keeps level 1 cleared, with
    ONLY_RESET_LEVELS set or unset (the gateway's setting is unknown; action_count > 0 makes it a level reset either way);
    the scorecard charges it as one action on the current level. The competition gateway only blocks a RESET at the very
    first action of the game, which the guard below never issues.
WHAT: (1) ACTION7 is shown to the model as UNDO (UNDO and ACTION7 both accepted); (2) a standalone action(['RESET']) is
executed through the same _execute_action path as the auto-reset (refused inside a batch, and refused right after another
RESET, when the level is already at its opening layout); (3) one bullet in the static priors section says so. Static
system-prompt text only: one prefix change for the whole run, nothing per turn. No game ids, rules or solutions.
READ: m99_summary()
"""
import threading as _th99

_LK99 = _th99.Lock()
_STAT99 = {"maps_patched": 0, "prompt_edit": False, "undo_requests": 0, "reset_requests": 0, "reset_executed": 0,
           "reset_refused_batch": 0, "reset_refused_repeat": 0, "errors": 0, "last_error": ""}

M99_PRIOR_OLD = "Any of these can be wrong in a given game.\n"
M99_PRIOR_NEW_UNUSED = (
    "Any of these can be wrong in a given game.\n"
    "- `UNDO`, offered by some games, is the standard ARC-AGI-3 undo: it reverts your most recent move. It does nothing "
    "when there is nothing to undo, so judge it only right after a move; a move followed by `UNDO` is a cheap, safe probe. "
    "`RESET` is always available although never listed: `action(['RESET'])` restarts the current level from its starting "
    "layout and keeps every level already cleared. Each costs one action. If you conclude that the level can no longer be "
    "won from the current state, RESET instead of continuing to act in it.\n")

M99_PRIOR_NEW = M99_PRIOR_OLD        # v18.1f: no priors bullet (no added text)


def _m99_err(exc):
    with _LK99:
        _STAT99["errors"] += 1
        _STAT99["last_error"] = f"{type(exc).__name__}: {exc}"[:200]


def _m99_names(arguments):
    items = arguments.get("actions") if isinstance(arguments, dict) else None
    if not isinstance(items, list):
        items = [{"action": (arguments or {}).get("action")}] if isinstance(arguments, dict) else []
    return [str(it.get("action", "") if isinstance(it, dict) else it).strip().upper() for it in items]


def install99():
    import sys as _sys99
    import arcengine as _ae99
    import inference.framework.solver as S
    sess = S._HarnessGameSession
    if getattr(sess.step_env, "_m99", False):
        raise RuntimeError("M99 installed twice")

    # (1) ACTION7 -> UNDO in every loaded copy of the mapping (v14.14 left it as the identity label)
    for name in list(_sys99.modules):
        mod = _sys99.modules.get(name)
        e2m = getattr(mod, "ENGINE_TO_MODEL_ACTION", None) if mod is not None else None
        m2e = getattr(mod, "MODEL_TO_ENGINE_ACTION", None) if mod is not None else None
        if isinstance(e2m, dict) and isinstance(m2e, dict):
            e2m["ACTION7"] = "UNDO"
            m2e["UNDO"] = "ACTION7"
            m2e["ACTION7"] = "ACTION7"
            _STAT99["maps_patched"] += 1
    import inference.agent.action_names as _A99
    checks = {"UNDO": "ACTION7", "ACTION7": "ACTION7", "UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3",
              "RIGHT": "ACTION4", "SPACE": "ACTION5", "MOUSE": "ACTION6", "RESET": "RESET"}
    bad = {k: _A99.to_engine_action(k) for k, v in checks.items() if _A99.to_engine_action(k) != v}
    if bad or _A99.to_model_action("ACTION7") != "UNDO" or _A99.to_model_action("ACTION1") != "UP":
        raise RuntimeError(f"M99: action mapping check failed: {bad}")

    # (2) the static priors section gains one bullet (read at every system-prompt build, like M94's edits)
    texts = globals()["_W90_TEXTS"]
    if texts["GAME_PRIORS_ADDENDUM"].count(M99_PRIOR_OLD) != 1:
        raise RuntimeError("M99: priors anchor not found exactly once")
    texts["GAME_PRIORS_ADDENDUM"] = texts["GAME_PRIORS_ADDENDUM"].replace(M99_PRIOR_OLD, M99_PRIOR_NEW)
    _STAT99["prompt_edit"] = True

    # (3) a standalone model RESET goes through the auto-reset's execution path
    orig_step = sess.step_env

    def step_env(self, arguments):
        try:
            names = _m99_names(arguments)
        except Exception as exc:
            _m99_err(exc)
            return orig_step(self, arguments)
        if "UNDO" in names or "ACTION7" in names:
            with _LK99:
                _STAT99["undo_requests"] += 1
        if "RESET" not in names:
            return orig_step(self, arguments)
        with _LK99:
            _STAT99["reset_requests"] += 1
        if len(names) != 1:
            with _LK99:
                _STAT99["reset_refused_batch"] += 1
            return self._error_payload("RESET was not executed: send it on its own, action(['RESET']). It restarts the "
                                       "current level; nothing else in this batch was executed.")
        if self.should_stop() or S._is_engine_game_over(self.game):
            return orig_step(self, arguments)
        if self.last_engine_action == "RESET":
            with _LK99:
                _STAT99["reset_refused_repeat"] += 1
            return self._error_payload("RESET was not executed: the last action was already a RESET, so the level is at "
                                       "its starting layout.")
        try:
            payload = self._execute_action(_ae99.ActionInput(id=_ae99.GameAction.RESET, data={}),
                                           batch_index=1, batch_size=1, flush_viewer_payload=False)
        except Exception as exc:
            _m99_err(exc)
            return self._error_payload(f"{type(exc).__name__}: {exc}")
        out = dict(payload)
        out.update(reward=float(payload.get("reward", 0.0) or 0.0), last_reward=payload.get("reward", 0.0),
                   batched=False, requested_count=1, executed_count=1, requested_actions=["RESET"],
                   executed_actions=[str(payload.get("action_display") or "RESET")], stopped_early=False)
        self.write_viewer_payload()
        with _LK99:
            _STAT99["reset_executed"] += 1
        return out

    step_env._m99 = True
    sess.step_env = step_env
    return True


def m99_summary():
    with _LK99:
        return dict(_STAT99)
install99()
print('V1900_M99 installed', __import__('json').dumps(m99_summary()), flush=True)

# ===== v19.2 = M102: SAY WHAT A GAME OVER MEANS, AND DO NOT LET ONE FAILED AUTO-RESET KILL A GAME =====
"""WHY (2026-09-27, six v18 runs):
  * 452 game-overs; 46% of all actions were spent in level attempts that ended in a game over. After each one the harness
    has already pressed RESET (the level restarted from its opening layout, earlier levels kept), but the next turn's
    message says only "You are still on the same level. / The game is over." The model then reasons about a finished
    game: tu93 "the harness says 'The game is over' but the image shows the player back at start position", bp35
    "Wait, the harness says 'The game is over.' Hmm! ... Something odd happened" (then ~20 min on the frame).
  * On Kaggle every action is an HTTP call to the gateway with a 10 s timeout and no retry (arc_agi remote_wrapper). A
    failure in a model's action comes back to the model as an error it can retry, but a failure in the harness's own
    auto-reset escapes play_slice, the game is marked crashed and M72 never resumes it. ~3 auto-resets per game ->
    a few hundred per hidden run. Repeating a RESET is harmless (another level reset).
WHAT: (1) the "The game is over." line in the per-turn message is replaced by an accurate one; (2) _execute_auto_reset
retries up to M102_RESET_TRIES times with backoff before giving up exactly as before.
READ: m102_summary()
"""
import threading as _th102
import time as _t102

M102_RESET_TRIES = 4
M102_OLD = "The game is over."
M102_NEW_UNUSED = ("GAME OVER on this level: the last sequence lost the level (for example its action budget ran out or "
            "something fatal happened). The harness has already pressed RESET for you, so the level restarted from its "
            "opening layout; levels already cleared stay cleared. The current frame is the fresh start of this level.")
M102_NEW = "Level restarted."   # v18.1f: true and as short as the line it replaces
_LK102 = _th102.Lock()
_STAT102 = {"messages_rewritten": 0, "reset_retries": 0, "reset_gave_up": 0, "errors": 0, "last_error": ""}


def install102():
    import inference.agent.tool_agent as T
    import inference.framework.solver as S
    A = T.ToolAgent
    orig_prompt = A._build_user_prompt
    if getattr(orig_prompt, "_m102", False):
        raise RuntimeError("M102 installed twice")

    def _build_user_prompt(self, action_num, *a, **k):
        out = orig_prompt(self, action_num, *a, **k)
        try:
            if isinstance(out, str) and M102_OLD in out:
                out = out.replace(M102_OLD, M102_NEW)
                with _LK102:
                    _STAT102["messages_rewritten"] += 1
        except Exception as exc:
            with _LK102:
                _STAT102["errors"] += 1
                _STAT102["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out

    for f in dir(orig_prompt):
        if f.startswith("_m") and not f.startswith("__"):
            try:
                setattr(_build_user_prompt, f, getattr(orig_prompt, f))
            except Exception:
                pass
    _build_user_prompt._m102 = True
    A._build_user_prompt = _build_user_prompt

    sess = S._HarnessGameSession
    orig_reset = sess._execute_auto_reset

    def _execute_auto_reset(self):
        for i in range(M102_RESET_TRIES):
            try:
                return orig_reset(self)
            except Exception as exc:
                with _LK102:
                    _STAT102["last_error"] = f"reset: {type(exc).__name__}: {exc}"[:200]
                    if i + 1 == M102_RESET_TRIES:
                        _STAT102["reset_gave_up"] += 1
                    else:
                        _STAT102["reset_retries"] += 1
                if i + 1 == M102_RESET_TRIES:
                    raise
                _t102.sleep(2.0 * (2 ** i))

    _execute_auto_reset._m102 = True
    sess._execute_auto_reset = _execute_auto_reset
    return True


def m102_summary():
    with _LK102:
        return dict(_STAT102)
install102()
print('V1920_M102 installed', __import__('json').dumps(m102_summary()), flush=True)

# ===== v18.1g: two phrases deleted from Tufa's prompt sections (the edge bar is the action budget, not noise) =====
import inference.agent.tool_agent as _T181g
_D181G = ((" If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is "
           "concrete evidence that it interacts with the puzzle mechanics."), "VISUAL_GAME_ADDENDUM"), \
         ((" and whether the last action affected gameplay state or only HUD elements such as countdown bars"), "COMPACT_TOOL_SESSION_ADDENDUM")
for _txt181g, _name181g in _D181G:
    _v181g = getattr(_T181g, _name181g)
    if _v181g.count(_txt181g) != 1:
        raise RuntimeError(f"v18.1g: phrase not found exactly once in {_name181g}")
    setattr(_T181g, _name181g, _v181g.replace(_txt181g, ""))
print('V181G_BAR_TEXT_DELETED', [len(t) for t, _ in _D181G], flush=True)
