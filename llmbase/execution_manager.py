import threading
import time


class ModuleGraph:
    def __init__(self, execution_order):
        """ModuleGraph: build, run and stop retico pipelines from a nested dictionary.
        Grammar (A, B, C... are retico modules):
            {A: B}                    A → B
            {A: [B, C, D]}            A → B → C → D            (a list is a chain)
            {A: {B: ..., C: ...}}     A → B and A → C           (a dict is a branch)
            {A: [B, {C: [D], E: F}]}  A → B → C → D and B → E → F
                                    (a dict inside a list branches from the element before it)
        Each top-level key is an independent set that can be run and stopped on its own.
        """
        self.sets = {}  # set name → list of (sender, receiver) edges
        for set_name, tree in execution_order.items():
            edges = []
            self._walk(tree, parent=None, edges=edges)
            self.sets[set_name] = edges
        self._connected = False

    def _walk(self, node, parent, edges):
        """Adds the edges of `node`, attached after `parent`. Returns the chain's last module."""
        if isinstance(node, dict):  # branches: every key starts from the same parent
            for module, rest in node.items():
                if parent is not None:
                    edges.append((parent, module))
                self._walk(rest, module, edges)
            return parent
        if isinstance(
            node, (list, tuple)
        ):  # chain: each element after the previous one
            last = parent
            for item in node:
                if isinstance(item, dict):
                    self._walk(item, last, edges)  # branch from the previous element
                else:
                    if last is not None:
                        edges.append((last, item))
                    last = item
            return last
        if node is None:
            return parent
        if parent is not None:  # a single module
            edges.append((parent, node))
        return node

    def connect(self):
        """Subscribes every edge (sender.subscribe(receiver)). Call once, before run()."""
        if self._connected:
            return
        for edges in self.sets.values():
            for sender, receiver in edges:
                sender.subscribe(receiver)
        self._connected = True

    def _modules(self, set_name):
        """Modules of a set, senders before receivers (topological order)."""
        edges = self.sets[set_name]
        modules = []
        for a, b in edges:
            for m in (a, b):
                if m not in modules:
                    modules.append(m)
        incoming = {m: 0 for m in modules}
        for _, b in edges:
            incoming[b] += 1
        order = [m for m in modules if incoming[m] == 0]  # sources (mics) first
        i = 0
        while i < len(order):
            for a, b in edges:
                if a is order[i]:
                    incoming[b] -= 1
                    if incoming[b] == 0:
                        order.append(b)
            i += 1
        return order

    def _names(self, set_name):
        return [set_name] if set_name is not None else list(self.sets)

    def run(self, set_name=None):
        """Starts a set (or all): receivers first, sources last."""
        self.connect()
        for name in self._names(set_name):
            for m in reversed(self._modules(name)):
                if not m._is_running:
                    m.run()

    def stop(self, set_name=None):
        """Stops a set (or all): sources first, so nothing new enters."""
        for name in self._names(set_name):
            for m in self._modules(name):
                m.stop(clear_buffer=False)

    def show(self):
        for name, edges in self.sets.items():
            print(f"[{name}]")
            for a, b in edges:
                print(f"  {a.name()}  →  {b.name()}")
