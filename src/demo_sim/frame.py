"""Small planar-frame finite-element engine used by all synthetic stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from . import constants as C


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float


@dataclass(frozen=True)
class Element:
    id: int
    i: int
    j: int
    kind: str
    storey: int
    bay: int
    area: float
    inertia: float


class FrameModel:
    """Linear 2D Euler-Bernoulli frame with three DOFs per node."""

    def __init__(self, nodes: list[Node], elements: list[Element]):
        self.nodes = nodes
        self.elements = elements
        self.ndof = 3 * len(nodes)
        self.fixed = np.array([3 * n.id + d for n in nodes if n.y == 0.0 for d in range(3)], dtype=int)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.mass = self._mass_matrix()

    @classmethod
    def regular_frame(cls) -> "FrameModel":
        nodes: list[Node] = []
        for s in range(C.N_STOREYS + 1):
            for b in range(C.N_BAYS + 1):
                nodes.append(Node(len(nodes), b * C.BAY_WIDTH, s * C.STOREY_HEIGHT))
        elements: list[Element] = []
        # Columns by storey, then beams by storey: 12 + 9 = 21 members.
        for s in range(1, C.N_STOREYS + 1):
            for b in range(C.N_BAYS + 1):
                i = (s - 1) * (C.N_BAYS + 1) + b
                j = s * (C.N_BAYS + 1) + b
                elements.append(Element(len(elements), i, j, "column", s, b, C.COLUMN_AREA, C.COLUMN_INERTIA))
        for s in range(1, C.N_STOREYS + 1):
            for b in range(C.N_BAYS):
                i = s * (C.N_BAYS + 1) + b
                j = i + 1
                elements.append(Element(len(elements), i, j, "beam", s, b, C.BEAM_AREA, C.BEAM_INERTIA))
        return cls(nodes, elements)

    def _mass_matrix(self) -> np.ndarray:
        m = np.zeros((self.ndof, self.ndof))
        for node in self.nodes:
            if node.y > 0.0:
                m[3 * node.id, 3 * node.id] = C.FLOOR_NODE_MASS
                m[3 * node.id + 1, 3 * node.id + 1] = C.FLOOR_NODE_MASS
                m[3 * node.id + 2, 3 * node.id + 2] = C.ROTATIONAL_MASS
        return m

    def _element_matrices(self, element: Element, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ni, nj = self.nodes[element.i], self.nodes[element.j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        length = float(np.hypot(dx, dy))
        c, s = dx / length, dy / length
        ea = C.E_CONCRETE * alpha * element.area
        ei = C.E_CONCRETE * alpha * element.inertia
        local = np.array([
            [ea / length, 0, 0, -ea / length, 0, 0],
            [0, 12 * ei / length**3, 6 * ei / length**2, 0, -12 * ei / length**3, 6 * ei / length**2],
            [0, 6 * ei / length**2, 4 * ei / length, 0, -6 * ei / length**2, 2 * ei / length],
            [-ea / length, 0, 0, ea / length, 0, 0],
            [0, -12 * ei / length**3, -6 * ei / length**2, 0, 12 * ei / length**3, -6 * ei / length**2],
            [0, 6 * ei / length**2, 2 * ei / length, 0, -6 * ei / length**2, 4 * ei / length],
        ])
        transform = np.array([
            [c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1],
        ])
        dofs = np.array([3 * element.i + d for d in range(3)] + [3 * element.j + d for d in range(3)])
        return transform.T @ local @ transform, local, dofs

    def stiffness(self, alpha: np.ndarray) -> np.ndarray:
        k = np.zeros((self.ndof, self.ndof))
        for element, a in zip(self.elements, alpha):
            kg, _, dofs = self._element_matrices(element, float(a))
            k[np.ix_(dofs, dofs)] += kg
        return k

    def modal(self, alpha: np.ndarray, n_modes: int = C.N_MODES) -> tuple[np.ndarray, np.ndarray]:
        k = self.stiffness(alpha)[np.ix_(self.free, self.free)]
        m = self.mass[np.ix_(self.free, self.free)]
        vals, vecs = eigh(k, m, subset_by_index=[0, n_modes - 1])
        freq = np.sqrt(np.maximum(vals, 0.0)) / (2.0 * np.pi)
        full = np.zeros((self.ndof, n_modes))
        full[self.free, :] = vecs
        return freq, full

    def static_response(self, alpha: np.ndarray, storey_force: float = C.DESIGN_STOREY_FORCE) -> tuple[np.ndarray, np.ndarray]:
        force = np.zeros(self.ndof)
        for s in range(1, C.N_STOREYS + 1):
            floor_nodes = [n for n in self.nodes if np.isclose(n.y, s * C.STOREY_HEIGHT)]
            for node in floor_nodes:
                force[3 * node.id] = storey_force * s / len(floor_nodes)
        k = self.stiffness(alpha)
        disp = np.zeros(self.ndof)
        disp[self.free] = np.linalg.solve(k[np.ix_(self.free, self.free)], force[self.free])
        moments = np.zeros(len(self.elements))
        for element, a in zip(self.elements, alpha):
            kg, local, dofs = self._element_matrices(element, float(a))
            del kg
            ni, nj = self.nodes[element.i], self.nodes[element.j]
            dx, dy = nj.x - ni.x, nj.y - ni.y
            length = float(np.hypot(dx, dy))
            c, s = dx / length, dy / length
            transform = np.array([
                [c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1],
            ])
            end_forces = local @ (transform @ disp[dofs])
            moments[element.id] = max(abs(end_forces[2]), abs(end_forces[5]))
        return disp, moments

    def sensor_dofs(self) -> np.ndarray:
        # Six lateral sensors: two nodes on each elevated floor, far fewer than 21 members.
        bays = C.SENSOR_BAYS
        node_ids = [s * (C.N_BAYS + 1) + b for s in range(1, C.N_STOREYS + 1) for b in bays]
        return np.array([3 * node_id for node_id in node_ids], dtype=int)

    def member_table(self):
        import pandas as pd
        return pd.DataFrame([{
            "member_id": e.id,
            "member": f"M{e.id:02d}",
            "kind": e.kind,
            "storey": e.storey,
            "bay": e.bay,
            "node_i": e.i,
            "node_j": e.j,
        } for e in self.elements])
