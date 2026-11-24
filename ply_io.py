"""Small PLY vertex reader/writer using NumPy; no mesh-library dependency."""
from pathlib import Path

import numpy as np

TYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}


def read_vertices(path: Path) -> tuple[np.ndarray, bool]:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not a PLY file: missing 'ply' header.")
        encoding = None
        elements = []
        current = None
        for _ in range(16_384):
            raw = stream.readline(65_537)
            if not raw or len(raw) > 65_536 or stream.tell() > 1_048_576:
                raise ValueError("PLY header is incomplete or too large.")
            line = raw.decode("ascii").strip().split()
            if not line or line[0] in {"comment", "obj_info"}:
                continue
            if line[0] == "end_header":
                break
            if line[0] == "format":
                if len(line) != 3 or line[2] != "1.0":
                    raise ValueError("Only PLY version 1.0 is supported.")
                encoding = line[1]
            elif line[0] == "element":
                if len(line) != 3 or int(line[2]) < 0:
                    raise ValueError("Invalid PLY element count.")
                current = {"name": line[1], "count": int(line[2]), "properties": []}
                elements.append(current)
            elif line[0] == "property":
                if current is None:
                    raise ValueError("PLY property without an element.")
                current["properties"].append(line[1:])
        else:
            raise ValueError("PLY header is too large.")
        if encoding not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise ValueError("Unsupported PLY encoding.")
        vertices = [e for e in elements if e["name"] == "vertex"]
        if len(vertices) != 1:
            raise ValueError("PLY must contain one vertex element.")
        vertex = vertices[0]
        if any(e["count"] for e in elements[:elements.index(vertex)]):
            raise ValueError("This reader requires the PLY vertex element before other nonempty elements. Export a point-cloud PLY first.")
        properties = []
        endian = ">" if encoding == "binary_big_endian" else "<"
        for prop in vertex["properties"]:
            if len(prop) != 2 or prop[0] not in TYPES:
                raise ValueError("PLY vertex properties must be scalar numbers; list-valued vertices are unsupported.")
            properties.append((prop[1], endian + TYPES[prop[0]]))
        if not {"x", "y", "z"} <= {name for name, _ in properties}:
            raise ValueError("PLY vertices need x, y and z properties.")
        if vertex["count"] == 0:
            raise ValueError("PLY contains no vertices.")
        dtype = np.dtype(properties)
        if encoding == "ascii":
            data = np.loadtxt(stream, dtype=dtype, max_rows=vertex["count"], comments=None, ndmin=1)
        else:
            remaining = path.stat().st_size - stream.tell()
            if vertex["count"] * dtype.itemsize > remaining:
                raise ValueError("PLY is truncated: fewer vertex bytes than declared.")
            data = np.fromfile(stream, dtype=dtype, count=vertex["count"])
        if len(data) != vertex["count"]:
            raise ValueError("PLY is truncated: fewer vertices than declared.")
        return data, any(e["name"] == "face" and e["count"] for e in elements)


def write_vertices(path: Path, vertices: np.ndarray) -> None:
    reverse = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}
    header = ["ply", "format binary_little_endian 1.0", "comment Hybrid Regrid Tk; regrid_generated=1 marks estimated points", f"element vertex {len(vertices)}"]
    fields = []
    for name in vertices.dtype.names:
        dt = vertices.dtype[name]
        code = dt.kind + str(dt.itemsize)
        header.append(f"property {reverse[code]} {name}")
        fields.append((name, "<" + code))
    header.append("end_header")
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        vertices.astype(np.dtype(fields), copy=False).tofile(stream)
