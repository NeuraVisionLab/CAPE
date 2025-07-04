
import argparse
from pathlib import Path
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle 
from typing import Union
from skimage.morphology import skeletonize

from utils.graph_from_skeleton_2D import graph_from_skeleton as graph_from_skeleton_2D
from utils.graph_from_skeleton_3D import graph_from_skeleton as graph_from_skeleton_3D

def write_gpickle(G: nx.Graph, path: Union[str, Path]) -> None:
    """Save a NetworkX graph using Python pickle."""
    path = Path(path)                     
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

def read_gpickle(path: Union[str, Path]) -> nx.Graph:
    """Load a graph saved with write_gpickle()."""
    path = Path(path)                      
    with open(path, "rb") as fh:
        return pickle.load(fh)
    
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path, help="Folder with .npy masks")
    p.add_argument("--dim", type=int, choices=[2, 3], default=2,
                   help="Image dimensionality")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Threshold for binarising the mask")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Destination for .gpickle files "
                        "(default: data_as_graph)")
    return p.parse_args()


def to_graph(mask: np.ndarray, three_d: bool):
    
    skeleton = skeletonize(mask)
    return graph_from_skeleton_3D(skeleton, angle_range=(175,185), verbose=False) if three_d else graph_from_skeleton_2D(skeleton, angle_range=(175,185), verbose=False)

def main():
    args = parse_args()
    out_dir = args.out_dir or Path("data_as_graph")
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(sorted(args.folder.glob("*.npy")), desc="Processing images"):
        mask = np.load(f).astype(np.float32)
        if mask.max() > 1:
            mask /= 255.0
        mask = (mask >= args.threshold).astype(np.uint8)

        G = to_graph(mask, args.dim == 3)
        
        write_gpickle(G, out_dir / f"{f.stem}.gpickle")

    print(f"Graphs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()