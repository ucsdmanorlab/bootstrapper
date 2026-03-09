import sys
import time
import random
import urllib.request
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Union, Set, Tuple

import neuroglancer
import numpy as np
import zarr
import torch
from skimage import color, util
from skimage.measure import label
from tqdm import tqdm
from funlib.persistence import open_ds

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path.home() / ".cache/active_learning"
SAM_CHECKPOINTS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
DEFAULT_MODEL = "vit_l"

def get_weights_path(model_type: str) -> Path:
    """Ensures model weights are downloaded and returns the path."""
    url = SAM_CHECKPOINTS.get(model_type, SAM_CHECKPOINTS["vit_l"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = CACHE_DIR / Path(url).name

    if not weight_path.exists():
        print(f"Downloading {model_type} weights to {weight_path}...")
        try:
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=weight_path.name) as t:
                def report_hook(b, bsize, tsize): t.update(bsize)
                urllib.request.urlretrieve(url, weight_path, reporthook=report_hook)
        except Exception as e:
            print(f"Failed to download weights: {e}")
            if weight_path.exists(): weight_path.unlink()
            raise
    return weight_path

def fast_remap(array: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """
    Remaps values in a large array using a dictionary without creating 
    massive dense lookup tables (safe for random uint64 IDs).
    """
    if not mapping:
        return array
    
    # Extract keys and values
    from_ids = np.array(list(mapping.keys()), dtype=array.dtype)
    to_ids = np.array(list(mapping.values()), dtype=array.dtype)
    
    # Create a dense lookup only for the values present in the array to save memory
    unique_vals, inverse_idx = np.unique(array, return_inverse=True)
    
    # Find which unique values need remapping
    # mask: boolean array where True indicates the unique value is in our map keys
    mask = np.isin(unique_vals, from_ids)
    
    if not np.any(mask):
        return array

    # Map the unique values
    # We use searchsorted to map 'from_ids' -> 'to_ids'
    # Sort from_ids for searchsorted
    sorter = np.argsort(from_ids)
    from_ids_sorted = from_ids[sorter]
    to_ids_sorted = to_ids[sorter]
    
    # Get indices of the unique values in the sorted keys
    idx = np.searchsorted(from_ids_sorted, unique_vals[mask])
    
    # Update the unique values
    unique_vals[mask] = to_ids_sorted[idx]
    
    # Reconstruct the array
    return unique_vals[inverse_idx].reshape(array.shape)

class SAM:
    def __init__(self, image_dataset: str, model_type: str = DEFAULT_MODEL):
        self.data = open_ds(image_dataset, "r")
        print(f"Data Loaded: {self.data.shape} ({self.data.dtype})")
        
        self.voxel_size = self.data.voxel_size
        self.dimensions = neuroglancer.CoordinateSpace(
            names=self.data.axis_names, units=self.data.units, scales=self.voxel_size
        )

        print(f"Voxel size: {self.voxel_size}, Dimensions: {self.dimensions}")

        # Layers
        self.raw_vol = neuroglancer.LocalVolume(
            data=self.data, dimensions=self.dimensions, 
            voxel_offset=self.data.offset / self.voxel_size
        )
        self.inf_results = zarr.zeros(
            self.data.shape, chunks=self.data.chunk_shape, dtype=np.uint64
        )
        self.inf_volume = neuroglancer.LocalVolume(
            data=self.inf_results, dimensions=self.dimensions
        )

        # Viewer
        self.viewer = neuroglancer.Viewer()
        self._setup_viewer()
        self._setup_bindings()

        # Model (Lazy loaded on first use if preferred, but loading here for readiness)
        self.model_type = model_type
        self._load_model()

        # State tracking
        self.current_ids: List[int] = []
        self.current_slices: Dict[str, slice] = {}
        self.current_offset = None
        self.raw_crop = None

    def _load_model(self):
        print(f"Loading SAM ({self.model_type}) on {DEVICE}...")
        checkpoint = get_weights_path(self.model_type)
        self._sam = sam_model_registry[self.model_type](checkpoint)
        self._sam.to(DEVICE)
        self._predictor = SamPredictor(self._sam)
        self._mask_gen = SamAutomaticMaskGenerator(self._sam)
        print("Model loaded.")

    def _setup_viewer(self):
        with self.viewer.txn() as s:
            s.layers["image"] = neuroglancer.ImageLayer(source=self.raw_vol)
            s.layers["labels"] = neuroglancer.SegmentationLayer(source=self.inf_volume)
            s.layers["merge_split"] = neuroglancer.LocalAnnotationLayer(
                linked_segmentation_layer={"segments": "labels"},
                dimensions=self.dimensions,
                annotation_color="#FC1DF4",
            )
            s.layout = "yz"
            s.position = [x // 2 for x in self.data.shape]

    def _setup_bindings(self):
        actions = {
            "segment": self._segment,
            "merge_labels": self._merge_labels,
            "unmerge_labels": self._unmerge_labels,
            "filter_labels": self._filter_labels,
            "omit_labels": self._omit_labels,
            "write_data": self._write_data
        }
        
        for name, func in actions.items():
            self.viewer.actions.add(name, func)

        keys = {"s": "segment", "m": "merge_labels", "u": "unmerge_labels", 
                "f": "filter_labels", "o": "omit_labels", "w": "write_data"}
        
        with self.viewer.config_state.txn() as s:
            for k, v in keys.items():
                s.input_event_bindings.data_view[f"key{k}"] = v

    # --- Helpers for Annotation Parsing ---

    def _get_selected_ids(self, s) -> Set[int]:
        """Extracts segmentation IDs from Point and Line annotations."""
        segments = set()
        for anno in s.viewer_state.layers["merge_split"].annotations:
            # Linked segmentation layers auto-populate 'segments' for points/lines
            if hasattr(anno, 'segments') and anno.segments:
                # Flatten list of lists if necessary
                ids = [item for sublist in anno.segments for item in sublist]
                segments.update(ids)
        return segments

    def _get_spatial_mask(self, s, shape, offset_global) -> np.ndarray:
        """
        Generates a boolean mask based on AxisAlignedBoundingBox or Ellipsoid annotations.
        Returns None if no spatial annotations exist.
        """
        # offset_global is the coordinate of the top-left of the current crop in nm
        mask = np.zeros(shape, dtype=bool)
        has_spatial = False
        
        # Convert crop bounds to global coordinates
        crop_offset = np.array(offset_global)
        
        for anno in s.viewer_state.layers["merge_split"].annotations:
            if isinstance(anno, neuroglancer.AxisAlignedBoundingBoxAnnotation):
                has_spatial = True
                # Global coords
                pA = np.array(anno.point_a)
                pB = np.array(anno.point_b)
                mn = np.minimum(pA, pB)
                mx = np.maximum(pA, pB)

                # Convert to local crop coordinates (pixels)
                # (Global_Coord - Crop_Offset) / Voxel_Size
                local_min = np.maximum(0, np.floor((mn - crop_offset) / self.voxel_size)).astype(int)
                local_max = np.minimum(shape, np.ceil((mx - crop_offset) / self.voxel_size)).astype(int)

                # Apply to mask (Handle ZYX order carefully)
                # Neuroglancer points are typically XYZ. Numpy is ZYX.
                # Assuming data.axis_names are consistent.
                
                # Slicing: [z_min:z_max, y_min:y_max, x_min:x_max]
                # Assuming axis 0 is Z, 1 is Y, 2 is X based on layout 'yz'
                mask[local_min[0]:local_max[0], local_min[1]:local_max[1], local_min[2]:local_max[2]] = True

            # Todo: Add Ellipsoid logic if strictly needed (complex math for interior check)
        
        return mask if has_spatial else None

    # --- Actions ---

    def _segment(self, s):
        pos = s.viewer_state.position
        if pos is None: return

        # Configuration for the patch
        PATCH_SHAPE = np.array([3, 512, 512], dtype=int) 
        
        # Calculate bounds
        pos_arr = np.array(pos)
        spos = (pos_arr - PATCH_SHAPE // 2).astype(int)
        epos = spos + PATCH_SHAPE

        # Ensure bounds are valid
        for i in range(3):
            spos[i] = max(0, spos[i])
            epos[i] = min(self.data.shape[i], epos[i])

        # Slice definition (ZYX)
        raw_slice = tuple(slice(s, e) for s, e in zip(spos, epos))
        # Write only to the middle slice to avoid Z-fighting on overlap, or define a specific write/inference logic
        # Current logic: Inference uses slice 1 (middle), Writes to slice 1
        inf_write_slice = (slice(spos[0]+1, epos[0]-1), slice(spos[1], epos[1]), slice(spos[2], epos[2]))
        
        self.current_slices = {'raw': raw_slice, 'inf': inf_write_slice}
        self.current_offset = [s * v for s, v in zip(spos, self.voxel_size)]

        # Load Data
        raw_data = self.data[raw_slice]
        if raw_data.shape[0] < 3: 
            print("Near boundary, not enough Z depth.")
            return

        self.raw_crop = raw_data.copy()
        
        # Preprocessing for SAM (RGB, normalized 0-255)
        # Taking middle slice for 2D SAM
        img_2d = raw_data[1] 
        img_rgb = color.gray2rgb(img_2d)
        if np.issubdtype(img_rgb.dtype, np.floating):
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-6)
        img_u8 = util.img_as_ubyte(img_rgb)

        # Inference
        self.current_ids = [] # Reset filter IDs
        
        # Determine name/path for saving later
        name = "_".join(map(str, map(int, pos)))
        self.zarr_path = f"training_crops/location_{name}.zarr"
        
        print(f"Predicting crop {name}...")
        start_t = time.time()
        
        preds = self._mask_gen.generate(img_u8)
        
        # Aggregate Masks
        # Start random ID base high enough to avoid conflicts with 0
        base_id = random.randint(100, 2**32) 
        
        # Efficiently combine masks
        # Note: overlapping masks sum up. We label connected components.
        combined_mask = np.zeros(img_2d.shape, dtype=np.uint64)
        for i, p in enumerate(preds):
            combined_mask += (p["segmentation"].astype(np.uint64) * (i + 1))

        # Relabel to unique IDs
        final_labels = label(combined_mask).astype(np.uint64)
        final_labels[final_labels > 0] += base_id

        print(f"Inference done in {time.time()-start_t:.2f}s")

        # Update Viewer
        # Only update the middle slice of the inference volume
        self.inf_results[inf_write_slice] = np.expand_dims(final_labels, axis=0)
        self.inf_volume.invalidate()

    def _merge_labels(self, s):
        segments = self._get_selected_ids(s)
        if not segments: return

        target_id = max(segments)
        with self.viewer.txn() as txn:
            for seg in segments:
                if seg != target_id:
                    txn.layers["labels"].equivalences.union(target_id, seg)
            
            # Clear point annotations used for merge
            txn.layers["merge_split"].annotations = [
                a for a in txn.layers["merge_split"].annotations 
                if not hasattr(a, 'segments') # Keep boxes/ellipsoids
            ]

    def _unmerge_labels(self, s):
        # Unmerge is tricky; usually done via graph split. 
        # NG default 'isolate_element' works if an equivalence exists.
        with self.viewer.txn() as txn:
            eq = txn.layers["labels"].equivalences
            try:
                # Logic depends on what is selected in the UI side panel
                if s.selected_values and "labels" in s.selected_values:
                    val = s.selected_values["labels"].value
                    key = val.key if hasattr(val, 'key') else val
                    eq.isolate_element(key)
            except Exception as e:
                print(f"Unmerge failed (select a segment first): {e}")

    def _omit_labels(self, s):
        """Sets selected segments or spatial regions to 0."""
        # 1. ID based omit
        segments_to_remove = self._get_selected_ids(s)
        
        # 2. Spatial based omit
        spatial_mask = None
        if 'inf' in self.current_slices:
            spatial_mask = self._get_spatial_mask(
                s, 
                shape=self.inf_results[self.current_slices['inf']].shape, 
                offset_global=self.current_offset
            )

        if not segments_to_remove and spatial_mask is None:
            return

        if 'inf' in self.current_slices:
            sl = self.current_slices['inf']
            data_crop = self.inf_results[sl].copy()
            
            if segments_to_remove:
                print(f"Omitting IDs: {segments_to_remove}")
                mask = np.isin(data_crop, list(segments_to_remove))
                data_crop[mask] = 0
            
            if spatial_mask is not None:
                print("Omitting spatial region")
                data_crop[spatial_mask] = 0

            self.inf_results[sl] = data_crop
            self.inf_volume.invalidate()

        # Clear annotations
        with self.viewer.txn() as txn:
            txn.layers["merge_split"].annotations = []

    def _filter_labels(self, s):
        """Keeps ONLY the selected segments or spatial regions."""
        segments_to_keep = self._get_selected_ids(s)
        
        spatial_mask = None
        if 'inf' in self.current_slices:
            spatial_mask = self._get_spatial_mask(
                s, 
                shape=self.inf_results[self.current_slices['inf']].shape, 
                offset_global=self.current_offset
            )
            
        if not segments_to_keep and spatial_mask is None:
            return

        # Update the list of "Approved" IDs for writing
        self.current_ids = list(segments_to_keep)

        # Visual update: Zero out everything else in the viewer? 
        # Usually better to just store the filter list for `write_data` 
        # so the user can change their mind without losing data immediately.
        # But if you want visual feedback:
        if 'inf' in self.current_slices:
            sl = self.current_slices['inf']
            data_crop = self.inf_results[sl].copy()
            
            keep_mask = np.zeros_like(data_crop, dtype=bool)
            
            if segments_to_keep:
                keep_mask |= np.isin(data_crop, list(segments_to_keep))
            
            if spatial_mask is not None:
                keep_mask |= spatial_mask

            data_crop[~keep_mask] = 0
            self.inf_results[sl] = data_crop
            self.inf_volume.invalidate()

        # Clear annotations
        with self.viewer.txn() as txn:
             txn.layers["merge_split"].annotations = []

    def _write_data(self, s):
        if not hasattr(self, 'zarr_path') or 'inf' not in self.current_slices:
            print("No segmentation active. Press 's' first.")
            return

        print("Saving data...")
        
        # 1. Get current mappings from Neuroglancer (user merges)
        with self.viewer.txn() as txn:
            eq = txn.layers["labels"].equivalences
            # Accessing internal map is risky but standard in NG python scripts
            # mapping: child -> parent
            mapping = {k: v for k, v in eq._parents.items() if k != v}

        # 2. Get data
        labels = self.inf_results[self.current_slices['inf']].copy()

        # 3. Apply ID filtering (if specific IDs were selected via filter_labels)
        if self.current_ids:
            # If we have a mapping, ensure we include the parents of our selected IDs
            valid_ids = set(self.current_ids)
            if mapping:
                # Add the parent IDs to valid list
                mapped_parents = {mapping.get(i, i) for i in valid_ids}
                valid_ids.update(mapped_parents)
            
            mask = np.isin(labels, list(valid_ids))
            labels[~mask] = 0

        # 4. Apply Equivalences (Merge)
        labels = fast_remap(labels, mapping)

        # 5. Save to Zarr
        try:
            container = zarr.open(self.zarr_path, "a")
            unlabelled = (labels > 0).astype(np.uint8)
            
            datasets = [
                ("raw", self.raw_crop, self.current_offset),
                ("labels", labels, [self.current_offset[0] + self.voxel_size[0], *self.current_offset[1:]]),
                ("unlabelled", unlabelled, [self.current_offset[0] + self.voxel_size[0], *self.current_offset[1:]])
            ]

            for name, data, offset in datasets:
                ds = container.require_dataset(name, shape=data.shape, dtype=data.dtype, overwrite=True)
                ds[:] = data
                ds.attrs.update({
                    "offset": offset,
                    "voxel_size": self.voxel_size,
                    "axis_names": ["z", "y", "x"],
                    "units": ["nm", "nm", "nm"]
                })
            
            print(f"Successfully wrote to {self.zarr_path}")

            # Cleanup Viewer
            with self.viewer.txn() as txn:
                txn.layers["labels"].equivalences.clear()
            
            self.inf_results[self.current_slices['inf']] = 0
            self.inf_volume.invalidate()
            
        except Exception as e:
            print(f"Failed to write data: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_zarr_image_dataset>")
        sys.exit(1)
        
    img_dataset = sys.argv[1]
    sam = SAM(img_dataset)
    print(sam.viewer)
