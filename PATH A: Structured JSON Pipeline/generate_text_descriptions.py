#!/usr/bin/env python3
"""
PHASE 4: Text Description Generator
Creates natural language descriptions from Phase 1 JSON files (AP214 format).
Outputs a CSV file: [text_description, graph_path]
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class TextDescriptionGenerator:
    """Generate template-based natural language descriptions from CAD geometry data."""

    def __init__(self, include_dimensions: bool = True, include_constraints: bool = False):
        """
        Args:
            include_dimensions: Whether to include approximate bounding box dimensions.
            include_constraints: Whether to mention detected constraints (if present).
        """
        self.include_dimensions = include_dimensions
        self.include_constraints = include_constraints

        # Map surface types to human-readable names
        self.surface_type_names = {
            'PLANE': 'planar',
            'PLANE_SIMPLE': 'planar',
            'CYLINDER': 'cylindrical',
            'CYLINDER_SIMPLE': 'cylindrical',
            'CONE': 'conical',
            'CONE_SIMPLE': 'conical',
            'SPHERE': 'spherical',
            'SPHERE_SIMPLE': 'spherical',
            'SPLINE_SURFACE': 'freeform',
            'TORUS': 'toroidal',
            'PROCESSING_ERROR': 'undefined',
            'UNKNOWN': 'undefined'
        }

        self.curve_type_names = {
            'LINE': 'straight',
            'LINE_SIMPLE': 'straight',
            'CIRCLE': 'circular',
            'CIRCLE_SIMPLE': 'circular',
            'ELLIPSE': 'elliptical',
            'ELLIPSE_SIMPLE': 'elliptical',
            'SPLINE': 'spline',
            'PROCESSING_ERROR': 'undefined',
            'UNKNOWN_CURVE': 'undefined'
        }

    def load_json(self, json_path: str) -> Dict[str, Any]:
        """Load Phase 1 JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)

    def compute_bounding_box(self, vertices: List[Dict]) -> Tuple[float, float, float]:
        """Compute approximate dimensions from vertex positions."""
        if not vertices:
            return 0.0, 0.0, 0.0

        xs = [v['point'][0] for v in vertices]
        ys = [v['point'][1] for v in vertices]
        zs = [v['point'][2] for v in vertices]

        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        dz = max(zs) - min(zs)

        return dx, dy, dz

    def format_dimensions(self, dx: float, dy: float, dz: float) -> str:
        """Format dimensions into human-readable string."""
        # Round to 2 decimal places, remove trailing zeros
        def fmt(v):
            return f"{v:.2f}".rstrip('0').rstrip('.')

        return f"{fmt(dx)} × {fmt(dy)} × {fmt(dz)} units"

    def count_surface_types(self, faces: List[Dict]) -> Dict[str, int]:
        """Count occurrences of each surface type category."""
        counts = {
            'planar': 0,
            'cylindrical': 0,
            'conical': 0,
            'spherical': 0,
            'freeform': 0,
            'other': 0
        }
        for face in faces:
            st = face.get('surface_type', 'UNKNOWN')
            base = st.split('_')[0] if '_' in st else st
            readable = self.surface_type_names.get(st, 'other')
            if readable in counts:
                counts[readable] += 1
            else:
                counts['other'] += 1
        return counts

    def count_edge_types(self, edges: List[Dict]) -> Dict[str, int]:
        """Count occurrences of edge curve types."""
        counts = {
            'straight': 0,
            'circular': 0,
            'elliptical': 0,
            'spline': 0,
            'other': 0
        }
        for edge in edges:
            ct = edge.get('curve_type', 'UNKNOWN_CURVE')
            readable = self.curve_type_names.get(ct, 'other')
            if readable in counts:
                counts[readable] += 1
            else:
                counts['other'] += 1
        return counts

    def identify_features(self, faces: List[Dict], edges: List[Dict]) -> List[str]:
        """Identify notable features like holes, fillets, etc."""
        features = []

        # Holes: cylindrical faces with small radius relative to model
        # (simplified heuristic)
        cyl_faces = [f for f in faces if f['surface_type'].startswith('CYLINDER')]
        if cyl_faces:
            # If many cylindrical faces, might be holes
            if len(cyl_faces) > 2:
                features.append(f"{len(cyl_faces)} holes/pockets")
            elif len(cyl_faces) > 0:
                features.append(f"cylindrical features")

        # Fillets/chamfers: small faces often connecting planar faces
        # (not robust, just for demonstration)
        if any(f['surface_type'] == 'SPLINE_SURFACE' for f in faces):
            features.append("blended edges")

        # Check for patterns (circular pattern of holes)
        # (very rough heuristic)
        if len(cyl_faces) >= 3:
            # Could be a circular pattern; check if centers lie on a circle
            centers = [f['centroid'] for f in cyl_faces]
            if len(centers) >= 3:
                # Placeholder: would need proper circle fitting
                features.append("pattern of holes")

        return features

    def generate_description(self, json_data: Dict[str, Any]) -> str:
        """Generate a natural language description from JSON data."""
        geometry = json_data.get('geometry_data', {})
        faces = geometry.get('faces', [])
        edges = geometry.get('edges', [])
        vertices = geometry.get('vertices', [])
        metadata = json_data.get('metadata', {})

        # Basic counts
        num_faces = len(faces)
        num_edges = len(edges)
        num_vertices = len(vertices)

        if num_faces == 0:
            return "A mechanical part with no detectable geometry."

        # Surface type distribution
        surf_counts = self.count_surface_types(faces)

        # Edge type distribution
        edge_counts = self.count_edge_types(edges)

        # Dimensions
        dims = ""
        if self.include_dimensions:
            dx, dy, dz = self.compute_bounding_box(vertices)
            if dx > 0 or dy > 0 or dz > 0:
                dims = f" Overall dimensions approximately {self.format_dimensions(dx, dy, dz)}."

        # Identify notable features
        features = self.identify_features(faces, edges)
        feature_str = ""
        if features:
            feature_str = " Features include: " + ", ".join(features) + "."

        # Build description
        desc_parts = []

        # Opening: what kind of part?
        if surf_counts['planar'] > 0.7 * num_faces:
            shape_desc = "prismatic mechanical part"
        elif surf_counts['cylindrical'] > 0.3 * num_faces:
            shape_desc = "mechanical part with cylindrical elements"
        elif surf_counts['freeform'] > 0:
            shape_desc = "freeform mechanical component"
        else:
            shape_desc = "mechanical part"

        desc_parts.append(f"A {shape_desc} with {num_faces} faces ({surf_counts['planar']} planar, "
                          f"{surf_counts['cylindrical']} cylindrical, {surf_counts['conical']} conical, "
                          f"{surf_counts['spherical']} spherical), {num_edges} edges, and {num_vertices} vertices.")

        if edge_counts['straight'] > 0 or edge_counts['circular'] > 0:
            desc_parts.append(f"The edges are primarily {edge_counts['straight']} straight and "
                              f"{edge_counts['circular']} circular.")

        if dims:
            desc_parts.append(dims.strip())

        if feature_str:
            desc_parts.append(feature_str.strip())

        # Optionally add constraint info if present in JSON (future Phase 3 integration)
        if self.include_constraints and 'constraints' in json_data:
            # Placeholder for constraint description
            pass

        # Combine parts
        description = " ".join(desc_parts)
        return description

    def process_directory(self, json_dir: str, graph_dir: str, output_csv: str):
        """
        Process all JSON files in a directory and generate CSV.

        Args:
            json_dir: Directory containing Phase 1 JSON files.
            graph_dir: Directory containing corresponding Phase 2 .pt graph files.
            output_csv: Path to output CSV file.
        """
        json_files = list(Path(json_dir).glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {json_dir}")
            return

        print(f"Found {len(json_files)} JSON files.")

        rows = []
        skipped = 0

        for json_path in json_files:
            try:
                data = self.load_json(json_path)
                description = self.generate_description(data)

                # Determine corresponding graph file path
                base_name = json_path.stem
                # The graph file may have suffix '_hybrid_graph' or similar; we'll match common patterns
                possible_graph_names = [
                    f"{base_name}.pt",
                    f"{base_name}_hybrid_graph.pt",
                    f"{base_name}_labeled_graph.pt"
                ]
                graph_path = None
                for name in possible_graph_names:
                    candidate = Path(graph_dir) / name
                    if candidate.exists():
                        graph_path = str(candidate)
                        break

                if graph_path is None:
                    print(f"Warning: No graph file found for {base_name}, skipping.")
                    skipped += 1
                    continue

                rows.append([description, graph_path])

            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                skipped += 1

        # Write CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'graph_path'])
            writer.writerows(rows)

        print(f"\nCSV saved to {output_csv}")
        print(f"Successfully generated {len(rows)} descriptions.")
        print(f"Skipped {skipped} files.")


def main():
    parser = argparse.ArgumentParser(description="Generate text descriptions from Phase 1 JSON files.")
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory containing Phase 1 JSON files.")
    parser.add_argument("--graph_dir", type=str, required=True,
                        help="Directory containing Phase 2 .pt graph files.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV file path.")
    parser.add_argument("--include_dimensions", action="store_true", default=True,
                        help="Include bounding box dimensions in description.")
    parser.add_argument("--include_constraints", action="store_true", default=False,
                        help="Include constraint information (if available).")

    args = parser.parse_args()

    generator = TextDescriptionGenerator(
        include_dimensions=args.include_dimensions,
        include_constraints=args.include_constraints
    )
    generator.process_directory(args.json_dir, args.graph_dir, args.output_csv)


if __name__ == "__main__":
    main()
