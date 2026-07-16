#!/usr/bin/env python3
"""
Generate detailed natural language descriptions from CAD action sequences.
Mimics annotation style of Text2CAD / DeepCAD benchmarks.
"""

import json
import os
import csv
import random
import math
from pathlib import Path
from typing import Dict, List


class ActionToTextConverter:
    def __init__(self, seed=42):
        random.seed(seed)

        # Templates for each action type. Several variants for diversity.
        self.templates = {
            "CreateSketch": [
                "Start by creating a sketch on the plane with normal [{normal}] and origin [{origin}]. The profile consists of: {profile}.",
                "Draw a sketch on the plane defined by normal [{normal}] at origin [{origin}]. The shape is composed of: {profile}.",
                "Sketch the following profile on the plane normal [{normal}] at [{origin}]: {profile}.",
                "On the plane with normal [{normal}] and passing through [{origin}], draw these elements: {profile}."
            ],
            "Extrude": [
                "Extrude the sketch by {distance} units in direction [{direction}] using {operation} operation.",
                "Pull the profile a distance of {distance} along [{direction}] ({operation}).",
                "Extrude the previous sketch {distance} mm along [{direction}] (operation: {operation}).",
                "Create a {operation} extrusion of length {distance} in the direction [{direction}]."
            ],
            "Revolve": [
                "Revolve the sketch around axis [{axis}] passing through [{origin}] by {angle} degrees ({operation}).",
                "Revolve the profile {angle}° around [{axis}] at [{origin}] using {operation}.",
            ],
            "AddHole": [
                "Add a hole of diameter {diameter} centered at [{center}], depth {depth}, along [{axis}].",
                "Drill a hole with radius {radius} at location [{center}], depth {depth}.",
                "Place a hole at [{center}] with diameter {diameter} and depth {depth} along axis [{axis}].",
            ],
            "Fillet": [
                "Apply a fillet of radius {radius} to all applicable edges.",
                "Fillet the edges with radius {radius}.",
                "Round the sharp edges using a fillet radius of {radius}.",
            ],
            "CircularPattern": [
                "Create a circular pattern of the previous features: {count} instances around axis [{axis}] through [{center}].",
                "Duplicate the previous features in a circular pattern with {count} copies, axis [{axis}], center [{center}].",
            ],
            # Add more as needed (LinearPattern, Chamfer, Boss, Cut, etc.)
        }

        # Fallback for unknown actions
        self.fallback_template = "Perform a {action} operation with parameters: {params}"

    def describe_profile(self, profile: List[Dict]) -> List[str]:
        """Convert a 2D profile (list of primitives) into a list of descriptive phrases."""
        parts = []
        for prim in profile:
            ptype = prim.get("type")
            if ptype == "line":
                parts.append(f"a line from {prim['start']} to {prim['end']}")
            elif ptype == "arc":
                parts.append(
                    f"an arc with center {prim.get('center')}, radius {prim.get('radius')}, "
                    f"from angle {prim.get('start_angle', 0):.2f} to {prim.get('end_angle', math.pi):.2f}"
                )
            elif ptype == "circle":
                parts.append(f"a circle with center {prim['center']}, radius {prim['radius']}")
            elif ptype == "rectangle":
                parts.append(f"a rectangle of width {prim.get('width')} and height {prim.get('height')}")
            else:
                parts.append(f"a {ptype} primitive: {json.dumps(prim)}")
        return parts

    def format_action(self, action: Dict) -> str:
        """Produce a natural language sentence for one action."""
        action_name = action["action"]
        params = action.get("params", {})

        # Choose a random template from the available ones
        templates = self.templates.get(action_name)
        if templates is None:
            return self.fallback_template.format(action=action_name, params=json.dumps(params))

        template = random.choice(templates)

        # Build a dict of values to substitute
        subs = {}
        for key, val in params.items():
            if isinstance(val, (list, tuple)):
                # Format as rounded numbers within brackets
                formatted = "[" + ", ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in val) + "]"
            elif isinstance(val, float):
                formatted = f"{val:.3f}"
            else:
                formatted = str(val)
            subs[key] = formatted

        # Special handling for "profile": convert list of primitives to text
        if "profile" in params:
            profile_desc = self.describe_profile(params["profile"])
            subs["profile"] = "; ".join(profile_desc)
        else:
            subs["profile"] = "a simple closed contour"

        # Map 'radius' to also provide 'diameter' for hole descriptions
        if 'radius' in params and 'diameter' not in subs:
            subs['diameter'] = f"{params['radius']*2:.3f}"

        # Replace placeholders
        try:
            sentence = template.format(**subs)
        except KeyError as e:
            # If some key missing, fallback
            sentence = self.fallback_template.format(action=action_name, params=json.dumps(params))
        return sentence

    def generate_description(self, actions: List[Dict]) -> str:
        """Generate a full description from a list of action dicts."""
        # Build description as a sequence of steps
        steps = []
        for act in actions:
            sentence = self.format_action(act)
            steps.append(sentence)

        # Join with a space; could also add paragraph breaks for readability
        description = " ".join(steps)
        return description


def process_actions_dir(actions_dir: str, output_csv: str):
    """Read all action JSON files, generate descriptions, save CSV."""
    converter = ActionToTextConverter()
    rows = []
    action_files = list(Path(actions_dir).glob("*_actions.json"))
    if not action_files:
        print(f"No action files found in {actions_dir}")
        return

    for afile in action_files:
        try:
            with open(afile, 'r') as f:
                actions = json.load(f)
            desc = converter.generate_description(actions)
            rows.append([desc, str(afile)])
        except Exception as e:
            print(f"Error processing {afile}: {e}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'action_file'])
        writer.writerows(rows)
    print(f"Saved {len(rows)} descriptions to {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions_dir", required=True, help="Directory with Phase 5 action JSON files")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV")
    args = parser.parse_args()
    process_actions_dir(args.actions_dir, args.output_csv)