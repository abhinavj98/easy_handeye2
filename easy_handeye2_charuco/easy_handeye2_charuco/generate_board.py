#!/usr/bin/env python3
"""Render the configured ChArUco board to a PNG at true scale for printing.

Board geometry defaults to config/charuco_board.yaml. Print at 100% (no
"fit to page"), then measure a square with calipers: it must equal
square_length, or update the yaml to what you measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

from easy_handeye2_charuco.board import make_board

_M_PER_INCH = 0.0254


def _default_config() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('easy_handeye2_charuco')) / 'config' / 'charuco_board.yaml'
    except Exception:  # noqa: BLE001 - not installed; fall back to source tree
        return Path(__file__).resolve().parents[1] / 'config' / 'charuco_board.yaml'


def load_board_params(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)['charuco_tf_publisher']['ros__parameters']


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=_default_config())
    p.add_argument('--out', type=Path, default=Path('charuco_board.png'))
    p.add_argument('--dpi', type=int, default=600)
    p.add_argument('--margin-mm', type=float, default=5.0, help='white border around the board')
    for key, typ in (('squares_x', int), ('squares_y', int), ('square_length', float),
                     ('marker_length', float), ('dictionary', str)):
        p.add_argument(f'--{key.replace("_", "-")}', dest=key, type=typ, default=None,
                       help='override the config value')
    return p.parse_args(argv)


def main(argv=None):
    cli = _parse_args(sys.argv[1:] if argv is None else argv)
    params = load_board_params(cli.config)
    for key in ('squares_x', 'squares_y', 'square_length', 'marker_length', 'dictionary'):
        if getattr(cli, key) is not None:
            params[key] = getattr(cli, key)

    board, _ = make_board(int(params['squares_x']), int(params['squares_y']),
                          float(params['square_length']), float(params['marker_length']),
                          str(params['dictionary']))
    px_per_m = cli.dpi / _M_PER_INCH
    width_m = params['squares_x'] * params['square_length'] + 2 * cli.margin_mm / 1000.0
    height_m = params['squares_y'] * params['square_length'] + 2 * cli.margin_mm / 1000.0
    size = (round(width_m * px_per_m), round(height_m * px_per_m))
    margin_px = round(cli.margin_mm / 1000.0 * px_per_m)
    img = board.generateImage(size, marginSize=margin_px, borderBits=1)
    try:  # embed DPI so print dialogs default to true scale
        from PIL import Image
        Image.fromarray(img).save(cli.out, dpi=(cli.dpi, cli.dpi))
    except ImportError:
        cv2.imwrite(str(cli.out), img)
        print(f'(Pillow not installed: PNG has no DPI tag; set {cli.dpi} dpi when printing)')

    print(f'Wrote {cli.out} ({size[0]}x{size[1]} px @ {cli.dpi} dpi = '
          f'{width_m * 1000:.1f} x {height_m * 1000:.1f} mm incl. {cli.margin_mm} mm margin)')
    print(f'Board: {params["squares_x"]}x{params["squares_y"]} {params["dictionary"]}, '
          f'square {params["square_length"] * 1000:.1f} mm, marker {params["marker_length"] * 1000:.1f} mm')
    print('Print at 100% scale and verify one square with calipers.')


if __name__ == '__main__':
    main()
