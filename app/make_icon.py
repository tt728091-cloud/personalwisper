"""Render the app icon (a white waveform on a blue rounded square) into an .iconset folder.

Usage: .venv/bin/python app/make_icon.py <output.iconset>
"""
import sys
from pathlib import Path

from AppKit import (NSBezierPath, NSBitmapImageRep, NSColor, NSGradient, NSGraphicsContext, NSImage,
                    NSFontWeightBold, NSImageSymbolConfiguration, NSMakeRect, NSPNGFileType)

DESTINATION_IN = 7  # NSCompositingOperationDestinationIn


def render(size):
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, "NSDeviceRGBColorSpace", 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))

    # macOS icon grid: the rounded square fills ~80% of the canvas.
    inset = size * 0.1
    rect = NSMakeRect(inset, inset, size - 2 * inset, size - 2 * inset)
    shape = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, size * 0.18, size * 0.18)
    gradient = NSGradient.alloc().initWithStartingColor_endingColor_(
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.35, 0.55, 1.0, 1.0),
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.15, 0.25, 0.85, 1.0))
    gradient.drawInBezierPath_angle_(shape, -90)

    config = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size * 0.42, NSFontWeightBold)
    symbol = NSImage.imageWithSystemSymbolName_accessibilityDescription_("waveform", None)
    symbol = symbol.imageWithSymbolConfiguration_(config)

    # Fill white, then keep only the pixels the symbol covers, giving a white symbol.
    w, h = symbol.size().width, symbol.size().height
    white = NSImage.alloc().initWithSize_((w, h))
    white.lockFocus()
    NSColor.whiteColor().set()
    NSBezierPath.fillRect_(NSMakeRect(0, 0, w, h))
    symbol.drawAtPoint_fromRect_operation_fraction_((0, 0), NSMakeRect(0, 0, 0, 0), DESTINATION_IN, 1.0)
    white.unlockFocus()
    white.drawInRect_(NSMakeRect((size - w) / 2, (size - h) / 2, w, h))

    NSGraphicsContext.restoreGraphicsState()
    return rep.representationUsingType_properties_(NSPNGFileType, {})


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for points in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            name = f"icon_{points}x{points}{'@2x' if scale == 2 else ''}.png"
            render(points * scale).writeToFile_atomically_(str(out / name), True)


if __name__ == "__main__":
    main(sys.argv[1])
