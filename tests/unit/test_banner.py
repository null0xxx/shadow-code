"""Tests for banner.py -- startup wordmark, gradient, and tips block."""

import unittest

from shadow_code import banner
from shadow_code.theme import THEME


class TestBannerArt(unittest.TestCase):
    def test_ascii_art_is_bounded_and_ascii(self):
        self.assertLessEqual(len(banner.BANNER_ART), 8)
        self.assertGreaterEqual(len(banner.BANNER_ART), 4)
        for line in banner.BANNER_ART:
            self.assertTrue(line.isascii())
            self.assertEqual(len(line), len(line.rstrip()))  # no trailing spaces
        self.assertEqual(banner.BANNER_ASCII_WIDTH, max(len(line) for line in banner.BANNER_ART))

    def test_art_spells_shadow_code(self):
        # Every letter slot contributes ink to every row band.
        joined = "\n".join(banner.BANNER_ART)
        self.assertIn("#####", joined)  # H/E crossbars
        self.assertNotIn("\t", joined)


class TestBlockArt(unittest.TestCase):
    def test_block_art_bounds_and_charset(self):
        self.assertGreaterEqual(len(banner.BLOCK_ART), 5)
        self.assertLessEqual(len(banner.BLOCK_ART), 7)
        self.assertLessEqual(banner.BANNER_WIDTH, 96)
        self.assertEqual(banner.BANNER_WIDTH, max(len(line) for line in banner.BLOCK_ART))
        for line in banner.BLOCK_ART:
            self.assertLessEqual(set(line), {" ", banner.FULL_BLOCK, banner.SHADOW_CHAR})
            self.assertEqual(len(line), len(line.rstrip()))  # no trailing spaces

    def test_block_art_has_ink_and_shadow(self):
        # Six letter rows of full-block ink, then one shadow row beneath.
        for line in banner.BLOCK_ART[:-1]:
            self.assertIn(banner.FULL_BLOCK, line)
            self.assertNotIn(banner.SHADOW_CHAR, line)
        shadow = banner.BLOCK_ART[-1]
        self.assertIn(banner.SHADOW_CHAR, shadow)
        self.assertNotIn(banner.FULL_BLOCK, shadow)
        # The shadow is the bottom row's silhouette shifted one column right.
        self.assertEqual(
            shadow,
            " " + banner.BLOCK_ART[-2].replace(banner.FULL_BLOCK, banner.SHADOW_CHAR),
        )

    def test_narrow_terminal_falls_back_to_plain_title(self):
        self.assertEqual(
            banner.banner_lines(banner.BANNER_WIDTH - 1, ascii_mode=False), ["SHADOW CODE"]
        )
        self.assertEqual(
            banner.banner_lines(banner.BANNER_WIDTH, ascii_mode=False), list(banner.BLOCK_ART)
        )
        self.assertEqual(banner.banner_lines(None, ascii_mode=False), list(banner.BLOCK_ART))
        self.assertEqual(
            banner.banner_lines(banner.BANNER_ASCII_WIDTH - 1, ascii_mode=True),
            ["SHADOW CODE"],
        )
        self.assertEqual(banner.banner_lines(None, ascii_mode=True), list(banner.BANNER_ART))

    def test_ascii_mode_autodetects_no_color_and_shadow_ascii(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(banner.banner_lines(None), list(banner.BANNER_ART))
        with patch.dict(os.environ, {"SHADOW_ASCII": "1"}):
            self.assertEqual(banner.banner_lines(None), list(banner.BANNER_ART))


class TestTips(unittest.TestCase):
    def test_tips_block_content(self):
        lines = banner.tips_lines()
        self.assertEqual(lines[0], "Tips for getting started:")
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[1].startswith("  1. "))
        self.assertIn("/help", lines[3])


class TestGradient(unittest.TestCase):
    def test_endpoints_reuse_theme_palette(self):
        self.assertEqual(banner.gradient_color(0.0), THEME.accent)
        self.assertEqual(banner.gradient_color(1.0), THEME.highlight)

    def test_midpoint_approaches_info(self):
        self.assertEqual(banner.gradient_color(0.5), THEME.info)

    def test_clamps_out_of_range(self):
        self.assertEqual(banner.gradient_color(-1), THEME.accent)
        self.assertEqual(banner.gradient_color(2), THEME.highlight)

    def test_monotonic_hue_drift(self):
        # A gradient, not a flat fill: many distinct colors across the width.
        colors = {banner.column_color(i) for i in range(banner.BANNER_WIDTH)}
        self.assertGreater(len(colors), banner.BANNER_WIDTH // 3)


class TestStyledFragments(unittest.TestCase):
    def test_banner_fragments_gradient_when_colored(self):
        fragments = banner.styled_banner_fragments(colors=True, ascii_mode=False)
        hex_styles = {style for style, _text in fragments if style.startswith("#")}
        self.assertGreater(len(hex_styles), 10)
        # Spaces never carry a style (terminal background shows through).
        for style, text in fragments:
            if text.strip() == "":
                self.assertEqual(style, "")

    def test_banner_fragments_shadow_is_muted(self):
        fragments = banner.styled_banner_fragments(colors=True, ascii_mode=False)
        shadow_fragments = [(s, t) for s, t in fragments if t == banner.SHADOW_CHAR]
        self.assertTrue(shadow_fragments)
        for style, _text in shadow_fragments:
            self.assertEqual(style, THEME.text_muted)
        # Ink takes the gradient instead.
        ink_styles = {s for s, t in fragments if t == banner.FULL_BLOCK}
        self.assertTrue(all(style.startswith("#") for style in ink_styles))

    def test_banner_fragments_ascii_mode_uses_hash_art(self):
        fragments = banner.styled_banner_fragments(colors=True, ascii_mode=True)
        plain = "".join(text for _style, text in fragments)
        self.assertEqual(plain, "\n".join(banner.BANNER_ART))
        self.assertNotIn(banner.FULL_BLOCK, plain)

    def test_banner_fragments_plain_without_colors(self):
        fragments = banner.styled_banner_fragments(colors=False, ascii_mode=False)
        self.assertTrue(all(style == "" for style, _text in fragments))
        plain = "".join(text for _style, text in fragments)
        self.assertEqual(plain, "\n".join(banner.BLOCK_ART))

    def test_banner_fragments_narrow_fallback(self):
        fragments = banner.styled_banner_fragments(colors=True, width=10)
        plain = "".join(text for _style, text in fragments)
        self.assertEqual(plain, "SHADOW CODE")

    def test_tips_fragments_accent_help_when_colored(self):
        fragments = banner.styled_tips_fragments(colors=True)
        self.assertIn(("class:tips-accent", "/help"), fragments)
        plain = "".join(text for _style, text in fragments)
        self.assertEqual(plain, "\n".join(banner.tips_lines()))

    def test_tips_fragments_plain_without_colors(self):
        fragments = banner.styled_tips_fragments(colors=False)
        self.assertTrue(all(style == "" for style, _text in fragments))


if __name__ == "__main__":
    unittest.main()
